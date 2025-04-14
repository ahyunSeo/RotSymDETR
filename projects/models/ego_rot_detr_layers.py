# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import warnings
import torch
from torch import Tensor
import torch.nn as nn
from mmengine import ConfigDict
from typing import Dict, Tuple, List, Optional, Union
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.registry import MODELS
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention, FFN
from mmdet.models import (
                DeformableDetrTransformerEncoder, 
                DeformableDetrTransformerDecoder,
                DeformableDetrTransformerEncoderLayer,
                DeformableDetrTransformerDecoderLayer,)
from mmdet.structures import SampleList
from mmengine.structures import InstanceData
from mmdet.utils import InstanceList, reduce_mean
from mmdet.models import inverse_sigmoid
from .ego_rot_detr_attn import *


@MODELS.register_module()
class EgoFormerEncoder(DeformableDetrTransformerEncoder):
    def __init__(self, pc_range, num_points_in_pillar, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = False

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            EgoFormerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    # @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, focal_length, reference_points, pc_range, img_shape, batch_input_shape):
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]
            
  
        # reference_points_cam
        # h, w = img_metas[0]['img_shape']
        img_shape = torch.tensor(img_shape).cuda()
        batch_input_shape = torch.tensor(batch_input_shape).cuda()
        h, w = img_shape[..., 0], img_shape[..., 1]
        cx, cy = w/2, h/2
        h, w = batch_input_shape[..., 0], batch_input_shape[..., 1]
        # (bs, n_pillar, n_pts, 1)
        u = focal_length * reference_points[..., 0] / reference_points[..., 2]
        v = focal_length * reference_points[..., 1] / reference_points[..., 2]

        u = u + cx.view(-1, 1, 1)
        v = v + cy.view(-1, 1, 1)

        u = u / w.view(-1, 1, 1)
        v = v / h.view(-1, 1, 1)

        reference_points_cam = torch.stack((u, v), dim=-1)#.flatten(1, 2)
        reference_points_cam = reference_points_cam.unsqueeze(3) # n_cams=1

        image_mask = ((reference_points_cam[..., 1:2] > 0.0) & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0) & (reference_points_cam[..., 0:1] > 0.0))

        # (B, D, N, n_cam, 2) ->  [bs, self.num_cams, max_len, D, 2])
        image_mask = image_mask.permute(0, 3, 2, 1, 4)
        reference_points_cam = reference_points_cam.permute(0, 3, 2, 1, 4)

        image_mask = image_mask.squeeze(-1)

        return reference_points_cam, image_mask

    # @auto_fp16()
    def forward(self,
                ego_query,
                key,
                value,
                focal_length,
                *args,
                ego_h=None,
                ego_w=None,
                key_pos=None,
                img_shape=None,
                batch_input_shape=None,
                ego_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_ego=None,
                shift=0.,
                **kwargs):
        output = ego_query
        intermediate = []

        ref_3d = self.get_reference_points(
            ego_h, ego_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, 
            dim='3d', bs=ego_query.size(1),  device=ego_query.device, dtype=ego_query.dtype)
        ref_2d = self.get_reference_points(
            ego_h, ego_w, dim='2d', bs=ego_query.size(1), device=ego_query.device, dtype=ego_query.dtype)
        reference_points_cam, ego_mask = self.point_sampling(
            focal_length, ref_3d, self.pc_range, img_shape, batch_input_shape)

        ego_query = ego_query.permute(1, 0, 2)
        ego_pos = ego_pos.permute(1, 0, 2)

        bs, len_ego, num_ego_level, _ = ref_2d.shape
        for lid, layer in enumerate(self.layers):
            output = layer(
                ego_query,
                key,
                value,
                *args,
                ref_2d=ref_2d,
                ref_3d=ref_3d,
                ego_h=ego_h,
                ego_w=ego_w,
                level_start_index=level_start_index,
                ego_mask=ego_mask,
                ego_pos=ego_pos,
                key_pos=key_pos,
                spatial_shapes=spatial_shapes,
                reference_points_cam=reference_points_cam,
                **kwargs)

            ego_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class EgoFormerEncoderLayer(DeformableDetrTransformerDecoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = build_attention(self.self_attn_cfg)
        self.cross_attn = build_attention(self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query,
                key=None,
                value=None,
                ego_pos=None,
                ego_mask=None,
                ego_h=None,
                ego_w=None,
                ref_2d=None,
                ref_3d=None,
                key_pos=None,
                reference_points_cam=None,
                key_padding_mask=None,
                level_start_index=None,
                spatial_shapes=None,
                **kwargs):
        
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=ego_pos,
            key_pos=ego_pos,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor([[ego_h, ego_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            ego_mask=ego_mask,
            key_pos=key_pos,
            key_padding_mask=key_padding_mask,
            reference_points=ref_3d,
            reference_points_cam=reference_points_cam,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query

class HalfDeformableDetrTransformerDecoder(DeformableDetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""
    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            HalfDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            reference_points = reference_points.half().double()
            valid_ratios = valid_ratios.half().double()

            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            elif reference_points.shape[-1] == 3:
                reference_points_input = reference_points[..., :2].unsqueeze(-2)
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
            reference_points_input = reference_points_input.float()
            if value is not None:
                value = value.half().float()
            if query_pos is not None:
                query_pos = query_pos.half().float()
            
            output = layer(
                output.half().float(),
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios.half().float(),
                reference_points=reference_points_input.half().float(),
                **kwargs).half().float()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points

class HalfDeformableDetrTransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = HalfMultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
