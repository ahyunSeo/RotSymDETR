# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
from torch import Tensor, nn
from torch.nn.init import normal_
import torch.nn.functional as F
from typing import Dict, Tuple
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import SampleList, OptSampleList
from mmdet.registry import MODELS
from mmdet.utils import InstanceList
from mmdet.models import DeformableDETR, LearnedPositionalEncoding, SinePositionalEncoding
from .ego_rot_detr_layers import *


@MODELS.register_module()
class EgoRotDETR(DeformableDETR):
    def __init__(self,
                ego_h=50,
                ego_w=50,
                focal_length=500,
                learnable_focal=False,
                  *args,
                 **kwargs) -> None:
        self.ego_h = ego_h
        self.ego_w = ego_w
        super().__init__(*args, **kwargs)
        if learnable_focal:
            self.focal_length = nn.Parameter(torch.tensor([focal_length]).float())
        else:
            self.focal_length = focal_length
        # self.bbox_head.focal_length = self.focal_length
        self.bbox_head.pc_range = self.encoder.pc_range

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = EgoFormerEncoder(**self.encoder)
        self.decoder = HalfDeformableDetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.ego_embedding = nn.Embedding(
            self.ego_h * self.ego_w, self.embed_dims)
        self.query_embedding = nn.Embedding(self.num_queries,
                                            self.embed_dims * 2)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.reference_points_fc = nn.Linear(self.embed_dims, 3)

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList) -> SampleList:
        for data_sample, pred_center in zip(data_samples, results_list):
            data_sample.pred_center = pred_center
        return data_samples
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x_bb = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x_bb)
        return x

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        
        encoder_inputs_dict.update(batch_input_shape=[ds.batch_input_shape for ds in batch_data_samples])
        encoder_inputs_dict.update(img_shape=[ds.img_shape for ds in batch_data_samples])
        encoder_inputs_dict.update(batch_img_metas=[ds.metainfo for ds in batch_data_samples])

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        decoder_inputs_dict.update(encoder_outputs_dict)

        decoder_outputs = self.forward_decoder(**decoder_inputs_dict)
        decoder_outputs['focal_length'] = self.focal_length
        return decoder_outputs

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        img_shape:None,
                        batch_input_shape:None,
                        batch_img_metas=None,
                        **kwargs) -> Dict:
        bs = feat.shape[0]
        dtype = feat.dtype
        ego_queries = self.ego_embedding.weight.to(dtype)
        ego_mask = torch.zeros((bs, self.ego_h, self.ego_w),
                               device=ego_queries.device).to(dtype)
        ego_pos = self.positional_encoding(ego_mask).to(dtype)
        ego_queries = ego_queries.unsqueeze(1).repeat(1, bs, 1)
        ego_pos = ego_pos.flatten(2).permute(2, 0, 1)

        feat = feat.unsqueeze(0).permute(0, 2, 1, 3)# (num_cam, H*W, bs, embed_dims)
        intrinsics = []

        if batch_img_metas is not None:
            for img_meta in batch_img_metas:
                img_h, img_w = img_meta['img_shape']

        ego_embed = self.encoder(
            ego_queries,
            feat,
            feat,
            focal_length=self.focal_length,
            ego_h=self.ego_h,
            ego_w=self.ego_w,
            ego_pos=ego_pos,
            key_pos=feat_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_ego=None,
            img_shape=img_shape,
            batch_input_shape=batch_input_shape,
        )
        encoder_outputs_dict = dict(
            ego_embed=ego_embed,
            ego_h=self.ego_h,
            ego_w=self.ego_w,
        )
        return encoder_outputs_dict

    def forward_decoder(self,
                ego_embed,
                ego_h,
                ego_w,
                **kwargs):
        bs = ego_embed.size(0)
        object_query_embeds = self.query_embedding.weight.to(ego_embed.dtype)
        query_pos, query = torch.split(
            object_query_embeds, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points_fc(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points
        del kwargs['spatial_shapes']
        del kwargs['level_start_index']

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=ego_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            key_padding_mask=None,
            spatial_shapes=torch.tensor([[ego_h, ego_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        references = [reference_points, *inter_references]
        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=references)
        return decoder_outputs_dict

