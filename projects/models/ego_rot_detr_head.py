# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor
import math
import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmengine.structures import InstanceData
from mmdet.utils import InstanceList, reduce_mean
from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.models import inverse_sigmoid
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import FFN

@MODELS.register_module()
class RotDETRHeadBase(DeformableDETRHead):
    def __init__(self,
                 bg_cls_weight=0.,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bg_cls_weight = bg_cls_weight
        self.loss_cls_v = self.loss_cls

    def get_targets(self, cls_scores: Tensor, point_preds: Tensor,
                    cls_scores_v: Tensor, point_preds_v: Tensor,
                    batch_gt_centers: InstanceList,
                    batch_gt_vertices: InstanceList,
                    batch_img_metas: List[dict]) -> tuple:
        targets = [self._get_targets_single_image(
            cls_scores[i], point_preds[i], cls_scores_v[i], point_preds_v[i], 
            batch_gt_centers[i], batch_gt_vertices[i], batch_img_metas[i]
        ) for i in range(cls_scores.shape[0])]
        
        assign_v_meta = [t[-1] for t in targets]
        targets = [t[0] + t[1] for t in targets]

        labels, label_weights, point_targets, point_weights, pos_inds, neg_inds, \
            labels_v, point_targets_v, point_weights_v, pos_inds_v, neg_inds_v \
             = [torch.cat(t, dim=0) for t in zip(*targets)]

        n_pos, n_neg = pos_inds.numel(), neg_inds.numel()
        n_pos_v, n_neg_v = pos_inds_v.numel(), neg_inds_v.numel()
        return (labels, label_weights, point_targets, point_weights, n_pos, n_neg), \
            (labels_v, point_targets_v, point_weights_v, n_pos_v, n_neg_v), assign_v_meta

    def _get_center_targets_single_image(self, point_pred, gt_instances, factor, assign_result):
        num_points = point_pred.size(0)
        gt_points = gt_instances.points
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_points = gt_points[pos_assigned_gt_inds.long(), :]
        pos_gt_points = pos_gt_points.to(point_pred.device)

        # label targets
        labels = gt_points.new_full((num_points, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        order_weights = gt_points.new_ones(num_points)

        # bbox targets
        point_targets = torch.zeros_like(point_pred, dtype=gt_points.dtype)
        point_weights = torch.zeros_like(point_pred, dtype=gt_points.dtype)
        point_weights[pos_inds] = 1.0

        pos_gt_points_normalized = (pos_gt_points.double() / factor.double())
        point_targets = point_targets.float()
        point_targets[pos_inds] = pos_gt_points_normalized.float()
        return [labels, order_weights, point_targets, point_weights, pos_inds, neg_inds]

    def _get_vertex_targets_single_image(self, point_pred_v, gt_points_v, factor, assign_result_v):
        ### assign_result_v
        pos_inds_v = torch.nonzero(assign_result_v.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds_v = torch.nonzero(assign_result_v.gt_inds == 0, as_tuple=False).squeeze(-1).unique()

        pos_assigned_gt_inds_v = assign_result_v.gt_inds[pos_inds_v] - 1
        pos_gt_points_v = gt_points_v[pos_assigned_gt_inds_v.long(), :]

        # label targets
        labels_v = gt_points_v.new_full((point_pred_v.size(0), point_pred_v.size(1)),
                                    1, dtype=torch.long).flatten(0, 1)
        labels_v[pos_inds_v] *= 0

        # bbox targets
        point_targets_v = torch.zeros_like(point_pred_v, dtype=gt_points_v.dtype).flatten(0, 1)
        point_weights_v = torch.zeros_like(point_pred_v, dtype=gt_points_v.dtype).flatten(0, 1)
        point_weights_v[pos_inds_v] = 1.0

        pos_gt_points_normalized_v = (pos_gt_points_v.double() / factor.double())
        point_targets_v = point_targets_v.float()
        point_targets_v[pos_inds_v] = pos_gt_points_normalized_v.float()
        point_targets_v = point_targets_v.view_as(point_pred_v)
        point_weights_v = point_weights_v.view_as(point_pred_v)
        return [labels_v, point_targets_v, point_weights_v, pos_inds_v, neg_inds_v]

    def _get_targets_single_image(self, cls_score: Tensor, point_pred: Tensor,
                            cls_score_v: Tensor, point_pred_v: Tensor,
                            gt_instances: InstanceData,
                            gt_vertices: InstanceData,
                            img_meta: dict) -> tuple:
        img_h, img_w = img_meta['img_shape']
        factor = point_pred.new_tensor([img_w, img_h]).unsqueeze(0)
        point_pred = (point_pred.double() * factor).float()
        point_pred_v = (point_pred_v.double() * factor).float()

        pred_instances = InstanceData(scores=cls_score, points=point_pred)
        pred_vertices = InstanceData(scores=cls_score_v, points=point_pred_v)

        # assigner and sampler
        assign_result, assign_result_v, assign_v_meta = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            pred_vertices=pred_vertices,
            gt_vertices=gt_vertices,
            img_meta=img_meta)

        center_cls_reg_targets = self._get_center_targets_single_image(
                point_pred, gt_instances, factor, assign_result)
        vertex_cls_reg_targets = self._get_vertex_targets_single_image(
             pred_vertices.points, gt_vertices.points, factor, assign_result_v)

        return center_cls_reg_targets, vertex_cls_reg_targets, assign_v_meta

    def loss_center_single_layer(self, cls_scores, point_preds, cls_reg_targets):
        (labels, label_weights, point_targets, point_weights,
        num_total_pos, num_total_neg) = cls_reg_targets

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        else:
            cls_avg_factor = cls_scores.new_tensor([cls_avg_factor])
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        point_preds = point_preds.flatten(0, 1)
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        # regression L1 loss
        loss_bbox = self.loss_bbox(
            point_preds, point_targets, point_weights, avg_factor=num_total_pos)

        return loss_cls, loss_bbox
    
    def loss_vertex_single_layer(self, cls_scores_v, point_preds_v, targets):
        labels_v, point_targets_v, point_weights_v, num_pos_v, num_neg_v = targets
        ### classification loss
        cls_scores_v = cls_scores_v.view(-1, self.cls_out_channels_v)
        cls_avg_factor_v = num_pos_v * 1.0 +  num_neg_v * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor_v = reduce_mean(cls_scores_v.new_tensor([cls_avg_factor_v]))
        else:
            cls_avg_factor_v = (cls_scores_v.new_tensor([cls_avg_factor_v]))
        cls_avg_factor_v = max(cls_avg_factor_v, 1)
        loss_cls_v = self.loss_cls_v(
            cls_scores_v, labels_v, avg_factor=cls_avg_factor_v)

        point_preds_v = point_preds_v.flatten(0, 1)
        num_pos_v = loss_cls_v.new_tensor([num_pos_v])
        num_pos_v = torch.clamp(reduce_mean(num_pos_v), min=1).item()
        loss_bbox_v = self.loss_bbox(
            point_preds_v, point_targets_v, point_weights_v, avg_factor=num_pos_v)
        return loss_cls_v, loss_bbox_v

    def loss_single_layer(self, cls_scores: Tensor, point_preds: Tensor,
                            cls_scores_v: Tensor, point_preds_v: Tensor,
                            point_preds_3d: Tensor, point_preds_v_3d: Tensor,
                            batch_gt_centers: InstanceList,
                            batch_gt_vertices: InstanceList,
                            batch_img_metas: List[dict],) -> Tuple[Tensor]:
                            
        center_cls_reg_targets, vertex_cls_reg_targets, assign_v_meta = self.get_targets(
            cls_scores, point_preds, cls_scores_v,
            point_preds_v, batch_gt_centers, batch_gt_vertices, batch_img_metas)
        
        loss_cls, loss_bbox = self.loss_center_single_layer(cls_scores, point_preds, center_cls_reg_targets)
        loss_cls_v, loss_bbox_v = self.loss_vertex_single_layer(cls_scores_v, point_preds_v, vertex_cls_reg_targets)
        return loss_cls, loss_bbox, loss_cls_v, loss_bbox_v

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             batch_data_samples: SampleList, **kwargs) -> dict:
        batch_img_metas = [ds.metainfo for ds in batch_data_samples]
        batch_gt_center = [ds.gt_center for ds in batch_data_samples]
        batch_gt_vertex = [ds.gt_vertex for ds in batch_data_samples]

        outs = self(hidden_states, references, batch_img_metas=batch_img_metas, **kwargs)

        losses = [self.loss_single_layer(score, pred, v_score, v_pred, pred_3d, v_pred_3d,
            batch_gt_center, batch_gt_vertex, batch_img_metas
            ) for score, pred, v_score, v_pred, pred_3d, v_pred_3d in zip(*outs)]
        
        loss_dict = dict()
        loss_dict['loss_cls'] = losses[-1][0]
        loss_dict['loss_reg'] = losses[-1][1]
        loss_dict['loss_cls_v'] = losses[-1][2]
        loss_dict['loss_reg_v'] = losses[-1][3]

        for d_i, _losses in enumerate(losses[:-1]):
            loss_cls_i, loss_reg_i, loss_cls_v_i, loss_reg_v_i = _losses
            loss_dict[f'd{d_i}.loss_cls'] = loss_cls_i
            loss_dict[f'd{d_i}.loss_reg'] = loss_reg_i
            loss_dict[f'd{d_i}.loss_cls_v'] = loss_cls_v_i
            loss_dict[f'd{d_i}.loss_reg_v'] = loss_reg_v_i
        
        return loss_dict


@MODELS.register_module()
class RotDETRHead(RotDETRHeadBase):
    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        n_pts_dim = 3
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, n_pts_dim))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
        self.reg_branches = nn.ModuleList([
            copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
        ])

        num_points = 8
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels_v = 1
        else:
            self.cls_out_channels_v = 2
        fc_cls = Linear(self.embed_dims, num_points*self.cls_out_channels_v)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, n_pts_dim*num_points))
        reg_branch = nn.Sequential(*reg_branch)

        self.v_cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
        self.v_reg_branches = nn.ModuleList([
            copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
        ])

    def init_weights(self) -> None:
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, bias_init)
        
        bias_init = bias_init_with_prob(0.01)
        for m in self.v_cls_branches:
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, bias_init)

        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        
        for m in self.v_reg_branches:
            constant_init(m[-1], 0, bias=0)
            
    def project_points(self, reference_points, img_shapes, batch_img_shapes, focal_length, scale=True):
        device = reference_points.device
        img_shapes = torch.stack(img_shapes).to(device)
        batch_shapes = torch.stack(batch_img_shapes).to(device)
        if scale:
            pc_range = self.pc_range
            reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
            reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
            reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        h, w = batch_shapes[:, 0], batch_shapes[:, 1]
        cx, cy = img_shapes[:, 1] / 2, img_shapes[:, 0] / 2
        if not torch.is_tensor(focal_length):
            focal_length = torch.tensor([focal_length], device=device)
        fx = fy = focal_length

        dims = reference_points.ndim
        view = [1, -1] + [1] * (dims - 2)
        h, w = h.view(*view), w.view(*view)
        cx, cy = cx.view(*view), cy.view(*view)
        fx, fy = fx.view(*view), fy.view(*view)

        u = fx * reference_points[..., 0:1] / reference_points[..., 2:3] + cx
        v = fy * reference_points[..., 1:2] / reference_points[..., 2:3] + cy
        reference_points_cam = torch.cat((u/w, v/h), dim=-1)
        return reference_points_cam, reference_points
    
    def inner_forward(self, hidden_states: Tensor,
                references: List[Tensor],
                half_prec=False, 
                focal_length=None,
                **kwargs) -> Tuple[Tensor, Tensor]:
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_outputs_classes_v = []
        all_layers_outputs_coords_v = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state) + reference
            v_outputs_class = self.v_cls_branches[layer_id](hidden_state)
            v_tmp_reg_preds = self.v_reg_branches[layer_id](hidden_state)
            v_tmp_reg_preds = v_tmp_reg_preds.view(
                v_tmp_reg_preds.shape[0], -1, 8, tmp_reg_preds.shape[-1]) + reference.unsqueeze(-2)

            outputs_coord = tmp_reg_preds.sigmoid()
            v_outputs_coord = v_tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            all_layers_outputs_classes_v.append(v_outputs_class)
            all_layers_outputs_coords_v.append(v_outputs_coord)

        return (
            torch.stack(all_layers_outputs_classes),
            torch.stack(all_layers_outputs_coords),
            torch.stack(all_layers_outputs_classes_v),
            torch.stack(all_layers_outputs_coords_v),
            True
        )

    def forward(self, hidden_states: Tensor,
                references: List[Tensor],
                half_prec=False,
                batch_img_metas=None,
                focal_length=None) -> Tuple[Tensor, Tensor]:
        all_layers_cls_scores, all_layers_bbox_preds, \
             all_layers_scores_v, all_layers_preds_v, scale_flag = \
                self.inner_forward(hidden_states, references)
        batch_img_shapes = []
        img_shapes = []
        for img_meta in batch_img_metas:
            if 'batch_input_shape' in img_meta:
                img_h, img_w = img_meta['batch_input_shape']
            else:
                img_h, img_w = img_meta['img_shape']
            batch_img_shapes.append(torch.tensor([img_h, img_w]))
            img_h, img_w = img_meta['img_shape']
            img_shapes.append(torch.tensor([img_h, img_w]))

        all_layers_bbox_preds, all_layers_bbox_preds_3d = self.project_points(
            all_layers_bbox_preds, img_shapes, img_shapes, focal_length, scale=scale_flag)
        all_layers_preds_v, all_layers_preds_v_3d = self.project_points(
            all_layers_preds_v, img_shapes, img_shapes, focal_length, scale=scale_flag)

        return all_layers_cls_scores, all_layers_bbox_preds, \
             all_layers_scores_v, all_layers_preds_v, \
             all_layers_bbox_preds_3d, all_layers_preds_v_3d
    
    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> InstanceList:
        batch_img_metas = [ds.metainfo for ds in batch_data_samples]
        results = self(hidden_states, references, batch_img_metas=batch_img_metas, **kwargs)
        results = [r[-1] for r in results] # last layer
        result_list = [self._predict_single_image(*[r[i] for r in results], \
            batch_img_metas[i], rescale) for i in range(len(batch_img_metas))]
        return result_list
        
    def _predict_single_image(self,
                                cls_score: Tensor,
                                point_pred: Tensor,
                                cls_score_v: Tensor,
                                point_pred_v: Tensor,
                                point_pred_3d: Tensor,
                                point_pred_v_3d: Tensor,
                                img_meta: dict,
                                rescale: bool = False) -> InstanceData:
        assert len(cls_score) == len(point_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        ori_shape = img_meta.get('ori_shape')
        assert ori_shape is not None, "Original image shape not provided in img_meta"

        # Process classification scores and retrieve top-k predictions
        cls_score = cls_score.sigmoid()
        scores, indices = cls_score.view(-1).topk(max_per_img)
        det_labels = indices % self.num_classes
        bbox_indices = indices // self.num_classes

        # Get predicted 2D and 3D center points
        det_points = self._rescale_to_ori(point_pred[bbox_indices], ori_shape)
        det_points_3d = point_pred_3d[bbox_indices]

        # Process vertex predictions
        cls_score_v = cls_score_v.sigmoid()
        scores_v = cls_score_v[bbox_indices]
        det_points_v = self._rescale_to_ori(point_pred_v[bbox_indices], ori_shape)
        det_points_v_3d = point_pred_v_3d[bbox_indices]

        # Create dummy scale values
        scales = torch.full((scores.shape[0],), fill_value=max(ori_shape), dtype=torch.float)

        return InstanceData(
            points=det_points,
            scores=scores,
            labels=det_labels,
            scales=scales,
            points_v=det_points_v,
            scores_v=scores_v,
            points_3d=det_points_3d,
            points_v_3d=det_points_v_3d
        )

    def _rescale_to_ori(self, points: Tensor, ori_shape: tuple) -> Tensor:
        """Rescale normalized points to original image coordinates and clamp to bounds."""
        points[..., 0] *= ori_shape[1]
        points[..., 1] *= ori_shape[0]
        points[..., 0].clamp_(min=0, max=ori_shape[1])
        points[..., 1].clamp_(min=0, max=ori_shape[0])
        return points


@MODELS.register_module()
class RotDETRPriorHead(RotDETRHead):
    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        n_pts_dim = 3
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        # x, y, z, dx, dy, dz
        reg_branch.append(Linear(self.embed_dims, n_pts_dim * 3 + 1))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
        self.reg_branches = nn.ModuleList([
            copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
        ])

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[3:6], -2.0)
        for m in self.reg_branches:
            bias_init = torch.tensor([0, 0, 1], dtype=torch.float32)
            m[-1].bias.data[6:9] = bias_init.to(m[-1].bias.data.device)

    def expand_vertices(self, outputs_class, outputs_coord, outputs_seed, outputs_axis, outputs_bias=None):
        apex = outputs_coord.unsqueeze(-2)
        seed = outputs_seed.unsqueeze(-2)
        axis = outputs_axis.unsqueeze(-2)

        pc_range = self.pc_range
        n_vertices = torch.tensor([4, 3, 4, 5, 6, 8]).cuda()[torch.argmax(outputs_class, dim=-1)]
        angle_bias = torch.tensor([1, 0, 0, 0, 0, 0]).cuda()[torch.argmax(outputs_class, dim=-1)]
        
        angle_bias = angle_bias.unsqueeze(-1).unsqueeze(-2) * \
            torch.tensor([0, 1, 0, 1, 0, 0, 0, 0]).cuda().view(1, 1, -1, 1)
        angle_bias = angle_bias * outputs_bias.unsqueeze(-1)

        angle_increment = torch.tensor(2 * torch.pi) / n_vertices
        angle_increment = angle_increment.unsqueeze(-1).unsqueeze(-2)
        angle = torch.arange(8, device=axis.device).view(1, 1, -1, 1) * angle_increment
        angle = angle + angle_bias

        axis = F.normalize(axis, p=2, dim=-1)
        point, origin = seed, apex
        point = point - origin
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        cross_prod = torch.cross(axis, point)
        dot_prod = torch.einsum('bmnc, bmnc -> bmn', axis, point).unsqueeze(-1)
        rotated_point = (point * cos_angle + cross_prod * sin_angle 
                         + axis * dot_prod * (1 - cos_angle)) + origin
        valid_mask = torch.arange(8, device=axis.device).view(1, 1, -1, 1).expand_as(rotated_point)
        n_vertices = n_vertices.unsqueeze(-1).unsqueeze(-2).expand_as(rotated_point)
        valid_mask = (valid_mask < n_vertices).float()

        return valid_mask[..., 0], rotated_point

    def expand_vertices_eval(self, outputs_class, outputs_coord, outputs_seed, outputs_axis, outputs_bias=None):
        apex = outputs_coord.unsqueeze(-2)
        seed = outputs_seed.unsqueeze(-2)
        axis = outputs_axis.unsqueeze(-2)
        point, origin = seed, apex

        pc_range = self.pc_range
        n_vertices = torch.tensor([4, 3, 4, 5, 6, 8]).cuda().view(1, 1, -1).expand_as(outputs_class)
        angle_bias = torch.tensor([1, 0, 0, 0, 0, 0]).cuda().view(1, 1, -1).expand_as(outputs_class)

        # (b, N, n_c), (8) -> (b, N, n_c, 8)
        angle_bias = torch.einsum('bna, m -> bnam', angle_bias, torch.tensor([0, 1, 0, 1, 0, 0, 0, 0]).cuda())
        angle_bias = (angle_bias * outputs_bias.unsqueeze(-1))
        angle_increment = torch.tensor(2 * torch.pi) / n_vertices
        angle = torch.einsum('bna, m -> bnam', angle_increment, torch.arange(8, device=axis.device))
        angle = angle + angle_bias
        # last dimension for n_pts_dim (bnamp)
        angle = angle.unsqueeze(-1)

        axis = F.normalize(axis, p=2, dim=-1)
        point = point - origin
        cross_prod = torch.cross(axis, point)
        dot_prod = torch.einsum('bnmc, bnmc -> bnm', axis, point).unsqueeze(2).unsqueeze(-1)

        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        axis, cross_prod = axis.unsqueeze(-2), cross_prod.unsqueeze(-2)
        point, origin = point.unsqueeze(-2), origin.unsqueeze(-2)
        rotated_point = (point * cos_angle + cross_prod * sin_angle 
                         + axis * dot_prod * (1 - cos_angle)) + origin
        valid_mask = torch.arange(8, device=axis.device).view(1, 1, 1, -1, 1).expand_as(rotated_point)

        n_vertices = n_vertices.unsqueeze(-1).unsqueeze(-2).expand_as(rotated_point)
        valid_mask = (valid_mask < n_vertices).float()

        return valid_mask[..., 0], rotated_point

    def inner_forward(self, hidden_states: Tensor,
                references: List[Tensor],
                half_prec=False, return_axis=False, **kwargs) -> Tuple[Tensor, Tensor]:
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []
        all_layers_outputs_classes_v = []
        all_layers_outputs_coords_v = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            tmp_reg_preds[..., :3] += reference
            tmp_reg_preds[..., 3:6] += reference

            _outputs_coord = tmp_reg_preds[..., :3].sigmoid()
            _outputs_seed = tmp_reg_preds[..., 3:6].sigmoid()
            outputs_axis = tmp_reg_preds[..., 6:9]
            _outputs_bias = tmp_reg_preds[..., 9:10].sigmoid()

            pc_range = self.pc_range
            outputs_coord = torch.cat([_outputs_coord[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0],
                        _outputs_coord[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1],
                        _outputs_coord[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2],], dim=-1)
            outputs_seed = torch.cat([_outputs_seed[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0],
                        _outputs_seed[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1],
                        _outputs_seed[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2],], dim=-1)

            # if self.legacy_angle:
            outputs_bias = (_outputs_bias - 0.5) * 2 * 90 # -90~90

            if self.training:
                outputs_class_v, outputs_coord_v = \
                    self.expand_vertices(outputs_class, outputs_coord, outputs_seed, outputs_axis, outputs_bias)
            else:
                outputs_class_v, outputs_coord_v = \
                    self.expand_vertices_eval(outputs_class, outputs_coord, outputs_seed, outputs_axis, outputs_bias)
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
            all_layers_outputs_classes_v.append(outputs_class_v)
            all_layers_outputs_coords_v.append(outputs_coord_v)

        return (
            torch.stack(all_layers_outputs_classes),
            torch.stack(all_layers_outputs_coords),
            torch.stack(all_layers_outputs_classes_v),
            torch.stack(all_layers_outputs_coords_v),
            False
        )

    def loss_vertex_single_layer(self, cls_scores_v, point_preds_v, targets):
        labels_v, point_targets_v, point_weights_v, num_pos_v, num_neg_v = targets
        point_preds_v = point_preds_v.flatten(0, 1)
        num_pos_v = point_preds_v.new_tensor([num_pos_v])
        num_pos_v = torch.clamp(reduce_mean(num_pos_v), min=1).item()
        loss_bbox_v = self.loss_bbox(
            point_preds_v, point_targets_v, point_weights_v, avg_factor=num_pos_v)
        return loss_bbox_v*0, loss_bbox_v
        
    def _predict_single_image(self,
                                cls_score: Tensor,
                                point_pred: Tensor,
                                cls_score_v: Tensor,
                                point_pred_v: Tensor,
                                point_pred_3d: Tensor,
                                point_pred_v_3d: Tensor,
                                img_meta: dict,
                                rescale: bool = False) -> InstanceData:
        assert len(cls_score) == len(point_pred)  # num_queries
        ori_shape = img_meta['ori_shape']
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))

        # Process classification scores and select top-k
        cls_score = cls_score.sigmoid()
        scores, indices = cls_score.view(-1).topk(max_per_img)
        det_labels = indices % self.num_classes
        bbox_index = indices // self.num_classes

        # Select and scale 2D/3D points
        det_points = self._rescale_to_ori(point_pred[bbox_index], ori_shape)
        det_points_3d = point_pred_3d[bbox_index]

        # Process vertex classification and points
        cls_score_v = cls_score_v.sigmoid()
        scores_v = cls_score_v.flatten(0, 1)[indices]
        det_points_v = self._rescale_to_ori(point_pred_v.flatten(0, 1)[indices], ori_shape)
        det_points_v_3d = point_pred_v_3d.flatten(0, 1)[indices]

        # Create dummy scale tensor
        scales = torch.full((scores.shape[0],), fill_value=max(ori_shape), dtype=torch.float)

        # Wrap results
        return InstanceData(
            points=det_points,
            scores=scores,
            labels=det_labels,
            scales=scales,
            points_v=det_points_v,
            scores_v=scores_v,
            points_3d=det_points_3d,
            points_v_3d=det_points_v_3d
        )

