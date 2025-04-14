# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union
import numpy as np
import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules import AssignResult
from mmdet.models.task_modules import BaseAssigner
from .match_cost import PointL1Cost


@TASK_UTILS.register_module()
class RotationAssigner(BaseAssigner):
    def __init__(
        self, match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                 ConfigDict]
    ) -> None:

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]
        self.l1_cost = PointL1Cost()        

    def assign_instance(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels), None, (num_gts, num_preds)

        # 2. compute weighted cost
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)

        cost = torch.stack(cost_list).sum(dim=0)

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels), matched_row_inds, matched_col_inds

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               pred_vertices: InstanceData,
               gt_vertices: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        
        assign_result, matched_row_inds, matched_col_inds = \
            self.assign_instance(pred_instances, gt_instances, img_meta)

        gt_instance_ids = gt_vertices.instance_ids
        n, c, _ = pred_vertices.points.shape
        num_preds = n*c
        device = gt_instance_ids.device
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)
        if matched_row_inds is None:
            num_gts, num_preds = matched_col_inds
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            assign_result_v = AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)
            return assign_result, assign_result_v, None

        vertex_cost = self.l1_cost(pred_vertices, gt_vertices, img_meta)
        vertex_cost_mask = torch.zeros_like(vertex_cost)
        num_gts = len(gt_instance_ids)
        assigned_gt_inds[:] = 0
        for row, col in zip(matched_row_inds, matched_col_inds):
            vertex_cost_mask[row] += (gt_instance_ids == col).float().unsqueeze(0)
        if vertex_cost.numel() < 1:
            # print('vertex_cost.numel() < 1:', matched_col_inds)
            assigned_gt_inds[:] = 0
            assign_result_v = AssignResult(
                num_gts=0,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)
            return assign_result, assign_result_v, None
        max_cost = vertex_cost.max() + 1
        vertex_cost = vertex_cost * vertex_cost_mask + max_cost * (1 - vertex_cost_mask)
        vertex_cost = vertex_cost.flatten(0, 1).detach().cpu()

        INFTY_COST = 1e+5

        vertex_cost = np.asarray(vertex_cost)
        nan = np.isnan(vertex_cost).any()
        if nan:
            vertex_cost[np.isnan(vertex_cost)]=INFTY_COST

        matched_row_inds_v, matched_col_inds_v = linear_sum_assignment(vertex_cost)
        matched_row_inds_v = torch.from_numpy(matched_row_inds_v).to(device)
        matched_col_inds_v = torch.from_numpy(matched_col_inds_v).to(device)
        assigned_gt_inds[matched_row_inds_v] = matched_col_inds_v + 1
        assigned_labels[matched_row_inds_v] += 1

        matched_col_ind_val, matched_col_inds_inds = matched_col_inds.sort()
        matched_row_inds = matched_row_inds[matched_col_inds_inds]
        matched_col_inds_v_val, matched_col_inds_v_inds = matched_col_inds_v.sort()
        matched_row_inds_v = matched_row_inds_v[matched_col_inds_v_inds]

        metadata = dict(
            matched_gt_inds=matched_col_ind_val,
            matched_pred_inds=matched_row_inds,
            matched_gt_inds_v=matched_col_inds_v_val,
            matched_pred_inds_v=matched_row_inds_v,
            gt_instance_ids=gt_instance_ids[matched_col_inds_v_val],
            gt_orders=gt_vertices.orders[matched_col_inds_v_val],
            c_gt_instance_ids=gt_instances.instance_ids[matched_col_ind_val],
            c_gt_orders=gt_instances.orders[matched_col_ind_val]
        )

        assign_result_v = AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
        return assign_result, assign_result_v, metadata

