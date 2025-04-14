from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from mmengine.structures import InstanceData
from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from scipy.optimize import linear_sum_assignment
import torch.distributed as dist

@TASK_UTILS.register_module()
class PointL1Cost(BaseMatchCost):
    def __init__(self,
                 factor_mode='detr',
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.factor_mode = factor_mode

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        pred_points = pred_instances.points
        gt_points = gt_instances.points
        if self.factor_mode in ['detr']:
            img_h, img_w = img_meta['img_shape']
        else:
            # normalized
            if 'batch_input_shape' in img_meta:
                img_h, img_w = img_meta['batch_input_shape']
            else:
                img_h, img_w = img_meta['img_shape']
        factor = pred_points.new_tensor([img_w, img_h]).unsqueeze(0)
        gt_points = gt_points.to(pred_points.device) # temp
        gt_points = gt_points / factor
        pred_points = pred_points / factor

        point_cost = torch.cdist(pred_points.float(), gt_points.float(), p=1)
        return point_cost * self.weight
