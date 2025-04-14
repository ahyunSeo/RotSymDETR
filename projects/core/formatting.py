# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence
import torch
import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData

from matplotlib.path import Path
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample


@TRANSFORMS.register_module()
class PackJointInputs(BaseTransform):
    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction', 'intrinsic')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        packed_results['inputs'] = results['image']

        data_sample = DetDataSample()
        center_data = InstanceData()
        vertex_data = InstanceData()

        pivot = results['is_center'].index(True)
        center_data['points'] = results['keypoints'][pivot:]
        center_data['instance_ids'] = results['instance_ids'][pivot:]
        center_data['orders'] = results['orders'][pivot:]
        center_data['labels'] = results['class_ids'][pivot:]
        vertex_data['points'] = results['keypoints'][:pivot]
        vertex_data['instance_ids'] = results['instance_ids'][:pivot]
        vertex_data['orders'] = results['orders'][:pivot]
        vertex_data['labels'] = results['class_ids'][:pivot]

        data_sample.gt_center = center_data
        data_sample.gt_vertex = vertex_data

        ori_center_data = InstanceData()
        ori_vertex_data = InstanceData()
        pivot = results['ori_is_center'].index(True)
        ori_center_data['points'] = results['ori_keypoints'][pivot:]
        ori_center_data['instance_ids'] = results['ori_instance_ids'][pivot:]
        ori_center_data['orders'] = results['ori_orders'][pivot:]
        ori_center_data['labels'] = results['ori_class_ids'][pivot:]
        ori_vertex_data['points'] = results['ori_keypoints'][:pivot]
        ori_vertex_data['instance_ids'] = results['ori_instance_ids'][:pivot]
        ori_vertex_data['orders'] = results['ori_orders'][:pivot]
        ori_vertex_data['labels'] = results['ori_class_ids'][:pivot]

        data_sample.ori_gt_center = ori_center_data
        data_sample.ori_gt_vertex = ori_vertex_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_sample
    
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

