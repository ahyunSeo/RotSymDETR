import torch
import cv2
import os
import mmcv
import mmengine
import random
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from mmdet.registry import DATASETS
from mmengine.dataset import Compose, BaseDataset
from .loading import LoadImageFromFile, LoadDepthFromFile, LoadSymmetryAnnotation

@DATASETS.register_module()
class JointDendiDataset(BaseDataset):
    CLASSES = None
    PALETTE = None

    def __init__(self,
                 rot_annfile,
                 split,
                 data_root,
                 input_size=(800, 800),
                 pipeline=None,
                 resize_val=False,
                 flip_aug=False,
                 rotate_aug=False,
                 rotate_deg=5,
                 intrinsic_aug=False,
                 rescale_aug=False,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 classes=[2, 3, 4, 5, 6, 7, 8],
                 n_samples=None,
                 remove_invisible=False,
                 **kwargs,
                 ):
        self.rot_annfile = rot_annfile
        self.data_root = data_root
        self.split = split
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.input_size = input_size
        self.classes = classes
        self.n_samples = n_samples
        self.resize_val = resize_val
        self.flip_aug = flip_aug
        self.rotate_aug = rotate_aug
        self.rotate_deg = rotate_deg
        self.intrinsic_aug = intrinsic_aug
        self.rescale_aug = rescale_aug
        self.remove_invisible = remove_invisible
        self._metainfo = {'classes': classes}
        self.load_annotations()
        self.loading = [
            LoadImageFromFile(channel_order='rgb'),
            LoadSymmetryAnnotation(classes=self.classes),
            ]
        self.compose_pipeline()
        self.pipeline = Compose(pipeline)
        random.seed(0)
        np.random.seed(0)

    def load_annotations(self):
        self.rot_data = torch.load(os.path.join(self.data_root,
                                                 self.rot_annfile))
        self.data_keys = list(self.rot_data.keys())
        self.filter_data()

    def full_init(self, ):
        pass

    def filter_data(self, ):
        new_data_keys = []
        cat_to_keys = {}
        for cat in self.classes:
            cat_to_keys[cat] = []

        for key in self.data_keys:
            orders = [ann['order'] for ann in self.rot_data[key]['anns']]
            check = [order in self.classes for order in orders]

            orders = set(orders)

            for cat in self.classes:
                if cat in orders:
                    cat_to_keys[cat].append(key)

            if True in check:
                new_data_keys.append(key)
                ann = [self.rot_data[key]['anns'][idx] for idx, flag, in enumerate(check) if flag == True]
                self.rot_data[key]['anns'] = ann

        print('Data filtered. %d -> %d' % (len(self.data_keys), len(new_data_keys)))
        self.data_keys = new_data_keys
        self.cat_to_keys = cat_to_keys

    def get_cat_ids(self, idx: int):
        instances = self.get_data_info(idx)
        return [ann['order'] for ann in instances['ann_info']]

    def compose_pipeline(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        keypoint_params = A.KeypointParams(
                            format='xy', 
                            remove_invisible=False,
                            label_fields=['instance_ids', 'orders', 'is_center', 'class_ids'])
        self.transforms_aug = None
        if self.split in ['train'] or self.resize_val:
            if self.split in ['train']:
                transforms_aug = [
                        A.LongestMaxSize(max_size=self.input_size[0]),
                        ]
                if self.flip_aug:
                    transforms_aug.append(A.HorizontalFlip(p=0.5))
                if self.rotate_aug:
                    transforms_aug.append(A.Rotate(limit=(-self.rotate_deg,self.rotate_deg), 
                                                border_mode=cv2.BORDER_CONSTANT,
                                                value=0, 
                                                mask_value=0,))
                self.transforms_aug = A.Compose(
                                        transforms_aug,
                                        keypoint_params=keypoint_params
                                                )
                self.transform = A.Compose([A.Normalize(mean, std), ToTensorV2()],
                        keypoint_params=keypoint_params
                        )
            else:
                self.transform = A.Compose([
                            A.LongestMaxSize(max_size=self.input_size[0]),
                            A.Normalize(mean, std),
                            ToTensorV2(), ],
                            keypoint_params=keypoint_params
                            )
        else:
            self.transform = A.Compose([
                            A.Normalize(mean, std),
                            ToTensorV2(), ],
                            keypoint_params=keypoint_params
                            )

    def get_data_info(self, idx):
        anno_index = self.data_keys[idx]
        img_path = self.rot_data[anno_index]['img_path'][:-4]
        img_info = {'filename': img_path, 'img_prefix': self.data_root}
        ann_info = self.rot_data[anno_index]['anns']
        results = dict(img_info=img_info, ann_info=ann_info)
        return results

    def __getitem__(self, idx):
        data = self.get_data_info(idx)
        data = self.do_pipeline(data, idx)
        data = self.pipeline(data)
        return data

    def _do_pipeline(self, x, transformed):
        img_shape = transformed['image'].shape
        transformed['img_shape'] = (int(img_shape[1]), int(img_shape[2]))
        transformed['ori_shape'] = (x['img'].shape[0], x['img'].shape[1])
        transformed['img_path'] = x['img_info']['filename']

        transformed['keypoints'] = torch.tensor(transformed['keypoints'])
        transformed['instance_ids'] = torch.tensor(transformed['instance_ids'])
        transformed['orders'] = torch.tensor(transformed['orders'])
        transformed['class_ids'] = torch.tensor(transformed['class_ids'])

        transformed['ori_keypoints'] = torch.tensor(x['sym_info']['keypoints'])
        transformed['ori_instance_ids'] = (transformed['instance_ids']).clone()
        transformed['ori_orders'] = (transformed['orders']).clone()
        transformed['ori_class_ids'] = (transformed['class_ids']).clone()
        transformed['ori_is_center'] = transformed['is_center']

        is_center = torch.tensor(transformed['is_center'])

        if self.remove_invisible:
            pts = transformed['ori_keypoints']
            ori_shape = transformed['ori_shape']
            pts_x = (pts[..., 0] >= 0) & (pts[..., 0] <= ori_shape[1])
            pts_y = (pts[..., 1] >= 0) & (pts[..., 1] <= ori_shape[0])
            new_pts = pts[pts_x & pts_y]
            if len(pts) != len(new_pts):
                transformed['ori_keypoints'] = new_pts
                transformed['ori_instance_ids'] = transformed['instance_ids'][pts_x & pts_y]
                transformed['ori_orders'] = transformed['orders'][pts_x & pts_y]
                transformed['ori_class_ids'] = transformed['class_ids'][pts_x & pts_y]
                transformed['ori_is_center'] = [d.item() for d in is_center[pts_x & pts_y]]

        if self.remove_invisible:
            pts = transformed['keypoints']
            pts_x = (pts[..., 0] >= -0.5) & (pts[..., 0] <= (img_shape[2] + 0.5))
            pts_y = (pts[..., 1] >= -0.5) & (pts[..., 1] <= (img_shape[1] + 0.5))
            pts_x1 = (pts[..., 0] < 0) | (pts[..., 0] > img_shape[2])
            pts_y1 = (pts[..., 1] < 0) | (pts[..., 1] > img_shape[1])
            new_pts = pts[pts_x & pts_y]
            new_pts[..., 0] = new_pts[..., 0].clamp(0, img_shape[2])
            new_pts[..., 1] = new_pts[..., 1].clamp(0, img_shape[1])
            if len(pts) != len(new_pts):
                transformed['keypoints'] = new_pts
                transformed['instance_ids'] = transformed['instance_ids'][pts_x & pts_y]
                transformed['orders'] = transformed['orders'][pts_x & pts_y]
                transformed['class_ids'] = transformed['class_ids'][pts_x & pts_y]
                transformed['is_center'] = [d.item() for d in is_center[pts_x & pts_y]]
        return transformed

    def do_pipeline(self, x, idx):
        for t in self.loading:
            x = t(x)

        if self.transforms_aug is not None:
            transformed = self.transforms_aug(image=x['img'], 
                                     **x['sym_info'])
            transformed = self.transform(**transformed)
        else:
            transformed = self.transform(image=x['img'], 
                                        **x['sym_info'])

        transformed = self._do_pipeline(x, transformed)

        if self.split == 'train':
            if True not in transformed['is_center']:
                transformed = self.transforms_aug(image=x['img'], 
                                        **x['sym_info'])
                transformed = self.transform(**transformed)
                transformed = self._do_pipeline(x, transformed)
                if True not in transformed['is_center']:
                    transformed = self.transform(image=x['img'], 
                                                **x['sym_info'])
                    transformed = self._do_pipeline(x, transformed)
            assert True in transformed['is_center']
        
        return transformed

    def __len__(self):
        """Total number of samples of data."""
        if self.n_samples is not None:
            return self.n_samples
        return len(self.data_keys)