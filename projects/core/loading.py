import torch
import cv2
import os
import mmcv
import mmengine
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A


class LoadImageFromFile:
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        if results['img_info']['img_prefix'] is not None:
            filename = osp.join(results['img_info']['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
 
        if os.path.isfile('%s.jpg' % filename):
            filename = '%s.jpg' % filename
        elif os.path.isfile('%s.png' % filename):
            filename = '%s.png' % filename

        img_bytes = mmengine.fileio.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        if 'intrinsic' in results['img_info']:
            results['intrinsic'] = results['img_info']['intrinsic']
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        
        return results


class LoadDepthFromFile:
    def __call__(self, results):
        root = results['depth_info']['img_prefix']
        fname = results['depth_info']['filename']
        img_path = Image.open(os.path.join(root, fname))
        depth = np.asarray(img_path).astype(np.uint16)
        depth_raw = depth / 256
        results['depth_path'] = img_path
        results['depth_raw'] = depth_raw
        
        return results

class LoadSymmetryAnnotation:
    def __init__(self,
                 classes
                 ):
        self.classes = classes

    def __call__(self, results):
        centers, c_instance_ids, c_orders = [], [], []
        vertices, instance_ids, orders = [], [], []
        c_class_ids, class_ids = [], []

        for a_i, ann in enumerate(results['ann_info']):
            order = ann['order']
            if order not in self.classes:
                continue
            centers.append(ann['center'])
            if 'vertices' in ann:
                _vertices = ann['vertices']
            else:
                _vertices = []
            vertices += _vertices
            instance_ids += [a_i] * len(_vertices)
            orders += [order] * len(_vertices)
            class_ids += [self.classes.index(order)] * len(_vertices)
            c_instance_ids.append(a_i)
            c_orders.append(order)
            c_class_ids.append(self.classes.index(order))

        is_center = [False]*len(vertices) + [True]*len(centers)
        keypoints = vertices + centers
        instance_ids += c_instance_ids
        orders += c_orders
        class_ids += c_class_ids

        sym_info = dict(keypoints=keypoints, 
                        instance_ids=instance_ids, 
                        orders=orders, 
                        is_center=is_center,
                        class_ids=class_ids)
    
        results['sym_info'] = sym_info
        return results


class LoadCenterAnnotation:
    def __init__(self,
                 classes
                 ):
        self.classes = classes

    def __call__(self, results):
        centers, c_instance_ids, c_orders = [], [], []
        vertices, instance_ids, orders = [], [], []
        c_class_ids, class_ids = [], []

        for a_i, ann in enumerate(results['ann_info']):
            order = ann['order']
            # if order not in self.classes:
            #     continue
            centers.append(ann['center'])
            c_instance_ids.append(a_i)
            c_orders.append(order)
            if order in self.classes:
                c_id = self.classes.index(order)
            else:
                c_id = -1
            c_class_ids.append(c_id)

        is_center = [True]*len(centers)
        keypoints = centers
        instance_ids += c_instance_ids
        orders += c_orders
        class_ids += c_class_ids

        sym_info = dict(keypoints=keypoints, 
                        instance_ids=instance_ids, 
                        orders=orders, 
                        is_center=is_center,
                        class_ids=class_ids)
    
        results['sym_info'] = sym_info
        return results