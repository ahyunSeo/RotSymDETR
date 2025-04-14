# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from collections import OrderedDict
from typing import List, Optional, Sequence, Union
from collections import defaultdict
import torch
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from scipy.optimize import linear_sum_assignment
from mmdet.registry import METRICS
from prettytable import PrettyTable
import torch.nn as nn
import math


@METRICS.register_module()
class DENDIVertexMetricV2(BaseMetric):
    def __init__(self,
                 tau_alpha=0.025, 
                 topk=300,
                 metric: Union[str, List[str]] = 'mAP',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 classes=[2, 3, 4, 5, 6, 7, 8],
                 ori_size_val= False,) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.tau_alpha = tau_alpha
        self.topk = topk
        self.metric = metric
        self.dataset_meta = dict(
            classes=classes
        )
        self.ori_size_val = ori_size_val

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        if type(data_samples[0]) == dict:
            for data_sample in data_samples:
                if self.ori_size_val:    
                    gt_center = data_sample['ori_gt_center']
                    gt_vertex = data_sample['ori_gt_vertex']
                else:
                    gt_center = data_sample['gt_center']
                    gt_vertex = data_sample['gt_vertex']

                ann = dict(
                    labels=gt_center['labels'].detach().cpu().numpy(),
                    points=gt_center['points'].detach().cpu().numpy(),
                    ids=gt_center['instance_ids'].detach().cpu().numpy(),
                    v_labels=gt_vertex['labels'].detach().cpu().numpy(),
                    v_points=gt_vertex['points'].detach().cpu().numpy(),
                    v_ids=gt_vertex['instance_ids'].detach().cpu().numpy(),)

                pred = data_sample['pred_center']
                pred_points = pred['points'].detach().cpu().numpy()
                pred_scores = pred['scores'].detach().cpu().numpy()
                pred_labels = pred['labels'].detach().cpu().numpy()
                pred_scale = pred['scales'].detach().cpu().numpy()
                if 'points_v' in pred:
                    pred_points_v = pred['points_v'].detach().cpu().numpy()
                    pred_scores_v = pred['scores_v'].detach().cpu().numpy()

                dets = []
                v_dets = []

                for label in range(len(self.dataset_meta['classes'])):
                    index = np.where(pred_labels == label)[0]
                    pred_dets = np.hstack(
                        [pred_points[index], 
                         pred_scores[index].reshape((-1, 1)),
                         pred_scale[index].reshape((-1, 1)),])
                    dets.append(pred_dets)
                    
                    if 'points_v' in pred:
                        v_pred_dets = np.concatenate(
                            [pred_points_v[index], 
                            pred_scores_v[index][..., np.newaxis]], axis=-1)
                        v_dets.append(v_pred_dets)

                self.results.append([dets, ann, v_dets])

    def _tpfp_symmetry(self, scores, preds, gts, img_sizes, sort_inds=None):
        if sort_inds is None:
            _, sort_inds = torch.sort(-scores)

        ### compute AP
        num_scales = 1
        tp = torch.zeros(num_scales, len(preds))
        fp = torch.zeros(num_scales, len(preds))
        taus = torch.cdist(preds.float(), gts.float(), p=2, compute_mode='donot_use_mm_for_euclid_dist')
        taus_min, taus_argmin = taus.min(dim=1)
        gt_covered = torch.zeros(len(gts), dtype=bool)
        tau_thr = self.tau_alpha * img_sizes

        k = 0
        pred_gt_pairs = {}
        for i in sort_inds:
            if taus_min[i] < tau_thr[i]:
                matched_gt = taus_argmin[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    pred_gt_pairs[i.item()] = (matched_gt.long().item(), tau_thr[i].item())
                    tp[k, i] = 1
                else:
                    fp[k, i] = 1
            else:
                fp[k, i] = 1

        return scores, tp, fp, gt_covered, pred_gt_pairs

    def update(self, dets, gts, nms_scale=0.0):
        tp_gt = torch.zeros(len(gts), dtype=bool)
        pred_gt_pairs = {}

        if len(dets) < 1:
            # No tp
            return None
        if len(gts) < 1:
            # No tp, All fp
            preds, scores, scales = dets[:, :2], dets[:, 2], dets[:, 3]
            scores = torch.tensor(scores)
            _, sort_inds = torch.sort(-scores)
            ### compute AP
            num_scales = 1
            tp = torch.zeros(num_scales, len(preds))
            fp = torch.ones(num_scales, len(preds))
            tp_gt = torch.zeros(len(gts), dtype=bool)
            pred_gt_pairs = {}
            return preds, scores, tp, fp, tp_gt, pred_gt_pairs

        preds, scores, scales = dets[:, :2], dets[:, 2], dets[:, 3]
        scores = torch.tensor(scores)
        preds = torch.tensor(preds)
        gts = torch.tensor(gts)
        scales = torch.tensor(scales)        
        sort_inds = None
        if nms_scale > 0:
            sort_inds = self._apply_nms(scores, preds, scales, nms_scale)
        scores, tp, fp, tp_gt, pred_gt_pairs = self._tpfp_symmetry(scores, preds, gts, scales, sort_inds)
        return preds, scores, tp, fp, tp_gt, pred_gt_pairs

    def _tpfp_symmetry_v(self, v_dets, v_gts, tau_thr):
        v_dets = torch.from_numpy(v_dets)
        v_gts = torch.from_numpy(v_gts)
        preds, scores = v_dets[..., :2], v_dets[..., 2]
        _, sort_inds = torch.sort(-scores)

        ### compute AP
        # scale 1
        tp = torch.zeros(len(preds))
        fp = torch.zeros(len(preds))
        taus = torch.cdist(preds.float(), v_gts.float(), p=2, compute_mode='donot_use_mm_for_euclid_dist')
        taus_min, taus_argmin = taus.min(dim=1)
        gt_covered = torch.zeros(len(v_gts), dtype=bool)

        for i in sort_inds:
            if taus_min[i] < tau_thr:
                matched_gt = taus_argmin[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        return scores, tp, fp

    def update_v(self, v_dets, v_gts, cls_ids, v_ids, pred_gt_pairs):
        scores, tps, fps = [], [], []
        num_gt = len(v_gts)
        pts = []
        gts = []

        if len(pred_gt_pairs) == 0:
            return None, None, None, num_gt, None

        for idx, gt in pred_gt_pairs.items():
            v_det = v_dets[idx]
            gt_id, tau = gt
            gt_id = cls_ids[gt_id]
            if len(v_gts[v_ids == gt_id]) > 0:
                v_gt = np.stack(v_gts[v_ids == gt_id])
                score, tp, fp = self._tpfp_symmetry_v(v_det, v_gt, tau)
                scores.append(score)
                tps.append(tp)
                fps.append(fp)
                pts.append(v_det)
                gts.append(v_gt)

        if len(scores) == 0:
            return None, None, None, num_gt, None

        return torch.cat(scores), torch.cat(tps).unsqueeze(0), torch.cat(fps).unsqueeze(0), num_gt, \
                (pred_gt_pairs, pts, scores, tps, fps, gts)

    def compute(self, cls_dets, cls_gts, v_dets, v_gts, cls_ids, v_ids, nms_scale=0.0):
        scores, tp, fp = [], [], []
        v_scores, v_tp, v_fp = [], [], []
        num_gt, num_gt_v = 0, 0
        v_infos = []
        
        for idx, (dets, gts) in enumerate(zip(cls_dets, cls_gts)):
            results = self.update(dets, gts, nms_scale)
            _num_gt = len(gts)
            num_gt += _num_gt

            # init
            pred_gt_pairs = {}
            if results is not None:
                # M detections 0 or N gts
                _preds, _scores, _tp, _fp, _tp_gt, pred_gt_pairs = results
                scores.append(_scores)
                tp.append(_tp)
                fp.append(_fp)

            _v_scores, _v_tp, _v_fp, _v_num_gt, _v_meta = self.update_v(
                v_dets[idx], v_gts[idx], cls_ids[idx], v_ids[idx], pred_gt_pairs)
            
            num_gt_v += _v_num_gt
            if _v_tp is not None:
                v_scores.append(_v_scores)
                v_tp.append(_v_tp)
                v_fp.append(_v_fp)
                v_infos.append((idx, _v_meta))

        ap = self.compute_ap(scores, tp, fp, num_gt)
        ap_v = self.compute_ap(v_scores, v_tp, v_fp, num_gt_v)

        return ap, ap_v, v_infos

    def compute_ap(self, scores, tp, fp, num_gt):
        if len(scores) < 1:
            return 0.0
        if num_gt == 0:
            return 0.0

        scores = torch.cat(scores, dim=-1)
        sort_inds = torch.argsort(-scores)
        tp = torch.cat(tp, dim=-1)
        fp = torch.cat(fp, dim=-1)
        tp = tp[:, sort_inds]
        fp = fp[:, sort_inds]
        tp = torch.cumsum(tp, dim=1)
        fp = torch.cumsum(fp, dim=1)
        recalls = tp / num_gt
        precisions = tp / torch.clamp((tp + fp), min=1e-5)

        num_scales = recalls.shape[0]
        ap = np.zeros(num_scales, dtype=np.float32)
        recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        
        for i in range(num_scales):
            for thr in recThrs:
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.numel() > 0 else 0
                ap[i] += prec
            
        ap /= 101 # replace 11 points (assume no ignorants)
        return ap[0]


    def get_cls_results(self, det_results, annotations, v_det_results, class_index, class_label):
        cls_dets = [det_result[class_index] for det_result in det_results]
        v_dets = [det_result[class_index] for det_result in v_det_results]
        cls_gts = []
        cls_ids = []
        v_gts = []
        v_ids = []
        cls_gt_inds_list = []

        for ann in annotations:
            cls_gt_inds = ann['labels'] == class_index
            cls_gts.append(ann['points'][cls_gt_inds, :])
            cls_ids.append(ann['ids'][cls_gt_inds])
            cls_gt_inds_list.append(cls_gt_inds)
            gt_inds = ann['v_labels'] == class_index
            v_gts.append(ann['v_points'][gt_inds, :])
            v_ids.append(ann['v_ids'][gt_inds])

        return cls_gt_inds_list, cls_dets, cls_gts, cls_ids, v_dets, v_gts, v_ids

    def compute_metrics(self, results=None) -> dict:
        if results is None:
            results = self.results

        preds, gts, v_preds = [list(x) for x in zip(*results)]        
        logger: MMLogger = MMLogger.get_current_instance()
        print_log('results:', logger)
        eval_results = OrderedDict()

        aps, aps_v = [], []
        cls_v_infos = []

        for i, c in enumerate(self.dataset_meta['classes']):
            cls_gt_inds, cls_dets, cls_gts, cls_ids, v_dets, v_gts, v_ids \
                = self.get_cls_results(preds, gts, v_preds, i, c)
            ap, ap_v, v_infos = self.compute(cls_dets, cls_gts, v_dets, v_gts, cls_ids, v_ids, nms_scale=0.0)
            eval_results['AP(%d)' % c] = ap
            eval_results['APV(%d)' % c] = ap_v
            aps.append(ap)
            aps_v.append(ap_v)
            cls_v_infos.append(v_infos)

        eval_results['AP(mean)'] = sum(aps) / len(aps)
        eval_results['APV(mean)'] = sum(aps_v) / len(aps_v)

        table_data = PrettyTable()
        for key, val in eval_results.items():
            if val == -1:
                table_data.add_column(key, ['N/A'])
            elif val == 0:
                table_data.add_column(key, [val])
            else:
                table_data.add_column(key, [round(val.item(), 5)])

        print_log('\n' + table_data.get_string(), logger=logger)

        return eval_results
