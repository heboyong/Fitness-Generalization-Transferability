# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.models.utils import (rename_loss_dict,
                                reweight_loss_dict)
from mmdet.structures.bbox import bbox2roi
from ..utils import unpack_gt_instances


from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector
from ..losses import KDLoss

def bbox_to_mask(batch_data_samples, N, H, W, class_names):
    batch_masks = torch.full((N, H, W), 0, dtype=torch.long)
    batch_labels = []
    for i in range(N):
        gt_instance = batch_data_samples[i].gt_instances
        bboxes = gt_instance["bboxes"]
        labels = gt_instance["labels"]
        sample_labels = set([class_names[label.item()] for label in labels])
        if sample_labels:
            label_string = "A photo of " + ", ".join(sample_labels)
        else:
            label_string = ""
        batch_labels.append(label_string)
        bbox_areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
        sorted_indices = sorted(range(len(bboxes)), key=lambda idx: bbox_areas[idx], reverse=True)
        for idx in sorted_indices:
            bbox = bboxes[idx]
            x1, y1, x2, y2 = bbox.int()
            batch_masks[i, y1:y2, x1:x2] = 1

    return batch_masks, batch_labels


@MODELS.register_module()
class DiffusionDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 auxiliary_branch_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.auxiliary_branch_cfg = auxiliary_branch_cfg
        
        self.loss_cls_kd = MODELS.build(self.auxiliary_branch_cfg['loss_cls_kd'])
        self.loss_reg_kd = MODELS.build(self.auxiliary_branch_cfg['loss_reg_kd'])
        self.apply_auxiliary_branch = self.auxiliary_branch_cfg['apply_auxiliary_branch']
        
        self.loss_feature = KDLoss(loss_weight=1.0, loss_type='mse')
        
        self.class_maps = backbone['diff_config']['classes']

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                    bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs: Tensor, ref_masks=None, ref_labels=None) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if ref_masks != None and ref_labels != None:
            x = self.backbone(batch_inputs, ref_masks, ref_labels)
        else:
            x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs,)
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             return_feature=False) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """

        # Extract feature with mask input
        ###########################################################################
        if self.apply_auxiliary_branch:
            N, _, H, W = batch_inputs.shape
            ref_masks, ref_labels = bbox_to_mask(batch_data_samples, N, H, W, self.class_maps)
            x_w_ref = self.extract_feat(batch_inputs, ref_masks, ref_labels)
            x_wo_ref = self.extract_feat(batch_inputs)
        ###########################################################################
        else:
            x_wo_ref = self.extract_feat(batch_inputs)

        losses = dict()

        
         # noref branch
        ###########################################################################
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list_noref = self.rpn_head.loss_and_predict(
                x_wo_ref, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rename_loss_dict('noref_', rpn_losses))
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list_noref = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x_wo_ref, rpn_results_list_noref,
                                        batch_data_samples)
        losses.update(rename_loss_dict('noref_', roi_losses))
        ###########################################################################
        
        # ref branch
        ###########################################################################
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list_ref = self.rpn_head.loss_and_predict(
                x_w_ref, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rename_loss_dict('ref_', rpn_losses))
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list_ref = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x_w_ref, rpn_results_list_ref,
                                        batch_data_samples)
        losses.update(rename_loss_dict('ref_', roi_losses))
        ##########################################################################

        # object-kd loss
        ##############################################################################################################
        if self.apply_auxiliary_branch:
            # Apply cross-kd in ROI head
            roi_losses_kd = self.roi_head_loss_with_kd(
                x_wo_ref, x_w_ref, rpn_results_list_ref, batch_data_samples)
            losses.update(roi_losses_kd)
        ##############################################################################################################
        
        # feature kd loss
        ##############################################################################################################
        if self.apply_auxiliary_branch:
            feature_loss = dict()
            feature_loss['pkd_feature_loss'] = 0
            for i, (x_wo, x_w) in enumerate(zip(x_wo_ref, x_w_ref)):
                layer_loss = self.loss_feature(x_wo, x_w)
                feature_loss['pkd_feature_loss'] += layer_loss/len(x_wo_ref)
            losses.update(feature_loss)
        ##############################################################################################################
        
        if not return_feature:
            return losses
        else:
            return losses, x_wo_ref

    def roi_head_loss_with_kd(self,
                              x_wo_ref, x_w_ref, rpn_results_list_ref, batch_data_samples):
        assert len(rpn_results_list_ref) == len(batch_data_samples)
         
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        roi_head = self.roi_head

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results_ref = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list_ref[i]
            # rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x_w_ref])
            sampling_results_ref.append(sampling_result)
                      
        losses = dict()
        # bbox head loss
        if roi_head.with_bbox:
            bbox_results = self.bbox_loss_with_kd(
                x_wo_ref, x_w_ref, sampling_results_ref)
            losses.update(bbox_results['loss_bbox_kd'])

        return losses

    def bbox_loss_with_kd(self, x_wo_ref, x_w_ref, sampling_results_ref):
        rois_ref = bbox2roi([res.priors for res in sampling_results_ref])

        roi_head = self.roi_head
        ref_bbox_results = roi_head._bbox_forward(x_w_ref, rois_ref)
        reused_bbox_results = roi_head._bbox_forward(x_wo_ref, rois_ref)

        losses_kd = dict()
        # classification KD
        reused_cls_scores = reused_bbox_results['cls_score']
        ref_cls_scores = ref_bbox_results['cls_score']
        avg_factor = sum([res.avg_factor for res in sampling_results_ref])
        loss_cls_kd = self.loss_cls_kd(
            ref_cls_scores,
            reused_cls_scores,
            avg_factor=avg_factor)
        losses_kd['loss_cls_kd'] = loss_cls_kd

        # l1 loss
        num_classes = roi_head.bbox_head.num_classes
        reused_bbox_preds = reused_bbox_results['bbox_pred']
        ref_bbox_preds = ref_bbox_results['bbox_pred']
        ref_cls_scores = ref_cls_scores.softmax(dim=1)[:, :num_classes]
        reg_weights, reg_distill_idx = ref_cls_scores.max(dim=1)
        if not roi_head.bbox_head.reg_class_agnostic:
            reg_distill_idx = reg_distill_idx[:, None, None].repeat(1, 1, 4)
            reused_bbox_preds = reused_bbox_preds.reshape(-1, num_classes, 4)
            reused_bbox_preds = reused_bbox_preds.gather(
                dim=1, index=reg_distill_idx)
            reused_bbox_preds = reused_bbox_preds.squeeze(1)
            ref_bbox_preds = ref_bbox_preds.reshape(-1, num_classes, 4)
            ref_bbox_preds = ref_bbox_preds.gather(
                dim=1, index=reg_distill_idx)
            ref_bbox_preds = ref_bbox_preds.squeeze(1)

        loss_reg_kd = self.loss_reg_kd(
            ref_bbox_preds,
            reused_bbox_preds,
            weight=reg_weights[:, None],
            avg_factor=reg_weights.sum() * 4)
        losses_kd['loss_reg_kd'] = loss_reg_kd

        bbox_results = dict()
        for key, value in ref_bbox_results.items():
            bbox_results['ref_' + key] = value
        for key, value in reused_bbox_results.items():
            bbox_results['reused_' + key] = value
        bbox_results['loss_bbox_kd'] = losses_kd
        return bbox_results

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True,
                return_feature=False):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)
        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        if not return_feature:
            return batch_data_samples
        else:
            return batch_data_samples, x
