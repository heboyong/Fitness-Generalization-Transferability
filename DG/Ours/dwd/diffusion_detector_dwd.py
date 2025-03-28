_base_ = [
    '../../_base_/models/faster-rcnn_diff_fpn.py',
    '../../_base_/dg_setting/dg_20k.py',
    '../../_base_/datasets/dwd/dwd_aug.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 7
detector.backbone.diff_config.classes = (
    'bicycle', 'bus', 'car', 'motorcycle', 'person', 'rider', 'truck')
