_base_ = [
    '../../_base_/models/faster-rcnn_diff_fpn.py',
    '../../_base_/da_setting/da_20k.py',
    '../../_base_/datasets/city_to_foggy/cityscapes_aug.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 8
