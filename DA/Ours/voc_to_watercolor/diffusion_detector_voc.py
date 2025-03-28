_base_ = [
    '../../_base_/models/faster-rcnn_diff_fpn.py',
    '../../_base_/da_setting/da_20k.py',
    '../../_base_/datasets/voc_to_comic/voc_aug.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 6


detector.backbone.diff_config.classes = ("bicycle", "bird", "car", "cat", "dog", "person",)

detector.backbone.diff_config.scheduler_timesteps = [400, 200]