_base_ = [
    '../../_base_/models/faster-rcnn_diff_fpn.py',
    '../../_base_/da_setting/da_20k.py',
    '../../_base_/datasets/voc_to_clipart/voc_aug.py'
]

detector = _base_.model
detector.roi_head.bbox_head.num_classes = 20

detector.backbone.diff_config.classes = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

detector.backbone.diff_config.scheduler_timesteps = [400, 200]