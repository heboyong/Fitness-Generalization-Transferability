_base_ = [
    '../../_base_/models/diffusion_guided_adaptation_faster_rcnn_r101_fpn.py',
    '../../_base_/da_setting/semi_20k.py',
    '../../_base_/datasets/city_to_bdd100k/semi_cityscapes_aug.py'
]

detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=64)

detector.detector.roi_head.bbox_head.num_classes = 8
detector.diff_model.config = 'DA/Ours/city_to_bdd100k/diffusion_detector_cityscapes_2.1.py'
detector.diff_model.pretrained_model = '.pth'

model = dict(
    _delete_=True,
    type='DomainAdaptationDetector',
    detector=detector,
    data_preprocessor=dict(
        type='MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    train_cfg=dict(
        detector_cfg=dict(type='SemiBaseDiff', burn_up_iters=_base_.burn_up_iters),
        
        feature_loss_cfg=dict(feature_loss_type='mse', feature_loss_weight=1.0)
        )

)
