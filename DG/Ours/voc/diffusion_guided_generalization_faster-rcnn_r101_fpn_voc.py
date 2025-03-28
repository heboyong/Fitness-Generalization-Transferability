_base_ = [
    '../../_base_/models/diffusion_guided_generalization_faster_rcnn_r101_fpn.py',
    '../../_base_/dg_setting/semi_20k.py',
    '../../_base_/datasets/voc/voc_aug.py'
]

detector = _base_.model
detector.data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=64)

detector.detector.roi_head.bbox_head.num_classes = 20
detector.diff_model.config = 'DG/Ours/voc/diffusion_detector_voc.py'
detector.diff_model.pretrained_model = '.pth'


model = dict(
    _delete_=True,
    type='DomainGeneralizationDetector',
    detector=detector,
    data_preprocessor=detector.data_preprocessor,
    train_cfg=dict(
    burn_up_iters=_base_.burn_up_iters,
    cross_loss_cfg=dict(
        enable_cross_loss=True,
        cross_loss_weight=0.5
    ),
    feature_loss_cfg=dict(
        enable_feature_loss=True,
        feature_loss_type='mse',
        feature_loss_weight=0.5
    ),
    kd_cfg=dict(
        loss_cls_kd=dict(type='KnowledgeDistillationKLDivLoss',
                            class_reduction='sum', T=3, loss_weight=1.0),
        loss_reg_kd=dict(type='L1Loss', loss_weight=1.0),
    ))
)

