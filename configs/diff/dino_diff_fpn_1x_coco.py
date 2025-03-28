_base_ = [
    '../_base_/models/dino_diff.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomChoice',
#         transforms=[
#             [
#                 dict(
#                     type='RandomChoiceResize',
#                     scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                             (736, 1333), (768, 1333), (800, 1333)],
#                     keep_ratio=True)
#             ],
#             [
#                 dict(
#                     type='RandomChoiceResize',
#                     # The radio of all image in train dataset < 7
#                     # follow the original implement
#                     scales=[(400, 4200), (500, 4200), (600, 4200)],
#                     keep_ratio=True),

#                 dict(
#                     type='RandomChoiceResize',
#                     scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                             (736, 1333), (768, 1333), (800, 1333)],
#                     keep_ratio=True),
#                 dict(
#                     type='RandomCrop',
#                     crop_type='absolute_range',
#                     crop_size=(384, 600),
#                     allow_negative_crop=True),
#             ]
#         ]),
#     dict(type='PackDetInputs')
# ]
# train_dataloader = dict(batch_size=1,
# dataset=dict(filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

# train_dataloader = dict(batch_size=8,
# dataset=dict(filter_cfg=dict(filter_empty_gt=False)))

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

auto_scale_lr = dict(enable=True, base_batch_size=16)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]
