# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
classes = ("bicycle", "bird", "car", "cat", "dog", "person",)

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1200, 600), keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute',
        crop_size=(512, 512),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1200, 600), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='VOC/VOC0712/train.json',
        data_prefix=dict(img='VOC/VOC0712/JPEGImages/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='VOC/VOC0712/test.json',
        data_prefix=dict(img='VOC/VOC0712/JPEGImages/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'VOC/VOC0712/test.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
