# model settings
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=64),
    backbone=dict(
        type='DIFF',
        diff_config=dict(aggregation_type="direct_aggregation",
                         fine_type = 'deep_fusion',
                         projection_dim=[2048, 2048, 1024, 512],
                        #  projection_dim=[2048, 1024, 512, 256],
                        #  fine_type = 'upsample',
                         projection_dim_x4=256,
                         model_id="../stable-diffusion-v1-5",
                         diffusion_mode="inversion",
                         input_resolution=[512, 512],
                         prompt="",
                         negative_prompt="",
                         guidance_scale=-1,
                         scheduler_timesteps=[50, 25],
                         save_timestep=[0],
                         num_timesteps=1,
                         idxs_resnet=[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [
                             1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],
                         idxs_ca=[[1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2]],
                         s_tmin=10,
                         s_tmax=250,
                         do_mask_steps=True,
                         classes=('bicycle', 'bus', 'car', 'motorcycle',
                                  'person', 'rider', 'train', 'truck')
                         )
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=True,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

