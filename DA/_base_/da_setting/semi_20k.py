default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1000, by_epoch=False,
                    max_keep_ckpts=1, save_best=['teacher/coco/bbox_mAP_50', 'student/coco/bbox_mAP_50']),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# env_cfg = dict(
#     cudnn_benchmark=False,
#     mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
#     dist_cfg=dict(backend='nccl'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=1, by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

# 进入半监督训练的iter数，如果burn_up_iters=0，则模型一开始进入半监督，源域和目标域一起进行训练，teacher模型进行EMA参数更新
# 如果burn_up_iters<max_iters, 则模型在指定iter进入半监督，源域和目标域一起进行训练，teacher模型进行EMA参数更新
# 如果burn_up_iters>max_iters, 则模型只进行源域训练

burn_up_iters = 12000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
val_cfg = dict(type='TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20000,
        by_epoch=False,
        milestones=[18000],
        gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=2, norm_type=2))

launcher = 'none'
auto_scale_lr = dict(enable=True, base_batch_size=32)
custom_hooks = [dict(type='AdaptiveTeacherHook', burn_up_iters=burn_up_iters, momentum=0.0004)]
find_unused_parameters = True
