default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=100, 
        # save_best=['AP(mean)', 'APV(mean)',],
        rule='greater'
        ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

num_classes = 6
classes = [2, 3, 4, 5, 6, 8]
## eval train data 
dataset_type = 'JointDendiDataset'
data_root = 'sym_datasets/DENDI'
train_pipeline = [
    dict(type='PackJointInputs')
]

seed = 0
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True, seed=seed),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        rot_annfile='rotation_refined_allfold_train.pt',
        split='train',
        input_size=(1333, 1333),
        classes=classes,
        resize_val=True,
        remove_invisible=True,
        flip_aug=True,
        rotate_aug=True,
        rotate_deg=20,
        pipeline=train_pipeline,))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        rot_annfile='rotation_refined_allfold_val.pt',
        split='val',
        input_size=(1333, 1333),
        classes=classes,
        remove_invisible=True,
        resize_val=True,
        pipeline=train_pipeline,))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        rot_annfile='rotation_refined_fix_test.pt',
        split='test',
        input_size=(1333, 1333),
        classes=classes,
        remove_invisible=True,
        resize_val=True,
        filter_off=True,
        pipeline=train_pipeline,))

val_evaluator = [dict(type='DENDIVertexMetricV2', classes=classes, ori_size_val=True),]
test_evaluator = [dict(type='DENDIVertexMetricV2', classes=classes, ori_size_val=True),]


# learning policy
max_epochs = 200
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

auto_scale_lr = dict(base_batch_size=32)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        })
        )
