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
    batch_size=2,
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
        n_samples=8,
        pipeline=train_pipeline,))
train_dataloader = val_dataloader
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

num_classes = 6
classes = [2, 3, 4, 5, 6, 8]
point_cloud_range = [-1, -1, 0.0, 1, 1, 4.0]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'


model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        pad_size_divisor=32,
        pad_seg=True,
        seg_pad_value=0),
    type='EgoRotDETR',
    focal_length=1000,
    num_queries=800,
    num_feature_levels=4,
    with_box_refine=False,
    as_two_stage=False,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(  # DeformableDetrTransformerEncoder
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        num_layers=6, layer_cfg=dict(
        self_attn_cfg=dict(
            type='HalfMultiScaleDeformableAttention',
            embed_dims=256,
            num_levels=1),
        cross_attn_cfg=dict(
            type='SpatialCrossAttention',
            num_cams=1,
            pc_range=point_cloud_range,
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=256,
                num_points=8,
                num_levels=4,
                ),
            embed_dims=256,
        )
    )
        ),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256,
                num_levels=1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type='RotDETRPriorHead',
        # vertex_mode=0,
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        # ori_size_val=True,
        # legacy_angle=True,
        # use_maxscore=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=10.0),
        ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='RotationAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=1.0),
                dict(type='PointL1Cost', weight=10.0,),
            ])),
    test_cfg=dict(max_per_img=800))

