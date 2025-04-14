_base_ = [
    './_base_.py',
]
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
        type='RotDETRHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
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

