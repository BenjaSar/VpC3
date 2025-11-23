# =============================================================================
# Swin Transformer + Mask R-CNN Configuration for CubiCasa5K
# Optimizado para NVIDIA Quadro P1000 (4GB VRAM)
# =============================================================================

import torch

# Model architecture
model = dict(
    type="MaskRCNN",
    # Backbone: Swin Transformer Tiny (para 4GB VRAM)
    backbone=dict(
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,  # Checkpoint para ahorrar memoria
        convert_weights=True,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
        ),
    ),
    # Neck: Feature Pyramid Network
    neck=dict(
        type="FPN", in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5
    ),
    # RPN Head
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    # RoI Head
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,  # CubiCasa5K room classes
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=dict(
            type="FCNMaskHead",
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=15,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
    ),
    # Training and testing cfg
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            mask_size=28,
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
        ),
    ),
)

# Dataset settings
dataset_type = "CubiCasaDataset"
data_root = "data/cubicasa5k/"

# CubiCasa5K Room Classes (15 principales)
classes = [
    "Bedroom",
    "Kitchen",
    "Living Room",
    "Bathroom",
    "Dining Room",
    "Corridor",
    "Balcony",
    "Storage",
    "Office",
    "Laundry",
    "Garage",
    "Terrace",
    "Closet",
    "Entrance",
    "Other",
]

# Image preprocessing pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomRotate", prob=0.5, angle=[-90, 90]),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type="Resize", scale=(512, 512), keep_ratio=True),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# Data loaders
data = dict(
    samples_per_gpu=2,  # Batch size optimizado para 4GB VRAM
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/train.json",
        img_prefix=data_root + "images/train/",
        pipeline=train_pipeline,
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/val.json",
        img_prefix=data_root + "images/val/",
        pipeline=test_pipeline,
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/test.json",
        img_prefix=data_root + "images/test/",
        pipeline=test_pipeline,
        classes=classes,
    ),
)

# Optimizer
optimizer = dict(
    type="AdamW",
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# Learning rate schedule
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11]
)

# Runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=12)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

evaluation = dict(interval=1, metric=["bbox", "segm"], save_best="bbox_mAP")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# GPU settings
gpu_ids = [0]
seed = 42
deterministic = False

# Mixed precision training (para optimizar 4GB VRAM)
fp16 = dict(loss_scale="dynamic")

# Working directory
work_dir = "./checkpoints"

# Load from checkpoint
load_from = None
resume_from = None
workflow = [("train", 1)]

# Custom settings
custom_hooks = []

# Distance to meters conversion (assumption: 1 pixel = 0.02 meters)
pixel_to_meter = 0.02
