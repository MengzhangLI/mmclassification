_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/nih_chestxrays_bs16.py',
    '../_base_/schedules/nih_bs16_swin.py',
    '../_base_/default_runtime.py',
]

# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/densenet/densenet121_4xb256_in1k_20220426-07450f99.pth'  # noqa
checkpoint = './pretrain/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'  # noqa
model = dict(
    type='ImageClassifier',
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    head=dict(
        _delete_=True,
        type='MultiLabelLinearClsHead',
        num_classes=14,
        in_channels=768,
        thr=0.5,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

# dataset settings
train_dataloader = dict(batch_size=16)

default_hooks = dict(
    # save last three checkpoints
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='auto',
        rule='greater'))

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (4 GPUs) x (256 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
