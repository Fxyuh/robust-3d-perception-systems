_base_ = [
    './centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py'
]

# 训练集外层还有一个 CBGSDataset，因此需要进入两层 dataset。
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        dataset=dict(
            metainfo=dict(version='v1.0-mini')
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        metainfo=dict(version='v1.0-mini')
    )
)

test_dataloader = val_dataloader

# 先训练2轮，验证完整训练链路。
train_cfg = dict(
    max_epochs=2,
    val_interval=1
)

# 原配置基准总批量为32；单卡batch size为2时自动缩放学习率。
auto_scale_lr = dict(
    enable=True,
    base_batch_size=32
)

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=2
    )
)
