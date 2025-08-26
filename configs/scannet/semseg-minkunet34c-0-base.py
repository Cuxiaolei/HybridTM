_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 3  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = True
enable_amp = True

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(type="MinkUNet34C", in_channels=6, out_channels=20),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 100
optimizer = dict(type="SGD", lr=0.05, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=10000.0,
)

# dataset settings
dataset_type = "ScanNetDataset"
data_root = "/root/autodl-tmp/data/data_scannet_tower"

data = dict(
    num_classes=3,
    ignore_index=-1,
    names=["class_0", "class_1", "class_2"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # 1. 坐标几何增强（核心）
            dict(type="CenterShift", apply_z=True),  # 坐标中心化（稳定几何基准）
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),  # 随机丢点，模拟遮挡
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),  # 绕z轴旋转（适应场景方向差异）
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),  # x/y轴微旋转（适应倾斜）
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),  # 尺度缩放（适应不同距离的点云）
            dict(type="RandomFlip", p=0.5),  # 随机翻转（提升对称性鲁棒性）
            dict(type="RandomJitter", sigma=0.005, clip=0.02),  # 坐标微抖动（抗噪声）

            # 2. 颜色增强（针对你的颜色特征）
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),  # 自动对比度调整
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),  # 颜色偏移（模拟光照变化）
            dict(type="ChromaticJitter", p=0.95, std=0.05),  # 颜色抖动（增强鲁棒性）

            # 3. 规整化与裁剪
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,  # 生成网格坐标，适配模型输入
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),  # 随机裁剪（控制点数）
            dict(type="SphereCrop", point_max=10000, mode="random"),  # 限制最大点数（防显存溢出）
            dict(type="CenterShift", apply_z=False),  # 二次中心化（微调坐标）
            dict(type="NormalizeColor"),  # 颜色归一化（将RGB映射到0-1范围）

            # 4. 数据转换
            dict(type="ToTensor"),  # 转为PyTorch张量
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),  # 保留坐标、网格坐标、标签
                feat_keys=("color", "normal",),  # 仅保留颜色特征（你的特征只有颜色）
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="center"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "color", "normal"),
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ],
            # aug_transform=[
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[0],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[1 / 2],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[1],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            #     [
            #         dict(
            #             type="RandomRotateTargetAngle",
            #             angle=[3 / 2],
            #             axis="z",
            #             center=[0, 0, 0],
            #             p=1,
            #         )
            #     ],
            # ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
        ),
    ),
)
