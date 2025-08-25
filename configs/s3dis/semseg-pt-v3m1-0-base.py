_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 3  # bs: total bs in all gpus
num_worker = 0
mix_prob = 0.8
empty_cache = True
enable_amp = True

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=3,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 100
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
dataset_type = "S3DISDataset"
data_root = "/root/autodl-tmp/data/data_s3dis_pointNeXt"

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
                type="Copy",
                keys_dict={"coord": "origin_coord", "segment": "origin_segment"},
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "origin_coord",
                    "segment",
                    "origin_segment",
                ),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
                feat_keys=("color", "normal",),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
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
                keys=("coord", "color", "normal"),
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ],
            # aug_transform=[
            #     [dict(type="RandomScale", scale=[0.9, 0.9])],
            #     [dict(type="RandomScale", scale=[0.95, 0.95])],
            #     [dict(type="RandomScale", scale=[1, 1])],
            #     [dict(type="RandomScale", scale=[1.05, 1.05])],
            #     [dict(type="RandomScale", scale=[1.1, 1.1])],
            #     [
            #         dict(type="RandomScale", scale=[0.9, 0.9]),
            #         dict(type="RandomFlip", p=1),
            #     ],
            #     [
            #         dict(type="RandomScale", scale=[0.95, 0.95]),
            #         dict(type="RandomFlip", p=1),
            #     ],
            #     [
            #         dict(type="RandomScale", scale=[1, 1]),
            #         dict(type="RandomFlip", p=1),
            #     ],
            #     [
            #         dict(type="RandomScale", scale=[1.05, 1.05]),
            #         dict(type="RandomFlip", p=1),
            #     ],
            #     [
            #         dict(type="RandomScale", scale=[1.1, 1.1]),
            #         dict(type="RandomFlip", p=1),
            #     ],
            # ],
        ),
    ),
)
