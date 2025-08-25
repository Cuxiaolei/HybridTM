import os
import numpy as np
from torch.utils.data import Dataset
from copy import deepcopy
from .builder import DATASETS
from .transform import Compose
@DATASETS.register_module()
class MyDataset(Dataset):
    def __init__(self,
                 split,
                 data_root,
                 transform=None,
                 test_mode=False,
                 ignore_index=-1,
                 loop=1,
                 coord_normalize=True,  # 是否归一化坐标
                 color_normalize=True):  # 是否归一化颜色
        self.split = split
        self.data_root = data_root
        self.transform = Compose(transform)
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.loop = loop if not test_mode else 1  # 测试时不循环
        self.coord_normalize = coord_normalize
        self.color_normalize = color_normalize

        # 加载场景列表
        split_file = os.path.join(data_root, f"{split}_scenes.txt")
        with open(split_file, 'r') as f:
            self.scene_list = [line.strip() for line in f.readlines()]

        # 过滤无效文件
        self.scene_list = [s for s in self.scene_list if os.path.exists(os.path.join(data_root, s))]

        # 缓存机制
        self.cache = {}

    def __len__(self):
        return len(self.scene_list) * self.loop

    def normalize_coordinates(self, coord):
        """归一化坐标到[-1, 1]范围"""
        if not self.coord_normalize:
            return coord

        # 计算边界框
        min_val = np.min(coord, axis=0)
        max_val = np.max(coord, axis=0)
        center = (min_val + max_val) / 2
        scale = np.max(max_val - min_val) / 2.0

        # 防止除以零
        if scale < 1e-6:
            scale = 1e-6

        # 归一化到中心在原点，尺度为1
        return (coord - center) / scale

    def normalize_colors(self, color):
        """归一化颜色值到[0, 1]范围（假设原始颜色是0-255）"""
        if not self.color_normalize:
            return color

        # 如果颜色值在0-255范围内，归一化到0-1
        if np.max(color) > 1.0 + 1e-6:  # 考虑浮点数误差
            return color / 255.0
        return color  # 已经归一化的情况

    def get_data(self, idx):
        idx = idx % len(self.scene_list)
        scene_path = self.scene_list[idx]
        full_path = os.path.join(self.data_root, scene_path)

        if idx in self.cache:
            return deepcopy(self.cache[idx])

        # 加载点云数据 (10个通道: 坐标3 + 颜色3 + 法向量3 + 标签1)
        data = np.load(full_path).astype(np.float32)

        # 关键：限制单样本最大点数量（根据内存调整，建议先设20万）
        MAX_POINTS = 200000  # 可逐步增大测试（如30万、50万）
        if len(data) > MAX_POINTS:
            # 随机采样固定数量的点（避免内存超限）
            indices = np.random.choice(len(data), MAX_POINTS, replace=False)
            data = data[indices]  # 裁剪点云

        # 解析数据（保持原始字段分离）
        coord = data[:, :3]  # 坐标
        color = data[:, 3:6]  # 颜色（单独保留）
        norm = data[:, 6:9]  # 法向量（单独保留）
        segment = data[:, 9].astype(np.int32)  # 标签

        # 对坐标和颜色进行归一化（保持不变）
        coord = self.normalize_coordinates(coord)
        color = self.normalize_colors(color)

        # 不合并为strength，而是单独存储color和norm
        data_dict = {
            'coord': coord,
            'color': color,  # 单独存储颜色
            'normal': norm,  # 单独存储法向量
            'segment': segment
        }

        # 测试模式下的修改（同步保留原始color和normal）
        if self.test_mode:
            data_dict['origin_segment'] = segment.copy()
            data_dict['raw_coord'] = data[:, :3].copy()
            data_dict['raw_color'] = data[:, 3:6].copy()  # 原始颜色
            data_dict['raw_normal'] = data[:, 6:9].copy()  # 原始法向量

        self.cache[idx] = deepcopy(data_dict)
        return data_dict

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        assert idx < len(self)
        data_dict = self.get_data(idx)

        # 应用测试变换
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        # 如果有测试增强配置
        if hasattr(self, 'test_cfg') and 'aug_transform' in self.test_cfg:
            data_dict_list = []
            for aug in self.test_cfg['aug_transform']:
                augmented = deepcopy(data_dict)
                for t in aug:
                    augmented = t(augmented)
                data_dict_list.append(augmented)

            # 应用后处理变换
            for i in range(len(data_dict_list)):
                for t in self.test_cfg['post_transform']:
                    data_dict_list[i] = t(data_dict_list[i])

            return dict(
                voting_list=data_dict_list,
                name=self.scene_list[idx % len(self.scene_list)]
            )

        # 普通测试处理
        if hasattr(self, 'test_cfg') and 'post_transform' in self.test_cfg:
            for t in self.test_cfg['post_transform']:
                data_dict = t(data_dict)
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def get_data_name(self, idx):
        return self.scene_list[idx % len(self.scene_list)]
