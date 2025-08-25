import os
import numpy as np
from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class S3DISDataset(DefaultDataset):
    def __init__(self, split, data_root, **kwargs):
        super().__init__(split=split, data_root=data_root, **kwargs)
        self.data_list = self._get_scene_list()
        # 打印数据加载配置信息
        print(
            f"[S3DISDataset] 初始化完成 | 数据集划分: {split} | 数据根目录: {data_root} | 场景数量: {len(self.data_list)}")

    def _get_scene_list(self):
        """从txt文件中读取场景路径列表，增加详细的错误检查"""
        split_file = os.path.join(self.data_root, f"{self.split}_scenes.txt")
        print(f"[读取场景列表] 路径: {split_file}")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"场景列表文件不存在: {split_file}\n请检查data_root是否正确或是否生成了划分文件")

        with open(split_file, "r") as f:
            scenes = [line.strip() for line in f.readlines() if line.strip()]

        if len(scenes) == 0:
            raise ValueError(f"场景列表文件为空: {split_file}\n请检查划分文件是否正确生成")

        # 验证场景文件路径并构建完整路径
        full_scene_paths = []
        for scene_rel_path in scenes:
            full_path = os.path.join(self.data_root, scene_rel_path)
            full_scene_paths.append(full_path)

            # 检查文件是否存在
            if not os.path.exists(full_path):
                print(f"[警告] 场景文件不存在: {full_path}")

        # 统计有效场景数量
        valid_count = sum(1 for path in full_scene_paths if os.path.exists(path))
        print(
            f"[场景统计] 总场景数: {len(full_scene_paths)} | 有效场景数: {valid_count} | 无效场景数: {len(full_scene_paths) - valid_count}")

        if valid_count == 0:
            raise RuntimeError("没有找到任何有效的场景文件，请检查数据路径是否正确")

        return full_scene_paths

    def load_data(self, idx):
        """加载并处理单一场景数据，对坐标、颜色、法向量进行归一化"""
        scene_path = self.data_list[idx]
        scene_name = os.path.basename(scene_path).replace(".npy", "")

        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"点云文件不存在: {scene_path}")

        # 加载10通道数据 (3坐标 + 3颜色 + 3法向量 + 1标签)
        data = np.load(scene_path).astype(np.float32)

        # 检查数据维度是否正确
        if data.ndim != 2 or data.shape[1] != 10:
            raise ValueError(f"数据格式错误: {scene_name} 应为 (N, 10) 的数组，实际为 {data.shape}")

        # 解析数据通道
        coord = data[:, :3]  # 坐标 (X, Y, Z)
        color = data[:, 3:6]  # 颜色 (R, G, B)
        normal = data[:, 6:9]  # 法向量 (NX, NY, NZ)
        segment = data[:, 9].astype(np.int32)  # 标签

        # 1. 坐标归一化: 缩放到 [0, 1] 范围
        coord_min = np.min(coord, axis=0)
        coord_max = np.max(coord, axis=0)
        coord_range = coord_max - coord_min
        # 处理坐标范围为0的特殊情况（避免除零）
        coord_range[coord_range < 1e-6] = 1.0
        coord_normalized = (coord - coord_min) / coord_range

        # 2. 颜色归一化: 将0-255的uint8转为0-1的float32
        # 处理可能的异常值（确保在0-255范围内）
        color = np.clip(color, 0, 255)
        color_normalized = color / 255.0  # 转换为[0, 1]范围

        # 3. 法向量归一化: 单位化法向量（确保长度为1）
        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
        # 处理法向量长度为0的特殊情况（避免除零）
        normal_norm[normal_norm < 1e-6] = 1.0
        normal_normalized = normal / normal_norm

        # 调试信息（每10个场景打印一次）
        if idx % 10 == 0:
            print(f"[加载场景] {scene_name} | 点数: {len(coord)} | "
                  f"坐标范围: [{coord_min.min():.2f}, {coord_max.max():.2f}] | "
                  f"颜色范围: [{color.min():.0f}, {color.max():.0f}] | "
                  f"法向量范围: [{normal.min():.2f}, {normal.max():.2f}]")

        return {
            "coord": coord_normalized,  # 归一化后的坐标
            "color": color_normalized,  # 归一化后的颜色
            "normal": normal_normalized,  # 归一化后的法向量
            "segment": segment,  # 原始标签
            "scene_name": scene_name,  # 场景名称
        }
