# pointcept/datasets/s3dis.py
import os
import numpy as np
from .builder import DATASETS
from .defaults import DefaultDataset  # 继承基础数据集类

@DATASETS.register_module()
class S3DISDataset(DefaultDataset):  # 自定义类名，避免与原有冲突
    def __init__(self, split, data_root, **kwargs):
        super().__init__(split=split, data_root=data_root,** kwargs)
        # 读取场景列表文件（train_scenes.txt/val_scenes.txt/test_scenes.txt）
        self.data_list = self._get_scene_list()

    def _get_scene_list(self):
        """从txt文件中读取场景路径列表"""
        split_file = os.path.join(self.data_root, f"{self.split}_scenes.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"场景列表文件不存在: {split_file}")
        with open(split_file, "r") as f:
            scenes = [line.strip() for line in f.readlines() if line.strip()]
        # 拼接完整路径（data_root + 相对路径，如"merged/Area_1.npy"）
        return [os.path.join(self.data_root, scene) for scene in scenes]

    def load_data(self, idx):
        scene_path = self.data_list[idx]
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"点云文件不存在: {scene_path}")

        # 加载10通道数据（3坐标 + 3特征 + 3法向量 + 1标签）
        data = np.load(scene_path).astype(np.float32)

        # 解析数据（根据10通道顺序调整索引）
        coord = data[:, :3]  # 前3维：坐标
        color = data[:, 3:6]  # 3-5维：特征（如颜色）
        normal = data[:, 6:9]  # 6-8维：法向量
        segment = data[:, 9].astype(np.int32)  # 第9维：标签

        # 坐标归一化：将坐标缩放到[0, 1]范围
        coord_min = np.min(coord, axis=0)
        coord_max = np.max(coord, axis=0)
        coord_range = coord_max - coord_min
        # 避免除以零
        coord_range[coord_range == 0] = 1.0
        coord = (coord - coord_min) / coord_range

        # 法向量归一化：单位化法向量
        normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
        # 避免除以零
        normal_norm[normal_norm == 0] = 1.0
        normal = normal / normal_norm

        return {
            "coord": coord,
            "color": color,
            "normal": normal,  # 新增法向量字段
            "segment": segment,
            "scene_name": os.path.basename(scene_path).replace(".npy", "")
        }
