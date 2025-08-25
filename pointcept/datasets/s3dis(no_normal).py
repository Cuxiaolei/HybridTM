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
        """加载单文件.npy，解析6特征+1标签"""
        scene_path = self.data_list[idx]
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"点云文件不存在: {scene_path}")
        
        # 加载数据（假设shape为(N, 7)：3坐标 + 3特征 + 1标签）
        data = np.load(scene_path).astype(np.float32)
        
        # 解析数据（根据你的实际维度调整索引！）
        coord = data[:, :3]  # 前3维：x, y, z坐标
        feat = data[:, 3:6]  # 中间3维：其他特征（如颜色/反射率等）
        segment = data[:, 6].astype(np.int32)  # 最后1维：标签（0/1/2）
        
        # 返回字典必须包含这些键（与transform和模型匹配）
        return {
            "coord": coord,
            "feat": feat,
            "segment": segment,
            "scene_name": os.path.basename(scene_path).replace(".npy", "")
        }