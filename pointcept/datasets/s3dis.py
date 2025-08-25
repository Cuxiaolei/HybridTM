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
        """从txt文件中读取场景路径列表（添加调试日志）"""
        # 1. 打印场景列表文件的完整路径，验证路径是否正确
        split_file = os.path.join(self.data_root, f"{self.split}_scenes.txt")
        # 从父类DefaultDataset继承了logger，直接用self.logger打印调试信息
        self.logger.info(f"[调试] 正在读取测试集场景列表文件 | split: {self.split} | 列表文件路径: {split_file}")

        # 2. 验证场景列表文件是否存在（保留原有报错逻辑，新增日志）
        if not os.path.exists(split_file):
            self.logger.error(f"[调试] 场景列表文件不存在！请检查路径是否正确 | 缺失路径: {split_file}")
            raise FileNotFoundError(f"场景列表文件不存在: {split_file}")
        self.logger.info(f"[调试] 场景列表文件存在，开始读取内容")

        # 3. 读取文件内容，打印有效场景数量（去空后）
        with open(split_file, "r") as f:
            # 过滤空行和纯空格行，保留有效场景路径
            scenes = [line.strip() for line in f.readlines() if line.strip()]
        self.logger.info(f"[调试] 场景列表文件总行数（去空后）: {len(scenes)}")

        # 4. 若场景列表为空，打印错误日志（提前预警样本数为0的原因）
        if len(scenes) == 0:
            self.logger.error(
                f"[调试] 场景列表文件内容为空！没有任何有效场景路径，请检查{split_file}文件内是否有场景相对路径（如merged/Area_5.npy）")
            return []  # 此时返回空列表，后续会触发样本数为0

        # 5. 拼接完整路径，验证每个场景文件是否存在（打印前5个场景的验证结果，避免日志过长）
        full_scene_paths = []
        valid_count = 0  # 统计路径存在的有效场景数
        for idx, scene_rel_path in enumerate(scenes):
            full_path = os.path.join(self.data_root, scene_rel_path)
            full_scene_paths.append(full_path)

            # 打印前5个场景的详细信息（路径+存在性）
            if idx < 5:
                file_exists = os.path.exists(full_path)
                if file_exists:
                    valid_count += 1
                    self.logger.info(
                        f"[调试] 场景{idx + 1} | 相对路径: {scene_rel_path} | 完整路径: {full_path} | 存在: ✅")
                else:
                    self.logger.warning(
                        f"[调试] 场景{idx + 1} | 相对路径: {scene_rel_path} | 完整路径: {full_path} | 存在: ❌（文件缺失，请检查数据放置路径）")

        # 6. 打印整体统计结果，明确有效场景数
        total_scene = len(full_scene_paths)
        final_valid_count = sum(1 for path in full_scene_paths if os.path.exists(path))
        self.logger.info(
            f"[调试] 场景列表统计 | 总场景数: {total_scene} | 有效场景数（文件存在）: {final_valid_count} | 无效场景数: {total_scene - final_valid_count}")

        # 7. 若有效场景数为0，打印严重错误日志（直接导致后续样本数为0）
        if final_valid_count == 0:
            self.logger.critical(
                f"[调试] 所有场景文件均不存在！请检查：1. data_root是否正确（当前: {self.data_root}） 2. 场景列表文件内的相对路径是否与实际数据文件匹配")

        return full_scene_paths


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
