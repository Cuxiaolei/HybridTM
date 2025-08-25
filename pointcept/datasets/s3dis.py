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
        """从txt文件中读取场景路径列表（添加调试打印）"""
        # 1. 拼接场景列表文件路径，打印关键信息
        split_file = os.path.join(self.data_root, f"{self.split}_scenes.txt")
        print(
            f"[调试] 正在读取测试集场景列表 | split: {self.split} | data_root: {self.data_root} | 列表文件路径: {split_file}")

        # 2. 验证场景列表文件是否存在
        if not os.path.exists(split_file):
            print(f"[调试-错误] 场景列表文件不存在！路径: {split_file}")
            raise FileNotFoundError(f"场景列表文件不存在: {split_file}")
        print(f"[调试] 场景列表文件存在，开始读取内容")

        # 3. 读取文件内容，过滤空行，打印有效场景数量
        with open(split_file, "r") as f:
            scenes = [line.strip() for line in f.readlines() if line.strip()]
        print(f"[调试] 场景列表文件总行数（去空后）: {len(scenes)}")

        # 4. 若场景列表为空，打印错误提示
        if len(scenes) == 0:
            print(f"[调试-错误] 场景列表文件内容为空！请检查 {split_file} 内是否有场景路径（如 merged/Area_5.npy）")
            return []

        # 5. 拼接完整路径，验证前5个场景文件是否存在
        full_scene_paths = []
        valid_count = 0
        for idx, scene_rel_path in enumerate(scenes):
            full_path = os.path.join(self.data_root, scene_rel_path)
            full_scene_paths.append(full_path)

            # 打印前5个场景的路径和存在性
            if idx < 5:
                file_exists = os.path.exists(full_path)
                if file_exists:
                    valid_count += 1
                    print(f"[调试] 场景{idx + 1} | 相对路径: {scene_rel_path} | 完整路径: {full_path} | 存在: ✅")
                else:
                    print(
                        f"[调试-警告] 场景{idx + 1} | 相对路径: {scene_rel_path} | 完整路径: {full_path} | 存在: ❌（文件缺失）")

        # 6. 打印整体统计结果
        total_scene = len(full_scene_paths)
        final_valid_count = sum(1 for path in full_scene_paths if os.path.exists(path))
        print(
            f"[调试] 场景列表统计 | 总场景数: {total_scene} | 有效场景数（文件存在）: {final_valid_count} | 无效场景数: {total_scene - final_valid_count}")

        # 7. 若有效场景数为0，打印严重错误
        if final_valid_count == 0:
            print(
                f"[调试-严重错误] 所有场景文件均不存在！请检查：1. data_root是否正确（当前: {self.data_root}） 2. 场景列表内的相对路径是否与实际数据匹配")

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
