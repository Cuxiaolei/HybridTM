import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)


@DATASETS.register_module()
class ScanNetDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
            self,
            lr_file=None,
            la_file=None, **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        # 新增调试日志：打印初始化参数
        print("\n" + "=" * 50)
        print(f"[ScanNetDataset 初始化]")
        print(f"  data_root: {kwargs.get('data_root')}")  # 打印数据根目录
        print(f"  split: {kwargs.get('split')}")  # 打印 split 参数（关键！）
        print(f"  lr_file: {lr_file}")  # 打印场景列表文件路径
        print(f"  父类 DefaultDataset 参数: {kwargs}")
        print("=" * 50 + "\n")
        super().__init__(**kwargs)

    # --------------------------
    # 关键修改：完全重写 get_data_list()
    # 不依赖 DefaultDataset，直接扫描 train 目录下的场景文件夹
    # --------------------------
    def get_data_list(self):
        print("\n" + "-" * 50)
        print(f"[get_data_list] 重写逻辑：直接扫描场景文件夹")

        # 1. 拼接完整的 train 目录路径（data_root + split）
        train_dir = os.path.join(self.data_root, self.split)
        print(f"  扫描的 train 目录路径: {train_dir}")

        # 2. 检查 train 目录是否存在
        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"train 目录不存在：{train_dir}\n"
                f"请确认路径是否正确，或场景文件夹是否已放入该目录"
            )

        # 3. 扫描 train 目录下的所有子文件夹（每个文件夹就是一个场景）
        scene_folders = []
        for fname in os.listdir(train_dir):
            scene_path = os.path.join(train_dir, fname)
            if os.path.isdir(scene_path):  # 核心判断：是否为文件夹
                scene_folders.append(scene_path)
                print(f"  找到场景文件夹: {fname}")  # 打印找到的场景名

        # 4. 检查是否找到场景
        if len(scene_folders) == 0:
            raise ValueError(
                f"在 {train_dir} 中未找到任何场景文件夹！\n"
                f"当前目录下的内容：{os.listdir(train_dir)}\n"
                f"请确认场景文件夹（如 scene0001_00）已放入该目录"
            )

        # 5. 输出最终结果
        print(f"  共找到 {len(scene_folders)} 个场景文件夹")
        print("-" * 50 + "\n")
        return scene_folders  # 返回场景路径列表

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        try:
            assets = os.listdir(data_path)
        except Exception as e:
            print(f"  读取文件夹失败: {str(e)}")
            return data_dict

        # 加载各类资产（coord/color/normal等）
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            try:
                data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
            except Exception as e:
                print(f"  加载 {asset} 失败: {str(e)}")

        # --------------------------
        # 新增：坐标归一化（复用 NormalizeCoord 逻辑）
        # 步骤：1. 中心化（减均值） 2. 标准化（除以最大距离）
        # --------------------------
        # if "coord" in data_dict:
        #     # # 1. 坐标中心化（消除绝对位置偏差）
        #     # centroid = np.mean(data_dict["coord"], axis=0)
        #     # data_dict["coord"] -= centroid
        #     # # 坐标标准化（统一尺度，最大距离为1）
        #     max_dist = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
        #     # 避免除以0（若所有点重合，max_dist为0，不执行标准化）
        #     if max_dist > 1e-6:
        #         data_dict["coord"] = data_dict["coord"] / max_dist
        #     # print(f"  [坐标归一化] 完成，处理后坐标范围：{data_dict['coord'].min(axis=0)} ~ {data_dict['coord'].max(axis=0)}")


        # --------------------------
        # 原有逻辑：数据类型转换 + 标签处理
        # --------------------------
        data_dict["name"] = name
        if "coord" in data_dict:
            data_dict["coord"] = data_dict["coord"].astype(np.float32)
        if "color" in data_dict:
            data_dict["color"] = data_dict["color"].astype(np.float32)
        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
            print(f"  警告：未找到 segment20 或 segment200 文件！")

        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["instance"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index

        return data_dict


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_200)