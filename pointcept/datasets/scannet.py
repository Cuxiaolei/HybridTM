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

    def get_data_list(self):
        # 新增调试日志：进入数据列表生成逻辑
        print("\n" + "-" * 50)
        print(f"[get_data_list] 开始生成数据列表")

        if self.lr is None:
            print(f"  未使用 lr_file，调用父类 DefaultDataset.get_data_list()")
            data_list = super().get_data_list()
            # 新增调试日志：打印父类返回的结果
            print(f"  父类返回的 data_list 长度: {len(data_list)}")
            if len(data_list) > 0:
                print(f"  父类返回的第一个场景路径: {data_list[0]}")
                # 验证路径是否存在
                print(f"  该路径是否存在: {os.path.exists(data_list[0])}")
            else:
                print(f"  父类返回空 data_list！")
        else:
            print(f"  使用 lr_file，场景名列表: {self.lr}")
            data_list = [os.path.join(self.data_root, "train", name) for name in self.lr]
            # 新增调试日志：打印生成的路径
            print(f"  生成的 data_list 长度: {len(data_list)}")
            if len(data_list) > 0:
                print(f"  生成的第一个场景路径: {data_list[0]}")
                print(f"  该路径是否存在: {os.path.exists(data_list[0])}")

        # 新增调试日志：最终数据列表信息
        print(f"[get_data_list] 最终 data_list 长度: {len(data_list)}")
        print("-" * 50 + "\n")
        return data_list

    def get_data(self, idx):
        # 新增调试日志：验证数据加载
        print(f"\n[get_data] 加载第 {idx} 个样本")
        data_path = self.data_list[idx % len(self.data_list)]
        print(f"  数据路径: {data_path}")
        print(f"  路径是否存在: {os.path.exists(data_path)}")

        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        try:
            assets = os.listdir(data_path)
            print(f"  场景文件夹下的文件: {assets}")  # 打印文件夹内的文件
        except Exception as e:
            print(f"  读取文件夹失败: {str(e)}")
            return data_dict

        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            try:
                data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
                print(f"  成功加载: {asset}，shape: {data_dict[asset[:-4]].shape}")
            except Exception as e:
                print(f"  加载 {asset} 失败: {str(e)}")

        # 后续处理逻辑不变...
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