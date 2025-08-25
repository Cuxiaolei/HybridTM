"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
import csv
from collections import defaultdict

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry


TRAINERS = Registry("trainers")


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

        # 初始化指标存储和CSV文件
        self.num_classes = cfg.data.num_classes
        self.class_names = cfg.data.names if hasattr(cfg.data, 'names') else [f"class_{i}" for i in range(self.num_classes)]

        # 创建保存指标的目录
        self.metrics_dir = os.path.join(cfg.save_path, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        # 初始化CSV文件
        self._init_metrics_csv()

    def _init_metrics_csv(self):
        """初始化训练和验证指标的CSV文件并写入表头"""
        # 训练指标CSV
        train_csv_path = os.path.join(self.metrics_dir, "train_metrics.csv")
        if not os.path.exists(train_csv_path) or self.start_epoch == 0:
            with open(train_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # 表头：epoch + 每个类别的IoU + 每个类别的准确率 + mIoU + OA
                header = ['epoch']
                for cls in self.class_names:
                    header.append(f"{cls}_iou")
                for cls in self.class_names:
                    header.append(f"{cls}_acc")
                header.extend(['mIoU', 'OA'])
                writer.writerow(header)

        # 验证指标CSV
        val_csv_path = os.path.join(self.metrics_dir, "val_metrics.csv")
        if not os.path.exists(val_csv_path) or self.start_epoch == 0:
            with open(val_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # 表头：epoch + 每个类别的IoU + 每个类别的准确率 + mIoU + OA
                header = ['epoch']
                for cls in self.class_names:
                    header.append(f"{cls}_iou")
                for cls in self.class_names:
                    header.append(f"{cls}_acc")
                header.extend(['mIoU', 'OA'])
                writer.writerow(header)

    def _compute_metrics(self, predictions, targets, num_classes):
        """
        计算每个类别的IoU、准确率，以及mIoU和OA

        Args:
            predictions: 模型预测结果，形状为[N]
            targets: 真实标签，形状为[N]
            num_classes: 类别数量

        Returns:
            metrics: 包含各类指标的字典
        """
        # 计算混淆矩阵
        confusion_matrix = torch.zeros((num_classes, num_classes), device=predictions.device)
        for p, t in zip(predictions, targets):
            confusion_matrix[p, t] += 1

        # 计算每个类别的IoU
        iou = torch.diag(confusion_matrix) / (
            confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - torch.diag(confusion_matrix) + 1e-10
        )

        # 计算每个类别的准确率
        acc = torch.diag(confusion_matrix) / (confusion_matrix.sum(dim=1) + 1e-10)

        # 计算mIoU (mean Intersection over Union)
        miou = iou.mean()

        # 计算OA (Overall Accuracy)
        oa = torch.diag(confusion_matrix).sum() / (confusion_matrix.sum() + 1e-10)

        return {
            'iou': iou.cpu().numpy(),
            'acc': acc.cpu().numpy(),
            'miou': miou.item(),
            'oa': oa.item()
        }

    def _save_metrics_to_csv(self, epoch, metrics, is_train=True):
        """将指标保存到CSV文件"""
        if not comm.is_main_process():
            return

        csv_path = os.path.join(self.metrics_dir, "train_metrics.csv" if is_train else "val_metrics.csv")

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch]

            # 添加每个类别的IoU
            for i in range(self.num_classes):
                row.append(metrics['iou'][i])

            # 添加每个类别的准确率
            for i in range(self.num_classes):
                row.append(metrics['acc'][i])

            # 添加mIoU和OA
            row.append(metrics['miou'])
            row.append(metrics['oa'])

            writer.writerow(row)

        # 同时将指标写入日志
        log_str = f"{'Train' if is_train else 'Val'} Epoch {epoch} Metrics: "
        log_str += f"mIoU: {metrics['miou']:.4f}, OA: {metrics['oa']:.4f}, "
        log_str += "IoUs: " + ", ".join([f"{self.class_names[i]}: {metrics['iou'][i]:.4f}" for i in range(min(5, self.num_classes))])
        if self.num_classes > 5:
            log_str += ", ..."  # 避免日志过长
        self.logger.info(log_str)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")

            # 用于收集训练集指标的变量
            self.train_predictions = []
            self.train_targets = []

            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)

                # 重置训练指标收集
                self.train_predictions = []
                self.train_targets = []

                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()

                    # 收集训练预测和目标
                    if 'pred' in self.comm_info["model_output_dict"] and 'segment' in self.comm_info["input_dict"]:
                        self.train_predictions.append(self.comm_info["model_output_dict"]['pred'].detach())
                        self.train_targets.append(self.comm_info["input_dict"]['segment'])

                # 计算并保存训练集指标
                if self.train_predictions and self.train_targets:
                    all_preds = torch.cat(self.train_predictions, dim=0)
                    all_targets = torch.cat(self.train_targets, dim=0)

                    # 忽略无效标签
                    valid_mask = all_targets != self.cfg.data.ignore_index
                    all_preds = all_preds[valid_mask]
                    all_targets = all_targets[valid_mask]

                    if all_preds.numel() > 0:  # 确保有有效数据
                        train_metrics = self._compute_metrics(all_preds, all_targets, self.num_classes)
                        self._save_metrics_to_csv(self.epoch, train_metrics, is_train=True)

                # 验证并计算验证集指标
                if self.val_loader is not None:
                    self.model.eval()
                    val_predictions = []
                    val_targets = []

                    with torch.no_grad():
                        for val_input in self.val_loader:
                            # 将数据移至GPU
                            for key in val_input.keys():
                                if isinstance(val_input[key], torch.Tensor):
                                    val_input[key] = val_input[key].cuda(non_blocking=True)

                            # 模型推理
                            val_output = self.model(val_input)
                            if 'pred' in val_output and 'segment' in val_input:
                                val_predictions.append(val_output['pred'])
                                val_targets.append(val_input['segment'])

                    # 计算并保存验证集指标
                    if val_predictions and val_targets:
                        all_val_preds = torch.cat(val_predictions, dim=0)
                        all_val_targets = torch.cat(val_targets, dim=0)

                        # 忽略无效标签
                        val_valid_mask = all_val_targets != self.cfg.data.ignore_index
                        all_val_preds = all_val_preds[val_valid_mask]
                        all_val_targets = all_val_targets[val_valid_mask]

                        if all_val_preds.numel() > 0:  # 确保有有效数据
                            val_metrics = self._compute_metrics(all_val_preds, all_val_targets, self.num_classes)
                            self._save_metrics_to_csv(self.epoch, val_metrics, is_train=False)

                # => after epoch
                self.after_epoch()

            # => after train
            self.after_train()

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        # if self.cfg.empty_cache_per_epoch:
        torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=False,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
