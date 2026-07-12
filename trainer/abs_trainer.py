#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.logger import print_log
from .resume_ema import (
    init_resume_ema,
    maybe_resume,
    after_optimizer_step,
    validation_ema,
    save_checkpoint,
)


class TrainConfig:
    def __init__(self, save_dir, lr, max_epoch, warmup=0,
                 metric_min_better=True, patience=3,
                 grad_clip=None, save_topk=-1,
                 **kwargs):
        self.save_dir = save_dir
        self.lr = lr
        self.max_epoch = max_epoch
        self.warmup = warmup
        self.metric_min_better = metric_min_better
        self.patience = patience
        self.grad_clip = grad_clip
        self.save_topk = save_topk
        self.__dict__.update(kwargs)

    def add_parameter(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config):
        self.model = model
        self.config = config
        self.optimizer = self.get_optimizer()
        sched_config = self.get_scheduler(self.optimizer)
        if sched_config is None:
            sched_config = {'scheduler': None, 'frequency': None}
        self.scheduler = sched_config['scheduler']
        self.sched_freq = sched_config['frequency']
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.local_rank = -1

        # log / run directory
        # Strict resume: continue writing into the original version directory.
        resume_checkpoint = str(getattr(self.config, "resume_checkpoint", "") or "").strip()
        if resume_checkpoint:
            resume_checkpoint = os.path.abspath(resume_checkpoint)
            if not os.path.isfile(resume_checkpoint):
                raise FileNotFoundError(f"resume_checkpoint not found: {resume_checkpoint}")

            ckpt_dir = os.path.dirname(resume_checkpoint)
            resume_run_dir = os.path.dirname(ckpt_dir)

            m = re.search(r"version_(\d+)$", os.path.basename(resume_run_dir))
            if m is None:
                raise ValueError(
                    "resume_checkpoint must be under a version_N/checkpoint directory, "
                    f"got: {resume_checkpoint}"
                )

            self.version = int(m.group(1))
            self.config.save_dir = resume_run_dir
            self.model_dir = ckpt_dir
            self.config.resume_checkpoint = resume_checkpoint
        else:
            self.version = self._get_version()
            self.config.save_dir = os.path.join(self.config.save_dir, f'version_{self.version}')
            self.model_dir = os.path.join(self.config.save_dir, 'checkpoint')

        self.writer = None
        self.writer_buffer = {}

        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.last_valid_metric = None
        self.topk_ckpt_map = []
        self.patience = self.config.patience
        
        self.last_state_path = None
        resume_checkpoint = str(getattr(self.config, "resume_checkpoint", "") or "").strip()
        if resume_checkpoint and os.path.basename(resume_checkpoint).startswith("last_step"):
            self.last_state_path = resume_checkpoint
        self.ema = None

        # Training-speed controls.  AMP is opt-in through config/env and is
        # implemented here so all project trainers inherit the same behavior.
        self.use_amp = bool(getattr(self.config, "amp", False))
        self.amp_dtype = str(getattr(self.config, "amp_dtype", "bf16")).lower()
        self.log_interval = max(1, int(getattr(self.config, "log_interval", 20) or 20))

        # GradScaler is only needed for fp16.  bf16 has a wider exponent range
        # and normally does not require scaling.
        self.grad_scaler = None

    @classmethod
    def to_device(cls, data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            data = type(data)([cls.to_device(item, device) for item in data])
        elif torch.is_tensor(data):
            data = data.to(device, non_blocking=True)
        elif hasattr(data, 'to'):
            data = data.to(device)
        return data

    def _amp_autocast(self, device):
        enabled = (
            bool(getattr(self, "use_amp", False))
            and device.type == "cuda"
        )
        if self.amp_dtype in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        elif self.amp_dtype in {"fp16", "float16", "half"}:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)

    def _should_log_step(self, step):
        return self._is_main_proc() and (int(step) % self.log_interval == 0)

    def _is_main_proc(self):
        return self.local_rank == 0 or self.local_rank == -1

    def _get_version(self):
        version, pattern = -1, r'version_(\d+)'
        if os.path.exists(self.config.save_dir):
            for fname in os.listdir(self.config.save_dir):
                ver = re.findall(pattern, fname)
                if len(ver):
                    version = max(int(ver[0]), version)
        return version + 1

    def _save_train_state(self, tag, metric=None):
        if not self._is_main_proc():
            return
        path = os.path.join(self.model_dir, f'{tag}_step{self.global_step}.pt')
        save_checkpoint(self, path, metric=metric)
        if tag == 'last' and self.last_state_path and os.path.exists(self.last_state_path):
            try:
                os.remove(self.last_state_path)
            except OSError:
                pass
        if tag == 'last':
            self.last_state_path = path

    def _save_eval_model(self, save_path):
        module_to_save = self.model.module if self.local_rank == 0 else self.model
        torch.save(module_to_save, save_path)

    def _train_epoch(self, device):
        if self.train_loader.sampler is not None and self.local_rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)

        t_iter = tqdm(
            self.train_loader,
            dynamic_ncols=True,
            mininterval=float(getattr(self.config, "tqdm_mininterval", 5.0)),
            leave=False,
        ) if self._is_main_proc() else self.train_loader

        for batch in t_iter:
            batch = self.to_device(batch, device)

            self.optimizer.zero_grad(set_to_none=True)

            with self._amp_autocast(device):
                loss = self.train_step(batch, self.global_step)

            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                if self.config.grad_clip is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.optimizer.step()

            after_optimizer_step(self)

            if self._should_log_step(self.global_step) and hasattr(t_iter, 'set_postfix'):
                t_iter.set_postfix(
                    loss=float(loss.detach().cpu()),
                    version=self.version,
                )

            self.global_step += 1

            if self.sched_freq == 'batch':
                self.scheduler.step()

        if self.sched_freq == 'epoch':
            self.scheduler.step()

    def _valid_epoch(self, device):
        metric_arr = []
        eval_path_to_save = None
        should_save_best = False
        valid_metric = None

        self.model.eval()
        with validation_ema(self):
            with torch.no_grad():
                t_iter = tqdm(
                    self.valid_loader,
                    dynamic_ncols=True,
                    mininterval=float(getattr(self.config, "tqdm_mininterval", 5.0)),
                    leave=False,
                ) if self._is_main_proc() else self.valid_loader
                for batch in t_iter:
                    batch = self.to_device(batch, device)
                    with self._amp_autocast(device):
                        metric = self.valid_step(batch, self.valid_global_step)
                    metric_arr.append(float(metric.detach().cpu()))
                    self.valid_global_step += 1

            valid_metric = float(np.mean(metric_arr))
            should_save_best = self._metric_better(valid_metric)

            if should_save_best:
                self.patience = self.config.patience
                if self._is_main_proc():
                    eval_path_to_save = os.path.join(
                        self.model_dir,
                        f'epoch{self.epoch}_step{self.global_step}.ckpt'
                    )
                    self._save_eval_model(eval_path_to_save)
            else:
                self.patience -= 1

        self.model.train()

        if should_save_best and self._is_main_proc():
            self._maintain_topk_checkpoint(valid_metric, eval_path_to_save)

        self.last_valid_metric = valid_metric

        if self._is_main_proc():
            for name, values in self.writer_buffer.items():
                value = float(np.mean(values))
                self.writer.add_scalar(name, value, self.epoch)

        self.writer_buffer = {}
        
    def _metric_better(self, new):
        old = self.last_valid_metric
        if old is None:
            return True
        return new < old if self.config.metric_min_better else old < new

    def _load_topk_checkpoint_map(self):
        self.topk_ckpt_map = []
        topk_map_path = os.path.join(self.model_dir, 'topk_map.txt')
        if not os.path.isfile(topk_map_path):
            return

        with open(topk_map_path, 'r') as fin:
            for line in fin:
                line = line.strip()
                if not line or ': ' not in line:
                    continue
                metric_text, path = line.split(': ', 1)
                try:
                    metric = float(metric_text)
                except ValueError:
                    continue
                if os.path.exists(path):
                    self.topk_ckpt_map.append((metric, path))

        if self.config.metric_min_better:
            self.topk_ckpt_map.sort(key=lambda x: x[0])
        else:
            self.topk_ckpt_map.sort(key=lambda x: x[0], reverse=True)
            
    def _maintain_topk_checkpoint(self, valid_metric, ckpt_path):
        topk = self.config.save_topk
        better = (lambda a, b: a < b) if self.config.metric_min_better else (lambda a, b: a > b)

        insert_pos = len(self.topk_ckpt_map)
        for i, (metric, _) in enumerate(self.topk_ckpt_map):
            if better(valid_metric, metric):
                insert_pos = i
                break

        self.topk_ckpt_map.insert(insert_pos, (valid_metric, ckpt_path))

        if topk > 0:
            while len(self.topk_ckpt_map) > topk:
                last_ckpt_path = self.topk_ckpt_map[-1][1]
                if os.path.exists(last_ckpt_path):
                    os.remove(last_ckpt_path)
                self.topk_ckpt_map.pop()

        topk_map_path = os.path.join(self.model_dir, 'topk_map.txt')
        with open(topk_map_path, 'w') as fout:
            for metric, path in self.topk_ckpt_map:
                fout.write(f'{metric}: {path}\n')

    def train(self, device_ids, local_rank):
        self.local_rank = local_rank

        if self._is_main_proc():
            self.writer = SummaryWriter(self.config.save_dir)
            os.makedirs(self.model_dir, exist_ok=True)
            with open(os.path.join(self.config.save_dir, 'namespace.json'), 'w') as fout:
                json.dump(self.config.__dict__, fout, indent=2)

        main_device_id = local_rank if local_rank != -1 else device_ids[0]
        device = torch.device('cpu' if main_device_id == -1 else f'cuda:{main_device_id}')

        self.model.to(device)

        if (
            self.use_amp
            and device.type == "cuda"
            and self.amp_dtype in {"fp16", "float16", "half"}
        ):
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.grad_scaler = None

        init_resume_ema(self)
        maybe_resume(self, device)
        
        if str(getattr(self.config, "resume_checkpoint", "") or "").strip():
            self._load_topk_checkpoint_map()

        if local_rank != -1:
            print_log(f'Using data parallel, local rank {local_rank}, all {device_ids}')
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                gradient_as_bucket_view=True,
            )
        else:
            print_log(f'training on {device_ids}')

        while self.epoch < self.config.max_epoch:
            print_log(f'epoch{self.epoch} starts') if self._is_main_proc() else 1
            self._train_epoch(device)
            
            print_log(f'validating ...') if self._is_main_proc() else 1
            self._valid_epoch(device)

            # Important: after validation, the current epoch is complete.
            # Increment first so the saved train-state records the next epoch to run.
            self.epoch += 1

            save_interval = int(getattr(self.config, 'save_interval', 1) or 0)
            if save_interval > 0 and self.epoch % save_interval == 0:
                self._save_train_state('last', metric=self.last_valid_metric)

            if self.patience <= 0:
                break

    def log(self, name, value, step, val=False):
        if not self._is_main_proc():
            return

        # Training scalar logging can synchronize CUDA if every tensor is
        # converted to a Python float every step.  Log at a fixed interval to keep
        # TensorBoard useful without throttling the GPU.
        if not val and (int(step) % self.log_interval != 0):
            return

        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu())
        if val:
            if name not in self.writer_buffer:
                self.writer_buffer[name] = []
            self.writer_buffer[name].append(value)
        else:
            self.writer.add_scalar(name, value, step)

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def get_scheduler(self, optimizer):
        lam = lambda epoch: 1 / (epoch + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
        return {'scheduler': scheduler, 'frequency': 'epoch'}

    def train_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/train', loss, batch_idx)
        return loss

    def valid_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/validation', loss, batch_idx, val=True)
        return loss