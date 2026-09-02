#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import traceback
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
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
        self.last_train_metric = None
        self.last_valid_metric = None
        self.last_test_metrics = {}
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

    def _scan_next_version_local(self):
        version, pattern = -1, r'version_(\d+)'
        if os.path.exists(self.config.save_dir):
            for fname in os.listdir(self.config.save_dir):
                ver = re.findall(pattern, fname)
                if len(ver):
                    version = max(int(ver[0]), version)
        return int(version + 1)

    def _get_version(self):
        """Choose exactly one run version for the whole DDP job.

        Previous diagnostic code let every rank scan the filesystem
        independently. Two ranks racing through startup could therefore choose
        different version_N directories (observed: rank0->version_0,
        rank1->version_1).

        In DDP, rank0 is now the sole version authority and broadcasts one
        integer to all ranks. This changes run-directory bookkeeping only.
        """
        if (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            rank = dist.get_rank()
            local_version = (
                self._scan_next_version_local() if rank == 0 else -1
            )

            backend = str(dist.get_backend()).lower()
            if "nccl" in backend:
                device = torch.device(
                    "cuda", torch.cuda.current_device()
                )
            else:
                device = torch.device("cpu")

            version_tensor = torch.tensor(
                [local_version],
                dtype=torch.long,
                device=device,
            )
            dist.broadcast(version_tensor, src=0)
            return int(version_tensor.item())

        return self._scan_next_version_local()

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

    def _optimizer_local_state_bytes(self):
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return 0
        local_optim = getattr(optimizer, "optim", optimizer)
        total = 0
        for state in getattr(local_optim, "state", {}).values():
            values = state.values() if isinstance(state, dict) else (state,)
            for value in values:
                if torch.is_tensor(value):
                    total += int(value.numel()) * int(value.element_size())
        return int(total)

    def _ema_state_bytes(self):
        ema = getattr(self, "ema", None)
        if ema is None:
            return 0
        return int(sum(
            int(v.numel()) * int(v.element_size())
            for v in getattr(ema, "shadow", {}).values()
            if torch.is_tensor(v)
        ))

    @staticmethod
    def _tensor_nbytes(value):
        if not torch.is_tensor(value):
            return 0
        return int(value.numel()) * int(value.element_size())

    def _registered_model_state_bytes(self):
        model = getattr(self, "model", None)
        raw = getattr(model, "module", model)
        if raw is None:
            return 0, 0, 0
        pb = gb = bb = 0
        for p in raw.parameters():
            pb += self._tensor_nbytes(p)
            if p.grad is not None:
                gb += self._tensor_nbytes(p.grad)
        for b in raw.buffers():
            bb += self._tensor_nbytes(b)
        return int(pb), int(gb), int(bb)

    def _runtime_python_tensor_owners(self):
        """Read-only direct Python Tensor ownership summary."""
        model = getattr(self, "model", None)
        raw = getattr(model, "module", model)
        if raw is None:
            return 0, []

        keywords = (
            "cache", "diagnostic", "probe", "pending",
            "_last_", "last_", "objective_tensor",
        )
        seen = set()
        owners = []

        def visit(obj, path, depth=0):
            if depth > 5:
                return 0
            if torch.is_tensor(obj):
                oid = id(obj)
                if oid in seen:
                    return 0
                seen.add(oid)
                n = self._tensor_nbytes(obj)
                if n:
                    owners.append((path, n, bool(obj.requires_grad)))
                return n
            if isinstance(obj, dict):
                return sum(
                    visit(v, f"{path}[{k!r}]", depth + 1)
                    for k, v in obj.items()
                )
            if isinstance(obj, (list, tuple)):
                return sum(
                    visit(v, f"{path}[{i}]", depth + 1)
                    for i, v in enumerate(obj)
                )
            return 0

        total = 0
        for module_name, module in raw.named_modules():
            prefix = module_name or "<root>"
            for name, value in module.__dict__.items():
                if name in {"_parameters", "_buffers", "_modules"}:
                    continue
                lname = name.lower()
                if not any(k in lname for k in keywords):
                    continue
                total += visit(value, f"{prefix}.{name}")

        owners.sort(key=lambda x: x[1], reverse=True)
        return int(total), owners[:12]

    def _runtime_log_path(self):
        rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else 0
        )
        return os.path.join(
            self.config.save_dir,
            f"runtime_memory_rank{rank}.log",
        )

    def _runtime_log_line(self, message):
        """Best-effort diagnostic file sink; never affects training."""
        def _on(name):
            return os.environ.get(
                name, "off"
            ).strip().lower() in {"1", "true", "yes", "y", "on"}

        if not (
            _on("ABFLOW_MEMORY_DIAGNOSTICS")
            or _on("ABFLOW_PERF_DIAGNOSTICS")
            or _on("ABFLOW_RUNTIME_TRACE")
        ):
            return
        try:
            os.makedirs(self.config.save_dir, exist_ok=True)
            with open(
                self._runtime_log_path(), "a", encoding="utf-8"
            ) as fout:
                fout.write(str(message).rstrip("\n") + "\n")
        except Exception:
            # Diagnostics are non-authoritative; never fail the run.
            pass

    def _runtime_exception_log(self, stage, step):
        rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else 0
        )
        header = (
            f"[RuntimeException] rank={rank} step={int(step)} "
            f"stage={stage}"
        )
        body = traceback.format_exc()
        self._runtime_log_line(header)
        for line in body.rstrip("\n").splitlines():
            self._runtime_log_line(line)

    def _cuda_memory_diag(self, device, tag, step):
        enabled = os.environ.get(
            "ABFLOW_MEMORY_DIAGNOSTICS", "off"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if device.type != "cuda" or not enabled:
            return
        limit = max(0, int(os.environ.get(
            "ABFLOW_MEMORY_DIAGNOSTIC_STEPS", "3"
        ) or 3))
        if int(step) >= limit:
            return

        rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else 0
        )
        mib = 1024.0 * 1024.0
        pb, gb, bb = self._registered_model_state_bytes()
        cache_b, owners = self._runtime_python_tensor_owners()

        _diag_line = (
            "[MemoryDiag] "
            f"rank={rank} step={int(step)} tag={tag} "
            f"alloc={torch.cuda.memory_allocated(device)/mib:.1f}MiB "
            f"reserved={torch.cuda.memory_reserved(device)/mib:.1f}MiB "
            f"peak={torch.cuda.max_memory_allocated(device)/mib:.1f}MiB "
            f"params={pb/mib:.1f}MiB grads={gb/mib:.1f}MiB "
            f"buffers={bb/mib:.1f}MiB "
            f"optimizer_state={self._optimizer_local_state_bytes()/mib:.1f}MiB "
            f"ema={self._ema_state_bytes()/mib:.1f}MiB "
            f"python_cache={cache_b/mib:.1f}MiB"
        )
        print(_diag_line, flush=True)
        self._runtime_log_line(_diag_line)

        owner_enabled = os.environ.get(
            "ABFLOW_MEMORY_OWNER_DIAGNOSTICS", "off"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}

        if owner_enabled and tag in {
            "after_forward", "after_backward", "after_ema_update"
        }:
            if owners:
                top = " | ".join(
                    f"{path}={n/mib:.1f}MiB"
                    f"{'*grad' if req else ''}"
                    for path, n, req in owners
                )
            else:
                top = "<none>"
            _owner_line = (
                f"[MemoryOwner] rank={rank} step={int(step)} "
                f"tag={tag} top={top}"
            )
            print(_owner_line, flush=True)
            self._runtime_log_line(_owner_line)

    def _consolidate_sharded_optimizer_for_checkpoint(self):
        consolidate = getattr(
            getattr(self, "optimizer", None),
            "consolidate_state_dict",
            None,
        )
        if callable(consolidate):
            consolidate(to=0)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    @staticmethod
    def _env_on(name, default="off"):
        return os.environ.get(
            name, default
        ).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _perf_step_enabled(self, step):
        if not self._env_on("ABFLOW_PERF_DIAGNOSTICS"):
            return False
        limit = max(
            0,
            int(os.environ.get("ABFLOW_PERF_DIAGNOSTIC_STEPS", "8") or 8),
        )
        return int(step) < limit

    def _consume_model_runtime_perf_stats(self):
        try:
            from models.modules.am_enc import consume_runtime_perf_stats
            return consume_runtime_perf_stats()
        except Exception:
            return {
                "full_calls": -1,
                "lma_calls": -1,
                "checkpoint_cleanup_calls": -1,
                "checkpoint_storage_tensors": -1,
            }

    def _effective_recycling_for_perf(self):
        raw = getattr(self.model, "module", self.model)
        gnn = getattr(raw, "gnn", None)
        return int(getattr(gnn, "_last_effective_recycling_steps", -1))

    def _write_perf_line(
        self,
        *,
        step,
        rank,
        data_wait_ms,
        forward_ms,
        backward_ms,
        optimizer_ms,
        ema_ms,
        wall_ms,
        peak_mib,
        end_alloc_mib,
        stats,
    ):
        line = (
            "[RuntimePerf] "
            f"rank={rank} step={int(step)} "
            f"recycle={self._effective_recycling_for_perf()} "
            f"data_wait_ms={data_wait_ms:.1f} "
            f"forward_ms={forward_ms:.1f} "
            f"backward_ms={backward_ms:.1f} "
            f"optimizer_ms={optimizer_ms:.1f} "
            f"ema_ms={ema_ms:.1f} "
            f"step_wall_ms={wall_ms:.1f} "
            f"peak={peak_mib:.1f}MiB "
            f"end_alloc={end_alloc_mib:.1f}MiB "
            f"attn_full={stats.get('full_calls', -1)} "
            f"attn_lma={stats.get('lma_calls', -1)} "
            f"ckpt_cleanup={stats.get('checkpoint_cleanup_calls', -1)} "
            f"ckpt_saved={stats.get('checkpoint_storage_tensors', -1)}"
        )
        # All ranks persist their own line. Only global rank0 prints to stdout.
        self._runtime_log_line(line)
        if rank == 0:
            print(line, flush=True)

    def _train_epoch(self, device):
        # Module 11: the Train phase exposes one epoch-level diagnostic scalar.
        # Accumulation is device-side and all-reduced once per epoch, so it does
        # not introduce a per-step CUDA synchronization or change optimization.
        epoch_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        epoch_loss_count = torch.zeros((), dtype=torch.float64, device=device)

        if self.train_loader.sampler is not None and self.local_rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)

        t_iter = tqdm(
            self.train_loader,
            dynamic_ncols=True,
            mininterval=float(getattr(self.config, "tqdm_mininterval", 5.0)),
            leave=False,
        ) if self._is_main_proc() else self.train_loader

        _previous_step_end = time.perf_counter()

        for batch in t_iter:
            _body_start = time.perf_counter()
            _data_wait_ms = (_body_start - _previous_step_end) * 1000.0
            _profile = self._perf_step_enabled(self.global_step)

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            batch = self.to_device(batch, device)

            self.optimizer.zero_grad(set_to_none=True)

            if _profile and device.type == "cuda":
                _ev0 = torch.cuda.Event(enable_timing=True)
                _ev1 = torch.cuda.Event(enable_timing=True)
                _ev2 = torch.cuda.Event(enable_timing=True)
                _ev3 = torch.cuda.Event(enable_timing=True)
                _ev4 = torch.cuda.Event(enable_timing=True)
                _ev0.record()
            else:
                _ev0 = _ev1 = _ev2 = _ev3 = _ev4 = None

            try:
                with self._amp_autocast(device):
                    loss = self.train_step(batch, self.global_step)
            except Exception:
                self._runtime_exception_log(
                    "train_forward", self.global_step
                )
                raise

            if _ev1 is not None:
                _ev1.record()

            # Keep only the scalar value needed for epoch statistics/progress.
            # This detached scalar has no grad_fn and therefore cannot own the
            # completed training graph.
            loss_detached = loss.detach()
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                if self.config.grad_clip is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                if _ev2 is not None:
                    _ev2.record()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                if _ev3 is not None:
                    _ev3.record()
            else:
                try:
                    loss.backward()
                except Exception:
                    self._runtime_exception_log(
                        "train_backward", self.global_step
                    )
                    raise
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                if _ev2 is not None:
                    _ev2.record()
                self.optimizer.step()
                if _ev3 is not None:
                    _ev3.record()

            after_optimizer_step(self)
            if _ev4 is not None:
                _ev4.record()

            epoch_loss_sum = epoch_loss_sum + loss_detached.double()
            epoch_loss_count = epoch_loss_count + 1.0

            if self._should_log_step(self.global_step) and hasattr(t_iter, 'set_postfix'):
                t_iter.set_postfix(
                    loss=float(loss_detached.cpu()),
                    version=self.version,
                )

            # PyTorch evaluates the RHS of the next
            #     loss = self.train_step(...)
            # before replacing this local variable.  Without this explicit
            # release, the previous step's loss.grad_fn remains referenced
            # throughout the next forward.  This matters on torch 1.11
            # non-reentrant activation checkpointing, whose recomputation
            # tensors live in checkpoint-owned storage associated with the
            # output graph.
            #
            # backward/optimizer/EMA and every use of the numerical loss value
            # have already completed above.  Releasing this Python reference
            # does not alter gradients, parameters, optimizer state, RNG,
            # batch composition, or model equations.
            del loss

            if _profile:
                rank = (
                    dist.get_rank()
                    if dist.is_available() and dist.is_initialized()
                    else 0
                )
                if device.type == "cuda":
                    # Synchronize only the first few explicit profile steps.
                    # Formal training keeps ABFLOW_PERF_DIAGNOSTICS=off.
                    _ev4.synchronize()
                    _forward_ms = _ev0.elapsed_time(_ev1)
                    _backward_ms = _ev1.elapsed_time(_ev2)
                    _optimizer_ms = _ev2.elapsed_time(_ev3)
                    _ema_ms = _ev3.elapsed_time(_ev4)
                    _peak_mib = (
                        torch.cuda.max_memory_allocated(device)
                        / (1024.0 * 1024.0)
                    )
                    _end_alloc_mib = (
                        torch.cuda.memory_allocated(device)
                        / (1024.0 * 1024.0)
                    )
                else:
                    _forward_ms = _backward_ms = 0.0
                    _optimizer_ms = _ema_ms = 0.0
                    _peak_mib = _end_alloc_mib = 0.0

                _stats = self._consume_model_runtime_perf_stats()
                _wall_ms = (time.perf_counter() - _body_start) * 1000.0
                self._write_perf_line(
                    step=self.global_step,
                    rank=rank,
                    data_wait_ms=_data_wait_ms,
                    forward_ms=_forward_ms,
                    backward_ms=_backward_ms,
                    optimizer_ms=_optimizer_ms,
                    ema_ms=_ema_ms,
                    wall_ms=_wall_ms,
                    peak_mib=_peak_mib,
                    end_alloc_mib=_end_alloc_mib,
                    stats=_stats,
                )
            else:
                # Reset counters even after the profile window so a later
                # diagnostic enable does not inherit stale calls.
                if self._env_on("ABFLOW_PERF_DIAGNOSTICS"):
                    self._consume_model_runtime_perf_stats()

            self.global_step += 1
            _previous_step_end = time.perf_counter()

            if self.sched_freq == 'batch':
                self.scheduler.step()

        if self.sched_freq == 'epoch':
            self.scheduler.step()

        train_stats = torch.stack([epoch_loss_sum, epoch_loss_count])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
        self.last_train_metric = float(
            (train_stats[0] / train_stats[1].clamp_min(1.0)).item()
        )

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
        
    def _test_epoch(self, device):
        """Optional third epoch phase. Base trainers do nothing by default."""
        self.last_test_metrics = {}
        return self.last_test_metrics

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
        # import ipdb; ipdb.set_trace()
        self.local_rank = local_rank

        # The version_N directory is already resolved in __init__.  Every DDP
        # rank may safely create the same directory.  AMEncoder reads this env
        # only for diagnostic file output.
        if (
            self._env_on("ABFLOW_MEMORY_DIAGNOSTICS")
            or self._env_on("ABFLOW_PERF_DIAGNOSTICS")
            or self._env_on("ABFLOW_RUNTIME_TRACE")
        ):
            os.makedirs(self.config.save_dir, exist_ok=True)
            os.environ["ABFLOW_RUNTIME_TRACE_DIR"] = self.config.save_dir
            self._runtime_log_line(
                "[RuntimeLog] "
                f"version_dir={self.config.save_dir} "
                f"local_rank={local_rank}"
            )

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

            # Module 11 formal protocol: every completed training epoch has
            # exactly three ordered phases: Train -> Validation -> Test.  Test is
            # a separate hook, not hidden inside validation, and cannot modify the
            # checkpoint-selection metric returned by Validation.
            print_log(f'testing ...') if self._is_main_proc() else 1
            self._test_epoch(device)

            # Only after all three phases complete is the epoch committed.
            self.epoch += 1

            save_interval = int(getattr(self.config, 'save_interval', 1) or 0)
            if save_interval > 0 and self.epoch % save_interval == 0:
                self._consolidate_sharded_optimizer_for_checkpoint()
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
        # import ipdb; ipdb.set_trace()
        loss = self.model(batch)
        self.log('Loss/train', loss, batch_idx)
        # import ipdb; ipdb.set_trace()
        return loss

    def valid_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/validation', loss, batch_idx, val=True)
        return loss