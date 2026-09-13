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


def _normalize_optional_path(value):
    """Normalize optional path values crossing JSON -> shell -> argparse.

    Historical launchers may serialize an empty JSON string as the literal
    tokens ``''`` or ``""``.  Those are scratch sentinels, not paths.
    Strip only whole-string matching quotes and common null sentinels; real
    non-empty checkpoint paths are left unchanged.
    """
    if value is None:
        return ""
    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    if text.lower() in {"", "none", "null"}:
        return ""
    return text


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
        resume_checkpoint = _normalize_optional_path(getattr(self.config, "resume_checkpoint", ""))
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
            # The formal R28-R30 launcher sets one explicit version for the whole
            # torchrun job.  Without this, two DDP ranks can race in _get_version():
            # rank0 observes no directory and chooses version_0 while rank1 sees
            # the just-created version_0 and chooses version_1.
            fixed_version = str(os.environ.get("ABFLOW_FIXED_VERSION", "") or "").strip()
            if fixed_version:
                try:
                    self.version = int(fixed_version)
                except ValueError as exc:
                    raise ValueError(
                        f"ABFLOW_FIXED_VERSION must be a non-negative integer, got {fixed_version!r}"
                    ) from exc
                if self.version < 0:
                    raise ValueError("ABFLOW_FIXED_VERSION must be non-negative")
            else:
                self.version = self._get_version()
            self.config.save_dir = os.path.join(
                self.config.save_dir, f'version_{self.version}'
            )
            self.model_dir = os.path.join(self.config.save_dir, 'checkpoint')

        self.writer = None
        self.writer_buffer = {}

        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.last_valid_metric = None
        self.topk_ckpt_map = []
        self.patience = self.config.patience

        # Runtime-only numerical diagnostics.  These values are deliberately
        # absent from checkpoints; they never affect optimizer/scheduler state.
        self._grad_norm_overflow_epoch = -1
        self._grad_norm_overflow_count = 0
        self._grad_norm_overflow_log_limit = max(
            1, int(os.environ.get("ABFLOW_GRAD_OVERFLOW_LOG_LIMIT", "3") or 3)
        )
        
        self.last_state_path = None
        resume_checkpoint = _normalize_optional_path(getattr(self.config, "resume_checkpoint", ""))
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

    def _runtime_guard_steps(self):
        return max(0, int(os.environ.get("ABFLOW_GRAD_FINITE_GUARD_STEPS", "8") or 0))

    def _autograd_anomaly_steps(self):
        # Diagnostic-only.  The first full forward+backward can be wrapped in PyTorch's
        # anomaly detector so the first autograd Function returning NaN is
        # reported with its forward traceback.  No optimizer/loss/model math is
        # changed, and the default is zero outside the V196 diagnostic configs.
        return max(0, int(os.environ.get("ABFLOW_AUTOGRAD_ANOMALY_STEPS", "0") or 0))

    def _backward_with_optional_anomaly(self, loss):
        # V197: anomaly mode is enabled before the forward in _train_epoch so
        # PyTorch can report the forward call site of the first bad backward op.
        loss.backward()

    def _train_memory_log(self, stage, device):
        """Early-step CUDA memory audit; diagnostic only, no training semantics."""
        if (
            stage != "after_backward"
            or not torch.cuda.is_available()
            or int(self.global_step) >= self._runtime_guard_steps()
            or not self._is_main_proc()
        ):
            return
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        alloc = torch.cuda.memory_allocated(dev) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
        peak_alloc = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved(dev) / (1024 ** 3)
        print_log(
            f"[RuntimeMemory] epoch={self.epoch} step={self.global_step} "
            f"alloc={alloc:.3f}GiB reserved={reserved:.3f}GiB "
            f"peak_alloc={peak_alloc:.3f}GiB peak_reserved={peak_reserved:.3f}GiB"
        )

    def _nonfinite_grad_summary(self, limit=24):
        """Failure-only forensic summary; never repairs or masks gradients."""
        from collections import Counter
        bad, groups = [], Counter()
        nan_elems = inf_elems = 0
        for name, param in self.model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            finite = torch.isfinite(grad)
            if bool(finite.all().detach().cpu().item()):
                continue
            bad.append(name)
            groups['.'.join(name.split('.')[:3])] += 1
            nan_elems += int(torch.isnan(grad).sum().detach().cpu().item())
            inf_elems += int(torch.isinf(grad).sum().detach().cpu().item())
        print_log(
            f"[NonFiniteGradSummary] epoch={self.epoch} step={self.global_step} "
            f"rank={self.local_rank} bad_params={len(bad)} nan_elems={nan_elems} "
            f"inf_elems={inf_elems} groups={groups.most_common(12)} "
            f"params={bad[:limit]}"
        )
        return bad

    @staticmethod
    def _stable_tensor_l2_norm(tensor):
        """L2 norm with scale separation so finite FP32 values cannot overflow.

        This is algebraically the usual Euclidean norm.  Only the reduction is
        performed in a numerically stable form; no gradient value is repaired or
        altered here.
        """
        value = tensor.detach().float()
        if value.numel() == 0:
            return torch.zeros((), dtype=torch.float64, device=value.device)
        max_abs = value.abs().amax()
        if not bool(torch.isfinite(max_abs).detach().cpu().item()):
            return max_abs.to(torch.float64)
        if float(max_abs.detach().cpu().item()) == 0.0:
            return max_abs.to(torch.float64)
        scaled = value / max_abs
        scaled_sq = scaled.square().sum(dtype=torch.float64)
        return max_abs.to(torch.float64) * torch.sqrt(scaled_sq)

    def _stable_finite_grad_l2_norm(self):
        """Stable global L2 norm for already-verified finite gradients."""
        grads = [
            p.grad for p in self.model.parameters()
            if p.grad is not None
        ]
        if not grads:
            device = next(self.model.parameters()).device
            return torch.zeros((), dtype=torch.float64, device=device)

        # One global scale gives the exact same norm while preventing g^2 from
        # overflowing FP32.  The scalar sum is accumulated in FP64.
        maxima = torch.stack([g.detach().float().abs().amax() for g in grads])
        global_max = maxima.amax()
        if not bool(torch.isfinite(global_max).detach().cpu().item()):
            return global_max.to(torch.float64)
        if float(global_max.detach().cpu().item()) == 0.0:
            return global_max.to(torch.float64)

        total_scaled_sq = torch.zeros(
            (), dtype=torch.float64, device=global_max.device
        )
        for grad in grads:
            scaled = grad.detach().float() / global_max
            total_scaled_sq = total_scaled_sq + scaled.square().sum(dtype=torch.float64)
        return global_max.to(torch.float64) * torch.sqrt(total_scaled_sq)

    def _finite_grad_magnitude_summary(self, limit=12):
        """Failure-only ranking of huge but finite parameter gradients."""
        rows = []
        for name, param in self.model.named_parameters():
            grad = param.grad
            if grad is None or grad.numel() == 0:
                continue
            finite = torch.isfinite(grad)
            if not bool(finite.all().detach().cpu().item()):
                continue
            norm64 = self._stable_tensor_l2_norm(grad)
            norm = float(norm64.detach().cpu().item())
            max_abs = float(grad.detach().float().abs().amax().cpu().item())
            rms = norm / max(float(grad.numel()) ** 0.5, 1.0)
            rows.append((norm, name, max_abs, rms, int(grad.numel())))
        rows.sort(key=lambda x: x[0], reverse=True)
        top = rows[:max(1, int(limit))]
        text = [
            {
                'name': name,
                'l2': norm,
                'absmax': max_abs,
                'rms': rms,
                'numel': numel,
            }
            for norm, name, max_abs, rms, numel in top
        ]
        print_log(
            f"[FiniteGradMagnitude] epoch={self.epoch} step={self.global_step} "
            f"rank={self.local_rank} top={text}"
        )
        return text

    def _stable_clip_finite_grad_norm(self, max_norm):
        """Apply mathematically intended clipping after FP32 norm overflow.

        Preconditions: every gradient element is finite.  The only recovered
        condition is a reduction overflow in ``clip_grad_norm_``.
        """
        total_norm = self._stable_finite_grad_l2_norm()
        if not bool(torch.isfinite(total_norm).detach().cpu().item()):
            raise FloatingPointError(
                "stable FP64 gradient norm is non-finite despite finite elements"
            )
        denom = total_norm + total_norm.new_tensor(1.0e-12)
        clip_coef = total_norm.new_tensor(float(max_norm)) / denom
        clip_value = float(clip_coef.detach().cpu().item())
        if clip_value < 1.0:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(clip_coef.to(
                        device=param.grad.device, dtype=param.grad.dtype
                    ))
        return total_norm, min(1.0, clip_value)

    def _checked_clip_grad_norm(self):
        """Parent grad clipping with stable recovery for finite-norm overflow."""
        max_norm = self.config.grad_clip
        if max_norm is None:
            return None
        try:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm, error_if_nonfinite=True
            )
        except TypeError:
            # torch versions without error_if_nonfinite: localize first, then clip.
            bad = [
                name for name, param in self.model.named_parameters()
                if param.grad is not None
                and not bool(torch.isfinite(param.grad).all().detach().cpu().item())
            ][:16]
            if bad:
                print_log(
                    f"[NonFiniteGrad] epoch={self.epoch} step={self.global_step} "
                    f"rank={self.local_rank} params={bad}"
                )
                raise FloatingPointError("non-finite gradient before optimizer.step")
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        except RuntimeError as exc:
            bad = self._nonfinite_grad_summary(limit=24)
            message = str(exc).lower()
            norm_overflow = (
                not bad
                and "total norm" in message
                and "non-finite" in message
            )
            if not norm_overflow:
                print_log(
                    f"[NonFiniteGrad] epoch={self.epoch} step={self.global_step} "
                    f"rank={self.local_rank} params={bad[:16] or ['<unable-to-localize>']}"
                )
                raise FloatingPointError("non-finite gradient before optimizer.step") from exc

            # Every gradient element is finite; only the FP32 norm reduction
            # overflowed.  Recompute the same Euclidean norm stably and apply
            # the configured clipping instead of aborting a mathematically
            # valid optimizer step.
            total_norm, clip_coef = self._stable_clip_finite_grad_norm(max_norm)
            current_epoch = int(self.epoch)
            if self._grad_norm_overflow_epoch != current_epoch:
                self._grad_norm_overflow_epoch = current_epoch
                self._grad_norm_overflow_count = 0
            self._grad_norm_overflow_count += 1
            if self._grad_norm_overflow_count <= self._grad_norm_overflow_log_limit:
                print_log(
                    f"[GradientNormOverflowRecovered] epoch={self.epoch} "
                    f"step={self.global_step} rank={self.local_rank} "
                    f"stable_preclip_norm={float(total_norm.detach().cpu().item()):.6e} "
                    f"clip={float(max_norm):.6g} clip_coef={clip_coef:.6e} "
                    f"ordinal={self._grad_norm_overflow_count}/{self._grad_norm_overflow_log_limit}"
                )
                self._finite_grad_magnitude_summary(limit=12)
            return total_norm

        if self._is_main_proc() and int(self.global_step) < self._runtime_guard_steps():
            print_log(
                f"[GradFinite] epoch={self.epoch} step={self.global_step} "
                f"preclip_total_norm={float(total_norm.detach().cpu().item()):.6g} "
                f"clip={float(max_norm):.6g}"
            )
        return total_norm

    def _check_parameters_finite_after_step(self):
        """Early-step optimizer guard; healthy path performs one host sync."""
        if int(self.global_step) >= self._runtime_guard_steps():
            return
        named = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        if not named:
            return
        flags = torch.stack([torch.isfinite(param.detach()).all() for _, param in named])
        if not bool(flags.all().detach().cpu().item()):
            bad_mask = (~flags).detach().cpu().tolist()
            bad = [name for (name, _), is_bad in zip(named, bad_mask) if is_bad][:16]
            print_log(
                f"[NonFiniteParamAfterStep] epoch={self.epoch} step={self.global_step} "
                f"rank={self.local_rank} params={bad}"
            )
            raise FloatingPointError("optimizer produced non-finite parameters")
        if self._is_main_proc():
            print_log(f"[ParamFinite] epoch={self.epoch} step={self.global_step} status=PASS")

    def _train_epoch(self, device):
        # import ipdb; ipdb.set_trace()
        if self.train_loader.sampler is not None and self.local_rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)

        _tqdm_on = str(os.environ.get("ABFLOW_TQDM", "off")).strip().lower() in {
            "1", "true", "yes", "y", "on"
        }
        t_iter = tqdm(
            self.train_loader,
            dynamic_ncols=True,
            mininterval=float(getattr(self.config, "tqdm_mininterval", 5.0)),
            leave=False,
        ) if self._is_main_proc() and _tqdm_on else self.train_loader

        for batch in t_iter:
            batch = self.to_device(batch, device)

            self.optimizer.zero_grad(set_to_none=True)
            if (
                torch.cuda.is_available()
                and int(self.global_step) < self._runtime_guard_steps()
            ):
                torch.cuda.reset_peak_memory_stats(device)

            anomaly_on = int(self.global_step) < self._autograd_anomaly_steps()
            if anomaly_on:
                print_log(
                    f"[AutogradAnomaly] epoch={self.epoch} step={self.global_step} "
                    f"rank={self.local_rank} status=ON scope=forward+backward detect_nan=1"
                )
                torch.autograd.set_detect_anomaly(True)
            try:
                with self._amp_autocast(device):
                    loss = self.train_step(batch, self.global_step)

                if not bool(torch.isfinite(loss.detach()).all().cpu().item()):
                    print_log(
                        f"[NonFiniteLoss] epoch={self.epoch} step={self.global_step} "
                        f"rank={self.local_rank} loss={loss.detach()}"
                    )
                    raise FloatingPointError("non-finite training loss before backward")
                self._train_memory_log("after_forward", device)

                if self.grad_scaler is not None:
                    scaled_loss = self.grad_scaler.scale(loss)
                    self._backward_with_optional_anomaly(scaled_loss)
                    if self.config.grad_clip is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                        self._checked_clip_grad_norm()
                    self._train_memory_log("after_backward", device)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self._backward_with_optional_anomaly(loss)
                    if self.config.grad_clip is not None:
                        self._checked_clip_grad_norm()
                    self._train_memory_log("after_backward", device)
                    self.optimizer.step()
            finally:
                if anomaly_on:
                    torch.autograd.set_detect_anomaly(False)

            self._check_parameters_finite_after_step()
            self._train_memory_log("after_optimizer", device)
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
                _tqdm_on = str(os.environ.get("ABFLOW_TQDM", "off")).strip().lower() in {
                    "1", "true", "yes", "y", "on"
                }
                t_iter = tqdm(
                    self.valid_loader,
                    dynamic_ncols=True,
                    mininterval=float(getattr(self.config, "tqdm_mininterval", 5.0)),
                    leave=False,
                ) if self._is_main_proc() and _tqdm_on else self.valid_loader
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
        # import ipdb; ipdb.set_trace()
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
            def _env_bool(name, default=False):
                raw = str(os.environ.get(name, 'on' if default else 'off')).strip().lower()
                if raw in {'1', 'true', 'yes', 'y', 'on'}:
                    return True
                if raw in {'0', 'false', 'no', 'n', 'off'}:
                    return False
                raise ValueError(f'{name} must be on/off, got {raw!r}')

            find_unused = _env_bool('ABFLOW_DDP_FIND_UNUSED_PARAMETERS', False)
            static_graph = _env_bool('ABFLOW_DDP_STATIC_GRAPH', False)
            # The formal relational trunk owns its checkpoint setting in JSON.
            # Read the live model instead of historical ABFLOW_ABX_* switches.
            native_trunk = getattr(self.model, 'native_trunk', None)
            trunk_cfg = getattr(getattr(native_trunk, 'trunk', None), 'config', None)
            relational_ckpt = bool(getattr(trunk_cfg, 'activation_checkpoint', False))

            if static_graph and find_unused:
                raise RuntimeError(
                    'DDP static_graph=True requires find_unused_parameters=False.'
                )
            if relational_ckpt and find_unused:
                raise RuntimeError(
                    'PyTorch 1.11 re-entrant relational checkpointing requires '
                    'find_unused_parameters=False.'
                )
            if relational_ckpt and not static_graph:
                raise RuntimeError(
                    'PyTorch 1.11 formal relational checkpointing requires '
                    'DDP static_graph=True.'
                )

            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                gradient_as_bucket_view=True,
                find_unused_parameters=find_unused,
            )

            if static_graph:
                if not hasattr(self.model, '_set_static_graph'):
                    raise RuntimeError(
                        'DDP static_graph=True was requested, but this torch DDP '
                        'implementation does not expose _set_static_graph().'
                    )
                self.model._set_static_graph()

            if self._is_main_proc():
                print_log(
                    '[DDPGraph] '
                    f'world={len(device_ids)} '
                    f'find_unused={int(find_unused)} '
                    f'static_graph={int(static_graph)} '
                    f'relational_checkpoint={int(relational_ckpt)}'
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
        # import ipdb; ipdb.set_trace()
        loss = self.model(batch)
        self.log('Loss/train', loss, batch_idx)
        # import ipdb; ipdb.set_trace()
        return loss

    def valid_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/validation', loss, batch_idx, val=True)
        return loss
