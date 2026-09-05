#!/usr/bin/python
# -*- coding:utf-8 -*-
"""AbFlow trainer with focused human epoch summaries and runtime diagnostics.

The formal human-facing record is one compact ``epoch_summary.csv``. Detailed
mechanism/runtime observations remain in rank-specific runtime logs/TensorBoard.
"""
from math import cos, pi, log, exp
from datetime import datetime
import json
import os
import tempfile
import time
import csv

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from .abs_trainer import Trainer
from .resume_ema import validation_ema, get_rng_state, set_rng_state



_ABFLOW_CONFIG_CACHE = None

def _abflow_config():
    """Load the formal JSON config once from ABFLOW_CONFIG_PATH."""
    global _ABFLOW_CONFIG_CACHE
    if _ABFLOW_CONFIG_CACHE is not None:
        return _ABFLOW_CONFIG_CACHE
    path = os.environ.get("ABFLOW_CONFIG_PATH", "").strip()
    if not path:
        _ABFLOW_CONFIG_CACHE = {}
        return _ABFLOW_CONFIG_CACHE
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _ABFLOW_CONFIG_CACHE = cfg if isinstance(cfg, dict) else {}
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load ABFLOW_CONFIG_PATH={path}: {exc}"
        ) from exc
    return _ABFLOW_CONFIG_CACHE


def _cfg_value(section, key, default):
    cfg = _abflow_config()
    block = cfg.get(section, {}) if isinstance(cfg, dict) else {}
    if isinstance(block, dict) and key in block:
        return block[key]
    return default

def _env_int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else int(default)


def _env_flag(name, default=False):
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "y", "on"}


def _env_float(name, default):
    value = os.environ.get(name, "").strip()
    return float(value) if value else float(default)


def _env_str(name, default):
    value = os.environ.get(name, "").strip()
    return value if value else str(default)



class AlphaFoldLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Exact AF3/MFDesign scheduler equation from the supplied source."""
    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        verbose=False,
        base_lr=0.0,
        max_lr=1.8e-3,
        warmup_no_steps=1000,
        start_decay_after_n_steps=50000,
        decay_every_n_steps=50000,
        decay_factor=0.95,
    ):
        if warmup_no_steps < 0 or start_decay_after_n_steps < 0:
            raise ValueError("scheduler step counts must be nonnegative")
        if warmup_no_steps > start_decay_after_n_steps:
            raise ValueError("warmup_no_steps must not exceed decay start")
        if warmup_no_steps == 0:
            raise ValueError("v132 requires at least one warmup step")
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.base_lr = float(base_lr)
        self.max_lr = float(max_lr)
        self.warmup_no_steps = int(warmup_no_steps)
        self.start_decay_after_n_steps = int(start_decay_after_n_steps)
        self.decay_every_n_steps = int(decay_every_n_steps)
        self.decay_factor = float(decay_factor)
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        step_no = self.last_epoch
        if step_no <= self.warmup_no_steps:
            lr = self.base_lr + (
                step_no / self.warmup_no_steps
            ) * self.max_lr
        elif step_no > self.start_decay_after_n_steps:
            steps_since_decay = step_no - self.start_decay_after_n_steps
            expn = (steps_since_decay // self.decay_every_n_steps) + 1
            lr = self.max_lr * (self.decay_factor ** expn)
        else:
            lr = self.max_lr
        return [lr for _ in self.optimizer.param_groups]

class _ExactDistributedValidationBatchSampler(Sampler):
    """Shard *logical validation batches* across ranks without padding.

    Why batch-level rather than sample-level sharding?  The pre-v3 DDP runtime
    already divided the configured global batch size by world_size and then
    evaluated the *entire* validation set independently on every rank.  Thus the
    rank-0 reference protocol consists of deterministic logical batches of that
    local batch size.

    Assigning those unchanged logical batches round-robin to ranks preserves:
      * exact sample order and batch boundaries;
      * every validation sample exactly once globally;
      * no DistributedSampler padding/duplication;
      * the historical mean-over-validation-batches checkpoint metric.

    Example for 5 logical batches and world_size=2:
      rank0 -> batches 0,2,4
      rank1 -> batches 1,3
    """

    def __init__(self, dataset_len, batch_size, rank, world_size):
        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.dataset_len < 0:
            raise ValueError("dataset_len must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not (0 <= self.rank < self.world_size):
            raise ValueError("invalid rank/world_size")
        self.num_logical_batches = (
            self.dataset_len + self.batch_size - 1
        ) // self.batch_size

    def __iter__(self):
        for logical_batch_idx in range(
            self.rank, self.num_logical_batches, self.world_size
        ):
            start = logical_batch_idx * self.batch_size
            end = min(start + self.batch_size, self.dataset_len)
            yield list(range(start, end))

    def __len__(self):
        if self.rank >= self.num_logical_batches:
            return 0
        return (
            self.num_logical_batches - 1 - self.rank
        ) // self.world_size + 1


class AbFlowTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        self.global_step = 0
        self.epoch = 0
        self.max_step = config.max_epoch * config.step_per_epoch
        self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        super().__init__(model, train_loader, valid_loader, config)

        # ================================================================
        # Module 9 — optimizer/schedule for the modern Pairformer backbone
        # ================================================================
        self._optimizer_name = _env_str("ABFLOW_OPTIMIZER", "adamw").lower()
        if self._optimizer_name not in {"adamw", "adam"}:
            raise ValueError("ABFLOW_OPTIMIZER must be adamw or adam")
        self._weight_decay = _env_float("ABFLOW_WEIGHT_DECAY", 0.01)
        if self._weight_decay < 0.0:
            raise ValueError("ABFLOW_WEIGHT_DECAY must be non-negative")
        self._warmup_epochs = max(
            0, _env_int("ABFLOW_WARMUP_EPOCHS", getattr(config, "warmup", 0))
        )
        self._warmup_steps = min(
            int(self.max_step),
            int(self._warmup_epochs) * int(config.step_per_epoch),
        )

        # ================================================================
        # Module 11 — formal three-phase Train -> Validation -> Test protocol
        # ================================================================
        self._three_phase_protocol = _env_flag(
            "ABFLOW_THREE_PHASE_PROTOCOL", True
        )

        self._diag_main_rank = int(getattr(self.config, "local_rank", -1)) in {-1, 0}
        requested_file_interval = _env_int("ABFLOW_DIAGNOSTIC_FILE_INTERVAL", 0)
        actual_train_steps = max(1, len(self.train_loader))
        self._diag_file_interval = (
            actual_train_steps
            if requested_file_interval <= 0
            else max(1, requested_file_interval)
        )
        self._diag_valid_interval = max(1, _env_int(
            "ABFLOW_DIAGNOSTIC_VALID_INTERVAL", 1
        ))
        requested_grad_interval = _env_int("ABFLOW_GRAD_DIAGNOSTIC_INTERVAL", 0)
        self._diag_grad_interval = (
            actual_train_steps
            if requested_grad_interval <= 0
            else max(1, requested_grad_interval)
        )
        self._diag_enabled = _env_flag("ABFLOW_DIAGNOSTIC_FILE", True)
        self._grad_diag_enabled = _env_flag(
            "ABFLOW_GRAD_CONFLICT_DIAGNOSTICS", False
        )

        # ================================================================
        # Exact DDP validation phase (evaluation infrastructure only)
        # ================================================================
        # v2 still repeated the full validation set on every rank.  v3 shards
        # the historical logical validation batches across the same DDP world
        # and reduces the batch metrics back to the exact global mean.
        self._ddp_validation_enabled = _env_flag(
            "ABFLOW_DDP_VALIDATION", True
        )
        self._ddp_validation_rank = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        self._ddp_validation_world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized() else 1
        )
        self._ddp_validation_active = (
            self._ddp_validation_enabled
            and self._ddp_validation_world_size > 1
        )
        self._validation_num_logical_batches = len(self.valid_loader)
        if self._ddp_validation_active:
            self._rebuild_exact_ddp_validation_loader()

        # ================================================================
        # Full epoch-wise Test phase (evaluation infrastructure only)
        # ================================================================
        # This does NOT change train/validation losses, optimizer, scheduler,
        # scientific model configuration, checkpoint selection or TopK logic.
        # It is a real model.sample + cal_metrics.py observation phase that is
        # inserted after the inherited validation phase.
        self._epoch_test_enabled = _env_flag("ABFLOW_EPOCH_TEST", False)
        self._epoch_test_interval = max(1, _env_int("ABFLOW_EPOCH_TEST_INTERVAL", 1))
        self._epoch_test_json = str(os.environ.get(
            "ABFLOW_EPOCH_TEST_JSON", ""
        ) or "").strip()
        self._epoch_test_pep = str(os.environ.get(
            "ABFLOW_EPOCH_TEST_PEP", ""
        ) or "").strip()
        self._epoch_test_surf = str(os.environ.get(
            "ABFLOW_EPOCH_TEST_SURF", ""
        ) or "").strip()
        self._epoch_test_batch_size = max(1, _env_int(
            "ABFLOW_EPOCH_TEST_BATCH_SIZE", 20
        ))
        self._epoch_test_n_steps = max(1, _env_int(
            "ABFLOW_EPOCH_TEST_N_STEPS", 10
        ))
        self._epoch_test_base_seed = _env_int("ABFLOW_EPOCH_TEST_BASE_SEED", 2023)
        self._epoch_test_metric_workers = max(1, _env_int(
            "ABFLOW_EPOCH_TEST_METRIC_WORKERS", 8
        ))
        self._epoch_test_show_sample_progress = _env_flag(
            "ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS", False
        )
        self._epoch_test_keep_structures = _env_flag(
            "ABFLOW_EPOCH_TEST_KEEP_STRUCTURES", False
        )
        self._epoch_test_fail_fast = _env_flag(
            "ABFLOW_EPOCH_TEST_FAIL_FAST", False
        )
        self._epoch_test_project_root = str(os.environ.get(
            "ABFLOW_PROJECT_ROOT", os.getcwd()
        ) or os.getcwd()).strip()
        self._epoch_test_dataset = None
        self._epoch_test_root = os.path.join(self.config.save_dir, "epoch_test")
        self._epoch_summary_csv = os.path.join(
            self.config.save_dir, "epoch_summary.csv"
        )
        self._best_val_selection_path = os.path.join(
            self.config.save_dir, "best_val_selection.json"
        )
        self._best_val_metric = None
        self._best_val_epoch = None
        self._best_val_checkpoint = None
        self._current_is_best_val = False
        self._last_epoch_train_losses = {}
        self._last_epoch_val_losses = {}
        self._epoch_loss_accum = {
            "train": self._new_epoch_loss_accum(),
            "validation": self._new_epoch_loss_accum(),
        }
        self._restore_best_val_selection()

        # All sparse runtime/scientific audits converge on one rank-specific
        # stream under the current version directory.
        os.environ["ABFLOW_RUNTIME_TRACE_DIR"] = os.path.abspath(
            self.config.save_dir
        )
        if self._three_phase_protocol:
            if not self._epoch_test_enabled:
                raise ValueError(
                    "ABFLOW_THREE_PHASE_PROTOCOL=on requires ABFLOW_EPOCH_TEST=on"
                )
            if int(self._epoch_test_interval) != 1:
                raise ValueError(
                    "Formal Train->Validation->Test requires "
                    "ABFLOW_EPOCH_TEST_INTERVAL=1"
                )
            if not self._epoch_test_json:
                raise ValueError(
                    "Formal Train->Validation->Test requires ABFLOW_EPOCH_TEST_JSON"
                )
            # Missing Test means the epoch protocol is incomplete; fail rather
            # than silently producing an epoch without Test observations.
            self._epoch_test_fail_fast = True

        self._diag_dir = os.path.join(self.config.save_dir, "diagnostics")
        if self._diag_main_rank and self._diag_enabled:
            os.makedirs(self._diag_dir, exist_ok=True)
            schema = {
                "purpose": "Diagnose modern generator objectives, joint structure-sequence state, refinement, optimizer stability and objective conflicts.",
                "files": {
                    "metrics.jsonl": "append-only batch/validation diagnostic records",
                    "latest_train.json": "latest train record",
                    "latest_validation.json": "latest validation record",
                    "alerts.log": "heuristic warnings; warnings are not stopping rules",
                    "../epoch_summary.csv": "focused per-epoch objective losses and core Test metrics",
                    "../runtime_memory_rankN.log": "runtime, gradient, validation, rollout and readout diagnostics",
                },
                "intervals": {
                    "train_steps": self._diag_file_interval,
                    "validation_batches": self._diag_valid_interval,
                    "gradient_probe_steps": self._diag_grad_interval,
                },
            }
            self._atomic_json(os.path.join(self._diag_dir, "schema.json"), schema)

    @staticmethod
    def _new_epoch_loss_accum():
        # Lazy on-device scalar accumulation.  Do NOT call .item()/.cpu() every
        # training step: that would introduce a CUDA synchronization solely for
        # logging.  We reduce to Python numbers once per epoch.
        return {
            "count": None,
            "total": None,
            "transport": None,
            "sequence": None,
            "aligned": None,
            "smooth_lddt": None,
            "distogram": None,
            "confidence": None,
        }

    @staticmethod
    def _detached_scalar(value, ref):
        if torch.is_tensor(value):
            out = value.detach().float()
            if out.numel() != 1:
                out = out.mean()
            return out.reshape(())
        try:
            return ref.new_tensor(float(value), dtype=torch.float32)
        except Exception:
            return ref.new_zeros((), dtype=torch.float32)

    def _runtime_log_line(self, line):
        try:
            rank = (
                int(dist.get_rank())
                if dist.is_available() and dist.is_initialized()
                else 0
            )
            trace_dir = os.environ.get(
                "ABFLOW_RUNTIME_TRACE_DIR", self.config.save_dir
            )
            os.makedirs(trace_dir, exist_ok=True)
            with open(
                os.path.join(trace_dir, f"runtime_memory_rank{rank}.log"),
                "a", encoding="utf-8"
            ) as f:
                f.write(str(line).rstrip("\n") + "\n")
        except Exception:
            pass

    def _restore_best_val_selection(self):
        if not os.path.isfile(self._best_val_selection_path):
            return
        try:
            with open(self._best_val_selection_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._best_val_metric = float(data["validation_loss"])
            self._best_val_epoch = int(data["epoch"])
            self._best_val_checkpoint = str(data.get("checkpoint", "")) or None
        except Exception:
            self._best_val_metric = None
            self._best_val_epoch = None
            self._best_val_checkpoint = None

    def _is_new_global_best_val(self, value):
        value = float(value)
        if self._best_val_metric is None:
            return True
        if bool(self.config.metric_min_better):
            return value < float(self._best_val_metric)
        return value > float(self._best_val_metric)

    def _accumulate_epoch_losses(self, split, total_loss, diagnostics):
        acc = self._epoch_loss_accum[split]
        ref = total_loss.detach().float().reshape(())
        if acc["count"] is None:
            acc["count"] = ref.new_zeros(())
            for key in (
                "total", "transport", "sequence", "aligned",
                "smooth_lddt", "distogram", "confidence",
            ):
                acc[key] = ref.new_zeros(())

        acc["count"] = acc["count"] + 1.0
        acc["total"] = acc["total"] + ref

        mapping = {
            "transport": "v132_weighted_transport",
            "sequence": "v132_weighted_sequence",
            "aligned": "v132_weighted_aligned",
            "smooth_lddt": "v132_weighted_smooth_lddt",
            "distogram": "v132_weighted_distogram",
            "confidence": "v132_weighted_confidence",
        }
        for out_name, diag_name in mapping.items():
            if split == "validation" and out_name == "confidence":
                continue
            acc[out_name] = acc[out_name] + self._detached_scalar(
                diagnostics.get(diag_name, 0.0), ref
            )

    def _finalize_epoch_losses(self, split, device):
        keys = [
            "total", "transport", "sequence", "aligned",
            "smooth_lddt", "distogram", "confidence",
        ]
        acc = self._epoch_loss_accum[split]
        if acc["count"] is None:
            self._epoch_loss_accum[split] = self._new_epoch_loss_accum()
            return {key: float("nan") for key in keys}

        stats = torch.stack(
            [acc[key].to(device=device) for key in keys]
            + [acc["count"].to(device=device)]
        ).double()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        count = max(1.0, float(stats[-1].item()))
        means = {
            key: float(stats[i].item()) / count
            for i, key in enumerate(keys)
        }
        self._epoch_loss_accum[split] = self._new_epoch_loss_accum()
        return means

    @staticmethod
    def _metric_any(metrics, *keys):
        for key in keys:
            if key in metrics:
                try:
                    return float(metrics[key])
                except Exception:
                    pass
        return float("nan")

    @staticmethod
    def _fmt5(value):
        try:
            value = float(value)
            if value != value or abs(value) == float("inf"):
                return ""
            return f"{value:.5f}"
        except Exception:
            return ""

    @staticmethod
    def _round5(value):
        try:
            value = float(value)
            if value != value or abs(value) == float("inf"):
                return None
            return round(value, 5)
        except Exception:
            return None

    def _write_epoch_summary_csv(self, metrics):
        if not self._is_main_proc():
            return
        tr = self._last_epoch_train_losses
        va = self._last_epoch_val_losses
        row = {
            "epoch": int(self.epoch),
            "train_loss": self._fmt5(tr.get("total")),
            "train_transport": self._fmt5(tr.get("transport")),
            "train_sequence": self._fmt5(tr.get("sequence")),
            "train_aligned": self._fmt5(tr.get("aligned")),
            "train_smooth_lddt": self._fmt5(tr.get("smooth_lddt")),
            "train_distogram": self._fmt5(tr.get("distogram")),
            "train_confidence": self._fmt5(tr.get("confidence")),
            "val_loss": self._fmt5(va.get("total")),
            "val_transport": self._fmt5(va.get("transport")),
            "val_sequence": self._fmt5(va.get("sequence")),
            "val_aligned": self._fmt5(va.get("aligned")),
            "val_smooth_lddt": self._fmt5(va.get("smooth_lddt")),
            "val_distogram": self._fmt5(va.get("distogram")),
            "AAR": self._fmt5(self._metric_any(metrics, "AAR_mean")),
            "CAAR": self._fmt5(self._metric_any(metrics, "CAAR_mean")),
            "H3raw": self._fmt5(self._metric_any(metrics, "RMSDCA_CDRH3_mean")),
            "H3aligned": self._fmt5(self._metric_any(
                metrics, "RMSDCA_CDRH3_aligned_mean",
                "RMSDCA_CDRH3_ALIGN_mean", "H3_aligned_RMSD_mean"
            )),
            "TM": self._fmt5(self._metric_any(
                metrics, "TMscore_mean", "TM_score_mean", "TM_mean"
            )),
            "lDDT": self._fmt5(self._metric_any(
                metrics, "LDDT_mean", "lDDT_mean", "lddt_mean"
            )),
            "DockQ": self._fmt5(self._metric_any(metrics, "DockQ_mean")),
        }
        os.makedirs(self.config.save_dir, exist_ok=True)
        write_header = not os.path.isfile(self._epoch_summary_csv)
        with open(self._epoch_summary_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        line = (
            "[EpochSummary] "
            f"epoch={row['epoch']} "
            f"train={row['train_loss']} val={row['val_loss']} "
            f"AAR={row['AAR']} CAAR={row['CAAR']} "
            f"H3raw={row['H3raw']} H3aligned={row['H3aligned']} "
            f"TM={row['TM']} lDDT={row['lDDT']} DockQ={row['DockQ']}"
        )
        print(line, flush=True)
        self._runtime_log_line(line)

    def _write_best_val_selection(self, metrics):
        if not self._is_main_proc() or not self._current_is_best_val:
            return
        core = {
            "AAR": self._round5(self._metric_any(metrics, "AAR_mean")),
            "CAAR": self._round5(self._metric_any(metrics, "CAAR_mean")),
            "H3raw": self._round5(self._metric_any(metrics, "RMSDCA_CDRH3_mean")),
            "H3aligned": self._round5(self._metric_any(
                metrics, "RMSDCA_CDRH3_aligned_mean",
                "RMSDCA_CDRH3_ALIGN_mean", "H3_aligned_RMSD_mean"
            )),
            "TM": self._round5(self._metric_any(
                metrics, "TMscore_mean", "TM_score_mean", "TM_mean"
            )),
            "lDDT": self._round5(self._metric_any(
                metrics, "LDDT_mean", "lDDT_mean", "lddt_mean"
            )),
            "DockQ": self._round5(self._metric_any(metrics, "DockQ_mean")),
        }
        record = {
            "epoch": int(self._best_val_epoch),
            "validation_loss": self._round5(self._best_val_metric),
            "checkpoint": self._best_val_checkpoint,
            "test": core,
        }
        self._atomic_json(self._best_val_selection_path, record)

    def _rebuild_exact_ddp_validation_loader(self):
        """Replace duplicated full-set validation with exact batch sharding.

        The original ``valid_loader.batch_size`` is already the per-rank batch
        size (56 global -> 28/rank in the formal 2-GPU runs).  We therefore
        preserve that batch size and only distribute the pre-existing logical
        batches across ranks.
        """
        old = self.valid_loader
        if old.batch_size is None:
            raise RuntimeError(
                "DDP validation requires a conventional validation DataLoader "
                "with batch_size set."
            )
        sampler = _ExactDistributedValidationBatchSampler(
            dataset_len=len(old.dataset),
            batch_size=int(old.batch_size),
            rank=self._ddp_validation_rank,
            world_size=self._ddp_validation_world_size,
        )
        kwargs = dict(
            dataset=old.dataset,
            batch_sampler=sampler,
            collate_fn=old.collate_fn,
            num_workers=int(old.num_workers),
            pin_memory=bool(old.pin_memory),
            timeout=old.timeout,
            worker_init_fn=old.worker_init_fn,
        )
        if int(old.num_workers) > 0:
            kwargs["prefetch_factor"] = (
                old.prefetch_factor if old.prefetch_factor is not None else 2
            )
            kwargs["persistent_workers"] = bool(old.persistent_workers)
        self.valid_loader = DataLoader(**kwargs)
        self._validation_num_logical_batches = sampler.num_logical_batches
        if self._diag_main_rank:
            print(
                "[DDPValidation] exact logical-batch sharding enabled: "
                f"world_size={self._ddp_validation_world_size} "
                f"logical_batches={sampler.num_logical_batches} "
                f"local_batch_size={old.batch_size}"
            )

    def _validation_reduce_metric(self, metric_arr, device):
        local_sum = float(sum(metric_arr))
        local_count = int(len(metric_arr))
        if self._ddp_validation_active:
            stats = torch.tensor(
                [local_sum, float(local_count)],
                dtype=torch.float64, device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_sum = float(stats[0].item())
            global_count = int(round(float(stats[1].item())))
        else:
            global_sum = local_sum
            global_count = local_count
        if global_count <= 0:
            raise RuntimeError("Validation produced zero logical batches.")
        return global_sum / float(global_count), global_count

    def _gather_validation_writer_buffer(self):
        """Gather per-batch validation scalars from every rank.

        ``share_step`` logs many scientifically useful scalars (overall loss,
        AAR, structure/interface losses and Score--Flow diagnostics).  With
        sharded validation, rank0 alone no longer sees the full set, so the
        buffer itself must be gathered before TensorBoard reduction.
        """
        local = {k: list(v) for k, v in self.writer_buffer.items()}
        if not self._ddp_validation_active:
            return local
        gathered = [None for _ in range(self._ddp_validation_world_size)]
        dist.all_gather_object(gathered, local)
        merged = {}
        for rank_buffer in gathered:
            if not rank_buffer:
                continue
            for name, values in rank_buffer.items():
                merged.setdefault(name, []).extend(values)
        return merged

    def log(self, name, value, step, val=False):
        """Preserve train logging; collect validation scalars on all ranks."""
        if val and self._ddp_validation_active:
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu())
            if name not in self.writer_buffer:
                self.writer_buffer[name] = []
            self.writer_buffer[name].append(value)
            return
        return super().log(name, value, step, val=val)

    def _epoch_test_should_run(self):
        return (
            bool(self._epoch_test_enabled)
            and int(self.epoch) % int(self._epoch_test_interval) == 0
        )

    def _ensure_epoch_test_dataset(self, raw_model):
        if self._epoch_test_dataset is not None:
            return self._epoch_test_dataset
        if not self._epoch_test_json:
            raise ValueError(
                "ABFLOW_EPOCH_TEST=on requires ABFLOW_EPOCH_TEST_JSON."
            )
        if not os.path.isfile(self._epoch_test_json):
            raise FileNotFoundError(
                f"ABFLOW_EPOCH_TEST_JSON not found: {self._epoch_test_json}"
            )

        from data.dataset import E2EDataset

        test_dir = os.path.dirname(os.path.abspath(self._epoch_test_json))
        pep_file = self._epoch_test_pep or os.path.join(test_dir, "test.pkl")
        surf_file = self._epoch_test_surf or os.path.join(test_dir, "test_surf.pkl")
        pep_file = pep_file if os.path.isfile(pep_file) else None
        surf_file = surf_file if os.path.isfile(surf_file) else None

        self._epoch_test_dataset = E2EDataset(
            self._epoch_test_json,
            pep_file=pep_file,
            surf_file=surf_file,
            cdr=raw_model.cdr_type,
        )
        if self._is_main_proc():
            print(
                "[EpochTest] dataset loaded: "
                f"n={len(self._epoch_test_dataset)} json={self._epoch_test_json} "
                f"pep={pep_file} surf={surf_file}"
            )
        return self._epoch_test_dataset

    def _write_epoch_test_error(self, message):
        if not self._is_main_proc():
            return
        os.makedirs(self._epoch_test_root, exist_ok=True)
        with open(
            os.path.join(self._epoch_test_root, "errors.log"),
            "a", encoding="utf-8"
        ) as f:
            f.write(
                f"{datetime.now().isoformat(timespec='seconds')} "
                f"epoch={self.epoch} global_step={self.global_step} "
                f"{message}\n"
            )

    def _run_epoch_test(self, device):
        """Real Test phase: EMA -> model.sample -> PDB -> cal_metrics.py.

        The full training RNG state is restored in ``finally``.  Therefore the
        observation phase cannot advance the stochastic stream used by the next
        training epoch.  This is essential for not perturbing the three current
        R01/R02/R03 optimization trajectories.
        """
        from utils.epoch_test import (
            TB_METRICS,
            assigned_logical_batches,
            cleanup_structures,
            dist_info,
            generate_distributed,
            run_cal_metrics_rank0,
        )

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        dataset = self._ensure_epoch_test_dataset(raw_model)
        epoch_dir = os.path.join(
            self._epoch_test_root, f"epoch_{int(self.epoch):04d}"
        )
        os.makedirs(epoch_dir, exist_ok=True)

        # Preserve all stochastic states so Test is observational only.
        rng_state_before_test = get_rng_state()
        was_training = bool(self.model.training)
        metrics = {}

        # Explicit observability for the SAME-WORLD DDP Test authority.
        #
        # Test intentionally assigns complete logical batches to ranks rather
        # than dividing one logical batch by world_size.  This preserves the
        # batch-keyed sampling RNG protocol across world sizes while still
        # allowing all GPUs to generate different test batches concurrently.
        test_rank, test_world_size = dist_info()
        test_assignments = assigned_logical_batches(
            len(dataset),
            self._epoch_test_batch_size,
            rank=test_rank,
            world_size=test_world_size,
        )
        test_local_samples = sum(len(indices) for _, indices in test_assignments)
        print(
            "[EpochTestDDP] "
            f"rank={test_rank}/{test_world_size} "
            f"device={device} "
            f"logical_batch_size={self._epoch_test_batch_size} "
            f"assigned_batches={len(test_assignments)} "
            f"assigned_samples={test_local_samples}",
            flush=True,
        )
        if self._is_main_proc():
            print(
                "[EpochTestDDP] cooperative_same_checkpoint=on "
                f"world_size={test_world_size} "
                "protocol=logical_batch_seeded_v1",
                flush=True,
            )

        try:
            self.model.eval()

            # V150 epoch-Test memory hygiene.
            #
            # V150 uses three task-level state-evolving outer rounds.  A
            # historical Test logical batch of 20 complexes can exceed a
            # 48-GiB GPU during model.sample.  Clear allocator cache before
            # generation and log the true allocator state.  This is evaluation
            # infrastructure only: parameters, EMA, RNG restoration, Test
            # cadence, n_steps and metrics are unchanged.
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
                torch.cuda.empty_cache()
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    pass
                _alloc_mb = torch.cuda.memory_allocated(device) / (1024.0 ** 2)
                _reserved_mb = torch.cuda.memory_reserved(device) / (1024.0 ** 2)
                _free_b, _total_b = torch.cuda.mem_get_info(device)
                _mem_line = (
                    "[EpochTestMemory] "
                    f"epoch={int(self.epoch)} rank={test_rank} "
                    f"before_alloc={_alloc_mb:.1f}MiB "
                    f"before_reserved={_reserved_mb:.1f}MiB "
                    f"free={_free_b/(1024.0**2):.1f}MiB "
                    f"total={_total_b/(1024.0**2):.1f}MiB "
                    f"logical_batch_size={self._epoch_test_batch_size}"
                )
                print(_mem_line, flush=True)
                self._runtime_log_line(_mem_line)

            # v103 validation_ema applies EMA whenever EMA exists.  This is the
            # same parameter snapshot that is serialized by validation .ckpt.
            with validation_ema(self):
                # Defense-in-depth: model.sample() is already @torch.no_grad()
                # and V150 now preserves the caller grad mode. inference_mode
                # ensures distributed Test cannot accidentally construct an
                # autograd graph in future inner modules.
                with torch.inference_mode():
                    generation = generate_distributed(
                        model=raw_model,
                        dataset=dataset,
                        device=device,
                        save_dir=epoch_dir,
                        batch_size=self._epoch_test_batch_size,
                        n_steps=self._epoch_test_n_steps,
                        base_seed=self._epoch_test_base_seed,
                        show_sample_progress=self._epoch_test_show_sample_progress,
                    )
                metrics = run_cal_metrics_rank0(
                    summary_file=generation.summary_file,
                    save_dir=epoch_dir,
                    project_root=self._epoch_test_project_root,
                    num_workers=self._epoch_test_metric_workers,
                )

            if self._is_main_proc():
                for metric_key, tb_name in TB_METRICS.items():
                    if metric_key in metrics and self.writer is not None:
                        self.writer.add_scalar(
                            tb_name, float(metrics[metric_key]), int(self.epoch)
                        )
                if self.writer is not None:
                    self.writer.flush()
                core = (
                    f"AAR={self._metric_any(metrics, 'AAR_mean'):.5f} "
                    f"CAAR={self._metric_any(metrics, 'CAAR_mean'):.5f} "
                    f"H3raw={self._metric_any(metrics, 'RMSDCA_CDRH3_mean'):.5f} "
                    f"H3aligned={self._metric_any(metrics, 'RMSDCA_CDRH3_aligned_mean', 'RMSDCA_CDRH3_ALIGN_mean', 'H3_aligned_RMSD_mean'):.5f} "
                    f"TM={self._metric_any(metrics, 'TMscore_mean', 'TM_score_mean', 'TM_mean'):.5f} "
                    f"lDDT={self._metric_any(metrics, 'LDDT_mean', 'lDDT_mean', 'lddt_mean'):.5f} "
                    f"DockQ={self._metric_any(metrics, 'DockQ_mean'):.5f}"
                )
                _epoch_test_line = f"[EpochTest] epoch={self.epoch} {core}"
                print(_epoch_test_line, flush=True)
                self._runtime_log_line(_epoch_test_line)

            if not self._epoch_test_keep_structures:
                cleanup_structures(epoch_dir)

        finally:
            # The order matters: restore EMA/raw parameters first (context exit),
            # then restore RNG and training/eval mode.  No optimizer or scheduler
            # state is ever touched by Test.
            set_rng_state(rng_state_before_test)
            if was_training:
                self.model.train()
            else:
                self.model.eval()

            # Release Test-only cached blocks before the next training epoch.
            # Every-epoch Test remains mandatory.
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                try:
                    torch.cuda.synchronize(device)
                except Exception:
                    pass
                torch.cuda.empty_cache()

        return metrics

    def _valid_epoch(self, device):
        """Exact EMA validation on the full set, sharded across DDP ranks.

        In DDP mode we deliberately evaluate through the raw per-rank module
        instead of the DDP wrapper.  DDP forward can synchronize buffers; with
        unequal numbers of validation batches per rank that can deadlock.  The
        underlying parameters are already synchronized by training, and EMA
        shadows are maintained identically on every rank.

        Checkpoint-selection semantics are preserved: the scalar validation
        metric is the mean of the same logical per-rank-sized batches that the
        pre-v3 rank0 validation evaluated over the entire validation set.
        """
        import numpy as np

        self._last_epoch_train_losses = self._finalize_epoch_losses(
            "train", device
        )
        self._epoch_loss_accum["validation"] = self._new_epoch_loss_accum()

        metric_arr = []
        eval_path_to_save = None
        should_save_best = False
        valid_metric = None
        start_valid_global_step = int(self.valid_global_step)

        ddp_model = self.model if hasattr(self.model, "module") else None
        raw_model = self.model.module if ddp_model is not None else self.model
        if ddp_model is not None and self._ddp_validation_active:
            self.model = raw_model

        self.model.eval()
        try:
            with validation_ema(self):
                with torch.no_grad():
                    t_iter = (
                        tqdm(
                            self.valid_loader,
                            dynamic_ncols=True,
                            mininterval=float(
                                getattr(self.config, "tqdm_mininterval", 5.0)
                            ),
                            leave=False,
                        )
                        if self._is_main_proc() else self.valid_loader
                    )
                    for local_batch_idx, batch in enumerate(t_iter):
                        batch = self.to_device(batch, device)
                        with self._amp_autocast(device):
                            metric = self.valid_step(
                                batch, start_valid_global_step + local_batch_idx
                            )
                        metric_arr.append(float(metric.detach().cpu()))

                valid_metric, global_batch_count = self._validation_reduce_metric(
                    metric_arr, device
                )
                # All ranks advance the diagnostic validation step by the same
                # total number of logical batches, preserving resume parity.
                self.valid_global_step = (
                    start_valid_global_step + int(global_batch_count)
                )

                should_save_best = self._metric_better(valid_metric)
                if should_save_best:
                    self.patience = self.config.patience
                    if self._is_main_proc():
                        eval_path_to_save = os.path.join(
                            self.model_dir,
                            f"epoch{self.epoch}_step{self.global_step}.ckpt",
                        )
                        # self.model may be the raw module during DDP validation.
                        torch.save(raw_model, eval_path_to_save)
                else:
                    self.patience -= 1
        finally:
            self.model.train()
            if ddp_model is not None and self._ddp_validation_active:
                self.model = ddp_model
                self.model.train()

        if should_save_best and self._is_main_proc():
            self._maintain_topk_checkpoint(valid_metric, eval_path_to_save)

        # Existing checkpoint/topk_map.txt remains untouched.  We only track the
        # all-time best validation scalar so its same-epoch Test can be recorded
        # in best_val_selection.json.  A true new global minimum necessarily
        # also beats the immediately preceding epoch, so its checkpoint is
        # already saved by the historical path above.
        self._current_is_best_val = self._is_new_global_best_val(valid_metric)
        if self._current_is_best_val:
            self._best_val_metric = float(valid_metric)
            self._best_val_epoch = int(self.epoch)
            if self._is_main_proc():
                self._best_val_checkpoint = eval_path_to_save

        self.last_valid_metric = float(valid_metric)
        self._last_epoch_val_losses = self._finalize_epoch_losses(
            "validation", device
        )

        merged_buffer = self._gather_validation_writer_buffer()
        if self._is_main_proc():
            for name, values in merged_buffer.items():
                if not values:
                    continue
                value = float(np.mean(values))
                self.writer.add_scalar(name, value, self.epoch)

            def _vmean(key):
                values = merged_buffer.get(key, [])
                return float(np.mean(values)) if values else float("nan")

            _val_line = (
                "[ValidationPhysical] "
                f"epoch={int(self.epoch)} val={float(valid_metric):.5f} "
                f"X1decode={_vmean('AbFlowDiag/val_proxy_physical_endpoint_decode/Validation'):.5f} "
                f"H3raw={_vmean('AbFlowDiag/val_proxy_round2_h3_ca_rmsd/Validation'):.5f}A "
                f"H3aligned={_vmean('AbFlowDiag/val_proxy_round2_h3_ca_aligned_rmsd/Validation'):.5f}A "
                f"contactF1={_vmean('AbFlowDiag/val_proxy_native_contact_f1/Validation'):.5f} "
                f"CAAR={_vmean('AbFlowDiag/val_proxy_caar/Validation'):.5f} "
                f"frameworkMove={_vmean('AbFlowDiag/v140_framework_update_from_template_rms/Validation'):.5f}A "
                f"fullAlignedRMSD={_vmean('AbFlowDiag/v149_full_antibody_aligned_rmsd_angstrom/Validation'):.5f}A "
                f"fullAlignedMSEscaled={_vmean('AbFlowDiag/v140_full_antibody_aligned_mse_scaled/Validation'):.5f} "
                f"full_sLDDT_loss={_vmean('AbFlowDiag/v149_full_antibody_smooth_lddt_loss/Validation'):.5f} "
                f"full_sLDDT_score={_vmean('AbFlowDiag/v149_full_antibody_smooth_lddt_score/Validation'):.5f} "
                f"full_hard_lDDT={_vmean('AbFlowDiag/v149_full_antibody_hard_lddt_score/Validation'):.5f}"
            )
            _frame_line = (
                "[FrameAudit] "
                f"epoch={int(self.epoch)} "
                f"templateH3_to_PCS_before={_vmean('AbFlowDiag/v143_frame_template_h3_to_pcs_before_rms/Validation'):.4f}A "
                f"templateH3_to_PCS_after={_vmean('AbFlowDiag/v143_frame_template_h3_to_pcs_after_rms/Validation'):.4f}A "
                f"rotation_deg={_vmean('AbFlowDiag/v143_frame_rotation_deg/Validation'):.3f} "
                f"translation={_vmean('AbFlowDiag/v143_frame_translation_rms/Validation'):.4f}A "
                f"anchor_atoms={_vmean('AbFlowDiag/v143_frame_anchor_atom_count/Validation'):.2f} "
                f"anchor_residues={_vmean('AbFlowDiag/v143_frame_anchor_residue_count/Validation'):.2f} "
                f"rank2={_vmean('AbFlowDiag/v143_frame_anchor_rank2_ratio/Validation'):.4f} "
                f"translation_fallback={_vmean('AbFlowDiag/v143_frame_translation_fallback_rate/Validation'):.4f} "
                "center_source=proposal_aligned_template "
                "frame_anchor=PCS_RC_only native_used_for_frame=0"
            )
            _sequence_authority_line = (
                "[SequenceAuthority] "
                f"epoch={int(self.epoch)} "
                "source=mfdesign_post_token_transformer "
                f"input_dim={_vmean('AbFlowDiag/v144_sequence_latent_input_dim/Validation'):.0f} "
                f"head_hidden={_vmean('AbFlowDiag/v144_sequence_head_hidden_dim/Validation'):.0f} "
                f"input_proj_identity={_vmean('AbFlowDiag/v144_sequence_input_projection_identity/Validation'):.0f} "
                f"raw_latent_rms={_vmean('AbFlowDiag/v144_sequence_raw_latent_rms/Validation'):.4f} "
                f"h3_latent_rms={_vmean('AbFlowDiag/v144_sequence_h3_raw_latent_rms/Validation'):.4f} "
                f"legacy_hidden_authority={_vmean('AbFlowDiag/v144_sequence_legacy_hidden_numerical_authority/Validation'):.0f} "
                "sequence_weight=0.4 terminal=argmax "
                "decision_rule=MAP_clean_endpoint "
                "sequence_process=masked_absorbing_mf_native "
                "denoiser_state_condition=off "
                "state_role=MF_single_stream_condition "
                "seq_input=pep_condition "
                "seq_proposal=pcs_rc_explicit_residual "
                "coord_proposal=pcs_rc_explicit_local "
                "complete_pcs_local_condition=1 "
                "structure_seq_adapter=off"
            )
            _sequence_process_line = (
                "[SequenceProcess] "
                f"epoch={int(self.epoch)} "
                "path=q_t:t_delta_clean+(1-t)_uniform20 "
                "target=clean_endpoint_CE "
                "reverse=exact_x0_marginalized_uniform_bridge "
                "denoiser=p_theta(S1|Xt,t,PCS_context) "
                "St_neural_condition=0 "
                "correctable=1 stochastic_terminal=integrated_state "
                "formal_readout=MAP_clean_endpoint"
            )
            _sequence_shortcut_line = (
                "[SequenceShortcutAudit] "
                f"epoch={int(self.epoch)} "
                f"state_clean={_vmean('AbFlowDiag/seq_state_clean_agreement/Validation'):.4f} "
                f"expected={_vmean('AbFlowDiag/seq_state_clean_agreement_expected/Validation'):.4f} "
                f"pred_copy_state={_vmean('AbFlowDiag/seq_pred_state_copy_rate/Validation'):.4f} "
                f"wrong_state_copy={_vmean('AbFlowDiag/seq_wrong_state_copy_rate/Validation'):.4f} "
                f"wrong_state_recovery={_vmean('AbFlowDiag/seq_wrong_state_clean_recovery/Validation'):.4f} "
                f"correct_state_retention={_vmean('AbFlowDiag/seq_correct_state_retention/Validation'):.4f} "
                f"clean_aar={_vmean('AbFlowDiag/seq_pred_clean_aar_from_state/Validation'):.4f} "
                f"neural_state_authority={_vmean('AbFlowDiag/seq_state_neural_authority/Validation'):.0f}"
            )
            _sequence_proposal_line = (
                "[SequenceProposalAudit] "
                f"epoch={int(self.epoch)} "
                f"proposal_valid={_vmean('AbFlowDiag/seq_proposal_valid_rate/Validation'):.4f} "
                f"proposal_AAR={_vmean('AbFlowDiag/seq_pep_vs_native_aar/Validation'):.4f} "
                f"pred_AAR={_vmean('AbFlowDiag/seq_pred_vs_native_on_pep_valid_aar/Validation'):.4f} "
                f"gain={_vmean('AbFlowDiag/seq_pred_gain_over_pep_aar/Validation'):.4f} "
                f"pred_pep_agree={_vmean('AbFlowDiag/seq_pred_vs_pep_aar/Validation'):.4f} "
                f"wrong_pep_copy={_vmean('AbFlowDiag/seq_wrong_pep_copy_rate/Validation'):.4f} "
                f"wrong_pep_recovery={_vmean('AbFlowDiag/seq_wrong_pep_clean_recovery/Validation'):.4f} "
                f"cond_valid={_vmean('AbFlowDiag/seq_condition_valid_rate/Validation'):.4f} "
                f"cond_residual={_vmean('AbFlowDiag/seq_condition_residual_ratio/Validation'):.4f} "
                f"val_seq_ce={_vmean('Seq/SNLL/Validation'):.4f} "
                f"val_seq_aar={_vmean('Seq/AAR/Validation'):.4f}"
            )
            _coordinate_proposal_line = (
                "[CoordinateProposalAudit] "
                f"epoch={int(self.epoch)} "
                f"explicit={_vmean('AbFlowDiag/coordinate_proposal_condition_explicit/Validation'):.0f} "
                f"cond_valid={_vmean('AbFlowDiag/coord_condition_valid_rate/Validation'):.4f} "
                f"cond_residual={_vmean('AbFlowDiag/coord_condition_residual_ratio/Validation'):.4f} "
                f"full_rmsd={_vmean('AbFlowDiag/v149_full_antibody_aligned_rmsd_angstrom/Validation'):.4f}A "
                f"soft_lddt={_vmean('AbFlowDiag/v149_full_antibody_smooth_lddt_score/Validation'):.4f} "
                f"hard_lddt={_vmean('AbFlowDiag/v149_full_antibody_hard_lddt_score/Validation'):.4f} "
                f"h3raw={_vmean('AbFlowDiag/val_proxy_round2_h3_ca_rmsd/Validation'):.4f}A "
                f"h3aligned={_vmean('AbFlowDiag/val_proxy_round2_h3_ca_aligned_rmsd/Validation'):.4f}A"
            )
            _state_recycle_line = (
                "[MFStatefulRecycleAudit] "
                f"epoch={int(self.epoch)} "
                f"r0_raw={_vmean('AbFlowDiag/val_proxy_round0_h3_ca_rmsd/Validation'):.4f}A "
                f"r1_raw={_vmean('AbFlowDiag/val_proxy_round1_h3_ca_rmsd/Validation'):.4f}A "
                f"r2_raw={_vmean('AbFlowDiag/val_proxy_round2_h3_ca_rmsd/Validation'):.4f}A "
                f"r0_aligned={_vmean('AbFlowDiag/val_proxy_round0_h3_ca_aligned_rmsd/Validation'):.4f}A "
                f"r1_aligned={_vmean('AbFlowDiag/val_proxy_round1_h3_ca_aligned_rmsd/Validation'):.4f}A "
                f"r2_aligned={_vmean('AbFlowDiag/val_proxy_round2_h3_ca_aligned_rmsd/Validation'):.4f}A "
                f"r0_AAR={_vmean('AbFlowDiag/val_proxy_round0_aar/Validation'):.4f} "
                f"r1_AAR={_vmean('AbFlowDiag/val_proxy_round1_aar/Validation'):.4f} "
                f"r2_AAR={_vmean('AbFlowDiag/val_proxy_round2_aar/Validation'):.4f} "
                f"raw_delta={_vmean('AbFlowDiag/val_proxy_refinement_raw_rmsd_delta/Validation'):.4f}A "
                f"aligned_delta={_vmean('AbFlowDiag/val_proxy_refinement_aligned_rmsd_delta/Validation'):.4f}A "
                "outer_rounds=1 outer_grad=off mf_internal_recycle=0 "
                "stateful_recycle=mf_native train_K=random_1_3 infer_K=3 "
                "proposal_start_round=0"
            )
            _mf_line = (
                "[MFBackboneContract] "
                f"epoch={int(self.epoch)} "
                "outer_rounds=1 mf_internal_recycling=0 "
                "pairformer_ckpt=nonreentrant_layer "
                "score_model_ckpt=nonreentrant_layer "
                "state_recycling=mf_native_clean_state "
                "intermediate_recycle=no_grad_detached final_recycle=grad "
                "train_depth=random_1_3 infer_depth=3 nested_recycling=off"
            )
            _bins_line = (
                "[ValidationPhysicalBins] "
                f"epoch={int(self.epoch)} "
                f"raw0={_vmean('AbFlowDiag/val_physical_x1_tbin0_raw_rmsd/Validation'):.5f} "
                f"raw1={_vmean('AbFlowDiag/val_physical_x1_tbin1_raw_rmsd/Validation'):.5f} "
                f"raw2={_vmean('AbFlowDiag/val_physical_x1_tbin2_raw_rmsd/Validation'):.5f} "
                f"raw3={_vmean('AbFlowDiag/val_physical_x1_tbin3_raw_rmsd/Validation'):.5f} "
                f"raw4={_vmean('AbFlowDiag/val_physical_x1_tbin4_raw_rmsd/Validation'):.5f}"
            )
            print(_val_line, flush=True)
            print(_frame_line, flush=True)
            print(_sequence_authority_line, flush=True)
            print(_sequence_process_line, flush=True)
            print(_sequence_shortcut_line, flush=True)
            print(_sequence_proposal_line, flush=True)
            print(_coordinate_proposal_line, flush=True)
            print(_state_recycle_line, flush=True)
            _transport_line = (
                "[TransportAuthorityAudit] "
                f"epoch={int(self.epoch)} "
                "Xt_fixed_inside_recycle=1 "
                "sampler_only_inter_time_update=1 "
                "clean_endpoint_recycled=1 soft_sequence_recycled=1 "
                "mf_s_z_recycled=1 train_depth=random_1_3 infer_depth=3 "
                "train_state_exposure=one_step_u02"
            )
            print(_transport_line, flush=True)
            print(_mf_line, flush=True)
            _closure_line = (
                "[TaskStateClosureAudit] "
                f"epoch={int(self.epoch)} "
                f"exposure_rate={_vmean('AbFlowDiag/v153_sampler_exposure_rate/Validation'):.4f} "
                f"exposure_step={_vmean('AbFlowDiag/v153_sampler_exposure_step_rms/Validation'):.4f}A "
                f"masked_seq_input={_vmean('AbFlowDiag/v153_mf_masked_sequence_input/Validation'):.0f} "
                f"masked_ce_rate={_vmean('AbFlowDiag/v153_sequence_masked_supervision_rate/Validation'):.4f} "
                f"framework_endpoint={_vmean('AbFlowDiag/v153_framework_endpoint_loss/Validation'):.4f} "
                "H3_authority=U02 framework_authority=clean_endpoint "
                "sequence_authority=MF_masked_seq"
            )
            print(_closure_line, flush=True)
            print(_bins_line, flush=True)
            self._runtime_log_line(_val_line)
            self._runtime_log_line(_frame_line)
            self._runtime_log_line(_sequence_authority_line)
            self._runtime_log_line(_sequence_process_line)
            self._runtime_log_line(_sequence_shortcut_line)
            self._runtime_log_line(_sequence_proposal_line)
            self._runtime_log_line(_coordinate_proposal_line)
            self._runtime_log_line(_state_recycle_line)
            self._runtime_log_line(_transport_line)
            self._runtime_log_line(_mf_line)
            self._runtime_log_line(_closure_line)
            self._runtime_log_line(_bins_line)
            if self.writer is not None:
                self.writer.flush()
        self.writer_buffer = {}

    def _test_epoch(self, device):
        """Formal third phase, separated from Validation.

        Test uses the same EMA/sample/cal_metrics core as standalone RAbD test,
        restores RNG afterwards, and is never allowed to change checkpoint
        membership or optimizer/scheduler state.
        """
        if not self._epoch_test_should_run():
            if self._three_phase_protocol:
                raise RuntimeError(
                    "Three-phase protocol requires Test on every epoch."
                )
            self.last_test_metrics = {}
            return self.last_test_metrics
        try:
            metrics = self._run_epoch_test(device)
            self.last_test_metrics = dict(metrics or {})
            self._write_epoch_summary_csv(self.last_test_metrics)
            self._write_best_val_selection(self.last_test_metrics)
            return self.last_test_metrics
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._write_epoch_test_error(message)
            if self._is_main_proc():
                print(f"[EpochTest][ERROR] {message}")
            if self._epoch_test_fail_fast or self._three_phase_protocol:
                raise
            self.last_test_metrics = {}
            return self.last_test_metrics

    def _optimizer_param_groups(self):
        """Decoupled weight decay without penalizing scalar/vector parameters.

        Matrix/tensor weights (ndim>=2) receive AdamW decay; biases, LayerNorm
        scales and other 1-D parameters do not.  The partition is exhaustive and
        deterministic, and introduces no model-specific hand tuning.
        """
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay.append(param)
            else:
                no_decay.append(param)
        return [
            {"params": decay, "weight_decay": float(self._weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def get_optimizer(self):
        """MFDesign optimizer family, task-scaled for scratch antibody design.

        MFDesign source uses Adam (not AdamW).  We copy that equation exactly:
        beta=(0.9,0.95), eps=1e-8 by default.  The max LR is supplied by the
        v132 JSON and is 3e-4 following the ABX scratch training scale.
        """
        optimizer_name = str(_cfg_value("optimizer_profile", "optimizer", _env_str("ABFLOW_OPTIMIZER", "adam"))).lower()
        if optimizer_name != "adam":
            raise ValueError("v132 formal profile requires ABFLOW_OPTIMIZER=adam")
        beta1 = float(_cfg_value("optimizer_profile", "beta1", _env_float("ABFLOW_ADAM_BETA1", 0.9)))
        beta2 = float(_cfg_value("optimizer_profile", "beta2", _env_float("ABFLOW_ADAM_BETA2", 0.95)))
        eps = float(_cfg_value("optimizer_profile", "eps", _env_float("ABFLOW_ADAM_EPS", 1.0e-8)))
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(
            params,
            betas=(beta1, beta2),
            eps=eps,
            lr=float(self.config.lr),
        )

    def get_scheduler(self, optimizer):
        """Exact AF3 scheduler equation with ABX scratch horizon ratios."""
        mode = str(_cfg_value("optimizer_profile", "scheduler", _env_str("ABFLOW_LR_SCHEDULER", "abx_taskscale_af3"))).lower()
        if mode != "abx_taskscale_af3":
            raise ValueError("v132 requires ABFLOW_LR_SCHEDULER=abx_taskscale_af3")

        total_steps = max(1, int(self.max_step))
        warmup_ratio = float(_cfg_value("optimizer_profile", "warmup_ratio", _env_float("ABFLOW_LR_WARMUP_RATIO", 0.05)))
        decay_start_ratio = float(_cfg_value("optimizer_profile", "decay_start_ratio", _env_float("ABFLOW_LR_DECAY_START_RATIO", 0.50)))
        decay_every_ratio = float(_cfg_value("optimizer_profile", "decay_every_ratio", _env_float("ABFLOW_LR_DECAY_EVERY_RATIO", 0.10)))
        decay_factor = float(_cfg_value("optimizer_profile", "decay_factor", _env_float("ABFLOW_LR_DECAY_FACTOR", 0.95)))
        base_lr = float(_cfg_value("optimizer_profile", "base_lr", _env_float("ABFLOW_LR_BASE", 0.0)))
        max_lr = float(self.config.lr)

        warmup_steps = max(1, int(round(total_steps * warmup_ratio)))
        decay_start = max(
            warmup_steps,
            int(round(total_steps * decay_start_ratio)),
        )
        decay_every = max(1, int(round(total_steps * decay_every_ratio)))
        decay_start = min(decay_start, total_steps)

        scheduler = AlphaFoldLRScheduler(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            warmup_no_steps=warmup_steps,
            start_decay_after_n_steps=decay_start,
            decay_every_n_steps=decay_every,
            decay_factor=decay_factor,
        )
        # TrainerBase constructs the scheduler before assigning self.local_rank.
        # Use the torchrun environment here; later training logs may safely use
        # self._is_main_proc().
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                "[V137Config][Optimizer] "
                f"Adam lr_max={max_lr:.3e} beta=("
                f"{optimizer.param_groups[0].get('betas', (None,None))[0]},"
                f"{optimizer.param_groups[0].get('betas', (None,None))[1]}) "
                f"warmup_steps={warmup_steps} decay_start={decay_start} "
                f"decay_every={decay_every} decay_factor={decay_factor}",
                flush=True,
            )
        return {"scheduler": scheduler, "frequency": "batch"}

    def train_step(self, batch, batch_idx):
        batch['context_ratio'] = self.get_context_ratio()
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        batch['context_ratio'] = 0
        return self.share_step(batch, batch_idx, val=True)

    ########## Override end ##########

    def get_context_ratio(self):
        step = self.global_step
        return 0.5 * (cos(step / self.max_step * pi) + 1) * 0.9

    @staticmethod
    def _scalar(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            return float(value.detach().float().cpu().item())
        if isinstance(value, (int, float, bool)):
            return float(value)
        return None

    @staticmethod
    def _atomic_json(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp_diag_", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _should_write(self, val, batch_idx):
        if not self._diag_enabled or not self._diag_main_rank:
            return False
        if val:
            return int(batch_idx) % self._diag_valid_interval == 0
        return int(self.global_step) % self._diag_file_interval == 0

    def _should_probe_grad(self, val):
        # interval=0 means once per actual train epoch, not on batch 0.
        step = int(self.global_step)
        return (
            (not val)
            and self._grad_diag_enabled
            and step > 0
            and step % self._diag_grad_interval == 0
        )

    @staticmethod
    def _diagnostic_alerts(record):
        """Heuristic warnings only; thresholds are deliberately conservative."""
        alerts = []
        get = lambda k: record.get(k, None)

        disagreement = get("diag/seq_state_token_disagreement_rate")
        state_res = get("diag/seq_state_residual_ratio")
        state_grad = get("grad/grad_probe_norm_seq")
        if disagreement is not None and disagreement > 0.05:
            if state_res is not None and state_res < 1e-6:
                alerts.append("SEQ_STATE_HIDDEN_PATH_NEAR_ZERO")
            if state_grad is not None and state_grad < 1e-10:
                alerts.append("SEQ_OBJECTIVE_GRAD_NEAR_ZERO")

        aux_ratio = get("dtm/scorefm_satc_aux_to_endpoint")
        if aux_ratio is not None and aux_ratio > 0.25:
            alerts.append("SATC_AUX_LARGE_RELATIVE_TO_ENDPOINT")

        neg_rate = get("dtm/scorefm_satc_normal_ratio_negative_rate")
        if neg_rate is not None and neg_rate > 0.50:
            alerts.append("SATC_CORRECTION_OFTEN_POINTS_AWAY_FROM_PATH")

        clip_rate = get("dtm/scorefm_satc_normal_ratio_clipped_rate")
        if clip_rate is not None and clip_rate > 0.25:
            alerts.append("SATC_PROJECTION_FREQUENTLY_CLIPPED")

        internal_fraction = get("dtm/scorefm_satc_perturb_internal_energy_fraction")
        if internal_fraction is not None and internal_fraction > 0.70:
            alerts.append("SATC_TUBE_DOMINATED_BY_INTERNAL_DEFORMATION")
        relative_tube = get("dtm/scorefm_satc_perturb_to_transport_rms")
        if relative_tube is not None and 0 < relative_tube < 0.01:
            alerts.append("SATC_TUBE_TINY_RELATIVE_TO_CLEAN_TRANSPORT")

        ess = get("dtm/scorefm_satc_interface_weight_ess")
        if ess is not None and ess > 0 and ess < 0.50:
            alerts.append("INTERFACE_WEIGHT_TOO_CONCENTRATED")

        round_delta = get("diag/val_proxy_refinement_raw_rmsd_delta")
        if round_delta is not None and round_delta > 0.20:
            alerts.append("LATE_REFINEMENT_DEGRADES_GLOBAL_H3_PLACEMENT")

        grad_failed = get("grad/grad_probe_failed")
        if grad_failed is not None and grad_failed > 0.5:
            alerts.append("GRADIENT_DIAGNOSTIC_FAILED_MAIN_TRAINING_CONTINUED")

        for key in (
            "grad/grad_probe_cos_endpoint_satc",
            "grad/grad_probe_cos_seq_satc",
            "grad/grad_probe_cos_structure_satc",
        ):
            value = get(key)
            if value is not None and value < -0.20:
                alerts.append("GRADIENT_CONFLICT:" + key.split("/", 1)[1])
        return alerts

    def _write_diagnostic_record(self, record):
        if not self._diag_main_rank or not self._diag_enabled:
            return
        record["alerts"] = self._diagnostic_alerts(record)
        jsonl = os.path.join(self._diag_dir, "metrics.jsonl")
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        latest = os.path.join(
            self._diag_dir,
            "latest_validation.json" if record["split"] == "validation" else "latest_train.json",
        )
        self._atomic_json(latest, record)
        if record["alerts"]:
            with open(os.path.join(self._diag_dir, "alerts.log"), "a", encoding="utf-8") as f:
                f.write(
                    f"{record['timestamp']} epoch={record['epoch']} "
                    f"step={record['global_step']} split={record['split']} "
                    + ",".join(record["alerts"]) + "\n"
                )

    def share_step(self, batch, batch_idx, val=False):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        # In sharded DDP validation every rank owns distinct logical batches.
        # Capture validation diagnostics on every rank so epoch-level TensorBoard
        # aggregation reflects the entire validation set.  File writing remains
        # rank0-only to avoid concurrent JSONL writes.
        should_probe_grad = bool(self._should_probe_grad(val))
        capture_diagnostics = (
            (bool(val) and bool(getattr(self, "_ddp_validation_active", False)))
            or self._should_write(val, batch_idx)
            or should_probe_grad
        )
        raw_model._diagnostic_capture = bool(capture_diagnostics)
        raw_model._diagnostic_validation_mode = bool(val and capture_diagnostics)
        raw_model._gradient_diagnostic_capture = should_probe_grad

        _perf_probe = (not val) and int(getattr(self, "global_step", 0)) < 8
        if _perf_probe and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        _forward_t0 = time.perf_counter() if _perf_probe else None
        loss, seq_detail, structure_detail, dock_detail, pdev_detail = self.model(**batch)
        if _perf_probe:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _forward_ms = (time.perf_counter() - _forward_t0) * 1000.0
            _peak_alloc_gib = (
                torch.cuda.max_memory_allocated() / (1024.0 ** 3)
                if torch.cuda.is_available() else 0.0
            )
            _peak_reserved_gib = (
                torch.cuda.max_memory_reserved() / (1024.0 ** 3)
                if torch.cuda.is_available() else 0.0
            )
        else:
            _forward_ms = None
            _peak_alloc_gib = None
            _peak_reserved_gib = None
        snll, aar = seq_detail
        struct_loss, xloss, bond_loss, sc_bond_loss = structure_detail
        dock_loss, interface_loss, ed_loss, r_ed_losses = dock_detail
        pdev_loss, prmsd_loss = pdev_detail

        if should_probe_grad and hasattr(raw_model, "compute_gradient_conflict_diagnostics"):
            raw_model.compute_gradient_conflict_diagnostics()
            grad_error = str(getattr(raw_model, "_last_gradient_diagnostic_error", "") or "")
            if grad_error and self._diag_main_rank and self._diag_enabled:
                os.makedirs(self._diag_dir, exist_ok=True)
                with open(
                    os.path.join(self._diag_dir, "gradient_diagnostic_errors.log"),
                    "a", encoding="utf-8"
                ) as f:
                    f.write(
                        f"{datetime.now().isoformat(timespec='seconds')} "
                        f"epoch={self.epoch} step={self.global_step} {grad_error}\n"
                    )
        else:
            raw_model.last_gradient_diagnostics = {}

        if hasattr(raw_model, "release_gradient_diagnostic_graph_refs"):
            raw_model.release_gradient_diagnostic_graph_refs()

        log_type = 'Validation' if val else 'Train'
        self.log(f'Overall/Loss/{log_type}', loss, batch_idx, val)
        self.log(f'Seq/SNLL/{log_type}', snll, batch_idx, val)
        self.log(f'Seq/AAR/{log_type}', aar, batch_idx, val)
        self.log(f'Struct/StructLoss/{log_type}', struct_loss, batch_idx, val)
        self.log(f'Struct/XLoss/{log_type}', xloss, batch_idx, val)
        self.log(f'Struct/BondLoss/{log_type}', bond_loss, batch_idx, val)
        self.log(f'Struct/SidechainBondLoss/{log_type}', sc_bond_loss, batch_idx, val)
        self.log(f'Dock/DockLoss/{log_type}', dock_loss, batch_idx, val)
        self.log(f'Dock/SPLoss/{log_type}', interface_loss, batch_idx, val)
        self.log(f'Dock/EDLoss/{log_type}', ed_loss, batch_idx, val)
        for i, l in enumerate(r_ed_losses):
            self.log(f'Dock/edloss{i}/{log_type}', l, batch_idx, val)
        if pdev_loss is not None:
            self.log(f'PDev/PDevLoss/{log_type}', pdev_loss, batch_idx, val)
            self.log(f'PDev/PRMSDLoss/{log_type}', prmsd_loss, batch_idx, val)

        scorefm_losses = getattr(raw_model, "last_scorefm_losses", None) or {}
        for name, value in scorefm_losses.items():
            self.log(f"DTM/{name}/{log_type}", value, batch_idx, val)

        # Modules 6--8 + Confidence diagnostics.  The model already returns the
        # correct optimization loss: training includes detached confidence
        # calibration, validation excludes it from checkpoint authority.
        modern_aux = getattr(
            raw_model, "last_modern_auxiliary_losses", None
        ) or {}
        for name, value in modern_aux.items():
            self.log(
                f"Modern/{name}/{log_type}",
                value,
                batch_idx,
                val,
            )

        abflow_diagnostics = getattr(raw_model, "last_abflow_diagnostics", None) or {}
        for name, value in abflow_diagnostics.items():
            self.log(f"AbFlowDiag/{name}/{log_type}", value, batch_idx, val)

        self._accumulate_epoch_losses(
            "validation" if val else "train",
            loss,
            abflow_diagnostics,
        )

        if _perf_probe:
            runtime_stats = {}
            if hasattr(raw_model, "gnn") and hasattr(
                raw_model.gnn, "consume_runtime_perf_stats"
            ):
                runtime_stats = raw_model.gnn.consume_runtime_perf_stats()

            def _dget(name, default=float("nan")):
                value = abflow_diagnostics.get(name, default)
                try:
                    if torch.is_tensor(value):
                        return float(value.detach().float().cpu().item())
                    return float(value)
                except Exception:
                    return float("nan")

            if self._is_main_proc():
                real_tok = float(runtime_stats.get("real_tokens", 0) or 0)
                padded_tok = float(runtime_stats.get("padded_tokens", 0) or 0)
                pad_eff = real_tok / padded_tok if padded_tok > 0 else float("nan")
                _step_line = (
                    "[V153Step] "
                    f"step={int(self.global_step)} forward_ms={_forward_ms:.1f} "
                    f"peak_alloc={_peak_alloc_gib:.2f}GiB "
                    f"peak_reserved={_peak_reserved_gib:.2f}GiB "
                    f"outer_rounds={int(getattr(raw_model, 'round', -1))} "
                    f"mf_recycle={getattr(raw_model.gnn, '_last_effective_recycling_steps', -1)} "
                    f"score_ckpt={getattr(raw_model.gnn.atom_structure, 'score_checkpoint_mode', 'off')} "
                    f"pad_eff={pad_eff:.3f} "
                    f"PFcalls={runtime_stats.get('batched_pairformer_calls', 0)} "
                    f"AtomCalls={runtime_stats.get('batched_atom_calls', 0)} "
                    f"SCteacher={runtime_stats.get('sc_teacher_calls', 0)}/"
                    f"graphs={runtime_stats.get('sc_teacher_graphs', 0)} "
                    f"SCformal={runtime_stats.get('sc_formal_calls', 0)} "
                    f"ConfCalls={runtime_stats.get('confidence_calls', 0)} "
                    f"wT={_dget('v132_weighted_transport'):.5f} "
                    f"wS={_dget('v132_weighted_sequence'):.5f} "
                    f"wA={_dget('v132_weighted_aligned'):.5f} "
                    f"wL={_dget('v132_weighted_smooth_lddt'):.5f} "
                    f"wD={_dget('v132_weighted_distogram'):.5f} "
                    f"wC={_dget('v132_weighted_confidence'):.5f} "
                    f"frameworkMove={_dget('v140_framework_update_from_template_rms'):.5f}A "
                    f"frameBefore={_dget('v143_frame_template_h3_to_pcs_before_rms'):.3f}A "
                    f"frameAfter={_dget('v143_frame_template_h3_to_pcs_after_rms'):.3f}A "
                    f"frameAtoms={_dget('v143_frame_anchor_atom_count'):.1f} "
                    f"frameResidues={_dget('v143_frame_anchor_residue_count'):.1f} "
                    f"frameFallback={_dget('v143_frame_translation_fallback_rate'):.2f} "
                    f"seqRaw={_dget('v144_sequence_authority_mfdesign_raw'):.0f} "
                    f"seqDim={_dget('v144_sequence_latent_input_dim'):.0f} "
                    f"seqHead={_dget('v144_sequence_head_hidden_dim'):.0f} "
                    f"seqProjId={_dget('v144_sequence_input_projection_identity'):.0f} "
                    f"seqH3RMS={_dget('v144_sequence_h3_raw_latent_rms'):.3f} "
                    f"seqStateNet={_dget('sequence_denoiser_state_conditioning'):.0f} "
                    f"seqStateSamplerOnly={_dget('sequence_state_sampler_only'):.0f} "
                    f"disto_gate={_dget('v137_distogram_gate_mean'):.3f} "
                    f"SCrate={self._scalar(modern_aux.get('self_condition_rate'))}"
                )
                print(_step_line, flush=True)
                self._runtime_log_line(_step_line)

        grad_diagnostics = getattr(raw_model, "last_gradient_diagnostics", None) or {}
        for name, value in grad_diagnostics.items():
            self.log(f"GradientDiag/{name}/{log_type}", value, batch_idx, val)

        if should_probe_grad and self._is_main_proc() and grad_diagnostics:
            def _g(name):
                value = grad_diagnostics.get(name, float("nan"))
                try:
                    if torch.is_tensor(value):
                        return float(value.detach().float().cpu().item())
                    return float(value)
                except Exception:
                    return float("nan")
            _gt = _g('grad_probe_norm_endpoint')
            _gs = _g('grad_probe_norm_seq')
            _gs_head = _g('grad_probe_norm_seq_head')
            _gs_pep = _g('grad_probe_norm_seq_pep_adapter')
            _gx_head = _g('grad_probe_norm_coord_head')
            _gx_pep = _g('grad_probe_norm_coord_pep_adapter')
            _ratio = _gt / _gs if abs(_gs) > 1.0e-30 else float('nan')
            _authority_line = (
                "[LossAuthority] "
                f"epoch={int(self.epoch)} step={int(self.global_step)} "
                f"|gT_shared|={_gt:.3e} "
                f"|gS_shared|={_gs:.3e} "
                f"|gS_head|={_gs_head:.3e} "
                f"|gS_pep|={_gs_pep:.3e} "
                f"|gX_head|={_gx_head:.3e} "
                f"|gX_pep|={_gx_pep:.3e} "
                f"|gA|={_g('grad_probe_norm_aligned'):.3e} "
                f"|gL|={_g('grad_probe_norm_smooth_lddt'):.3e} "
                f"|gD|={_g('grad_probe_norm_distogram'):.3e} "
                f"gT/gS={_ratio:.3f} "
                f"cos(T,A)={_g('grad_probe_cos_endpoint_aligned'):.3f} "
                f"cos(T,L)={_g('grad_probe_cos_endpoint_smooth_lddt'):.3f} "
                f"cos(A,L)={_g('grad_probe_cos_aligned_smooth_lddt'):.3f} "
                f"cos(T,S)={_g('grad_probe_cos_endpoint_seq'):.3f} "
                f"framework_scope=cmask fullAb_geometry=on outer_rounds=1 mf_internal_recycle=0 mf_stateful=1 train_K=random1-3 infer_K=3 Xt_fixed=1 terminal=argmax frame=pcs_h3_kabsch seq_latent=mfdesign_2x_token seq_process=uniform_reversible seq_state_net=off seq_readout=map seq_input=pep_condition seq_proposal=explicit coord_proposal=explicit"
            )
            print(_authority_line, flush=True)
            self._runtime_log_line(_authority_line)

        lr = None
        if not val:
            lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()[0]
            self.log('lr', lr, batch_idx, val)
            self.log('context_ratio', batch['context_ratio'], batch_idx, val)

        if self._should_write(val, batch_idx):
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "split": "validation" if val else "train",
                "epoch": int(getattr(self, "epoch", -1)),
                "global_step": int(getattr(self, "global_step", -1)),
                "batch_idx": int(batch_idx),
                "loss/overall": self._scalar(loss),
                "loss/seq_snll": self._scalar(snll),
                "metric/aar": self._scalar(aar),
                "loss/structure": self._scalar(struct_loss),
                "loss/x": self._scalar(xloss),
                "loss/bond": self._scalar(bond_loss),
                "loss/sidechain_bond": self._scalar(sc_bond_loss),
                "loss/dock": self._scalar(dock_loss),
                "loss/interface": self._scalar(interface_loss),
                "loss/edge": self._scalar(ed_loss),
                "lr": None if lr is None else float(lr),
                "context_ratio": float(batch.get("context_ratio", 0)),
            }
            for prefix, values in (
                ("dtm", scorefm_losses),
                ("modern", modern_aux),
                ("diag", abflow_diagnostics),
                ("grad", grad_diagnostics),
            ):
                for name, value in values.items():
                    scalar = self._scalar(value)
                    if scalar is not None:
                        record[f"{prefix}/{name}"] = scalar
            self._write_diagnostic_record(record)
        return loss
