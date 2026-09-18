#!/usr/bin/python
# -*- coding:utf-8 -*-
"""AbFlow trainer with low-overhead, machine-readable diagnostics.

The TensorBoard logging behavior is preserved.  Main-rank JSONL/latest files are
added so each epoch can be inspected without opening TensorBoard or evaluating
all test checkpoints.  Periodic gradient-conflict probes are observational only.
"""
from math import cos, pi, log, exp
from datetime import datetime
import json
import os
import tempfile

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from .abs_trainer import Trainer
from .resume_ema import validation_ema, get_rng_state, set_rng_state


def _env_int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else int(default)


def _env_flag(name, default=False):
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "y", "on"}


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

        self._diag_dir = os.path.join(self.config.save_dir, "diagnostics")
        if self._diag_main_rank and self._diag_enabled:
            os.makedirs(self._diag_dir, exist_ok=True)
            schema = {
                "purpose": "Diagnose joint state, proposal influence, SATC scale, refinement rounds and objective conflicts.",
                "files": {
                    "metrics.jsonl": "append-only batch/validation diagnostic records",
                    "latest_train.json": "latest train record",
                    "latest_validation.json": "latest validation record",
                    "alerts.log": "heuristic warnings; warnings are not stopping rules",
                },
                "intervals": {
                    "train_steps": self._diag_file_interval,
                    "validation_batches": self._diag_valid_interval,
                    "gradient_probe_steps": self._diag_grad_interval,
                },
            }
            self._atomic_json(os.path.join(self._diag_dir, "schema.json"), schema)

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
            append_epoch_metrics,
            cleanup_structures,
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
        try:
            self.model.eval()
            # v103 validation_ema applies EMA whenever EMA exists.  This is the
            # same parameter snapshot that is serialized by validation .ckpt.
            with validation_ema(self):
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

            append_epoch_metrics(
                root_dir=self._epoch_test_root,
                epoch=self.epoch,
                global_step=self.global_step,
                metrics=metrics,
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
                    f"AAR={metrics.get('AAR_mean', float('nan')):.4f} "
                    f"CAAR={metrics.get('CAAR_mean', float('nan')):.4f} "
                    f"H3raw={metrics.get('RMSDCA_CDRH3_mean', float('nan')):.4f} "
                    f"DockQ={metrics.get('DockQ_mean', float('nan')):.4f}"
                )
                print(f"[EpochTest] epoch={self.epoch} {core}")

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

        self.last_valid_metric = float(valid_metric)

        merged_buffer = self._gather_validation_writer_buffer()
        if self._is_main_proc():
            for name, values in merged_buffer.items():
                if not values:
                    continue
                value = float(np.mean(values))
                self.writer.add_scalar(name, value, self.epoch)
            if self.writer is not None:
                self.writer.flush()
        self.writer_buffer = {}

        # Third epoch phase: real EMA rollout Test.  This is observation-only
        # and leaves the scientific R01/R02/R03 configurations untouched.
        if not self._epoch_test_should_run():
            return
        try:
            self._run_epoch_test(device)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._write_epoch_test_error(message)
            if self._is_main_proc():
                print(f"[EpochTest][ERROR] {message}")
            if self._epoch_test_fail_fast:
                raise

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def get_scheduler(self, optimizer):
        log_alpha = self.log_alpha
        lr_lambda = lambda step: exp(log_alpha * (step + 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {'scheduler': scheduler, 'frequency': 'batch'}

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
        capture_diagnostics = (
            (bool(val) and bool(getattr(self, "_ddp_validation_active", False)))
            or self._should_write(val, batch_idx)
            or self._should_probe_grad(val)
        )
        raw_model._diagnostic_capture = bool(capture_diagnostics)
        raw_model._diagnostic_validation_mode = bool(val and capture_diagnostics)

        loss, seq_detail, structure_detail, dock_detail, pdev_detail = self.model(**batch)
        snll, aar = seq_detail
        struct_loss, xloss, bond_loss, sc_bond_loss = structure_detail
        dock_loss, interface_loss, ed_loss, r_ed_losses = dock_detail
        pdev_loss, prmsd_loss = pdev_detail

        if self._should_probe_grad(val) and hasattr(raw_model, "compute_gradient_conflict_diagnostics"):
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

        abflow_diagnostics = getattr(raw_model, "last_abflow_diagnostics", None) or {}
        for name, value in abflow_diagnostics.items():
            self.log(f"AbFlowDiag/{name}/{log_type}", value, batch_idx, val)

        grad_diagnostics = getattr(raw_model, "last_gradient_diagnostics", None) or {}
        for name, value in grad_diagnostics.items():
            self.log(f"GradientDiag/{name}/{log_type}", value, batch_idx, val)

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
                ("diag", abflow_diagnostics),
                ("grad", grad_diagnostics),
            ):
                for name, value in values.items():
                    scalar = self._scalar(value)
                    if scalar is not None:
                        record[f"{prefix}/{name}"] = scalar
            self._write_diagnostic_record(record)
        return loss
