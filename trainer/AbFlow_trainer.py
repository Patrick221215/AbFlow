#!/usr/bin/python
# -*- coding:utf-8 -*-
"""AbFlow trainer with low-overhead, machine-readable diagnostics.

The TensorBoard logging behavior is preserved.  Main-rank JSONL/latest files are
added so each epoch can be inspected without opening TensorBoard or evaluating
all test checkpoints.  Periodic gradient-conflict probes are observational only.
"""
from math import cos, pi, log, exp, isfinite
import csv
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from .abs_trainer import Trainer
from .resume_ema import validation_ema, get_rng_state, set_rng_state


def _env_int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else int(default)


def _env_float(name, default):
    value = os.environ.get(name, "").strip()
    return float(value) if value else float(default)


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

        # Epoch-level scientific summaries.  These are observational only and
        # never participate in gradient computation or checkpoint selection.
        # We keep only the formal top-level objective components so the canonical
        # epoch table stays compact and directly comparable across R08/R09/R10.
        self._train_component_names = (
            "loss", "seq", "structure", "interface", "edge",
            "distogram", "smooth_lddt"
        )
        self._epoch_train_acc_epoch = -1
        self._epoch_train_sums = {name: 0.0 for name in self._train_component_names}
        self._epoch_train_counts = {name: 0 for name in self._train_component_names}
        self._last_validation_summary = {}
        self._last_epoch_test_metrics = {}

        # Diagnostics are intentionally console-first.  The launcher tees the
        # complete stdout/stderr stream into version_0/run_time.log, so we do not
        # maintain a second fragmented metrics/latest/alerts file tree.
        self._diag_main_rank = int(getattr(self.config, "local_rank", -1)) in {-1, 0}
        requested_grad_interval = _env_int("ABFLOW_GRAD_DIAGNOSTIC_INTERVAL", 0)
        actual_train_steps = max(1, len(self.train_loader))
        self._diag_grad_interval = (
            actual_train_steps
            if requested_grad_interval <= 0
            else max(1, requested_grad_interval)
        )
        self._grad_diag_enabled = _env_flag(
            "ABFLOW_GRAD_CONFLICT_DIAGNOSTICS", False
        )

        # Epoch-0 R08/R10 summaries exposed rare ~1e5-1e6 structure-loss means
        # that were invisible in rank-0 tqdm.  Record the first true per-rank
        # outlier in run_time.log instead of guessing whether this is aggregation
        # error or a real hard sample on another DDP rank.  Observational only.
        self._train_loss_outlier_threshold = _env_float(
            "ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD", 1.0e4
        )
        self._train_loss_outlier_logged_epoch = -1

        self._epoch_summary_path = os.path.join(
            self.config.save_dir, "epoch_summary.csv"
        )
        self._best_val_metric = None
        self._best_val_epoch = None
        self._best_val_test = {}
        self._restore_best_val_summary_if_present()

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

        if self._diag_main_rank:
            print(
                "[LossContract] "
                f"sequence={getattr(model, 'loss_sequence_weight', float('nan')):.4g} "
                f"structure={getattr(model, 'loss_structure_weight', float('nan')):.4g} "
                f"interface={getattr(model, 'loss_interface_weight', float('nan')):.4g} "
                f"edge={getattr(model, 'loss_edge_weight', float('nan')):.4g} "
                f"distogram={getattr(model, 'loss_distogram_weight', float('nan')):.4g} "
                "authority=R05_parent_plus_optional_pair_aux"
            )


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
        # Errors are printed by the caller and captured in run_time.log.
        return

    def _run_epoch_test(self, device):
        """Real Test phase: EMA -> model.sample -> PDB -> cal_metrics.py.

        The full training RNG state is restored in ``finally``.  Therefore the
        observation phase cannot advance the stochastic stream used by the next
        training epoch.  This is essential for not perturbing the three current
        R01/R02/R03 optimization trajectories.
        """
        from utils.epoch_test import (
            TB_METRICS,
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


            if self._is_main_proc():
                # TensorBoard also stays compact: only the same seven test metrics
                # that appear in epoch_summary.csv are recorded.
                core_metric_keys = {
                    "AAR_mean", "CAAR_mean", "RMSDCA_CDRH3_mean",
                    "RMSDCA_CDRH3_aligned_mean", "TMscore_mean",
                    "LDDT_mean", "DockQ_mean",
                }
                for metric_key, tb_name in TB_METRICS.items():
                    if (
                        metric_key in core_metric_keys
                        and metric_key in metrics
                        and self.writer is not None
                    ):
                        self.writer.add_scalar(
                            tb_name, float(metrics[metric_key]), int(self.epoch)
                        )
                if self.writer is not None:
                    self.writer.flush()

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
            # Epoch Test and metric collection can leave a rank-specific CUDA
            # caching-allocator high-water mark.  Releasing only unused cached
            # blocks here does not change tensors, gradients, RNG, or optimizer
            # state, but prevents nvidia-smi from reporting a stale rank0 peak.
            if torch.cuda.is_available():
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

        train_summary = self._reduce_train_epoch_summary(device)
        merged_buffer = self._gather_validation_writer_buffer()
        validation_summary = self._build_validation_epoch_summary(
            merged_buffer, valid_metric
        )
        self._last_validation_summary = validation_summary
        if self._is_main_proc():
            for name, values in merged_buffer.items():
                if not values:
                    continue
                value = float(np.mean(values))
                self.writer.add_scalar(name, value, self.epoch)
            if self.writer is not None:
                self.writer.flush()
            self._print_validation_audits(validation_summary)
        self.writer_buffer = {}

        # Third epoch phase: real EMA rollout Test.  It is observation-only.
        test_metrics = {}
        if self._epoch_test_should_run():
            try:
                test_metrics = self._run_epoch_test(device) or {}
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                self._write_epoch_test_error(message)
                if self._is_main_proc():
                    print(f"[EpochTest][ERROR] {message}")
                if self._epoch_test_fail_fast:
                    raise
        self._last_epoch_test_metrics = dict(test_metrics)
        self._finalize_epoch_summary(
            train_summary=train_summary,
            validation_summary=validation_summary,
            test_metrics=test_metrics,
            is_new_checkpoint_improvement=bool(should_save_best),
        )

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
    def _buffer_mean(buffer, key, default=float("nan")):
        values = buffer.get(key, []) if isinstance(buffer, dict) else []
        clean = []
        for value in values:
            try:
                value = float(value)
            except Exception:
                continue
            if isfinite(value):
                clean.append(value)
        return (sum(clean) / len(clean)) if clean else float(default)

    @staticmethod
    def _buffer_sum(buffer, key):
        values = buffer.get(key, []) if isinstance(buffer, dict) else []
        total = 0.0
        found = False
        for value in values:
            try:
                value = float(value)
            except Exception:
                continue
            if isfinite(value):
                total += value
                found = True
        return total if found else 0.0

    @staticmethod
    def _fmt(value, digits=4, suffix=""):
        if value is None:
            return "NA" + suffix
        try:
            value = float(value)
        except Exception:
            return "NA" + suffix
        if not isfinite(value):
            return "nan" + suffix
        return f"{value:.{int(digits)}f}{suffix}"

    @staticmethod
    def _fmte(value, digits=3):
        if value is None:
            return "NA"
        try:
            value = float(value)
        except Exception:
            return "NA"
        if not isfinite(value):
            return "nan"
        return f"{value:.{int(digits)}e}"

    def _weighted_timebin_metric(self, buffer, bin_idx, aligned=False):
        prefix = f"AbFlowDiag/val_proxy_timebin{int(bin_idx)}_"
        if aligned:
            count_key = prefix + "aligned_count/Validation"
            sum_key = prefix + "h3_ca_aligned_rmsd_sum/Validation"
        else:
            count_key = prefix + "count/Validation"
            sum_key = prefix + "h3_ca_rmsd_sum/Validation"
        count = self._buffer_sum(buffer, count_key)
        total = self._buffer_sum(buffer, sum_key)
        if count <= 0:
            return float("nan"), 0
        return total / count, int(round(count))

    def _build_validation_epoch_summary(self, merged_buffer, valid_metric):
        m = lambda key: self._buffer_mean(merged_buffer, key)
        summary = {
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "validation_metric": float(valid_metric),
            "loss_overall": m("Overall/Loss/Validation"),
            "loss_seq": m("Seq/SNLL/Validation"),
            "aar": m("Seq/AAR/Validation"),
            "loss_structure": m("Struct/StructLoss/Validation"),
            "loss_interface": m("Dock/SPLoss/Validation"),
            "loss_edge": m("Dock/EDLoss/Validation"),
            "mf_distogram_loss": m("DTM/mf_distogram_loss/Validation"),
            "mf_smooth_lddt_loss": m("DTM/mf_smooth_lddt_loss/Validation"),
            "mf_smooth_lddt_intra_loss": m("AbFlowDiag/mf_smooth_lddt_intra_loss/Validation"),
            "mf_smooth_lddt_scaffold_loss": m("AbFlowDiag/mf_smooth_lddt_scaffold_loss/Validation"),
            "mf_smooth_lddt_antigen_loss": m("AbFlowDiag/mf_smooth_lddt_antigen_loss/Validation"),
            "mf_smooth_lddt_intra_pairs": m("AbFlowDiag/mf_smooth_lddt_intra_pairs/Validation"),
            "mf_smooth_lddt_scaffold_pairs": m("AbFlowDiag/mf_smooth_lddt_scaffold_pairs/Validation"),
            "mf_smooth_lddt_antigen_pairs": m("AbFlowDiag/mf_smooth_lddt_antigen_pairs/Validation"),
            "mf_smooth_lddt_perfect_floor": m("AbFlowDiag/mf_smooth_lddt_perfect_floor/Validation"),
            "mf_smooth_lddt_excess": m("AbFlowDiag/mf_smooth_lddt_excess/Validation"),
            "h3ca_raw_r0": m("AbFlowDiag/val_proxy_round0_h3_ca_rmsd/Validation"),
            "h3ca_raw_r1": m("AbFlowDiag/val_proxy_round1_h3_ca_rmsd/Validation"),
            "h3ca_raw_r2": m("AbFlowDiag/val_proxy_round2_h3_ca_rmsd/Validation"),
            "h3ca_aligned_r0": m("AbFlowDiag/val_proxy_round0_h3_ca_aligned_rmsd/Validation"),
            "h3ca_aligned_r1": m("AbFlowDiag/val_proxy_round1_h3_ca_aligned_rmsd/Validation"),
            "h3ca_aligned_r2": m("AbFlowDiag/val_proxy_round2_h3_ca_aligned_rmsd/Validation"),
            "aar_r0": m("AbFlowDiag/val_proxy_round0_aar/Validation"),
            "aar_r1": m("AbFlowDiag/val_proxy_round1_aar/Validation"),
            "aar_r2": m("AbFlowDiag/val_proxy_round2_aar/Validation"),
            "refinement_raw_delta": m("AbFlowDiag/val_proxy_refinement_raw_rmsd_delta/Validation"),
            "refinement_aligned_delta": m("AbFlowDiag/val_proxy_refinement_aligned_rmsd_delta/Validation"),
            "contact_f1": m("AbFlowDiag/val_proxy_native_contact_f1/Validation"),
            "caar": m("AbFlowDiag/val_proxy_caar/Validation"),
            "seq_entropy": m("AbFlowDiag/val_proxy_seq_entropy/Validation"),
            "seq_max_prob": m("AbFlowDiag/val_proxy_seq_max_prob/Validation"),
            "seq_dominant_map_fraction": m("AbFlowDiag/val_proxy_seq_dominant_map_fraction/Validation"),
            "seq_unique_map_classes": m("AbFlowDiag/val_proxy_seq_unique_map_classes/Validation"),
            "proposal_aar": m("AbFlowDiag/seq_pep_vs_native_aar/Validation"),
            "pred_vs_proposal_aar": m("AbFlowDiag/seq_pred_vs_pep_aar/Validation"),
            "mf_single_rms": m("AbFlowDiag/mf_single_rms/Validation"),
            "mf_pair_rms": m("AbFlowDiag/mf_pair_rms/Validation"),
            "mf_pair_count": m("AbFlowDiag/mf_pair_count/Validation"),
            "mf_pair_count_graph_mean": m("AbFlowDiag/mf_pair_count_graph_mean/Validation"),
            "mf_pair_count_graph_max": m("AbFlowDiag/mf_pair_count_graph_max/Validation"),
            "mf_local_token_count_graph_mean": m("AbFlowDiag/mf_local_token_count_graph_mean/Validation"),
            "mf_local_token_count_graph_max": m("AbFlowDiag/mf_local_token_count_graph_max/Validation"),
            "mf_allatom_pair_rbf_rms": m("AbFlowDiag/mf_allatom_pair_rbf_rms/Validation"),
            "mf_opm_update_rms": m("AbFlowDiag/mf_opm_update_rms/Validation"),
            "mf_triangle_update_rms": m("AbFlowDiag/mf_triangle_update_rms/Validation"),
            "mf_pair_atom_update_rms": m("AbFlowDiag/mf_pair_atom_update_rms/Validation"),
            "mf_pair_atom_residual_rms": m("AbFlowDiag/mf_pair_atom_residual_rms/Validation"),
            "mf_pair_atom_adapter_weight_rms": m("AbFlowDiag/mf_pair_atom_adapter_weight_rms/Validation"),
            "mf_base_residual_rms": m("AbFlowDiag/mf_base_residual_rms/Validation"),
            "mf_seq_residual_rms": m("AbFlowDiag/mf_seq_residual_rms/Validation"),
            "mf_single_round_delta_rms": m("AbFlowDiag/mf_single_round_delta_rms/Validation"),
            "mf_pair_round_delta_rms": m("AbFlowDiag/mf_pair_round_delta_rms/Validation"),
            "mf_base_adapter_weight_rms": m("AbFlowDiag/mf_base_adapter_weight_rms/Validation"),
            "mf_seq_adapter_weight_rms": m("AbFlowDiag/mf_seq_adapter_weight_rms/Validation"),
            "t_mean": m("AbFlowDiag/t_mean/Validation"),
            "t_min": m("AbFlowDiag/t_min/Validation"),
            "t_max": m("AbFlowDiag/t_max/Validation"),
        }
        for ridx in range(3):
            for name in (
                "mf_single_rms", "mf_pair_rms", "mf_base_residual_rms",
                "mf_seq_residual_rms", "mf_single_round_delta_rms",
                "mf_pair_round_delta_rms",
            ):
                summary[f"round{ridx}_{name}"] = m(
                    f"AbFlowDiag/round{ridx}_{name}/Validation"
                )
        for bidx in range(5):
            raw, count = self._weighted_timebin_metric(
                merged_buffer, bidx, aligned=False
            )
            aligned, aligned_count = self._weighted_timebin_metric(
                merged_buffer, bidx, aligned=True
            )
            summary[f"timebin{bidx}_h3ca_raw"] = raw
            summary[f"timebin{bidx}_h3ca_aligned"] = aligned
            summary[f"timebin{bidx}_count"] = count
            summary[f"timebin{bidx}_aligned_count"] = aligned_count
        return summary

    def _print_validation_audits(self, summary):
        if not self._is_main_proc():
            return
        print(
            "[ValidationPhysical] "
            f"epoch={self.epoch} val={self._fmt(summary.get('validation_metric'), 5)} "
            f"H3CAraw={self._fmt(summary.get('h3ca_raw_r2'), 5, 'A')} "
            f"H3CAaligned={self._fmt(summary.get('h3ca_aligned_r2'), 5, 'A')} "
            f"contactF1={self._fmt(summary.get('contact_f1'), 5)} "
            f"CAAR={self._fmt(summary.get('caar'), 5)}"
        )
        print(
            "[R05RoundAudit] "
            f"epoch={self.epoch} "
            f"r0_raw={self._fmt(summary.get('h3ca_raw_r0'), 4, 'A')} "
            f"r1_raw={self._fmt(summary.get('h3ca_raw_r1'), 4, 'A')} "
            f"r2_raw={self._fmt(summary.get('h3ca_raw_r2'), 4, 'A')} "
            f"r0_aligned={self._fmt(summary.get('h3ca_aligned_r0'), 4, 'A')} "
            f"r1_aligned={self._fmt(summary.get('h3ca_aligned_r1'), 4, 'A')} "
            f"r2_aligned={self._fmt(summary.get('h3ca_aligned_r2'), 4, 'A')} "
            f"r0_AAR={self._fmt(summary.get('aar_r0'), 4)} "
            f"r1_AAR={self._fmt(summary.get('aar_r1'), 4)} "
            f"r2_AAR={self._fmt(summary.get('aar_r2'), 4)} "
            f"raw_delta={self._fmt(summary.get('refinement_raw_delta'), 4, 'A')} "
            f"aligned_delta={self._fmt(summary.get('refinement_aligned_delta'), 4, 'A')}"
        )
        print(
            "[SequenceForensic] "
            f"epoch={self.epoch} val_AAR={self._fmt(summary.get('aar'), 4)} "
            f"val_CE={self._fmt(summary.get('loss_seq'), 4)} "
            f"entropy={self._fmt(summary.get('seq_entropy'), 4)} "
            f"max_prob={self._fmt(summary.get('seq_max_prob'), 4)} "
            f"dominant_MAP={self._fmt(summary.get('seq_dominant_map_fraction'), 4)} "
            f"unique_MAP={self._fmt(summary.get('seq_unique_map_classes'), 1)} "
            f"proposal_AAR={self._fmt(summary.get('proposal_aar'), 4)} "
            f"pred_vs_proposal={self._fmt(summary.get('pred_vs_proposal_aar'), 4)}"
        )
        print(
            "[MFRepresentationAudit] "
            f"epoch={self.epoch} "
            f"single_rms={self._fmt(summary.get('mf_single_rms'), 5)} "
            f"pair_rms={self._fmt(summary.get('mf_pair_rms'), 5)} "
            f"pairs={self._fmt(summary.get('mf_pair_count'), 1)} "
            f"pairs_graph_mean={self._fmt(summary.get('mf_pair_count_graph_mean'), 1)} "
            f"pairs_graph_max={self._fmt(summary.get('mf_pair_count_graph_max'), 1)} "
            f"tokens_graph_mean={self._fmt(summary.get('mf_local_token_count_graph_mean'), 2)} "
            f"tokens_graph_max={self._fmt(summary.get('mf_local_token_count_graph_max'), 2)} "
            f"atom_pair={self._fmt(summary.get('mf_allatom_pair_rbf_rms'), 5)} "
            f"opm={self._fmt(summary.get('mf_opm_update_rms'), 6)} "
            f"triangle={self._fmt(summary.get('mf_triangle_update_rms'), 6)} "
            f"pair_atom={self._fmt(summary.get('mf_pair_atom_update_rms'), 6)} "
            f"pair_atom_res={self._fmt(summary.get('mf_pair_atom_residual_rms'), 6)} "
            f"pair_atom_w={self._fmt(summary.get('mf_pair_atom_adapter_weight_rms'), 6)} "
            f"base_res={self._fmt(summary.get('mf_base_residual_rms'), 6)} "
            f"seq_res={self._fmt(summary.get('mf_seq_residual_rms'), 6)} "
            f"single_delta={self._fmt(summary.get('mf_single_round_delta_rms'), 6)} "
            f"pair_delta={self._fmt(summary.get('mf_pair_round_delta_rms'), 6)} "
            f"base_w={self._fmt(summary.get('mf_base_adapter_weight_rms'), 6)} "
            f"seq_w={self._fmt(summary.get('mf_seq_adapter_weight_rms'), 6)} "
            f"disto={self._fmt(summary.get('mf_distogram_loss'), 5)} "
            f"slddt={self._fmt(summary.get('mf_smooth_lddt_loss'), 5)} "
            f"slddt_intra={self._fmt(summary.get('mf_smooth_lddt_intra_loss'), 5)} "
            f"slddt_scaf={self._fmt(summary.get('mf_smooth_lddt_scaffold_loss'), 5)} "
            f"slddt_ag={self._fmt(summary.get('mf_smooth_lddt_antigen_loss'), 5)} "
            f"slddt_pairs=({self._fmt(summary.get('mf_smooth_lddt_intra_pairs'), 0)},"
            f"{self._fmt(summary.get('mf_smooth_lddt_scaffold_pairs'), 0)},"
            f"{self._fmt(summary.get('mf_smooth_lddt_antigen_pairs'), 0)}) "
            f"slddt_floor={self._fmt(summary.get('mf_smooth_lddt_perfect_floor'), 5)} "
            f"slddt_excess={self._fmt(summary.get('mf_smooth_lddt_excess'), 5)}"
        )
        bins = []
        for bidx in range(5):
            bins.append(
                f"b{bidx}={self._fmt(summary.get(f'timebin{bidx}_h3ca_raw'), 4, 'A')}"
                f"(n={int(summary.get(f'timebin{bidx}_count', 0) or 0)})"
            )
        print(
            f"[ValidationPhysicalBins] epoch={self.epoch} " + " ".join(bins)
        )


    def _print_loss_authority(self, grad_diagnostics):
        if not self._is_main_proc() or not grad_diagnostics:
            return
        def g(name):
            return self._scalar(grad_diagnostics.get(name))
        print(
            "[LossAuthority] "
            f"epoch={self.epoch} step={self.global_step} "
            f"|gT|={self._fmte(g('grad_probe_norm_endpoint'), 3)} "
            f"|gS|={self._fmte(g('grad_probe_norm_seq'), 3)} "
            f"|gStruct|={self._fmte(g('grad_probe_norm_structure'), 3)} "
            f"|gEdge|={self._fmte(g('grad_probe_norm_edge'), 3)} "
            f"|gD|={self._fmte(g('grad_probe_norm_distogram'), 3)} "
            f"|gL|={self._fmte(g('grad_probe_norm_smooth_lddt'), 3)} "
            f"cos(T,S)={self._fmt(g('grad_probe_cos_endpoint_seq'), 3)} "
            f"cos(T,D)={self._fmt(g('grad_probe_cos_endpoint_distogram'), 3)} "
            f"cos(S,D)={self._fmt(g('grad_probe_cos_seq_distogram'), 3)} "
            f"cos(Struct,D)={self._fmt(g('grad_probe_cos_structure_distogram'), 3)} "
            f"cos(T,L)={self._fmt(g('grad_probe_cos_endpoint_smooth_lddt'), 3)} "
            f"cos(Struct,L)={self._fmt(g('grad_probe_cos_structure_smooth_lddt'), 3)}"
        )


    def _accumulate_train_component(self, name, value):
        scalar = self._scalar(value)
        if scalar is None or not isfinite(scalar):
            return
        self._epoch_train_sums[name] += float(scalar)
        self._epoch_train_counts[name] += 1

    def _reduce_train_epoch_summary(self, device):
        """Return global DDP means of the formal top-level train losses."""
        summary = {}
        for name in self._train_component_names:
            pair = torch.tensor(
                [
                    float(self._epoch_train_sums.get(name, 0.0)),
                    float(self._epoch_train_counts.get(name, 0)),
                ],
                dtype=torch.float64,
                device=device,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(pair, op=dist.ReduceOp.SUM)
            total = float(pair[0].item())
            count = float(pair[1].item())
            summary[name] = total / count if count > 0 else float("nan")
        return summary

    @staticmethod
    def _test_core(test_metrics):
        metrics = test_metrics or {}
        return {
            "AAR": metrics.get("AAR_mean", float("nan")),
            "CAAR": metrics.get("CAAR_mean", float("nan")),
            "H3raw": metrics.get("RMSDCA_CDRH3_mean", float("nan")),
            "H3aligned": metrics.get("RMSDCA_CDRH3_aligned_mean", float("nan")),
            "TM": metrics.get("TMscore_mean", float("nan")),
            "lDDT": metrics.get("LDDT_mean", float("nan")),
            "DockQ": metrics.get("DockQ_mean", float("nan")),
        }

    @classmethod
    def _epoch_summary_fields(cls):
        fields = [
            "epoch",
            "train_loss", "train_seq", "train_structure", "train_interface",
            "train_edge", "train_distogram", "train_smooth_lddt",
            "val_loss", "val_seq", "val_structure", "val_interface",
            "val_edge", "val_distogram", "val_smooth_lddt",
            "test_AAR", "test_CAAR", "test_H3raw", "test_H3aligned",
            "test_TM", "test_lDDT", "test_DockQ",
            "best_val_epoch", "best_val_loss",
            "best_test_AAR", "best_test_CAAR", "best_test_H3raw",
            "best_test_H3aligned", "best_test_TM", "best_test_lDDT",
            "best_test_DockQ",
        ]
        return fields

    def _restore_best_val_summary_if_present(self):
        if not os.path.isfile(self._epoch_summary_path):
            return
        try:
            with open(self._epoch_summary_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return
            last = rows[-1]
            val = float(last.get("best_val_loss", "nan"))
            epoch = int(float(last.get("best_val_epoch", "nan")))
            if isfinite(val):
                self._best_val_metric = val
                self._best_val_epoch = epoch
                self._best_val_test = {
                    key: float(last.get(f"best_test_{key}", "nan"))
                    for key in ("AAR", "CAAR", "H3raw", "H3aligned", "TM", "lDDT", "DockQ")
                }
        except Exception as exc:
            if self._diag_main_rank:
                print(f"[EpochSummary][WARN] could not restore previous summary: {exc}")

    def _validation_is_global_best(self, value):
        try:
            value = float(value)
        except Exception:
            return False
        if not isfinite(value):
            return False
        if self._best_val_metric is None:
            return True
        if bool(getattr(self.config, "metric_min_better", True)):
            return value < float(self._best_val_metric)
        return value > float(self._best_val_metric)

    def _finalize_epoch_summary(
        self, train_summary, validation_summary, test_metrics,
        is_new_checkpoint_improvement=False,
    ):
        # ``is_new_checkpoint_improvement`` preserves the inherited checkpoint
        # semantics but does not define the all-time best row.  The latter is
        # tracked independently because the historical _metric_better compares
        # only against the immediately previous validation value.
        del is_new_checkpoint_improvement
        current_val = float(validation_summary.get("validation_metric", float("nan")))
        current_test = self._test_core(test_metrics)
        if self._validation_is_global_best(current_val):
            self._best_val_metric = current_val
            self._best_val_epoch = int(self.epoch)
            self._best_val_test = dict(current_test)

        row = {
            "epoch": int(self.epoch),
            "train_loss": train_summary.get("loss", float("nan")),
            "train_seq": train_summary.get("seq", float("nan")),
            "train_structure": train_summary.get("structure", float("nan")),
            "train_interface": train_summary.get("interface", float("nan")),
            "train_edge": train_summary.get("edge", float("nan")),
            "train_distogram": train_summary.get("distogram", float("nan")),
            "train_smooth_lddt": train_summary.get("smooth_lddt", float("nan")),
            "val_loss": validation_summary.get("validation_metric", float("nan")),
            "val_seq": validation_summary.get("loss_seq", float("nan")),
            "val_structure": validation_summary.get("loss_structure", float("nan")),
            "val_interface": validation_summary.get("loss_interface", float("nan")),
            "val_edge": validation_summary.get("loss_edge", float("nan")),
            "val_distogram": validation_summary.get("mf_distogram_loss", float("nan")),
            "val_smooth_lddt": validation_summary.get("mf_smooth_lddt_loss", float("nan")),
        }
        for key, value in current_test.items():
            row[f"test_{key}"] = value
        row["best_val_epoch"] = (
            float("nan") if self._best_val_epoch is None else int(self._best_val_epoch)
        )
        row["best_val_loss"] = (
            float("nan") if self._best_val_metric is None else float(self._best_val_metric)
        )
        for key in ("AAR", "CAAR", "H3raw", "H3aligned", "TM", "lDDT", "DockQ"):
            row[f"best_test_{key}"] = self._best_val_test.get(key, float("nan"))

        if self._is_main_proc():
            os.makedirs(os.path.dirname(self._epoch_summary_path), exist_ok=True)
            exists = os.path.isfile(self._epoch_summary_path)
            with open(self._epoch_summary_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._epoch_summary_fields())
                if not exists or os.path.getsize(self._epoch_summary_path) == 0:
                    writer.writeheader()
                writer.writerow(row)

            print(
                "[EpochSummary] "
                f"epoch={row['epoch']} "
                f"train={self._fmt(row['train_loss'], 5)} "
                f"seq={self._fmt(row['train_seq'], 5)} "
                f"struct={self._fmt(row['train_structure'], 5)} "
                f"interface={self._fmt(row['train_interface'], 5)} "
                f"edge={self._fmt(row['train_edge'], 5)} "
                f"disto={self._fmt(row['train_distogram'], 5)} "
                f"slddt={self._fmt(row['train_smooth_lddt'], 5)} | "
                f"val={self._fmt(row['val_loss'], 5)} "
                f"vseq={self._fmt(row['val_seq'], 5)} "
                f"vstruct={self._fmt(row['val_structure'], 5)} "
                f"vinterface={self._fmt(row['val_interface'], 5)} "
                f"vedge={self._fmt(row['val_edge'], 5)} "
                f"vdisto={self._fmt(row['val_distogram'], 5)} "
                f"vslddt={self._fmt(row['val_smooth_lddt'], 5)} | "
                f"AAR={self._fmt(row['test_AAR'], 5)} "
                f"CAAR={self._fmt(row['test_CAAR'], 5)} "
                f"H3raw={self._fmt(row['test_H3raw'], 5, 'A')} "
                f"H3aligned={self._fmt(row['test_H3aligned'], 5, 'A')} "
                f"TM={self._fmt(row['test_TM'], 5)} "
                f"lDDT={self._fmt(row['test_lDDT'], 5)} "
                f"DockQ={self._fmt(row['test_DockQ'], 5)} | "
                f"best_val_epoch={row['best_val_epoch']} "
                f"best_val={self._fmt(row['best_val_loss'], 5)} "
                f"best_AAR={self._fmt(row['best_test_AAR'], 5)} "
                f"best_CAAR={self._fmt(row['best_test_CAAR'], 5)} "
                f"best_H3raw={self._fmt(row['best_test_H3raw'], 5, 'A')} "
                f"best_H3aligned={self._fmt(row['best_test_H3aligned'], 5, 'A')} "
                f"best_TM={self._fmt(row['best_test_TM'], 5)} "
                f"best_lDDT={self._fmt(row['best_test_lDDT'], 5)} "
                f"best_DockQ={self._fmt(row['best_test_DockQ'], 5)}"
            )

    def _should_probe_grad(self, val):
        # interval=0 means once per actual train epoch, not on batch 0.
        step = int(self.global_step)
        return (
            (not val)
            and self._grad_diag_enabled
            and step > 0
            and step % self._diag_grad_interval == 0
        )

    def share_step(self, batch, batch_idx, val=False):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        # Validation always captures the model-side diagnostics needed for the
        # epoch mechanism audit.  Training captures them only for the periodic
        # gradient-authority probe.  No per-step diagnostic files are written.
        capture_diagnostics = bool(val) or self._should_probe_grad(val)
        raw_model._diagnostic_capture = bool(capture_diagnostics)
        raw_model._diagnostic_validation_mode = bool(val and capture_diagnostics)

        loss, seq_detail, structure_detail, dock_detail, pdev_detail = self.model(**batch)
        snll, aar = seq_detail
        struct_loss, xloss, bond_loss, sc_bond_loss = structure_detail
        dock_loss, interface_loss, ed_loss, r_ed_losses = dock_detail
        pdev_loss, prmsd_loss = pdev_detail

        if not val and self._train_loss_outlier_logged_epoch != int(self.epoch):
            struct_scalar = self._scalar(struct_loss)
            if (
                struct_scalar is not None
                and isfinite(struct_scalar)
                and abs(float(struct_scalar)) >= float(self._train_loss_outlier_threshold)
            ):
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                loss_scalar = self._scalar(loss)
                print(
                    "[TrainLossOutlier] "
                    f"epoch={self.epoch} step={self.global_step} rank={rank} "
                    f"loss={self._fmte(loss_scalar, 6)} "
                    f"struct={self._fmte(struct_scalar, 6)} "
                    f"threshold={self._fmte(self._train_loss_outlier_threshold, 3)}"
                )
                self._train_loss_outlier_logged_epoch = int(self.epoch)

        if self._should_probe_grad(val) and hasattr(raw_model, "compute_gradient_conflict_diagnostics"):
            raw_model.compute_gradient_conflict_diagnostics()
            grad_error = str(getattr(raw_model, "_last_gradient_diagnostic_error", "") or "")
            if grad_error and self._diag_main_rank:
                print(
                    f"[LossAuthority][ERROR] epoch={self.epoch} "
                    f"step={self.global_step} {grad_error}"
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
        if self._should_probe_grad(val):
            self._print_loss_authority(grad_diagnostics)
            # autograd.grad creates temporary per-objective gradient tensors.
            # They are observational only; release their cached CUDA blocks once
            # the probe has been materialized/logged so rank-specific diagnostics
            # do not become a persistent memory asymmetry.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        lr = None
        if not val:
            lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()[0]
            self.log('lr', lr, batch_idx, val)
            self.log('context_ratio', batch['context_ratio'], batch_idx, val)


        if not val:
            if self._epoch_train_acc_epoch != int(self.epoch):
                self._epoch_train_acc_epoch = int(self.epoch)
                self._epoch_train_sums = {
                    name: 0.0 for name in self._train_component_names
                }
                self._epoch_train_counts = {
                    name: 0 for name in self._train_component_names
                }
            self._accumulate_train_component("loss", loss)
            self._accumulate_train_component("seq", snll)
            self._accumulate_train_component("structure", struct_loss)
            self._accumulate_train_component("interface", interface_loss)
            self._accumulate_train_component("edge", ed_loss)
            self._accumulate_train_component(
                "distogram", scorefm_losses.get("mf_distogram_loss")
            )
            self._accumulate_train_component(
                "smooth_lddt", scorefm_losses.get("mf_smooth_lddt_loss")
            )
        return loss
