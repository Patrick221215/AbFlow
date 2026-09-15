#!/usr/bin/python
# -*- coding:utf-8 -*-
# V211_EVIDENCE_DRIVEN_R28_PARENT: R28 is R05+Pair; R29 is R28+Distogram;
# R30 is R28+donor smooth-lDDT. Run settings, module switches and every loss
# coefficient are read from the selected JSON. The V207.1 zero-init-aware
# Distogram gradient contract is retained.
# V203_FORMAL_TRAIN_VAL_TEST_EVERY_EPOCH
# Fixed project protocol: every epoch executes Train -> Val -> formal EMA Test generation.
# Infrastructure Test failures are fail-fast; invalid model outputs are recorded explicitly.
"""AbFlow trainer: stable v163 training/validation behavior with minimal V185 diagnostics.

V185 deliberately restores the previously stable trainer instead of using the
heavily trimmed V182-V184 trainer. Optimizer/scheduler/DDP-validation/EMA/checkpoint
semantics are preserved. Only observational logging is adapted to the new
AbX-native single/pair/time representation and auxiliary loss names.
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
        # epoch table stays compact and directly comparable across R28/R29/R30.
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
        # Diagnostic code must never gate or backpropagate through formal training.
        # Retain only the ordinary-forward zero-start/live representation bridge.
        self._live_bridge_contract_verified = False
        self._bridge_cold_start_observed = False
        # V235: failed AMP PairGradientAudit removed.  Diagnostics below use only
        # quantities produced by the ordinary scientific forward.
        self._singlefield_contract_verified = False
        # V185: diagnostics-only cadence. First few steps verify the new
        # representation/time/edge routing without changing optimization.
        self._science_log_first_steps = max(0, _env_int(
            "ABFLOW_SCI_LOG_FIRST_STEPS", 3
        ))
        self._science_log_interval = max(0, _env_int(
            "ABFLOW_SCI_LOG_INTERVAL", 0
        ))

        # Earlier R28/R30 summaries exposed rare ~1e5-1e6 structure-loss means
        # that were invisible in rank-0 tqdm.  Record the first true per-rank
        # outlier in run_time.log instead of guessing whether this is aggregation
        # error or a real hard sample on another DDP rank.  Observational only.
        self._train_loss_outlier_threshold = _env_float(
            "ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD", 1.0e4
        )
        self._train_loss_outlier_max_per_epoch = max(1, _env_int(
            "ABFLOW_TRAIN_LOSS_OUTLIER_MAX_PER_EPOCH", 3
        ))
        self._train_loss_outlier_epoch = -1
        self._train_loss_outlier_count = 0

        # V213 authority telemetry.  V212 correctly bounded only the relational
        # coordinate residual; R33 showed that the shared/base controller can
        # still enter a high-gain regime without crossing the 1e4 loss alert.
        # These thresholds are observational only and never alter optimization.
        self._geometry_authority_interval = max(0, _env_int(
            "ABFLOW_GEOMETRY_AUTHORITY_INTERVAL", 20
        ))
        self._geometry_authority_base_alert = _env_float(
            "ABFLOW_GEOMETRY_AUTHORITY_BASE_ALERT", 64.0
        )
        self._geometry_authority_update_alert = _env_float(
            "ABFLOW_GEOMETRY_AUTHORITY_UPDATE_ALERT", 1.0e4
        )
        self._geometry_authority_alert_max_per_epoch = max(1, _env_int(
            "ABFLOW_GEOMETRY_AUTHORITY_ALERT_MAX_PER_EPOCH", 3
        ))
        self._geometry_authority_alert_epoch = -1
        self._geometry_authority_alert_count = 0

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
            "ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS", True
        )
        self._epoch_test_keep_structures = _env_flag(
            "ABFLOW_EPOCH_TEST_KEEP_STRUCTURES", False
        )
        self._epoch_test_fail_fast = _env_flag(
            "ABFLOW_EPOCH_TEST_FAIL_FAST", False
        )
        self._epoch_test_model_invalid_policy = str(
            os.environ.get(
                "ABFLOW_EPOCH_TEST_MODEL_INVALID_POLICY",
                "record_and_continue",
            )
            or "record_and_continue"
        ).strip().lower()
        if self._epoch_test_model_invalid_policy not in {
            "record_and_continue", "fail_fast"
        }:
            raise ValueError(
                "ABFLOW_EPOCH_TEST_MODEL_INVALID_POLICY must be "
                "'record_and_continue' or 'fail_fast', got "
                f"{self._epoch_test_model_invalid_policy!r}."
            )
        self._epoch_test_project_root = str(os.environ.get(
            "ABFLOW_PROJECT_ROOT", os.getcwd()
        ) or os.getcwd()).strip()
        self._epoch_test_dataset = None
        self._epoch_test_root = os.path.join(self.config.save_dir, "epoch_test")

        # V203 fixed project protocol: Test is a mandatory third phase of every epoch.
        # The launcher also validates this contract before torchrun; keeping the
        # trainer-side assertion prevents a direct train.py invocation from silently
        # producing NaN test columns.
        if not self._epoch_test_enabled:
            raise RuntimeError(
                "V203 formal protocol requires ABFLOW_EPOCH_TEST=on: "
                "every epoch must execute Train -> Val -> formal Test generation."
            )
        if int(self._epoch_test_interval) != 1:
            raise RuntimeError(
                "V203 formal protocol requires ABFLOW_EPOCH_TEST_INTERVAL=1, "
                f"got {self._epoch_test_interval}."
            )
        if int(self._epoch_test_n_steps) != 10:
            raise RuntimeError(
                "V203 formal protocol requires 10-step generation, "
                f"got ABFLOW_EPOCH_TEST_N_STEPS={self._epoch_test_n_steps}."
            )
        if not self._epoch_test_fail_fast:
            raise RuntimeError(
                "V203 formal protocol requires ABFLOW_EPOCH_TEST_FAIL_FAST=on "
                "for infrastructure/protocol failures."
            )
        if not self._epoch_test_json:
            raise RuntimeError(
                "V203 formal protocol requires ABFLOW_EPOCH_TEST_JSON."
            )
        if self._diag_main_rank:
            raw_model = model.module if hasattr(model, "module") else model
            trunk = getattr(raw_model, "native_trunk", None)
            trunk_cfg = getattr(getattr(trunk, "trunk", None), "config", None)
            repr_cfg = getattr(trunk, "representation_config", {}) if trunk is not None else {}
            geom_cfg = repr_cfg.get("geometry", {}) if isinstance(repr_cfg, dict) else {}
            print(
                "[FormalModelContract] "
                f"rounds={getattr(raw_model, 'round', 'NA')} "
                f"single={getattr(trunk_cfg, 'seq_channel', 'NA')} "
                f"pair={getattr(trunk_cfg, 'pair_channel', 'NA')} "
                f"time=R05:1/AbX:{int(bool(getattr(trunk_cfg, 'time_embed', False)))} "
                "recycling=0 "
                f"frame={geom_cfg.get('frame', 'NA')} "
                "context=full_antibody+dataset_epitope "
                "bridge=zero_start_residual "
                f"distogram={getattr(raw_model, 'loss_distogram_weight', 0.0):.4g} "
                f"smooth_lddt={getattr(raw_model, 'loss_smooth_lddt_weight', 0.0):.4g} "
                f"smooth_lddt_source={getattr(raw_model, 'smooth_lddt_prediction_source', 'pred_design_endpoint')}"
            )
            print(
                "[FormalEvalContract] "
                "ckpt=validation test=observation_only "
                f"test_batch={self._epoch_test_batch_size} "
                f"test_steps={self._epoch_test_n_steps} "
                f"seed={self._epoch_test_base_seed} fail_fast=infra "
                f"model_invalid={self._epoch_test_model_invalid_policy}"
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

    @staticmethod
    def _is_model_output_invalid_test_error(exc):
        """Return True only for failures caused by the generated structure itself.

        Infrastructure, DDP, dataset and evaluator-code failures must still abort.
        The marker is emitted by the shared epoch-test coordinate guard.  The
        Bio.PDB phrase keeps backward compatibility with already-generated bad
        structures from the current R28/R29 runs.
        """
        text = str(exc)
        markers = (
            "[ModelOutputInvalid]",
            "Generated coordinates contain",
            "PDBConstructionException: Invalid or missing coordinate(s)",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _compact_test_error(exc, limit=800):
        text = " ".join(str(exc).split())
        if len(text) > int(limit):
            text = text[: int(limit) - 3] + "..."
        return f"{type(exc).__name__}: {text}"

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
                if (
                    bool(getattr(raw_model, 'sample_authority_diagnostics', False))
                    and hasattr(raw_model, 'reset_sample_authority_diagnostics')
                ):
                    raw_model.reset_sample_authority_diagnostics()
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
                self._validate_formal_epoch_test_metrics(metrics, device)
                if (
                    bool(getattr(raw_model, 'sample_authority_diagnostics', False))
                    and hasattr(raw_model, 'consume_sample_authority_diagnostics')
                ):
                    local_trace = raw_model.consume_sample_authority_diagnostics()
                    self._print_epoch_test_authority_trace(local_trace)


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

    def _validate_formal_epoch_test_metrics(self, metrics, device):
        """Fail-fast if the formal Test phase did not return the seven core metrics."""
        required = (
            "AAR_mean", "CAAR_mean", "RMSDCA_CDRH3_mean",
            "RMSDCA_CDRH3_aligned_mean", "TMscore_mean",
            "LDDT_mean", "DockQ_mean",
        )
        local_ok = True
        local_message = ""
        if self._is_main_proc():
            missing = [k for k in required if k not in metrics]
            nonfinite = []
            if not missing:
                for key in required:
                    try:
                        value = float(metrics[key])
                    except Exception:
                        nonfinite.append(key)
                        continue
                    if not isfinite(value):
                        nonfinite.append(key)
            if missing or nonfinite:
                local_ok = False
                local_message = (
                    f"formal Test metrics invalid: missing={missing} nonfinite={nonfinite}; "
                    f"available={sorted(metrics.keys()) if isinstance(metrics, dict) else type(metrics)}"
                )

        status = torch.tensor(1 if local_ok else 0, dtype=torch.int32, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(status, src=0)
        if int(status.item()) != 1:
            if self._is_main_proc() and local_message:
                print(f"[FormalEpochTestFAIL] epoch={self.epoch} {local_message}")
            raise RuntimeError(local_message or "formal Test metric validation failed on rank0")


    def _print_epoch_test_authority_trace(self, local_records):
        """Aggregate model.sample authority diagnostics across Test ranks.

        This is observation-only: records were collected from the exact forwards
        already performed by the formal Test sampler.  No extra model query, RNG
        call, sampler step, or metric computation is introduced.
        """
        gathered = [local_records]
        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local_records)
        if not self._is_main_proc():
            return
        rows = []
        for item in gathered:
            if item:
                rows.extend(item)
        if not rows:
            return
        fields = (
            'seq_raw_ratio', 'seq_applied_ratio', 'seq_clip', 'seq_map_change',
            'base_raw_ratio', 'base_applied_ratio', 'base_clip',
            'edge_raw_ratio', 'edge_applied_ratio', 'edge_clip',
            'coord_field_rms_A',
        )
        by_step = {}
        for row in rows:
            step = int(row.get('step', -1))
            if step < 0:
                continue
            by_step.setdefault(step, []).append(row)

        def mean_at(step, field):
            vals = []
            for row in by_step.get(step, []):
                try:
                    val = float(row.get(field, float('nan')))
                except Exception:
                    continue
                if isfinite(val):
                    vals.append(val)
            return sum(vals) / len(vals) if vals else float('nan')

        steps = sorted(by_step)
        def arr(field, nd=3):
            return '(' + ','.join(self._fmt(mean_at(st, field), nd) for st in steps) + ')'
        tvals = []
        for st in steps:
            vals = []
            for row in by_step[st]:
                try:
                    v = float(row.get('t', float('nan')))
                except Exception:
                    continue
                if isfinite(v):
                    vals.append(v)
            tvals.append(sum(vals) / len(vals) if vals else float('nan'))
        print(
            '[TestTrajectoryAudit] '
            f'epoch={self.epoch} '
            f't=(' + ','.join(self._fmt(v, 2) for v in tvals) + ') '
            f'seq_raw={arr("seq_raw_ratio")} '
            f'seq_applied={arr("seq_applied_ratio")} '
            f'seq_clip={arr("seq_clip")} '
            f'seq_map_change={arr("seq_map_change")} '
            f'base_raw={arr("base_raw_ratio")} '
            f'base_applied={arr("base_applied_ratio")} '
            f'base_clip={arr("base_clip")} '
            f'edge_raw={arr("edge_raw_ratio")} '
            f'edge_applied={arr("edge_applied_ratio")} '
            f'edge_clip={arr("edge_clip")} '
            f'coord_field_A={arr("coord_field_rms_A", 4)}'
        )

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
                        metric_value = float(metric.detach().cpu())
                        metric_arr.append(metric_value)
                        if self._is_main_proc() and hasattr(t_iter, "set_postfix"):
                            t_iter.set_postfix(
                                val_loss=f"{metric_value:.5f}",
                                version=self.version,
                            )

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

        # V203 fixed third phase: every epoch performs formal EMA rollout Test.
        # There is deliberately no silent skip path and no NaN fallback.
        if not self._epoch_test_should_run():
            raise RuntimeError(
                f"V203 formal Test was unexpectedly disabled/skipped at epoch={self.epoch}."
            )
        try:
            test_metrics = self._run_epoch_test(device) or {}
            test_metrics["_status"] = "ok"
            test_metrics["_error"] = ""
        except Exception as exc:
            message = self._compact_test_error(exc)
            self._write_epoch_test_error(message)
            is_model_invalid = self._is_model_output_invalid_test_error(exc)
            if (
                is_model_invalid
                and self._epoch_test_model_invalid_policy == "record_and_continue"
            ):
                test_metrics = {
                    "_status": "model_output_invalid",
                    "_error": message,
                }
                if self._is_main_proc():
                    print(
                        "[FormalEpochTestINVALID] "
                        f"epoch={self.epoch} status=model_output_invalid "
                        f"action=record_and_continue reason={message}"
                    )
            else:
                if self._is_main_proc():
                    print(f"[EpochTest][ERROR] {message}")
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

    def _formal_train_batch_indices(self, batch):
        """Reconstruct current formal sampler indices without touching global RNG.

        The R28/R29/R30 launcher uses CostBalancedDistributedSampler, whose
        ``__iter__`` is a pure function of seed+epoch via a private Generator.
        For unknown sampler types we deliberately return no guess.
        """
        sampler = getattr(self.train_loader, 'sampler', None)
        if sampler is None or sampler.__class__.__name__ != 'CostBalancedDistributedSampler':
            return []
        local_bs = getattr(self.train_loader, 'batch_size', None)
        if local_bs is None:
            return []
        try:
            ordered = list(iter(sampler))
            epoch_steps = max(1, len(self.train_loader))
            step_in_epoch = int(self.global_step) - int(self.epoch) * epoch_steps
            if not (0 <= step_in_epoch < epoch_steps):
                step_in_epoch = int(self.global_step) % epoch_steps
            n_graph = int(batch['lengths'].numel()) if torch.is_tensor(batch.get('lengths')) else int(local_bs)
            start = step_in_epoch * int(local_bs)
            return [int(v) for v in ordered[start:start + n_graph]]
        except Exception:
            return []

    def _formal_dataset_labels(self, logical_indices):
        dataset = getattr(self.train_loader, 'dataset', None)
        if dataset is None:
            return [str(v) for v in logical_indices]
        labels = []
        for logical_idx in logical_indices:
            label = f'idx:{int(logical_idx)}'
            try:
                raw_idx = (
                    int(dataset.idx_mapping[int(logical_idx)])
                    if hasattr(dataset, 'idx_mapping') else int(logical_idx)
                )
                obj = dataset.data[raw_idx] if hasattr(dataset, 'data') else None
                if obj is not None and hasattr(obj, 'get_id'):
                    label = str(obj.get_id()).split('(')[0]
                elif obj is not None and hasattr(obj, 'pdb_id'):
                    label = str(obj.pdb_id)
            except Exception:
                pass
            labels.append(label)
        return labels

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
            "distogram_loss": m("DTM/distogram_loss/Validation"),
            "smooth_lddt_loss": m("DTM/smooth_lddt_loss/Validation"),
            "smooth_lddt_DD": m("DTM/smooth_lddt_DD/Validation"),
            "smooth_lddt_DF": m("DTM/smooth_lddt_DF/Validation"),
            "smooth_lddt_DA": m("DTM/smooth_lddt_DA/Validation"),
            "relational_single_rms": m("DTM/relational_single_rms/Validation"),
            "relational_pair_rms": m("DTM/relational_pair_rms/Validation"),
            "relational_antigen_keep_fraction": m("DTM/relational_antigen_keep_fraction/Validation"),
            "distogram_da_ce": m("DTM/distogram_da_ce/Validation"),
            "distogram_da_contact_precision": m("DTM/distogram_da_contact_precision/Validation"),
            "distogram_head_weight_rms": m("DTM/distogram_head_weight_rms/Validation"),
            "distogram_task_pair_fraction": m("DTM/distogram_task_pair_fraction/Validation"),
            "distogram_DD_ce": m("DTM/distogram_DD_ce/Validation"),
            "distogram_DF_ce": m("DTM/distogram_DF_ce/Validation"),
            "distogram_DA_ce": m("DTM/distogram_DA_ce/Validation"),
            "distogram_context_context_optimized_pairs": m(
                "DTM/distogram_context_context_optimized_pairs/Validation"
            ),
            "r05_parent_static_bio_rms": m("AbFlowDiag/r05_parent_static_bio_rms/Validation"),
            "r05_parent_dynamic_bio_rms": m("AbFlowDiag/r05_parent_dynamic_bio_rms/Validation"),
            "abx_single_bio_rms": m("AbFlowDiag/abx_single_bio_rms/Validation"),
            "r05_time_embed_on": m("AbFlowDiag/r05_time_embed_on/Validation"),
            "abx_time_embed_on": m("AbFlowDiag/abx_time_embed_on/Validation"),
            "explicit_time_authority_count": m(
                "AbFlowDiag/explicit_time_authority_count/Validation"
            ),
            "abx_ctx_edge_attr_rms": m("AbFlowDiag/abx_ctx_edge_attr_rms/Validation"),
            "abx_inter_edge_attr_rms": m("AbFlowDiag/abx_inter_edge_attr_rms/Validation"),
            "abx_surf_edge_attr_rms": m("AbFlowDiag/abx_surf_edge_attr_rms/Validation"),
            "abx_ctx_edge_count": m("AbFlowDiag/abx_ctx_edge_count/Validation"),
            "abx_inter_edge_count": m("AbFlowDiag/abx_inter_edge_count/Validation"),
            "abx_surf_edge_count": m("AbFlowDiag/abx_surf_edge_count/Validation"),
            "bridge_single_delta_to_base_ratio": m("AbFlowDiag/bridge_single_delta_to_base_ratio/Validation"),
            "bridge_pair_delta_to_base_ratio_mean": m("AbFlowDiag/bridge_pair_delta_to_base_ratio_mean/Validation"),
            "bridge_pair_delta_to_base_ratio_max": m("AbFlowDiag/bridge_pair_delta_to_base_ratio_max/Validation"),
            "sf_chart_gap_A": m("AbFlowSF/final_chart_gap_A/Validation"),
            "sf_canonical_active_rate": m("AbFlowSF/canonical_active_rate/Validation"),
            "sf_latent_gap_r0_A": m("AbFlowSF/latent_native_gap_r0_A/Validation"),
            "sf_latent_gap_r1_A": m("AbFlowSF/latent_native_gap_r1_A/Validation"),
            "sf_latent_gap_r2_A": m("AbFlowSF/latent_native_gap_r2_A/Validation"),
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
            "proposal_change_rate": m("AbFlowDiag/seq_change_from_proposal_rate/Validation"),
            "proposal_preservation_rate": m("AbFlowDiag/seq_proposal_correct_preservation_rate/Validation"),
            "proposal_damage_rate": m("AbFlowDiag/seq_proposal_correct_damage_rate/Validation"),
            "proposal_correction_rate": m("AbFlowDiag/seq_proposal_wrong_correction_rate/Validation"),
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
            "mf_base_residual_applied_rms": m("AbFlowDiag/mf_base_residual_applied_rms/Validation"),
            "mf_base_raw_to_parent_ratio": m("AbFlowDiag/mf_base_raw_to_parent_ratio/Validation"),
            "mf_base_applied_to_parent_ratio": m("AbFlowDiag/mf_base_applied_to_parent_ratio/Validation"),
            "mf_base_authority_clip_fraction": m("AbFlowDiag/mf_base_authority_clip_fraction/Validation"),
            "mf_base_authority_mean_scale": m("AbFlowDiag/mf_base_authority_mean_scale/Validation"),
            "mf_edge_raw_ratio": m("AbFlowDiag/mf_edge_raw_ratio/Validation"),
            "mf_edge_applied_ratio": m("AbFlowDiag/mf_edge_applied_ratio/Validation"),
            "mf_edge_clip_fraction": m("AbFlowDiag/mf_edge_clip_fraction/Validation"),
            "mf_edge_mean_scale": m("AbFlowDiag/mf_edge_mean_scale/Validation"),
            "mf_seq_residual_rms": m("AbFlowDiag/mf_seq_residual_rms/Validation"),
            "mf_seq_parent_logit_rms": m("AbFlowDiag/mf_seq_parent_logit_rms/Validation"),
            "mf_seq_final_logit_rms": m("AbFlowDiag/mf_seq_final_logit_rms/Validation"),
            "mf_seq_raw_residual_to_parent_ratio": m("AbFlowDiag/mf_seq_raw_residual_to_parent_ratio/Validation"),
            "mf_seq_residual_to_parent_ratio": m("AbFlowDiag/mf_seq_residual_to_parent_ratio/Validation"),
            "mf_seq_parent_residual_cos": m("AbFlowDiag/mf_seq_parent_residual_cos/Validation"),
            "mf_seq_parent_to_final_map_change_rate": m("AbFlowDiag/mf_seq_parent_to_final_map_change_rate/Validation"),
            "mf_seq_authority_clip_fraction": m("AbFlowDiag/mf_seq_authority_clip_fraction/Validation"),
            "mf_seq_authority_mean_scale": m("AbFlowDiag/mf_seq_authority_mean_scale/Validation"),
            "mf_single_round_delta_rms": m("AbFlowDiag/mf_single_round_delta_rms/Validation"),
            "mf_pair_round_delta_rms": m("AbFlowDiag/mf_pair_round_delta_rms/Validation"),
            "mf_base_adapter_weight_rms": m("AbFlowDiag/mf_base_adapter_weight_rms/Validation"),
            "mf_seq_adapter_weight_rms": m("AbFlowDiag/mf_seq_adapter_weight_rms/Validation"),
            "sequence_path_contract_exact": m("AbFlowDiag/sequence_path_contract_exact/Validation"),
            "sequence_loss_scope_path_noisy": m("AbFlowDiag/sequence_loss_scope_path_noisy/Validation"),
            "sequence_path_mask_rate": m("AbFlowDiag/sequence_path_mask_rate/Validation"),
            "sequence_loss_mask_rate": m("AbFlowDiag/sequence_loss_mask_rate/Validation"),
            "sequence_path_participation_rate": m("AbFlowDiag/sequence_path_participation_rate/Validation"),
            "sequence_loss_coverage_design": m("AbFlowDiag/sequence_loss_coverage_design/Validation"),
            "sequence_external_native_context_rate": m("AbFlowDiag/sequence_external_native_context_rate/Validation"),
            "sequence_noisy_branch_rate": m("AbFlowDiag/sequence_noisy_branch_rate/Validation"),
            "sequence_clean_branch_rate": m("AbFlowDiag/sequence_clean_branch_rate/Validation"),
            "sequence_loss_on_noisy_precision": m("AbFlowDiag/sequence_loss_on_noisy_precision/Validation"),
            "sequence_state_native_fraction": m("AbFlowDiag/sequence_state_native_fraction/Validation"),
            "t_mean": m("AbFlowDiag/t_mean/Validation"),
            "t_min": m("AbFlowDiag/t_min/Validation"),
            "t_max": m("AbFlowDiag/t_max/Validation"),
        }
        for ridx in range(3):
            for name in (
                "mf_single_rms", "mf_pair_rms", "mf_base_residual_rms",
                "mf_base_residual_applied_rms",
                "mf_base_raw_to_parent_ratio", "mf_base_applied_to_parent_ratio",
                "mf_base_authority_clip_fraction", "mf_base_authority_mean_scale",
                "mf_edge_raw_ratio", "mf_edge_applied_ratio",
                "mf_edge_clip_fraction", "mf_edge_mean_scale",
                "mf_seq_residual_rms", "mf_seq_parent_logit_rms",
                "mf_seq_final_logit_rms", "mf_seq_raw_residual_to_parent_ratio",
                "mf_seq_residual_to_parent_ratio",
                "mf_seq_parent_residual_cos",
                "mf_seq_parent_to_final_map_change_rate",
                "mf_seq_authority_clip_fraction", "mf_seq_authority_mean_scale",
                "mf_single_round_delta_rms", "mf_pair_round_delta_rms",
            ):
                summary[f"round{ridx}_{name}"] = m(
                    f"AbFlowDiag/round{ridx}_{name}/Validation"
                )
            for name in (
                "seq_entropy", "seq_max_prob", "seq_native_prob",
                "seq_top1_margin",
            ):
                summary[f"round{ridx}_{name}"] = m(
                    f"AbFlowDiag/val_proxy_round{ridx}_{name}/Validation"
                )
        for a, b, key in ((0, 1, "aar_delta01"), (1, 2, "aar_delta12")):
            va, vb = summary.get(f"aar_r{a}"), summary.get(f"aar_r{b}")
            summary[key] = (
                float(vb) - float(va)
                if va is not None and vb is not None
                and isfinite(float(va)) and isfinite(float(vb))
                else float("nan")
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
        # New observers use existing exact DDP validation buffers. CE/AAR bins
        # use token-count-weighted sums; empty bins have no fabricated score.
        prefix = 'DTM/r05v173_'
        for key in merged_buffer:
            if key.startswith(prefix) and key.endswith('/Validation'):
                short = key[len(prefix):-len('/Validation')]
                summary['r05v173_'+short] = m(key)
        for b in range(5):
            count = self._buffer_sum(merged_buffer, prefix+'seq_bin%d_count/Validation'%b)
            if count > 0:
                ce_sum = self._buffer_sum(merged_buffer, prefix+'seq_bin%d_ce_sum/Validation'%b)
                hits = self._buffer_sum(merged_buffer, prefix+'seq_bin%d_correct/Validation'%b)
                summary['r05v173_seq_bin%d_ce'%b] = ce_sum/count
                summary['r05v173_seq_bin%d_aar'%b] = hits/count
                summary['r05v173_seq_bin%d_count'%b] = count
        return summary

    def _print_validation_audits(self, summary):
        if not self._is_main_proc():
            return
        # Proxy validation is used for checkpoint selection.  Do not print
        # unavailable generator metrics as "nan"; free-running metrics are
        # reported by the mandatory observational Test phase below.
        print(
            "[Validation] "
            f"epoch={self.epoch} val={self._fmt(summary.get('validation_metric'), 5)} "
            f"seq={self._fmt(summary.get('loss_seq'), 5)} "
            f"struct={self._fmt(summary.get('loss_structure'), 5)} "
            f"interface={self._fmt(summary.get('loss_interface'), 5)} "
            f"edge={self._fmt(summary.get('loss_edge'), 5)} "
            f"aar={self._fmt(summary.get('aar'), 5)}"
        )
        print(
            "[GeometryValidation] "
            f"epoch={self.epoch} "
            f"raw_A=[{self._fmt(summary.get('h3ca_raw_r0'),4)},"
            f"{self._fmt(summary.get('h3ca_raw_r1'),4)},"
            f"{self._fmt(summary.get('h3ca_raw_r2'),4)}] "
            f"aligned_A=[{self._fmt(summary.get('h3ca_aligned_r0'),4)},"
            f"{self._fmt(summary.get('h3ca_aligned_r1'),4)},"
            f"{self._fmt(summary.get('h3ca_aligned_r2'),4)}] "
            f"refine_raw_A={self._fmt(summary.get('refinement_raw_delta'),4)} "
            f"refine_aligned_A={self._fmt(summary.get('refinement_aligned_delta'),4)} "
            f"caar={self._fmt(summary.get('caar'),5)} "
            f"contact_f1={self._fmt(summary.get('contact_f1'),5)} "
            f"bridge_s={self._fmt(summary.get('bridge_single_delta_to_base_ratio'),4)} "
            f"bridge_z={self._fmt(summary.get('bridge_pair_delta_to_base_ratio_mean'),4)} "
            f"chart_gap_A={self._fmt(summary.get('sf_chart_gap_A'),6)} "
            f"latent_gap_A=[{self._fmt(summary.get('sf_latent_gap_r0_A'),3)},"
            f"{self._fmt(summary.get('sf_latent_gap_r1_A'),3)},"
            f"{self._fmt(summary.get('sf_latent_gap_r2_A'),3)}]"
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
            "test_status", "test_error",
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
            "val_distogram": validation_summary.get("distogram_loss", float("nan")),
            "val_smooth_lddt": validation_summary.get("smooth_lddt_loss", float("nan")),
        }
        row["test_status"] = str(test_metrics.get("_status", "ok"))
        row["test_error"] = str(test_metrics.get("_error", ""))
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
                f"epoch={row['epoch']} status={row['test_status']} "
                f"train={self._fmt(row['train_loss'], 5)} "
                f"val={self._fmt(row['val_loss'], 5)} "
                f"AAR={self._fmt(row['test_AAR'], 5)} CAAR={self._fmt(row['test_CAAR'], 5)} "
                f"H3raw={self._fmt(row['test_H3raw'], 4, 'A')} H3aligned={self._fmt(row['test_H3aligned'], 4, 'A')} "
                f"TM={self._fmt(row['test_TM'], 5)} lDDT={self._fmt(row['test_lDDT'], 5)} DockQ={self._fmt(row['test_DockQ'], 5)} "
                f"best_val_epoch={row['best_val_epoch']} best_val={self._fmt(row['best_val_loss'], 5)}"
            )

    def _requires_live_bridge_contract(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        return bool(getattr(raw_model, "relational_trunk_enabled", False))

    def _print_singlefield_authority_audit(self, raw_model, val=False):
        diag = getattr(raw_model, 'last_singlefield_diagnostics', None) or {}
        mode = str(getattr(raw_model, 'physical_authority_mode', 'legacy_split'))
        if mode == 'legacy_split' or not diag:
            return

        def scalar(name):
            return self._scalar(diag.get(name))
        def arr(name):
            v = diag.get(name)
            if not torch.is_tensor(v):
                return '[]'
            vals = v.detach().float().reshape(-1).cpu().tolist()
            return '[' + ','.join('NA' if not isfinite(float(x)) else f'{float(x):.5f}' for x in vals) + ']'

        final_gap = scalar('final_pred_vs_carrier_x1_rms_A')
        carrier_rt = scalar('final_carrier_roundtrip_rms_A')
        endpoint_rt = scalar('final_endpoint_roundtrip_rms_A')
        phase = 'val' if val else 'train'
        # V238: the detailed authority line is a startup semantic contract, not a
        # routine training trace. Long-run mechanism tracking is aggregated in
        # [GeometryValidation] instead.
        if (not val) and (not self._singlefield_contract_verified) and self._diag_main_rank:
            print(
                '[SingleFieldAuthorityAudit] '
                f'phase={phase} epoch={self.epoch} step={self.global_step} '
                f'mode={mode} physical_dof=1 rounds=3 '
                f'structure_authority=analytic_endpoint_chart '
                f'transport_authority=analytic_carrier_chart '
                f'sample_terminal=integrated_carrier '
                f'final_chart_gap_A={self._fmt(final_gap,6)} '
                f'carrier_roundtrip_A={self._fmt(carrier_rt,6)} '
                f'endpoint_roundtrip_A={self._fmt(endpoint_rt,6)} '
                f'active={self._fmt(scalar("canonical_active_rate"),4)} '
                f'round_gt_A={arr("round_endpoint_gt_rms_A")} '
                f'round_step_A={arr("round_endpoint_step_rms_A")} '
                f'round_carrier_target_A={arr("round_carrier_target_rms_A")} '
                f'latent_native_gap_A={arr("round_discarded_endpoint_proposal_gap_rms_A")}',
                flush=True,
            )

        # First ordinary training forward is a distributed fail-fast semantic
        # contract.  Tolerance is deliberately physical (0.5 A) to accommodate
        # BF16 arithmetic while still catching a wrong frame/chart by orders of
        # magnitude.  This check never alters the forward/loss.
        if (not val) and not self._singlefield_contract_verified:
            vals = [final_gap, carrier_rt, endpoint_rt]
            local_ok = all(v is not None and isfinite(float(v)) and abs(float(v)) <= 0.5 for v in vals)
            device = next(raw_model.parameters()).device
            status = torch.tensor(1 if local_ok else 0, dtype=torch.int32, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(status, op=dist.ReduceOp.MIN)
            if int(status.item()) != 1:
                raise RuntimeError(
                    'Single-field analytic authority contract failed: '
                    f'mode={mode} final_gap={final_gap} carrier_rt={carrier_rt} endpoint_rt={endpoint_rt}'
                )
            self._singlefield_contract_verified = True
            if self._diag_main_rank:
                print(
                    '[SingleFieldContract] PASS '
                    f'mode={mode} physical_dof=1 frame_aware=1 analytic_roundtrip=PASS '
                    f'tolerance_A=0.5 all_ranks=PASS',
                    flush=True,
                )

    def share_step(self, batch, batch_idx, val=False):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        # Validation captures epoch diagnostics. Training captures diagnostics
        # only at compact science cadence or for the zero-start bridge check.
        science_step_diag = (not val) and (
            int(self.global_step) < self._science_log_first_steps
            or (self._science_log_interval > 0 and int(self.global_step) % self._science_log_interval == 0)
        )
        bridge_contract_probe = bool(
            (not val)
            and self._requires_live_bridge_contract()
            and not self._live_bridge_contract_verified
        )
        capture_diagnostics = bool(val) or science_step_diag or bridge_contract_probe
        raw_model._diagnostic_capture = bool(capture_diagnostics)
        raw_model._diagnostic_validation_mode = bool(val and capture_diagnostics)

        loss, seq_detail, structure_detail, dock_detail, pdev_detail = self.model(**batch)
        snll, aar = seq_detail
        struct_loss, xloss, bond_loss, sc_bond_loss = structure_detail
        dock_loss, interface_loss, ed_loss, r_ed_losses = dock_detail
        pdev_loss, prmsd_loss = pdev_detail

        if capture_diagnostics and ((not val) or int(batch_idx) == 0):
            self._print_singlefield_authority_audit(raw_model, val=val)

        if not val:
            current_epoch = int(self.epoch)
            if self._train_loss_outlier_epoch != current_epoch:
                self._train_loss_outlier_epoch = current_epoch
                self._train_loss_outlier_count = 0

            struct_scalar = self._scalar(struct_loss)
            is_outlier = (
                struct_scalar is not None
                and isfinite(struct_scalar)
                and abs(float(struct_scalar)) >= float(self._train_loss_outlier_threshold)
            )
            if (
                is_outlier
                and self._train_loss_outlier_count < self._train_loss_outlier_max_per_epoch
            ):
                self._train_loss_outlier_count += 1
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                loss_scalar = self._scalar(loss)
                print(
                    "[TrainLossOutlier] "
                    f"epoch={self.epoch} step={self.global_step} rank={rank} "
                    f"ordinal={self._train_loss_outlier_count}/{self._train_loss_outlier_max_per_epoch} "
                    f"loss={self._fmte(loss_scalar, 6)} "
                    f"struct={self._fmte(struct_scalar, 6)} "
                    f"threshold={self._fmte(self._train_loss_outlier_threshold, 3)}"
                )

                abdiag = getattr(raw_model, "last_abflow_diagnostics", None) or {}
                print(
                    "[TrainLossOutlierDetails] "
                    f"epoch={self.epoch} step={self.global_step} rank={rank} "
                    f"xloss={self._fmte(self._scalar(xloss), 6)} "
                    f"bond={self._fmte(self._scalar(bond_loss), 6)} "
                    f"scbond={self._fmte(self._scalar(sc_bond_loss), 6)} "
                    f"interface={self._fmte(self._scalar(interface_loss), 6)} "
                    f"edge={self._fmte(self._scalar(ed_loss), 6)} "
                    f"t_min={self._fmt(self._scalar(abdiag.get('t_min')), 5)} "
                    f"t_mean={self._fmt(self._scalar(abdiag.get('t_mean')), 5)} "
                    f"t_max={self._fmt(self._scalar(abdiag.get('t_max')), 5)} "
                    f"path_cov={self._fmt(self._scalar(abdiag.get('sequence_path_mask_rate')), 5)} "
                    f"loss_cov={self._fmt(self._scalar(abdiag.get('sequence_loss_mask_rate')), 5)} "
                    f"context_ratio={self._fmt(self._scalar(batch.get('context_ratio')), 5)}"
                )

                logical_indices = self._formal_train_batch_indices(batch)
                labels = self._formal_dataset_labels(logical_indices)
                geom = getattr(raw_model, 'last_geometry_forensics', None) or {}
                worst_idx = geom.get('worst_graph_index')
                try:
                    worst_idx = int(worst_idx.detach().cpu().item()) if torch.is_tensor(worst_idx) else int(worst_idx)
                except Exception:
                    worst_idx = None

                def _graph_scalar(key):
                    value = geom.get(key)
                    if value is None or worst_idx is None:
                        return None
                    try:
                        if torch.is_tensor(value):
                            return float(value.detach().float().reshape(-1)[worst_idx].cpu().item())
                        return float(value[worst_idx])
                    except Exception:
                        return None

                worst_label = (
                    labels[worst_idx]
                    if worst_idx is not None and worst_idx < len(labels)
                    else 'NA'
                )
                lengths = batch.get('lengths')
                worst_length = None
                if torch.is_tensor(lengths) and worst_idx is not None:
                    try:
                        worst_length = int(lengths.detach().reshape(-1)[worst_idx].cpu().item())
                    except Exception:
                        pass
                print(
                    "[TrainGeometryOutlier] "
                    f"epoch={self.epoch} step={self.global_step} rank={rank} "
                    f"worst_graph={worst_idx} dataset_index={logical_indices[worst_idx] if worst_idx is not None and worst_idx < len(logical_indices) else 'NA'} "
                    f"name={worst_label!r} length={worst_length if worst_length is not None else 'NA'} "
                    f"t={self._fmt(_graph_scalar('per_graph_t'), 5)} "
                    f"pred_design_rms_A={self._fmt(_graph_scalar('per_graph_pred_design_rms_A'), 5)} "
                    f"pred_design_absmax_A={self._fmt(_graph_scalar('per_graph_pred_design_absmax_A'), 5)} "
                    f"carrier_target_rms_A={self._fmt(_graph_scalar('per_graph_carrier_target_rms_A'), 5)} "
                    f"carrier_absmax_A={self._fmt(_graph_scalar('per_graph_carrier_absmax_A'), 5)}"
                )

                def _round_graph_values(key):
                    value = geom.get(key)
                    if value is None or worst_idx is None:
                        return []
                    try:
                        if torch.is_tensor(value):
                            vv = value.detach().float().cpu()
                            if vv.ndim != 2 or worst_idx >= vv.shape[1]:
                                return []
                            return [float(x) for x in vv[:, worst_idx].tolist()]
                        return [float(row[worst_idx]) for row in value]
                    except Exception:
                        return []

                round_delta = _round_graph_values('per_round_graph_delta_rms_A')
                round_absmax = _round_graph_values('per_round_graph_absmax_A')
                if round_delta or round_absmax:
                    print(
                        "[TrainGeometryRounds] "
                        f"epoch={self.epoch} step={self.global_step} rank={rank} "
                        f"worst_graph={worst_idx} "
                        f"delta_rms_A={[round(v, 5) for v in round_delta]} "
                        f"absmax_A={[round(v, 5) for v in round_absmax]}"
                    )

                auth_rms = _round_graph_values('per_round_authority_endpoint_rms_A')
                auth_abs = _round_graph_values('per_round_authority_endpoint_absmax_A')
                auth_car = _round_graph_values('per_round_authority_carrier_target_rms_A')
                disc_ep = _round_graph_values('per_round_discarded_endpoint_gap_rms_A')
                disc_car = _round_graph_values('per_round_discarded_carrier_gap_rms_A')
                if auth_rms or auth_car:
                    sf = getattr(raw_model, 'last_singlefield_diagnostics', None) or {}
                    print(
                        "[TrainAuthorityRounds] "
                        f"epoch={self.epoch} step={self.global_step} rank={rank} "
                        f"mode={getattr(raw_model, 'physical_authority_mode', 'NA')} "
                        f"worst_graph={worst_idx} "
                        f"endpoint_gt_rms_A={[round(v, 5) for v in auth_rms]} "
                        f"endpoint_absmax_A={[round(v, 5) for v in auth_abs]} "
                        f"carrier_target_rms_A={[round(v, 5) for v in auth_car]} "
                        f"discard_endpoint_gap_A={[round(v, 5) for v in disc_ep]} "
                        f"discard_carrier_gap_A={[round(v, 5) for v in disc_car]} "
                        f"final_chart_gap_A={self._fmt(self._scalar(sf.get('final_pred_vs_carrier_x1_rms_A')), 6)}"
                    )

                # Exact stage-wise actuator trace from the SAME forward that
                # produced the outlier.  No legacy key aliases are used.
                round_egnn = getattr(raw_model, '_last_round_egnn_diagnostics', None) or []
                def _diag_number(value):
                    try:
                        if torch.is_tensor(value):
                            return float(value.detach().float().cpu().item())
                        return float(value)
                    except Exception:
                        return None
                def _stage(coord_diag, stage, stream):
                    p = stage + '.'
                    return {
                        'a_rms': _diag_number(coord_diag.get(p+'coord_state_coeff_raw_rms')),
                        'a_max': _diag_number(coord_diag.get(p+'coord_state_coeff_raw_absmax')),
                        'base_rms': _diag_number(coord_diag.get(p+'coord_base_coeff_raw_rms')),
                        'base_max': _diag_number(coord_diag.get(p+'coord_base_coeff_raw_absmax')),
                        'dpair_rms': _diag_number(coord_diag.get(p+'coord_direct_pair_coeff_delta_rms')),
                        'dpair_max': _diag_number(coord_diag.get(p+'coord_direct_pair_coeff_delta_absmax')),
                        'lever_rms': _diag_number(coord_diag.get(p+'coord_diff_norm_rms')),
                        'lever_max': _diag_number(coord_diag.get(p+'coord_diff_norm_absmax')),
                        'dx_rms': _diag_number(coord_diag.get(p+stream+'_design_update_rms')),
                        'dx_max': _diag_number(coord_diag.get(p+stream+'_design_update_absmax')),
                        'w1': _diag_number(coord_diag.get(p+'coord_head_w1_opnorm')),
                        'w2': _diag_number(coord_diag.get(p+'coord_head_w2_opnorm')),
                    }
                def _stage_text(name, vals):
                    return (
                        f'{name}[a={self._fmt(vals["a_rms"],5)}/{self._fmt(vals["a_max"],5)} '
                        f'base={self._fmt(vals["base_rms"],5)}/{self._fmt(vals["base_max"],5)} '
                        f'dpair={self._fmt(vals["dpair_rms"],5)}/{self._fmt(vals["dpair_max"],5)} '
                        f'lever={self._fmt(vals["lever_rms"],5)}/{self._fmt(vals["lever_max"],5)} '
                        f'dx={self._fmt(vals["dx_rms"],5)}/{self._fmt(vals["dx_max"],5)} '
                        f'w={self._fmt(vals["w1"],5)}/{self._fmt(vals["w2"],5)}]'
                    )

                n_layers = int(getattr(getattr(raw_model, 'gnn', None), 'n_layers', 0))
                for rec in round_egnn:
                    coord_diag = rec.get('coord', {}) or {}
                    bridge_diag = rec.get('bridge', {}) or {}
                    native_names = [f'ctx_{i}' for i in range(n_layers)] + ['out']
                    carrier_names = []
                    for i in range(n_layers):
                        carrier_names.extend([f'inter_{i}', f'surf_{i}'])
                    native_vals = [(st, _stage(coord_diag, st, 'native')) for st in native_names]
                    carrier_vals = [(st, _stage(coord_diag, st, 'carrier')) for st in carrier_names]
                    finite_dx = [
                        (v['dx_max'], st) for st, v in native_vals + carrier_vals
                        if v['dx_max'] is not None and isfinite(v['dx_max'])
                    ]
                    worst = max(finite_dx, default=(None, 'NA'))
                    print(
                        "[TrainStageOutlier] "
                        f"epoch={self.epoch} step={self.global_step} rank={rank} "
                        f"physical_round={rec.get('round_idx', 'NA')} "
                        f"authority={getattr(raw_model, 'physical_authority_mode', 'NA')} "
                        f"worst_stage={worst[1]} worst_dx_absmax={self._fmt(worst[0],6)} "
                        f"single_ratio={self._fmt(_diag_number(bridge_diag.get('bridge_single_delta_to_base_ratio')),6)} "
                        f"pair_sem_ratio={self._fmt(_diag_number(bridge_diag.get('bridge_pair_delta_to_base_ratio_mean')),6)} "
                        f"pair_coord_ratio={self._fmt(_diag_number(bridge_diag.get('bridge_pair_coordinate_delta_to_base_ratio_mean')),6)} "
                        "native=" + ';'.join(_stage_text(st,v) for st,v in native_vals) + " "
                        "carrier=" + ';'.join(_stage_text(st,v) for st,v in carrier_vals)
                    )

                def _safe_preview(value, limit=8):
                    try:
                        if torch.is_tensor(value):
                            v = value.detach().cpu()
                            if v.numel() <= limit:
                                return str(v.reshape(-1).tolist())
                            return str(v.reshape(-1)[:limit].tolist()) + "..."
                        if isinstance(value, (list, tuple)):
                            return repr(list(value[:limit])) + ("..." if len(value) > limit else "")
                        if isinstance(value, (str, int, float, bool)):
                            return repr(value)
                    except Exception:
                        pass
                    return None

                meta = []
                if logical_indices:
                    meta.append(f"dataset_indices={logical_indices[:8]}{'...' if len(logical_indices) > 8 else ''}")
                if labels:
                    meta.append(f"dataset_names={labels[:8]}{'...' if len(labels) > 8 else ''}")
                for key in (
                    "names", "pdb", "pdb_id", "complex_id", "sample_id", "id",
                    "name", "summary", "lengths"
                ):
                    if key in batch:
                        preview = _safe_preview(batch[key])
                        if preview is not None:
                            meta.append(f"{key}={preview}")
                if not meta:
                    meta.append("keys=" + ",".join(sorted(map(str, batch.keys()))))
                print(
                    "[TrainLossOutlierBatch] "
                    f"epoch={self.epoch} step={self.global_step} rank={rank} "
                    + " ".join(meta)
                )

        # Common R28/R29/R30 bridge contract.  Step 0 must be the exact R05
        # function (zero Single and Pair deltas); after one optimizer update,
        # both donor routes must be measurably active.  This makes a detached or
        # accidentally bypassed donor trunk fail before an expensive epoch.
        if (
            not val
            and self._requires_live_bridge_contract()
            and not self._live_bridge_contract_verified
        ):
            bridge_diag = getattr(raw_model, "last_abflow_diagnostics", None) or {}
            single_ratio = self._scalar(
                bridge_diag.get("bridge_single_delta_to_base_ratio")
            )
            pair_ratio = self._scalar(
                bridge_diag.get("bridge_pair_delta_to_base_ratio_mean")
            )
            finite_ratios = bool(
                single_ratio is not None and pair_ratio is not None
                and isfinite(float(single_ratio)) and isfinite(float(pair_ratio))
            )
            is_resume_state = bool(
                int(self.global_step) > 0 and not self._bridge_cold_start_observed
            )
            local_cold = bool(
                (not is_resume_state)
                and not self._bridge_cold_start_observed
                and finite_ratios
                and abs(float(single_ratio)) <= 1.0e-12
                and abs(float(pair_ratio)) <= 1.0e-12
            )
            local_live = bool(
                self._bridge_cold_start_observed
                and finite_ratios
                and float(single_ratio) > 1.0e-12
                and float(pair_ratio) > 1.0e-12
            )
            local_resume_live = bool(
                is_resume_state
                and finite_ratios
                and float(single_ratio) > 1.0e-12
                and float(pair_ratio) > 1.0e-12
            )
            states = loss.detach().new_tensor(
                [
                    1 if local_cold else 0,
                    1 if local_live else 0,
                    1 if local_resume_live else 0,
                ],
                dtype=torch.int32,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(states, op=dist.ReduceOp.MIN)
            cold_all, live_all, resume_live_all = [
                bool(int(v)) for v in states.cpu().tolist()
            ]
            if cold_all:
                self._bridge_cold_start_observed = True
                if self._diag_main_rank:
                    print(
                        "[BridgeContract] phase=cold_start PASS "
                        f"step={self.global_step} single_ratio={self._fmte(single_ratio, 3)} "
                        f"pair_ratio={self._fmte(pair_ratio, 3)} "
                        "parent_identity=exact next_batch_requires_live=1"
                    )
            elif live_all:
                self._live_bridge_contract_verified = True
                if self._diag_main_rank:
                    print(
                        "[BridgeContract] phase=live PASS "
                        f"step={self.global_step} single_ratio={self._fmte(single_ratio, 3)} "
                        f"pair_ratio={self._fmte(pair_ratio, 3)} all_ranks=PASS"
                    )
            elif resume_live_all:
                self._live_bridge_contract_verified = True
                if self._diag_main_rank:
                    print(
                        "[BridgeContract] phase=resume_live PASS "
                        f"step={self.global_step} single_ratio={self._fmte(single_ratio, 3)} "
                        f"pair_ratio={self._fmte(pair_ratio, 3)} "
                        "checkpoint_bridge_already_live=1 all_ranks=PASS"
                    )
            else:
                raise RuntimeError(
                    "V211 Single/Pair bridge contract failed: a fresh run must "
                    "show exact zero deltas then live deltas; a resumed run must "
                    "restore finite non-zero bridge deltas immediately. "
                    f"step={self.global_step}, single_ratio={single_ratio}, "
                    f"pair_ratio={pair_ratio}, "
                    f"cold_start_seen={self._bridge_cold_start_observed}."
                )

        # V213: periodic + threshold-triggered observation of the *base* Cartesian
        # controller.  This closes the blind spot that let R33 epoch30 move the
        # model state substantially while staying below the 1e4 loss threshold.
        if not val:
            round_egnn_authority = getattr(raw_model, '_last_round_egnn_diagnostics', None) or []

            def _auth_num(value):
                try:
                    if torch.is_tensor(value):
                        return float(value.detach().float().cpu().item())
                    return float(value)
                except Exception:
                    return None

            def _auth_stage_max(diag, suffix):
                vals = []
                for key, value in diag.items():
                    if not key.endswith(suffix):
                        continue
                    fv = _auth_num(value)
                    if fv is not None and isfinite(fv):
                        vals.append(fv)
                return max(vals) if vals else None

            authority_rows = []
            for rec in round_egnn_authority:
                cd = rec.get('coord', {}) or {}
                authority_rows.append({
                    'round': rec.get('round_idx', 'NA'),
                    'base': _auth_stage_max(cd, '.coord_base_coeff_absmax'),
                    'state': _auth_stage_max(cd, '.coord_state_coeff_absmax'),
                    'pair_bounded': _auth_stage_max(cd, '.coord_pair_delta_bounded_absmax'),
                    'update': _auth_num(cd.get('coord_update_absmax_max')),
                })

            max_base = max(
                [r['base'] for r in authority_rows if r['base'] is not None],
                default=None,
            )
            max_update = max(
                [r['update'] for r in authority_rows if r['update'] is not None],
                default=None,
            )
            authority_interval_hit = bool(
                self._geometry_authority_interval > 0
                and int(self.global_step) % self._geometry_authority_interval == 0
            )
            authority_alert = bool(
                (max_base is not None and max_base >= self._geometry_authority_base_alert)
                or (max_update is not None and max_update >= self._geometry_authority_update_alert)
            )
            current_epoch = int(self.epoch)
            if self._geometry_authority_alert_epoch != current_epoch:
                self._geometry_authority_alert_epoch = current_epoch
                self._geometry_authority_alert_count = 0
            allow_alert = (
                authority_alert
                and self._geometry_authority_alert_count
                < self._geometry_authority_alert_max_per_epoch
            )
            if allow_alert:
                self._geometry_authority_alert_count += 1

            if (authority_interval_hit and self._diag_main_rank) or allow_alert:
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                tag = 'GeometryAuthorityAlert' if authority_alert else 'GeometryAuthority'
                print(
                    f'[{tag}] '
                    f'epoch={self.epoch} step={self.global_step} rank={rank} '
                    f"base_coeff_absmax={[None if r['base'] is None else round(r['base'], 6) for r in authority_rows]} "
                    f"state_coeff_absmax={[None if r['state'] is None else round(r['state'], 6) for r in authority_rows]} "
                    f"pair_bounded_absmax={[None if r['pair_bounded'] is None else round(r['pair_bounded'], 6) for r in authority_rows]} "
                    f"coord_update_absmax={[None if r['update'] is None else round(r['update'], 6) for r in authority_rows]} "
                    f'base_alert={self._geometry_authority_base_alert:g} '
                    f'update_alert={self._geometry_authority_update_alert:g}'
                )

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

        # V238 compact single-field validation observers. These are detached
        # diagnostics only; they never enter the objective or checkpoint rule.
        if val:
            sf_diag = getattr(raw_model, "last_singlefield_diagnostics", None) or {}
            for src, dst in (
                ("final_pred_vs_carrier_x1_rms_A", "final_chart_gap_A"),
                ("canonical_active_rate", "canonical_active_rate"),
            ):
                value = sf_diag.get(src)
                if value is not None:
                    self.log(f"AbFlowSF/{dst}/Validation", value, batch_idx, True)
            latent = sf_diag.get("round_discarded_endpoint_proposal_gap_rms_A")
            if torch.is_tensor(latent):
                flat = latent.reshape(-1)
                for ridx in range(min(3, int(flat.numel()))):
                    self.log(
                        f"AbFlowSF/latent_native_gap_r{ridx}_A/Validation",
                        flat[ridx], batch_idx, True
                    )

        if not val and (
            int(self.global_step) < self._science_log_first_steps
            or (self._science_log_interval > 0 and int(self.global_step) % self._science_log_interval == 0)
        ) and self._diag_main_rank:
            def _sf(name):
                return self._scalar(scorefm_losses.get(name))
            def _ad(name):
                return self._scalar(abflow_diagnostics.get(name))
            print(
                "[RelationalStep] "
                f"epoch={self.epoch} step={self.global_step} "
                f"loss={self._fmt(self._scalar(loss), 5)} "
                f"seq={self._fmt(self._scalar(snll), 5)} "
                f"struct={self._fmt(self._scalar(struct_loss), 5)} "
                f"interface={self._fmt(self._scalar(interface_loss), 5)} "
                f"edge={self._fmt(self._scalar(ed_loss), 5)} "
                f"t={self._fmt(_ad('t_mean'), 3)} "
                f"s={self._fmt(_sf('relational_single_rms'), 4)} "
                f"z={self._fmt(_sf('relational_pair_rms'), 4)} "
                f"edge_z=({self._fmt(_ad('abx_ctx_edge_attr_rms'), 4)},"
                f"{self._fmt(_ad('abx_inter_edge_attr_rms'), 4)},"
                f"{self._fmt(_ad('abx_surf_edge_attr_rms'), 4)}) "
                f"bridge_s={self._fmt(_ad('bridge_single_delta_to_base_ratio'), 6)} "
                f"bridge_z={self._fmt(_ad('bridge_pair_delta_to_base_ratio_mean'), 6)}"
            )
            if bool(getattr(raw_model, "distogram_enabled", False)):
                print(
                    "[Distogram] "
                    f"epoch={self.epoch} step={self.global_step} "
                    f"scope={getattr(raw_model, 'distogram_pair_scope', 'NA')} "
                    f"raw={self._fmt(_sf('distogram_loss'), 6)} "
                    f"weighted={self._fmt(_sf('distogram_weighted_loss'), 6)} "
                    f"head_rms={self._fmt(_sf('distogram_head_weight_rms'), 6)} "
                    f"task_fraction={self._fmt(_sf('distogram_task_pair_fraction'), 4)} "
                    f"pairs=DD:{self._fmt(_sf('distogram_DD_pairs'),0)},"
                    f"DF:{self._fmt(_sf('distogram_DF_pairs'),0)},"
                    f"DA:{self._fmt(_sf('distogram_DA_pairs'),0)} "
                    f"ctxctx={self._fmt(_sf('distogram_context_context_optimized_pairs'),0)} "
                    f"CE=DD:{self._fmt(_sf('distogram_DD_ce'),4)},"
                    f"DF:{self._fmt(_sf('distogram_DF_ce'),4)},"
                    f"DA:{self._fmt(_sf('distogram_DA_ce'),4)} "
                    f"DA_contactP={self._fmt(_sf('distogram_da_contact_precision'), 4)}"
                )
            if bool(getattr(raw_model, "smooth_lddt_enabled", False)):
                print(
                    "[SmoothLDDT] "
                    f"epoch={self.epoch} step={self.global_step} "
                    f"source={getattr(raw_model, 'smooth_lddt_prediction_source', 'pred_design_endpoint')} "
                    f"raw={self._fmt(_sf('smooth_lddt_loss'), 6)} "
                    f"weighted={self._fmt(_sf('smooth_lddt_weighted_loss'), 6)} "
                    f"endpoint_rms_A={self._fmt(_sf('smooth_lddt_endpoint_rms_A'), 5)} "
                    f"endpoint_absmax_A={self._fmt(_sf('smooth_lddt_endpoint_absmax_A'), 5)} "
                    f"design_rows={self._fmt(_sf('smooth_lddt_design_rows'), 0)} "
                    f"coord_rows={self._fmt(_sf('smooth_lddt_coord_rows'), 0)} "
                    f"coord_outside_design={self._fmt(_sf('smooth_lddt_coord_outside_design_rows'), 0)} "
                    f"DD={self._fmt(_sf('smooth_lddt_DD'), 5)} "
                    f"DF={self._fmt(_sf('smooth_lddt_DF'), 5)} "
                    f"DA={self._fmt(_sf('smooth_lddt_DA'), 5)}"
                )
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
                "distogram", scorefm_losses.get("distogram_loss")
            )
            self._accumulate_train_component(
                "smooth_lddt", scorefm_losses.get("smooth_lddt_loss")
            )
        return loss
