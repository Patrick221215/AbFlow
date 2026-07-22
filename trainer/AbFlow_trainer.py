#!/usr/bin/python
# -*- coding:utf-8 -*-
"""AbFlow trainer with low-overhead, machine-readable diagnostics.

The TensorBoard logging behavior is preserved.  Main-rank JSONL/latest files are
added so each epoch can be inspected without opening TensorBoard or evaluating
all test checkpoints.  Periodic gradient-conflict probes are observational only.
"""
from math import cos, pi, log, exp, isfinite
from datetime import datetime
import json
import os
import tempfile

import torch
from evaluation.rmsd import kabsch_torch
from .abs_trainer import Trainer


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

        # ---------------------------------------------------------
        # Fixed-subset in-memory validation rollout
        # ---------------------------------------------------------
        # This is intentionally separated from the external AutoTopK evaluator.
        # It uses the current in-memory model and current validation batch:
        #
        #   sample(n_steps) -> tensor metrics -> JSON
        #
        # It does NOT reload checkpoints, write PDB files, create summary.json,
        # or call generate.py/cal_metrics.py/OpenMM/DockQ.
        self._rollout_enabled = _env_flag(
            "ABFLOW_INMEM_ROLLOUT_VALIDATION", True
        )
        self._rollout_start_epoch = max(
            0, _env_int("ABFLOW_INMEM_ROLLOUT_START_EPOCH", 0)
        )
        self._rollout_interval_epochs = max(
            1, _env_int("ABFLOW_INMEM_ROLLOUT_INTERVAL_EPOCHS", 5)
        )
        self._rollout_max_batches = max(
            1, _env_int("ABFLOW_INMEM_ROLLOUT_MAX_BATCHES", 1)
        )

        # AbFlowModel.sample() itself honors ABFLOW_SAMPLE_N_STEPS when it is
        # non-empty. Mirror that precedence here so the recorded n_steps is exact.
        sample_steps_env = os.environ.get(
            "ABFLOW_SAMPLE_N_STEPS", ""
        ).strip()
        self._rollout_n_steps = (
            int(sample_steps_env)
            if sample_steps_env
            else max(1, _env_int("ABFLOW_INMEM_ROLLOUT_N_STEPS", 10))
        )
        self._rollout_seed = _env_int(
            "ABFLOW_INMEM_ROLLOUT_SEED", 20260723
        )
        self._rollout_contact_cutoff = _env_float(
            "ABFLOW_INMEM_ROLLOUT_CONTACT_CUTOFF", 8.0
        )
        if self._rollout_contact_cutoff <= 0.0:
            raise ValueError(
                "ABFLOW_INMEM_ROLLOUT_CONTACT_CUTOFF must be positive."
            )

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
                    "rollout_metrics.jsonl": "real multi-step validation rollout metrics; no structures are written",
                    "latest_rollout_validation.json": "latest in-memory rollout probe",
                    "rollout_errors.log": "fail-soft rollout diagnostic errors",
                },
                "intervals": {
                    "train_steps": self._diag_file_interval,
                    "validation_batches": self._diag_valid_interval,
                    "gradient_probe_steps": self._diag_grad_interval,
                    "rollout_start_epoch": self._rollout_start_epoch,
                    "rollout_epochs": self._rollout_interval_epochs,
                    "rollout_batches": self._rollout_max_batches,
                    "rollout_n_steps": self._rollout_n_steps,
                },
            }
            self._atomic_json(os.path.join(self._diag_dir, "schema.json"), schema)

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


    def _should_rollout_probe(self, val, batch_idx):
        """Select a fixed, tiny validation subset.

        Validation loaders are expected to be non-shuffled.  Under that standard
        setting, ``batch_idx < max_batches`` selects the same complexes at every
        probe epoch.  The seed is also fixed across epochs, so metric changes are
        attributable to the checkpoint rather than a new source/CTMC draw.
        """
        if (
            not val
            or not self._rollout_enabled
            or not self._diag_enabled
            or not self._diag_main_rank
        ):
            return False

        epoch = int(getattr(self, "epoch", 0))
        if epoch < self._rollout_start_epoch:
            return False
        if (
            (epoch - self._rollout_start_epoch)
            % self._rollout_interval_epochs
            != 0
        ):
            return False
        return int(batch_idx) < self._rollout_max_batches

    @staticmethod
    def _batch_id_from_lengths(lengths, ref_tensor):
        lengths = torch.as_tensor(
            lengths, device=ref_tensor.device, dtype=torch.long
        ).reshape(-1)
        if lengths.numel() == 0:
            raise ValueError("Validation rollout received empty lengths.")
        if int(lengths.sum().item()) != int(ref_tensor.shape[0]):
            raise ValueError(
                "Sum(lengths) does not match residue count: "
                f"{int(lengths.sum().item())} vs "
                f"{int(ref_tensor.shape[0])}."
            )

        batch_id = torch.zeros(
            ref_tensor.shape[0],
            device=ref_tensor.device,
            dtype=torch.long,
        )
        starts = torch.cumsum(lengths, dim=0)[:-1]
        if starts.numel() > 0:
            batch_id[starts] = 1
        return batch_id.cumsum(dim=0), lengths

    @staticmethod
    def _mean_or_nan(values, ref_tensor):
        if not values:
            return ref_tensor.new_tensor(
                float("nan"), dtype=torch.float32
            )
        return torch.stack([value.float() for value in values]).mean()

    @torch.no_grad()
    def _compute_inmemory_rollout_metrics(
            self, raw_model, batch, gen_X, gen_S):
        """Compute rollout metrics directly from tensors.

        These are low-cost validation diagnostics, not replacements for the
        canonical final TM-score/lDDT/DockQ pipeline.

        Metrics:
          - rollout_aar:
                per-complex recovery over the design mask;
          - rollout_native_contact_caar:
                recovery on native H3 residues contacting antigen at the
                configured CA cutoff;
          - rollout_h3_ca_rmsd_raw:
                complex-frame H3 placement error;
          - rollout_h3_ca_rmsd_aligned:
                H3 local shape error after H3-only Kabsch alignment;
          - rollout_native_contact_precision/recall/f1:
                native-vs-generated H3--antigen CA contact recovery.
        """
        true_X = batch["X"].detach()
        true_S = batch["S"].detach().long()
        gen_X = gen_X.detach()
        gen_S = gen_S.detach().long()
        paratope_mask = batch["paratope_mask"].detach().bool()
        design_mask = batch["smask"].detach().bool()

        batch_id, lengths = self._batch_id_from_lengths(
            batch["lengths"], true_S
        )
        n_graph = int(lengths.numel())

        segment_ids = raw_model.aa_feature._construct_segment_ids(
            true_S
        )
        antigen_mask = (
            (segment_ids == raw_model.aa_feature.ag_seg_id)
            & (true_S != raw_model.aa_feature.boa_idx)
        )

        ca_idx = 1 if true_X.shape[1] > 1 else 0
        true_ca = true_X[:, ca_idx].float()
        pred_ca = gen_X[:, ca_idx].float()
        cutoff = float(self._rollout_contact_cutoff)
        eps = 1e-8

        aar_values = []
        caar_values = []
        raw_rmsd_values = []
        aligned_rmsd_values = []
        contact_precision_values = []
        contact_recall_values = []
        contact_f1_values = []

        valid_graphs = 0
        for graph_idx in range(n_graph):
            graph_mask = batch_id == graph_idx
            h3_mask = graph_mask & paratope_mask
            antigen_graph_mask = graph_mask & antigen_mask
            seq_mask = graph_mask & design_mask

            if bool(seq_mask.any()):
                aar_values.append(
                    (gen_S[seq_mask] == true_S[seq_mask])
                    .float().mean()
                )

            if not bool(h3_mask.any()):
                continue

            valid_graphs += 1
            pred_h3_ca = pred_ca[h3_mask]
            true_h3_ca = true_ca[h3_mask]

            raw_rmsd_values.append(torch.sqrt(
                (pred_h3_ca - true_h3_ca)
                .pow(2).sum(dim=-1).mean().clamp_min(0.0)
            ))

            if pred_h3_ca.shape[0] >= 3:
                try:
                    _, rotation, translation = kabsch_torch(
                        pred_h3_ca, true_h3_ca
                    )
                    pred_h3_aligned = (
                        torch.matmul(pred_h3_ca, rotation.T)
                        + translation
                    )
                    aligned_rmsd_values.append(torch.sqrt(
                        (pred_h3_aligned - true_h3_ca)
                        .pow(2).sum(dim=-1).mean().clamp_min(0.0)
                    ))
                except Exception:
                    # A degenerate local alignment must not invalidate the raw
                    # placement metric or terminate expensive training.
                    pass

            if not bool(antigen_graph_mask.any()):
                continue

            antigen_ca = true_ca[antigen_graph_mask]
            native_contact = (
                torch.cdist(true_h3_ca, antigen_ca) < cutoff
            )
            generated_contact = (
                torch.cdist(pred_h3_ca, antigen_ca) < cutoff
            )

            tp = (native_contact & generated_contact).float().sum()
            fp = ((~native_contact) & generated_contact).float().sum()
            fn = (native_contact & (~generated_contact)).float().sum()

            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = (
                2.0 * precision * recall
                / (precision + recall + eps)
            )
            contact_precision_values.append(precision)
            contact_recall_values.append(recall)
            contact_f1_values.append(f1)

            native_contact_residue = native_contact.any(dim=-1)
            if bool(native_contact_residue.any()):
                pred_h3_seq = gen_S[h3_mask]
                true_h3_seq = true_S[h3_mask]
                caar_values.append(
                    (
                        pred_h3_seq[native_contact_residue]
                        == true_h3_seq[native_contact_residue]
                    ).float().mean()
                )

        return {
            "rollout_aar": self._mean_or_nan(
                aar_values, true_X
            ),
            "rollout_native_contact_caar": self._mean_or_nan(
                caar_values, true_X
            ),
            "rollout_h3_ca_rmsd_raw": self._mean_or_nan(
                raw_rmsd_values, true_X
            ),
            "rollout_h3_ca_rmsd_aligned": self._mean_or_nan(
                aligned_rmsd_values, true_X
            ),
            "rollout_native_contact_precision": self._mean_or_nan(
                contact_precision_values, true_X
            ),
            "rollout_native_contact_recall": self._mean_or_nan(
                contact_recall_values, true_X
            ),
            "rollout_native_contact_f1": self._mean_or_nan(
                contact_f1_values, true_X
            ),
            "rollout_num_graphs": true_X.new_tensor(
                float(valid_graphs), dtype=torch.float32
            ),
        }

    @torch.no_grad()
    def _run_inmemory_rollout_probe(
            self, raw_model, batch, batch_idx):
        """Run the real sampler without checkpoint reload or structure output."""
        required = (
            "X", "S", "cmask", "smask", "paratope_mask",
            "X_pep", "S_pep", "surface", "residue_pos",
            "template", "lengths",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(
                "Validation rollout batch is missing keys: "
                + ", ".join(missing)
            )

        device = batch["X"].device
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ]

        # Keep source noise and categorical CTMC draws identical across epochs.
        seed = int(self._rollout_seed) + int(batch_idx)

        was_training = bool(raw_model.training)
        old_capture = bool(
            getattr(raw_model, "_diagnostic_capture", False)
        )
        old_validation_mode = bool(
            getattr(raw_model, "_diagnostic_validation_mode", False)
        )
        raw_model.eval()
        raw_model._diagnostic_capture = False
        raw_model._diagnostic_validation_mode = False

        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)

                gen_X, gen_S, _ = raw_model.sample(
                    X=batch["X"],
                    S=batch["S"],
                    cmask=batch["cmask"],
                    smask=batch["smask"],
                    paratope_mask=batch["paratope_mask"],
                    X_pep=batch["X_pep"],
                    S_pep=batch["S_pep"],
                    surface=batch["surface"],
                    residue_pos=batch["residue_pos"],
                    template=batch["template"],
                    lengths=batch["lengths"],
                    n_steps=self._rollout_n_steps,
                    return_hidden=False,
                    show_progress=False,
                )

            metrics = self._compute_inmemory_rollout_metrics(
                raw_model, batch, gen_X, gen_S
            )
        finally:
            # sample() normally clears this cache. Explicit cleanup protects the
            # next training/validation batch after a diagnostic exception.
            if hasattr(raw_model, "_clean_batch_constants"):
                raw_model._clean_batch_constants()
            raw_model._diagnostic_capture = old_capture
            raw_model._diagnostic_validation_mode = (
                old_validation_mode
            )
            if was_training:
                raw_model.train()

        return metrics, seed

    def _write_rollout_record(
            self, metrics, batch_idx, seed):
        if not self._diag_main_rank or not self._diag_enabled:
            return

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "split": "validation_rollout",
            "epoch": int(getattr(self, "epoch", -1)),
            "global_step": int(getattr(self, "global_step", -1)),
            "validation_batch_idx": int(batch_idx),
            "n_steps": int(self._rollout_n_steps),
            "seed": int(seed),
            "contact_cutoff_angstrom": float(
                self._rollout_contact_cutoff
            ),
            "writes_pdb": False,
            "reloads_checkpoint": False,
            "calls_generate_py": False,
            "calls_cal_metrics_py": False,
        }
        for name, value in metrics.items():
            scalar = self._scalar(value)
            if scalar is not None and isfinite(scalar):
                record[name] = scalar

        jsonl_path = os.path.join(
            self._diag_dir, "rollout_metrics.jsonl"
        )
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                record, ensure_ascii=False, sort_keys=True
            ) + "\n")

        self._atomic_json(
            os.path.join(
                self._diag_dir,
                "latest_rollout_validation.json",
            ),
            record,
        )

    def _write_rollout_error(self, batch_idx, error):
        if not self._diag_main_rank or not self._diag_enabled:
            return
        os.makedirs(self._diag_dir, exist_ok=True)
        with open(
            os.path.join(self._diag_dir, "rollout_errors.log"),
            "a", encoding="utf-8",
        ) as f:
            f.write(
                f"{datetime.now().isoformat(timespec='seconds')} "
                f"epoch={getattr(self, 'epoch', -1)} "
                f"step={getattr(self, 'global_step', -1)} "
                f"validation_batch={batch_idx} "
                f"error={repr(error)}\n"
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
        capture_diagnostics = self._should_write(val, batch_idx) or self._should_probe_grad(val)
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

        # Fixed-subset real rollout.  No TensorBoard/all-rank logging is done
        # here because only rank 0 executes the extra sample() call; JSON output
        # avoids DDP collective mismatches.
        if self._should_rollout_probe(val, batch_idx):
            try:
                rollout_metrics, rollout_seed = (
                    self._run_inmemory_rollout_probe(
                        raw_model, batch, batch_idx
                    )
                )
                self._write_rollout_record(
                    rollout_metrics, batch_idx, rollout_seed
                )
            except Exception as rollout_error:
                # Diagnostic failure is fail-soft and cannot terminate training.
                self._write_rollout_error(
                    batch_idx, rollout_error
                )

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
