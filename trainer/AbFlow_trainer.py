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
from .abs_trainer import Trainer


def _env_int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else int(default)


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
        self._diag_file_interval = (
            max(1, int(getattr(config, "step_per_epoch", 52)))
            if requested_file_interval <= 0
            else max(1, requested_file_interval)
        )
        self._diag_valid_interval = max(1, _env_int(
            "ABFLOW_DIAGNOSTIC_VALID_INTERVAL", 1
        ))
        requested_grad_interval = _env_int("ABFLOW_GRAD_DIAGNOSTIC_INTERVAL", 0)
        self._diag_grad_interval = (
            max(1, int(getattr(config, "step_per_epoch", 52)))
            if requested_grad_interval <= 0
            else max(1, requested_grad_interval)
        )
        self._diag_enabled = _env_flag("ABFLOW_DIAGNOSTIC_FILE", True)
        self._grad_diag_enabled = _env_flag(
            "ABFLOW_GRAD_CONFLICT_DIAGNOSTICS", False
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
                },
                "intervals": {
                    "train_steps": self._diag_file_interval,
                    "validation_batches": self._diag_valid_interval,
                    "gradient_probe_steps": self._diag_grad_interval,
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
        return (
            (not val)
            and self._grad_diag_enabled
            and int(self.global_step) % self._diag_grad_interval == 0
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
