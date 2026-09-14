#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_mean

from data.pdb_utils import VOCAB
from utils.nn_utils import (
    SeparatedAminoAcidFeature, ProteinFeature, GMEdgeConstructor,
    SeperatedCoordNormalizer, _knn_edges, get_timestep_embedding,
    _abflow_ca_fill_observed_mask,
)
from evaluation.rmsd import kabsch_torch

from ..modules.am_enc import AMEncoder
from ..modules.am_egnn import AMEGNN
from .abflow_conditional_matcher import AbFlowConditionalMatcher
from .abflow_components import (
    AbFlowR3Matcher, NativeTrunk, design_region_smooth_lddt_loss,
)
from configs import normalize_regions

class AbFlowModel(nn.Module):


    def __init__(self, embed_size, hidden_size, n_channel, num_classes, num_verts,
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6,
                 n_layers=3, iter_round=3, dropout=0.1,
                 pep_seq=True, pep_struct=True, struct_only=False,
                 backbone_only=False, fix_channel_weights=False, pred_edge_dist=True,
                 keep_memory=True, cdr_type='H3', paratope='H3', relative_position=False,
                 model_config=None, loss_config=None):
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.bind_dist_cutoff = bind_dist_cutoff
        self.k_neighbors = k_neighbors
        self.round = iter_round
        self.pep_seq = pep_seq
        self.pep_struct = pep_struct
        self.struct_only = struct_only
        self.backbone_only = backbone_only
        self.fix_channel_weights = fix_channel_weights
        self.pred_edge_dist = pred_edge_dist
        self.keep_memory = keep_memory
        self.n_channel = 4 if backbone_only else n_channel
        self.cdr_type = list(normalize_regions(cdr_type)) or cdr_type
        self.paratope = list(normalize_regions(paratope)) or paratope
        self.eps = 1e-8
        model_config = model_config or {}
        loss_config = loss_config or {}
        r05_config = model_config.get("r05", {})
        source_config = r05_config.get("source", {})
        flow_config = r05_config.get("flow", {})
        r3_config = r05_config.get("r3", {})
        representation_config = model_config.get("representation", {}).get("single_pair", {})
        pair_coord_config = representation_config.get("pair_coordinate", {})
        self.pair_coord_mode = str(
            pair_coord_config.get("mode", "bounded_residual")
        ).strip().lower()
        self.pair_coord_delta_bound = float(
            pair_coord_config.get("delta_bound", 1.0)
        )

        # V219 formal controller: preserve Pair as a first-class EGNN edge condition
        # and restore the original R05 raw Cartesian relative-vector operator.
        # Stability is introduced only at the representation->action boundary by
        # LayerNorm before the coordinate scalar head; the scalar remains
        # unsaturated and no clipping/trust-radius/manual movement scale is used.
        coord_controller = representation_config.get("coordinate_controller", {})
        self.coord_controller_mode = str(
            coord_controller.get("mode", "legacy_unbounded")
        ).strip().lower()
        _coord_controller_modes = {
            "legacy_unbounded", "egnn_tanh", "egnn_tanh_normalized",
            "egnn_unit_direction", "egnn_prenorm_raw",
        }
        if self.coord_controller_mode not in _coord_controller_modes:
            raise ValueError(
                "model.representation.single_pair.coordinate_controller.mode "
                f"must be one of {sorted(_coord_controller_modes)}, got "
                f"{self.coord_controller_mode!r}."
            )
        self.coord_tanh = self.coord_controller_mode in {"egnn_tanh", "egnn_tanh_normalized"}
        self.coord_normalize = self.coord_controller_mode in {
            "egnn_tanh_normalized", "egnn_unit_direction"
        }
        self.coord_prenorm = self.coord_controller_mode == "egnn_prenorm_raw"

        # V223: close the recurrent H3 coordinate semantics around ONE clean
        # endpoint.  The local Score--Flow carrier and the full-atom endpoint
        # remain two analytic parameterizations of that same endpoint instead
        # of two independently drifting geometry authorities.
        state_cfg = representation_config.get("coordinate_state", {})
        self.coordinate_state_mode = str(
            state_cfg.get("mode", "legacy_dual") or "legacy_dual"
        ).strip().lower()
        _state_modes = {"legacy_dual", "scoreflow_endpoint_fused", "scoreflow_single_endpoint"}
        if self.coordinate_state_mode not in _state_modes:
            raise ValueError(
                "model.representation.single_pair.coordinate_state.mode must be one of "
                f"{sorted(_state_modes)}, got {self.coordinate_state_mode!r}."
            )
        self.generated_region_only_loss = bool(
            loss_config.get("generated_region_only", False)
        )
        self._last_state_closure_audit = {}

        if self.coord_controller_mode in {"egnn_unit_direction", "egnn_prenorm_raw"} and self.pair_coord_mode != "direct_shared":
            raise ValueError(
                f"{self.coord_controller_mode} requires pair_coordinate.mode='direct_shared': "
                "Pair must condition the actual EGNN edge message used by both node and coordinate updates."
            )

        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size, relative_position=relative_position,
            edge_constructor=GMEdgeConstructor, fix_atom_weights=fix_channel_weights,
            backbone_only=backbone_only)
        self.protein_feature = ProteinFeature(backbone_only=backbone_only)

        if keep_memory:
            self.memory_ffn = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, hidden_size),
                nn.SiLU(), nn.Linear(hidden_size, embed_size))

        if pred_edge_dist:
            if keep_memory:
                self.edge_H_ffn = nn.Sequential(
                    nn.SiLU(), nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(), nn.Linear(hidden_size, hidden_size))
            self.edge_dist_ffn = nn.Sequential(
                nn.SiLU(), nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(), nn.Linear(hidden_size, 1))
            self.init_gnn = AMEGNN(
                embed_size, hidden_size, hidden_size, self.n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, n_layers=n_layers, residual=True,
                dropout=dropout, dense=False)

        if struct_only:
            self.prmsd_ffn = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, hidden_size),
                nn.SiLU(), nn.Linear(hidden_size, 1))
        else:
            self.ffn_residue = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, hidden_size),
                nn.SiLU(), nn.Linear(hidden_size, num_classes))

        distogram_config = loss_config.get("distogram", {})
        smooth_lddt_config = loss_config.get("smooth_lddt", {})
        self.loss_distogram_weight = float(distogram_config.get("weight", 0.0))
        self.distogram_pair_scope = str(
            distogram_config.get("pair_scope", "all_resolved") or "all_resolved"
        ).strip().lower()
        _distogram_scopes = {"all_resolved", "design_anchored", "generation_anchored"}
        if self.distogram_pair_scope not in _distogram_scopes:
            raise ValueError(
                "loss.distogram.pair_scope must be one of "
                f"{sorted(_distogram_scopes)}, got {self.distogram_pair_scope!r}."
            )
        self.loss_smooth_lddt_weight = float(
            smooth_lddt_config.get("weight", 0.0)
        )
        coarse_anchor_config = loss_config.get("coarse_anchor_distance", {})
        self.loss_coarse_anchor_weight = float(
            coarse_anchor_config.get("weight", 0.0)
        )
        self.coarse_anchor_cutoff_A = float(
            coarse_anchor_config.get("cutoff_A", distogram_config.get("max_bin", 21.6875))
        )
        self.coarse_anchor_relation_balance = str(
            coarse_anchor_config.get("relation_balance", "DF_DA_equal") or "DF_DA_equal"
        ).strip()
        if self.coarse_anchor_relation_balance != "DF_DA_equal":
            raise ValueError(
                "loss.coarse_anchor_distance.relation_balance must be 'DF_DA_equal'"
            )
        self.smooth_lddt_cutoff = float(
            smooth_lddt_config.get("cutoff", 15.0)
        )
        # V213: smooth-lDDT must supervise the coordinate object whose semantics
        # match the intended experiment.  The historical/default target keeps
        # backward compatibility with R33.  The aligned target converts the
        # sampler carrier into its implied clean endpoint before computing lDDT.
        self.smooth_lddt_prediction_source = str(
            smooth_lddt_config.get(
                "prediction_source",
                smooth_lddt_config.get("target", "pred_design_endpoint"),
            )
            or "pred_design_endpoint"
        ).strip().lower()
        _smooth_lddt_sources = {
            "pred_design_endpoint",
            "carrier_implied_endpoint",
            "terminal_fused_endpoint",
        }
        if self.smooth_lddt_prediction_source not in _smooth_lddt_sources:
            raise ValueError(
                "loss.smooth_lddt.prediction_source must be one of "
                f"{sorted(_smooth_lddt_sources)}, got "
                f"{self.smooth_lddt_prediction_source!r}."
            )
        self.distogram_enabled = self.loss_distogram_weight > 0.0
        self.smooth_lddt_enabled = self.loss_smooth_lddt_weight > 0.0
        self.coarse_anchor_enabled = self.loss_coarse_anchor_weight > 0.0
        self.relational_trunk_enabled = True

        # Representation initialization is isolated from the R05 RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(
                int(representation_config["init_seed"])
            )
            self.aa_feature.configure_single_pair(representation_config)
            self.native_trunk = NativeTrunk(
                representation_config=representation_config,
                distogram_config=distogram_config,
                forward_seed=int(representation_config["forward_seed"]),
            )

        self.gnn = AMEncoder(
            embed_size, hidden_size, hidden_size, self.n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=self.native_trunk.pair_dim,
            in_single_nf=self.native_trunk.single_dim,
            num_verts=num_verts, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False,
            pair_coord_mode=self.pair_coord_mode,
            pair_coord_delta_bound=self.pair_coord_delta_bound,
            coord_tanh=self.coord_tanh, coord_normalize=self.coord_normalize,
            coord_prenorm=self.coord_prenorm)

        self.normalizer = SeperatedCoordNormalizer()
        self.batch_constants = {}

        self.flow_matcher = AbFlowConditionalMatcher(
            min_sigma=float(flow_config["min_sigma"]),
            eps=self.eps,
        )
        self.r3_matcher = AbFlowR3Matcher(
            transport_fraction=float(r3_config["transport_fraction"]),
            path_min_sigma=float(r3_config["path_min_sigma"]),
            eps=self.eps,
        )

        self.flow_coordinate_scaling = float(r3_config["coordinate_scaling"])
        self.r3_fixed_g_scaled = float(r3_config["fixed_g_scaled"])
        self.f01_hybrid_t_min = float(flow_config["hybrid_t_min"])
        self.proposal_adapter_start_round = int(
            source_config["proposal_adapter_start_round"]
        )

        self.loss_sequence_weight = float(loss_config["sequence"])
        self.loss_structure_weight = float(loss_config["structure"])
        self.loss_interface_weight = float(loss_config["interface"])
        self.loss_edge_weight = float(loss_config["edge"])

        self.flow_time_mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size), nn.SiLU(),
            nn.Linear(embed_size, embed_size))

        self.coord_pep_condition_dim = 6
        self.coord_pep_condition_adapter = nn.Sequential(
            nn.Linear(embed_size + self.coord_pep_condition_dim, embed_size),
            nn.SiLU(), nn.Linear(embed_size, embed_size))
        nn.init.zeros_(self.coord_pep_condition_adapter[-1].weight)
        nn.init.zeros_(self.coord_pep_condition_adapter[-1].bias)

        self.seq_pep_condition_embedding = nn.Embedding(num_classes, embed_size)
        self.seq_pep_condition_adapter = nn.Sequential(
            nn.Linear(2 * embed_size, embed_size), nn.SiLU(),
            nn.Linear(embed_size, embed_size))
        nn.init.zeros_(self.seq_pep_condition_adapter[-1].weight)
        nn.init.zeros_(self.seq_pep_condition_adapter[-1].bias)

        # Lightweight trainer diagnostics.  These tensors are observational
        # only and are populated only on explicitly requested diagnostic steps.
        self.last_scorefm_losses = {}
        self.last_abflow_diagnostics = {}
        self._last_trunk_state = {}
        self._last_message_diagnostics = {}
        self._last_round_egnn_diagnostics = []
        self.grad_conflict_diagnostics = False
        self._diagnostic_capture = False
        self._diagnostic_validation_mode = False

        # Geometry forensics are observational only.  These are plain Python
        # attributes (not Parameters / buffers), so strict resume state_dict
        # compatibility is unchanged.
        _gf = str(os.environ.get("ABFLOW_GEOMETRY_FORENSICS", "off") or "off").strip().lower()
        self.geometry_forensics_enabled = _gf in {"1", "true", "yes", "y", "on"}
        _sf = str(os.environ.get("ABFLOW_SAMPLE_FORENSICS", "off") or "off").strip().lower()
        self.sample_forensics_enabled = _sf in {"1", "true", "yes", "y", "on"}
        self.sample_forensics_threshold_A = float(
            os.environ.get("ABFLOW_SAMPLE_FORENSICS_THRESHOLD_A", "500") or 500.0
        )
        self.last_geometry_forensics = {}
        self._coord_audit_train_call = 0
        self.coord_audit_interval = max(1, int(os.environ.get('ABFLOW_COORD_AUDIT_INTERVAL', '20') or 20))
        self.coord_audit_first_steps = max(0, int(os.environ.get('ABFLOW_COORD_AUDIT_FIRST_STEPS', '5') or 5))
        self._sample_forensic_context = {}
        self._sample_forensic_records = []
        self._sample_forensic_alerted = False

        # Plain runtime metadata; not part of state_dict.
        self.geometry_coupling_contract = {
            'pair_coord_mode': self.pair_coord_mode,
            'pair_coord_delta_bound': self.pair_coord_delta_bound,
            'coord_controller_mode': self.coord_controller_mode,
            'coord_tanh': self.coord_tanh,
            'coord_normalize': self.coord_normalize,
        }

    @staticmethod
    def _coord_diag_scalar(diag, key):
        value = (diag or {}).get(key)
        if value is None:
            return float('nan')
        try:
            return float(value.detach().float().item()) if torch.is_tensor(value) else float(value)
        except Exception:
            return float('nan')

    def _maybe_log_coordinate_controller_audit(self, round_egnn_diagnostics):
        """Compact V221 controller audit (diagnostic-only).

        Scientific controller is unchanged from V219.  RMS/max pairs distinguish
        distribution-wide gain drift from sparse extreme actions.
        """
        if not self.training or not self.geometry_forensics_enabled:
            return
        call = int(self._coord_audit_train_call)
        self._coord_audit_train_call += 1
        if not (call < self.coord_audit_first_steps or call % self.coord_audit_interval == 0):
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        rows = []
        for rec in (round_egnn_diagnostics or []):
            d = rec.get('coord', {}) or {}
            rows.append((
                int(rec.get('round_idx', len(rows))),
                self._coord_diag_scalar(d, 'coord_state_coeff_raw_rms_max'),
                self._coord_diag_scalar(d, 'coord_state_coeff_raw_absmax_max'),
                self._coord_diag_scalar(d, 'coord_pair_delta_raw_rms_max'),
                self._coord_diag_scalar(d, 'coord_update_rms_max'),
                self._coord_diag_scalar(d, 'coord_update_absmax_max'),
            ))
        fmt = lambda v: 'nan' if not math.isfinite(v) else f'{v:.6g}'
        payload = ';'.join(
            f'r{r}:alpha={fmt(ar)}/{fmt(am)} pair_rms={fmt(pr)} '
            f'dx={fmt(xr)}/{fmt(xm)}'
            for r, ar, am, pr, xr, xm in rows
        )
        print(
            '[ControllerAudit] '
            f'train_call={call} mode={self.coord_controller_mode} '
            f'fields=rms/max {payload}',
            flush=True,
        )

    def set_sample_forensic_context(self, logical_batch_id=None, global_indices=None, names=None):
        """Attach evaluation identity to the next ``sample`` call.

        Runtime-only metadata: it is never inserted into ``state_dict`` and has
        no effect on the model/sampler computation.
        """
        self._sample_forensic_context = {
            'logical_batch_id': None if logical_batch_id is None else int(logical_batch_id),
            'global_indices': [] if global_indices is None else [int(v) for v in global_indices],
            'names': [] if names is None else [str(v) for v in names],
        }

    def reset_sample_forensics(self):
        self._sample_forensic_records = []
        self._sample_forensic_alerted = False

    def consume_sample_forensics(self):
        records = list(self._sample_forensic_records)
        self._sample_forensic_records = []
        self._sample_forensic_alerted = False
        return records

    @staticmethod
    def _forensic_graph_stats(value, graph_id, n_graph):
        """Return JSON-safe per-graph statistics without changing ``value``."""
        value = value.detach().float()
        graph_id = graph_id.detach().long()
        rows = []
        for gid in range(int(n_graph)):
            part = value[graph_id == gid]
            if part.numel() == 0:
                rows.append({
                    'finite': True, 'absmax': 0.0, 'rms': 0.0,
                    'min': 0.0, 'max': 0.0,
                })
                continue
            finite = bool(torch.isfinite(part).all().item())
            if finite:
                part32 = part.float()
                rows.append({
                    'finite': True,
                    'absmax': float(part32.abs().amax().item()),
                    'rms': float(part32.square().mean().sqrt().item()),
                    'min': float(part32.amin().item()),
                    'max': float(part32.amax().item()),
                })
            else:
                finite_part = part[torch.isfinite(part)]
                rows.append({
                    'finite': False,
                    'absmax': float(finite_part.abs().amax().item()) if finite_part.numel() else float('nan'),
                    'rms': float(finite_part.square().mean().sqrt().item()) if finite_part.numel() else float('nan'),
                    'min': float(finite_part.amin().item()) if finite_part.numel() else float('nan'),
                    'max': float(finite_part.amax().item()) if finite_part.numel() else float('nan'),
                })
        return rows

    @staticmethod
    def _diag_float(diag, key):
        value = (diag or {}).get(key)
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            return float(value.detach().float().item())
        try:
            return float(value)
        except Exception:
            return None

    def _sample_forensic_identity(self, local_graph):
        ctx = self._sample_forensic_context or {}
        global_indices = ctx.get('global_indices', []) or []
        names = ctx.get('names', []) or []
        return {
            'logical_batch_id': ctx.get('logical_batch_id'),
            'local_graph': int(local_graph),
            'global_index': (
                int(global_indices[local_graph])
                if local_graph < len(global_indices) else None
            ),
            'name': str(names[local_graph]) if local_graph < len(names) else '',
        }

    def _append_sample_forensic_record(self, row):
        if not self.sample_forensics_enabled:
            return
        self._sample_forensic_records.append(dict(row))
        if self._sample_forensic_alerted:
            return

        bad_field = None
        bad_value = None
        ordered = (
            'xt_absmax_A', 'carrier_absmax_A', 'implied_x1_absmax_A',
            'xnext_absmax_A', 'pred_final_absmax_A',
            'gen_pre_align_absmax_A', 'gen_post_align_absmax_A',
        )
        for field in ordered:
            value = row.get(field)
            if value is None:
                continue
            try:
                fv = float(value)
            except Exception:
                continue
            if (not math.isfinite(fv)) or abs(fv) > float(self.sample_forensics_threshold_A):
                bad_field, bad_value = field, fv
                break
        if row.get('finite') is False and bad_field is None:
            bad_field, bad_value = 'nonfinite', float('nan')

        if bad_field is not None:
            self._sample_forensic_alerted = True
            print(
                '[SampleGeometryOutlier] '
                f"logical_batch_id={row.get('logical_batch_id')} "
                f"global_index={row.get('global_index')} name={row.get('name', '')!r} "
                f"stage={row.get('stage')} step={row.get('step')} "
                f"t={row.get('t')} field={bad_field} value_A={bad_value:.6g} "
                f"threshold_A={float(self.sample_forensics_threshold_A):.6g}",
                flush=True,
            )

            # Failure-only trajectory: all preceding sampler steps for exactly
            # this sample.  No extra forward/RNG call is introduced.
            gid = row.get('global_index')
            name = row.get('name', '')
            trace = [
                r for r in self._sample_forensic_records
                if r.get('global_index') == gid and r.get('name', '') == name
            ]
            step_rows = [r for r in trace if r.get('stage') == 'sampling_step']
            if step_rows:
                step_rows.sort(key=lambda r: int(r.get('step', -1)))
                def arr(key, nd=3):
                    vals = []
                    for rr in step_rows:
                        vv = rr.get(key)
                        try:
                            fv = float(vv)
                            vals.append('nan' if not math.isfinite(fv) else f'{fv:.{nd}g}')
                        except Exception:
                            vals.append('nan')
                    return '(' + ','.join(vals) + ')'
                print(
                    '[SampleOutlierTrajectory] '
                    f'global_index={gid} name={name!r} '
                    f't={arr("t", 3)} '
                    f'xt={arr("xt_absmax_A", 4)} '
                    f'carrier={arr("carrier_absmax_A", 4)} '
                    f'x1={arr("implied_x1_absmax_A", 4)} '
                    f'xnext={arr("xnext_absmax_A", 4)} '
                    f'step_rms={arr("step_delta_rms_A", 4)}',
                    flush=True,
                )

    @staticmethod
    def _per_graph_coord_rms(pred, target, atom_mask, graph_id, n_graph):
        """Vectorized coordinate RMS (Angstrom) for training diagnostics."""
        pred32 = pred.detach().float()
        target32 = target.detach().float()
        atom_mask = atom_mask.detach().bool()
        graph_id = graph_id.detach().long()
        per_res_ss = ((pred32 - target32).square().sum(dim=-1) * atom_mask.float()).sum(dim=-1)
        per_res_count = atom_mask.float().sum(dim=-1) * 3.0
        ss = pred32.new_zeros(int(n_graph))
        count = pred32.new_zeros(int(n_graph))
        ss.scatter_add_(0, graph_id, per_res_ss)
        count.scatter_add_(0, graph_id, per_res_count)
        return torch.sqrt(ss / count.clamp_min(1.0))

    @staticmethod
    def _per_graph_absmax(value, graph_id, n_graph):
        value = value.detach().float()
        graph_id = graph_id.detach().long()
        out = []
        for gid in range(int(n_graph)):
            part = value[graph_id == gid]
            out.append(part.abs().amax() if part.numel() else value.new_zeros(()))
        return torch.stack(out) if out else value.new_zeros((0,))

    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return (X, S)

    def _valid_proposal_backbone(self, coords):
        """Return a per-residue validity mask for proposal N/CA/C coordinates.

        The project runtime is PyTorch 1.11, whose ``Tensor.all`` accepts only
        one reduction dimension.  Flattening the N/CA/C xyz block keeps the
        original predicate exactly while remaining runtime-compatible:

            all coordinates finite AND total backbone magnitude > eps.

        This helper is the single authority for PCS-RC proposal-coordinate
        validity across initialization and recurrent proposal replacement.
        """
        n_bb = min(3, int(coords.shape[1]))
        bb_flat = coords[:, :n_bb].reshape(coords.shape[0], -1)
        return (
            torch.isfinite(bb_flat).all(dim=-1)
            & (bb_flat.abs().sum(dim=-1) > self.eps)
        )


    def replace_pep(self, X, S, paratope_mask, X_pep, S_pep,
                    replace_seq=True, replace_struct=True):
        """Build the recurrent PCS-RC proposal context."""
        if replace_seq and self.pep_seq and S_pep is not None and S_pep.numel() == int(paratope_mask.sum()):
            pep_S = S_pep.to(device=S.device, dtype=torch.long)
            valid = (pep_S >= 0) & (pep_S < self.num_classes)
            local = S[paratope_mask].clone()
            S[paratope_mask] = torch.where(valid, pep_S, local)
        if replace_struct and self.pep_struct and X_pep is not None and X_pep.shape == X[paratope_mask].shape:
            pep_X = X_pep.to(device=X.device, dtype=X.dtype)
            valid = self._valid_proposal_backbone(pep_X)
            local = X[paratope_mask].clone()
            X[paratope_mask] = torch.where(valid[:, None, None], pep_X, local)
        return X, S


    def _condition_initial_interface(self, interface_X, interface_S, X_pep, S_pep):
        """Use valid proposal residues as the R05 source state."""
        if self.pep_struct and X_pep is not None and X_pep.shape == interface_X.shape:
            pep_X = X_pep.to(device=interface_X.device, dtype=interface_X.dtype)
            valid = self._valid_proposal_backbone(pep_X)
            interface_X = torch.where(valid[:, None, None], pep_X, interface_X)
        if not self.struct_only and self.pep_seq and S_pep is not None and S_pep.shape == interface_S.shape:
            pep_S = S_pep.to(device=interface_S.device, dtype=torch.long)
            valid = (pep_S >= 0) & (pep_S < self.num_classes)
            interface_S = torch.where(valid, pep_S, interface_S)
        return interface_X, interface_S


    def _sample_categorical_path(self, clean_S, base_S, t_graph,
                                 interface_batch_id, corrupt_mask=None):
        """Sample the linear categorical source-to-native path."""
        t = torch.as_tensor(t_graph, device=clean_S.device, dtype=torch.float32)
        prob = t.reshape(1).expand_as(clean_S) if t.numel() == 1 else t[interface_batch_id]
        if self.training:
            u = torch.rand(clean_S.shape, device=clean_S.device)
        else:
            idx = torch.arange(clean_S.numel(), device=clean_S.device, dtype=torch.float32)
            u = torch.frac(torch.sin((idx + 1.0) * 12.9898) * 43758.5453).abs()
        sampled = torch.where(u < prob.clamp(0.0, 1.0), clean_S, base_S).long()
        if corrupt_mask is None:
            return sampled
        return torch.where(corrupt_mask.to(clean_S.device).bool(), sampled, clean_S).long()

    def align_epi_ab(self, local_inter_edges, local_is_ab):
        row, col = local_inter_edges
        row_is_ab, col_is_ab = local_is_ab[row], local_is_ab[col]
        swap = row_is_ab & ~col_is_ab
        aligned = local_inter_edges.clone()
        aligned[0, swap], aligned[1, swap] = col[swap], row[swap]
        epi_index = torch.nonzero(~local_is_ab, as_tuple=False).reshape(-1)
        return aligned, epi_index


    def _sample_flow_times(self, batch_size, device, dtype=torch.float32):
        """R05 uses per-complex uniform time; validation is deterministic."""
        n = int(batch_size)
        if self.training:
            return torch.rand(n, device=device, dtype=dtype)
        if n == 1:
            return torch.full((1,), 0.5, device=device, dtype=dtype)
        return (torch.arange(n, device=device, dtype=dtype) + 0.5) / float(n)

    def _time_for_interface(self, t_graph, interface_batch_id, ref_tensor):
        if t_graph is None:
            return None
        t = torch.as_tensor(t_graph, device=ref_tensor.device, dtype=ref_tensor.dtype)
        if t.numel() == 1:
            return t.reshape(1, 1, 1)
        return t[interface_batch_id].reshape(-1, 1, 1)

    def _flow_time_embedding_for_residues(self, flow_t, batch_id, H_0):
        if flow_t is None:
            return None
        t = torch.as_tensor(flow_t, device=H_0.device, dtype=H_0.dtype)
        if t.numel() == 1:
            n_graph = int(batch_id.max()) + 1 if batch_id.numel() else 1
            t = t.reshape(1).expand(n_graph)
        emb = get_timestep_embedding(t, H_0.shape[-1]).to(H_0)
        return self.flow_time_mlp(emb)[batch_id]

    def _build_coord_pep_condition_for_residues(
            self, pep_X_model, interface_X, paratope_mask, pep_coord_valid=None):
        if pep_X_model is None or pep_X_model.shape != interface_X.shape:
            return None, None
        if interface_X.shape[1] < 3 or int(paratope_mask.sum()) != interface_X.shape[0]:
            return None, None

        n_int = interface_X.shape[0]
        if pep_coord_valid is None:
            valid = torch.ones(n_int, device=interface_X.device, dtype=torch.bool)
        else:
            valid = torch.as_tensor(pep_coord_valid, device=interface_X.device, dtype=torch.bool).reshape(-1)
            if valid.numel() != n_int:
                return None, None

        delta = pep_X_model - interface_X
        ca_delta = delta[:, 1]
        ca_dist = torch.norm(ca_delta, dim=-1, keepdim=True)

        n_vec = pep_X_model[:, 0] - pep_X_model[:, 1]
        c_vec = pep_X_model[:, 2] - pep_X_model[:, 1]
        c_norm = torch.norm(c_vec, dim=-1, keepdim=True)
        e1 = F.normalize(c_vec, dim=-1, eps=self.eps)
        n_orth = n_vec - (n_vec * e1).sum(-1, keepdim=True) * e1
        n_orth_norm = torch.norm(n_orth, dim=-1, keepdim=True)
        e2 = F.normalize(n_orth, dim=-1, eps=self.eps)
        e3 = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=self.eps)

        local_delta = torch.stack([
            (ca_delta * e1).sum(-1),
            (ca_delta * e2).sum(-1),
            (ca_delta * e3).sum(-1),
        ], dim=-1)
        local_delta = local_delta * (torch.log1p(ca_dist) / ca_dist.clamp_min(self.eps))
        frame_valid = (
            torch.isfinite(c_norm.squeeze(-1))
            & torch.isfinite(n_orth_norm.squeeze(-1))
            & (c_norm.squeeze(-1) > 1e-4)
            & (n_orth_norm.squeeze(-1) > 1e-4)
        )
        local_delta = torch.where(
            (valid & frame_valid)[:, None], local_delta, torch.zeros_like(local_delta))

        n_bb = min(4, interface_X.shape[1])
        bb_dist = torch.norm(delta[:, :n_bb], dim=-1)
        dist_feat = torch.cat([
            ca_dist,
            bb_dist.mean(-1, keepdim=True),
            torch.sqrt((bb_dist ** 2).mean(-1, keepdim=True) + self.eps),
        ], dim=-1)
        feat = torch.cat([local_delta, torch.log1p(dist_feat.clamp_min(0.0))], dim=-1)
        feat = torch.nan_to_num(feat) * valid[:, None].to(feat.dtype)

        full = interface_X.new_zeros((paratope_mask.shape[0], self.coord_pep_condition_dim))
        mask = torch.zeros(paratope_mask.shape[0], device=interface_X.device, dtype=torch.bool)
        full[paratope_mask], mask[paratope_mask] = feat, valid
        return full, mask


    def _build_seq_pep_condition_for_residues(self, S_pep, paratope_mask, ref_tensor):
        if S_pep is None or S_pep.numel() != int(paratope_mask.sum()):
            return None, None
        pep = S_pep.to(device=ref_tensor.device, dtype=torch.long).reshape(-1)
        valid = (pep >= 0) & (pep < self.num_classes)
        if not bool(valid.any()):
            return None, None
        idx = paratope_mask.nonzero(as_tuple=False).reshape(-1)[valid]
        tokens = torch.zeros(paratope_mask.shape[0], device=ref_tensor.device, dtype=torch.long)
        tokens = tokens.index_copy(0, idx, pep[valid])
        mask = torch.zeros_like(paratope_mask, dtype=torch.bool).index_fill(0, idx, True)
        emb = self.seq_pep_condition_embedding(tokens).to(ref_tensor.dtype)
        return emb * mask[:, None].to(emb.dtype), mask


    def message_passing(self, X, S, residue_pos, interface_X, surf, paratope_mask,
                        batch_id, memory_H=None, smooth_prob=None, smooth_mask=None,
                        flow_t=None, coord_pep_condition=None,
                        coord_pep_condition_mask=None, seq_pep_condition=None,
                        seq_pep_condition_mask=None, trunk_state=None):
        H_0, (ctx_edges, _), (atom_embeddings, atom_weights) = self.aa_feature(
            X, S, batch_id, self.k_neighbors, residue_pos,
            smooth_prob=smooth_prob, smooth_mask=smooth_mask)

        time_emb = self._flow_time_embedding_for_residues(flow_t, batch_id, H_0)
        if time_emb is not None:
            H_0 = H_0 + time_emb

        if coord_pep_condition is not None and coord_pep_condition_mask is not None:
            feat = coord_pep_condition.to(H_0)
            mask = coord_pep_condition_mask.to(H_0.device).bool()
            residual = self.coord_pep_condition_adapter(torch.cat([H_0, feat], dim=-1))
            H_0 = H_0 + residual * mask[:, None].to(H_0.dtype)
        else:
            H_0 = H_0 + 0.0 * sum(p.sum() for p in self.coord_pep_condition_adapter.parameters())

        if seq_pep_condition is not None and seq_pep_condition_mask is not None:
            feat = seq_pep_condition.to(H_0)
            mask = seq_pep_condition_mask.to(H_0.device).bool()
            residual = self.seq_pep_condition_adapter(torch.cat([H_0, feat], dim=-1))
            H_0 = H_0 + residual * mask[:, None].to(H_0.dtype)
        else:
            dummy = sum(p.sum() for p in self.seq_pep_condition_adapter.parameters())
            dummy = dummy + sum(p.sum() for p in self.seq_pep_condition_embedding.parameters())
            H_0 = H_0 + 0.0 * dummy

        if not self.keep_memory:
            memory_H = None
        if memory_H is not None:
            H_0 = H_0 + self.memory_ffn(memory_H)

        if self.pred_edge_dist:
            if memory_H is None:
                edge_H, dummy_X = self.init_gnn(
                    H_0, X, ctx_edges, channel_attr=atom_embeddings,
                    channel_weights=atom_weights)
                X = X + 0.0 * dummy_X
            else:
                edge_H = self.edge_H_ffn(memory_H)

        X = self.aa_feature.update_global_coordinates(X, S)
        local_mask = self.batch_constants['local_mask']
        local_is_ab = self.batch_constants['local_is_ab']
        local_batch_id = self.batch_constants['local_batch_id']
        local_X = X[local_mask].clone()
        local_X[local_is_ab] = interface_X

        local_ctx_edges = self.batch_constants['local_ctx_edges']
        local_inter_edges = self.batch_constants['local_inter_edges']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        offsets, max_n, gni2lni = self.batch_constants['local_edge_infos']
        edge_info = (offsets, local_batch_id, max_n, gni2lni)
        local_ctx_edges = _knn_edges(
            local_X, atom_pos, local_ctx_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors, edge_info)

        if self.pred_edge_dist:
            local_H = edge_H[local_mask]
            src, dst = local_H[local_inter_edges[0]], local_H[local_inter_edges[1]]
            p_edge_dist = (
                self.edge_dist_ffn(torch.cat([src, dst], dim=-1))
                + self.edge_dist_ffn(torch.cat([dst, src], dim=-1))).squeeze(-1)
        else:
            p_edge_dist = None

        local_inter_edges = _knn_edges(
            local_X, atom_pos, local_inter_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors, edge_info,
            given_dist=p_edge_dist)
        local_edges = torch.cat([local_ctx_edges, local_inter_edges], dim=1)
        surf_edges, epi_index = self.align_epi_ab(local_inter_edges, local_is_ab)

        trunk_single = trunk_state['single_global'].to(H_0)
        ctx_pair_attr = self.native_trunk.gather_pair(
            trunk_state['pair_dense'], ctx_edges,
            trunk_state['node_graph'], trunk_state['node_local']).to(H_0)
        local_global = torch.nonzero(local_mask, as_tuple=False).flatten()
        inter_global = local_global[local_edges]
        surf_global = local_global[surf_edges]
        inter_pair_attr = self.native_trunk.gather_pair(
            trunk_state['pair_dense'], inter_global,
            trunk_state['node_graph'], trunk_state['node_local']).to(H_0)
        surf_pair_attr = self.native_trunk.gather_pair(
            trunk_state['pair_dense'], surf_global,
            trunk_state['node_graph'], trunk_state['node_local']).to(H_0)

        capture_diag = bool(
            getattr(self, '_diagnostic_capture', False)
            or getattr(self, 'geometry_forensics_enabled', False)
            or getattr(self, 'sample_forensics_enabled', False)
        )
        H, pred_X, pred_local_X = self.gnn(
            H_0, X, ctx_edges, local_mask, local_X, surf, local_edges,
            paratope_mask, local_is_ab, surf_edges, epi_index,
            channel_attr=atom_embeddings, channel_weights=atom_weights,
            ctx_edge_attr=ctx_pair_attr, inter_edge_attr=inter_pair_attr,
            surf_edge_attr=surf_pair_attr, single_attr=trunk_single,
            capture_bridge_diagnostics=capture_diag)
        if capture_diag:
            def _rms(value):
                if value is None or value.numel() == 0:
                    return H_0.new_zeros(())
                return value.detach().float().square().mean().sqrt().to(H_0.dtype)
            self._last_message_diagnostics = {
                'abx_ctx_edge_attr_rms': _rms(ctx_pair_attr),
                'abx_inter_edge_attr_rms': _rms(inter_pair_attr),
                'abx_surf_edge_attr_rms': _rms(surf_pair_attr),
            }
            self._last_message_diagnostics.update(
                getattr(self.gnn, 'last_bridge_diagnostics', {}) or {}
            )
        pred_logits = None if self.struct_only else self.ffn_residue(H)
        return pred_logits, pred_X, pred_local_X[local_is_ab], H, p_edge_dist

    @torch.no_grad()
    def init_interface(self, X, S, paratope_mask, batch_id, init_noise=None):
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]
        init_local_X = torch.zeros_like(X[paratope_mask])
        init_local_X = init_local_X + ag_centers[batch_id[paratope_mask]].unsqueeze(1)
        noise = torch.randn_like(init_local_X) if init_noise is None else init_noise
        ca_noise = noise[:, 1]
        noise = noise / 10 + ca_noise.unsqueeze(1)
        noise[:, 1] = ca_noise
        init_local_X = init_local_X + noise
        init_local_S = torch.randint(0, self.num_classes, (paratope_mask.sum(),), device=X.device, dtype=torch.long)
        return (init_local_X, init_local_S)

    @torch.no_grad()
    def _prepare_batch_constants(self, S, paratope_mask, lengths):
        batch_id = torch.zeros_like(S)
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)
        self.batch_constants['batch_id'] = batch_id
        self.batch_constants['batch_size'] = torch.max(batch_id) + 1
        segment_ids = self.aa_feature._construct_segment_ids(S)
        self.batch_constants['segment_ids'] = segment_ids
        is_ag = segment_ids == self.aa_feature.ag_seg_id
        not_ag_global = S != self.aa_feature.boa_idx
        local_mask = torch.logical_or(paratope_mask, torch.logical_and(is_ag, not_ag_global))
        local_segment_ids = segment_ids[local_mask]
        local_is_ab = local_segment_ids != self.aa_feature.ag_seg_id
        local_batch_id = batch_id[local_mask]
        self.batch_constants['is_ag'] = is_ag
        self.batch_constants['local_mask'] = local_mask
        self.batch_constants['local_is_ab'] = local_is_ab
        self.batch_constants['local_batch_id'] = local_batch_id
        self.batch_constants['local_segment_ids'] = local_segment_ids
        (row, col), (offsets, max_n, gni2lni) = self.aa_feature.edge_constructor.get_batch_edges(local_batch_id)
        row_segment_ids, col_segment_ids = (local_segment_ids[row], local_segment_ids[col])
        row_is_ag = row_segment_ids == self.aa_feature.ag_seg_id
        col_is_ag = col_segment_ids == self.aa_feature.ag_seg_id
        is_inter = torch.logical_xor(row_is_ag, col_is_ag)
        is_ctx = torch.logical_not(is_inter)
        self.batch_constants['local_ctx_edges'] = torch.stack([row[is_ctx], col[is_ctx]])
        self.batch_constants['local_inter_edges'] = torch.stack([row[is_inter], col[is_inter]])
        self.batch_constants['local_edge_infos'] = (offsets, max_n, gni2lni)
        interface_batch_id = batch_id[paratope_mask]
        self.batch_constants['interface_batch_id'] = interface_batch_id

    def _clean_batch_constants(self):
        self.batch_constants = {}

    @torch.no_grad()
    def _get_inter_edge_dist(self, X, S):
        """Native minimum atom distance for interface edges."""
        local_mask = self.batch_constants['local_mask']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        src_dst = self.batch_constants['local_inter_edges'].T
        dist = X[local_mask][src_dst]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)
        dist = torch.norm(dist, dim=-1)
        pos_pad = atom_pos[src_dst] == self.aa_feature.atom_pos_pad_idx
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))
        dist = dist + pos_pad * 10000000000.0
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]
        return dist

    def _raw_interface_to_model_frame(self, interface_X, paratope_mask, batch_id):
        """Map raw paratope coordinates to the internal frame."""
        interface_batch_id = batch_id[paratope_mask]
        ag_centers = self.normalizer.ag_centers[interface_batch_id]
        return self.normalizer.normalize(interface_X - ag_centers.unsqueeze(1))

    def _interface_valid_graph_mask(self, interface_batch_id, n_graph, device):
        valid = torch.zeros(n_graph, device=device, dtype=torch.bool)
        if interface_batch_id.numel() > 0:
            valid[torch.unique(interface_batch_id)] = True
        return valid

    def _masked_residue_smooth_l1_per_graph(self, pred, target, atom_mask, interface_batch_id):
        """Per-complex masked coordinate SmoothL1."""
        if interface_batch_id.numel() == 0:
            return (pred.new_zeros(1), torch.zeros(1, device=pred.device, dtype=torch.bool))
        n_graph = int(interface_batch_id.max().item()) + 1
        atom_mask_f = atom_mask.to(pred.dtype)
        err = F.smooth_l1_loss(pred, target, reduction='none').sum(dim=-1)
        err = err * atom_mask_f
        per_res = err.sum(dim=-1) / (3.0 * atom_mask_f.sum(dim=-1).clamp_min(1.0))
        per_graph = scatter_mean(per_res, interface_batch_id, dim=0, dim_size=n_graph)
        valid_graph = self._interface_valid_graph_mask(interface_batch_id, n_graph, pred.device)
        return (per_graph, valid_graph)


    def _deterministic_standard_normal(self, shape, device, dtype):
        n = math.prod(int(d) for d in shape)
        idx = torch.arange(n, device=device, dtype=torch.float32) + 1.0
        u = torch.frac(torch.sin(idx * 12.9898 + 78.233) * 43758.5453).abs().clamp(1e-4, 1.0 - 1e-4)
        return (math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)).reshape(*shape).to(dtype)

    def _r05_primary_path(self, source_X0, target_X1, t_graph, t_int,
                          interface_batch_id):
        """Residue-level fixed-g FoldFlow R3 path used by R05."""
        mu = self.r3_matcher.linear_mean(source_X0, target_X1, t_int)
        if interface_batch_id.numel() == 0:
            return mu, {'r3_sigma_mean': target_X1.new_zeros(())}
        n_graph = int(interface_batch_id.max()) + 1
        t = torch.as_tensor(t_graph, device=target_X1.device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(n_graph)
        g_raw = self.r3_matcher.foldflow_scaled_g_to_raw(
            g_scaled=self.r3_fixed_g_scaled,
            coordinate_scaling=self.flow_coordinate_scaling)
        g = torch.full_like(t, g_raw)
        sigma = self.r3_matcher.sigma_t(t, g)
        n_res = int(interface_batch_id.numel())
        if self.training:
            eps = torch.randn((n_res, 3), device=target_X1.device, dtype=torch.float32)
        else:
            eps = self._deterministic_standard_normal(
                (n_res, 3), target_X1.device, torch.float32)
        shift = (sigma[interface_batch_id, None] * eps).to(mu.dtype)
        Xt = mu + shift[:, None, :]
        return Xt, {'r3_sigma_mean': sigma.mean().to(target_X1.dtype)}

    @torch.no_grad()
    def _r05_coordinate_target(self, Xt, source_X0, target_X1, t_int):
        """U02 target: endpoint below t=0.2, canonical carrier above it."""
        canonical = self.r3_matcher.canonical_carrier_target_gfree(
            Xt, source_X0, target_X1, t_int,
            boundary_eps=self.f01_hybrid_t_min)
        t = torch.as_tensor(t_int, device=Xt.device, dtype=Xt.dtype)
        while t.dim() < Xt.dim():
            t = t.unsqueeze(-1)
        active = t >= self.f01_hybrid_t_min
        target = torch.where(active, canonical, target_X1)
        return target, {'r3_canonical_active_rate': active.reshape(active.shape[0], -1).any(-1).float().mean()}

    def _coordinate_training_objective(self, pred, target, atom_mask,
                                       interface_batch_id):
        per_graph, valid = self._masked_residue_smooth_l1_per_graph(
            pred, target, atom_mask, interface_batch_id)
        loss = per_graph[valid].mean() if bool(valid.any()) else pred.new_zeros(())
        return loss

    @staticmethod
    def _rms_tensor(x):
        return torch.sqrt(x.float().square().mean().clamp_min(0.0))

    def _differentiable_design_kabsch(self, moving, target):
        """Rigidly place ``moving`` onto ``target`` without changing shape.

        Both inputs are [M,3] in the same centered/model frame.  The operation
        is differentiable through the SVD and contains no clipping, learned
        scale, trust radius or coordinate normalization heuristic.
        """
        if moving.shape != target.shape or moving.dim() != 2 or moving.shape[-1] != 3:
            raise RuntimeError(
                f"V223 Kabsch shape mismatch: moving={tuple(moving.shape)} "
                f"target={tuple(target.shape)}"
            )
        if moving.shape[0] < 3:
            raise RuntimeError(
                f"V223 Kabsch requires >=3 backbone points, got {moving.shape[0]}"
            )
        out_dtype = moving.dtype
        # torch<=1.11 AMP can route linalg ops through low precision.  The
        # explicit FP32 island changes only numerical linear algebra precision.
        with torch.cuda.amp.autocast(enabled=False):
            p = moving.float()
            q = target.float()
            pc = p.mean(dim=0, keepdim=True)
            qc = q.mean(dim=0, keepdim=True)
            p0, q0 = p - pc, q - qc
            cov = p0.transpose(0, 1).matmul(q0)
            U, _, Vh = torch.linalg.svd(cov, full_matrices=False)
            V = Vh.transpose(-2, -1)
            raw_R = V.matmul(U.transpose(-2, -1))
            sign = torch.where(
                torch.det(raw_R) < 0,
                raw_R.new_tensor(-1.0), raw_R.new_tensor(1.0),
            )
            diag = torch.stack([
                raw_R.new_tensor(1.0), raw_R.new_tensor(1.0), sign
            ])
            R = V.matmul(torch.diag(diag)).matmul(U.transpose(-2, -1))
            trans = qc.squeeze(0) - pc.squeeze(0).matmul(R.transpose(0, 1))
            aligned = p.matmul(R.transpose(0, 1)) + trans
        return aligned.to(out_dtype), R.to(out_dtype), trans.to(out_dtype)

    def _close_scoreflow_endpoint_state(
            self, pred_X, carrier, paratope_mask, interface_batch_id,
            flow_xt_model, flow_x0_model, flow_t):
        """Project the two R05 coordinate parameterizations onto one endpoint.

        The GNN carrier is first inverted through the exact U02/F01 chart to the
        clean endpoint it implies.  The full-atom ``pred_X`` contributes only
        internal H3 shape; its H3 backbone is rigidly placed onto that implied
        endpoint.  The fused clean endpoint is then mapped analytically back to
        the carrier chart.  Therefore the next recurrent round sees two *known
        parameterizations of one endpoint*, not two independent H3 geometries.
        """
        if flow_xt_model is None or flow_x0_model is None or flow_t is None:
            raise RuntimeError(
                "scoreflow_endpoint_fused requires flow Xt, source X0 and flow_t"
            )
        pred_design = pred_X[paratope_mask]
        if tuple(pred_design.shape) != tuple(carrier.shape):
            raise RuntimeError(
                f"V223 generated/carrier shape mismatch: pred={tuple(pred_design.shape)} "
                f"carrier={tuple(carrier.shape)}"
            )
        t_int = self._time_for_interface(flow_t, interface_batch_id, carrier)
        implied = self.r3_matcher.endpoint_from_canonical_carrier_gfree(
            x_t=flow_xt_model, x0=flow_x0_model, carrier=carrier, t=t_int,
            boundary_eps=self.f01_hybrid_t_min,
        )
        active = torch.as_tensor(t_int, device=carrier.device, dtype=carrier.dtype) >= float(self.f01_hybrid_t_min)
        while active.dim() < carrier.dim():
            active = active.unsqueeze(-1)
        implied = torch.where(active, implied, carrier)

        fused = pred_design.clone()
        centroid_gaps, aligned_shape_rms, rotation_deg = [], [], []
        for gid in range(int(interface_batch_id.max().item()) + 1 if interface_batch_id.numel() else 0):
            idx = interface_batch_id == gid
            if not bool(idx.any()):
                continue
            moving_bb = pred_design[idx, :4].reshape(-1, 3)
            target_bb = implied[idx, :4].reshape(-1, 3)
            aligned_bb, R, trans = self._differentiable_design_kabsch(
                moving_bb, target_bb
            )
            # Apply the same rigid placement to all atom14 coordinates.
            p = pred_design[idx].float()
            fused[idx] = (
                p.matmul(R.float().transpose(0, 1)) + trans.float()
            ).to(fused.dtype)
            with torch.no_grad():
                centroid_gaps.append((
                    moving_bb.detach().float().mean(0)
                    - target_bb.detach().float().mean(0)
                ).norm())
                aligned_shape_rms.append(self._rms_tensor(
                    aligned_bb.detach().float() - target_bb.detach().float()
                ))
                tr = torch.trace(R.detach().float())
                cosang = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
                rotation_deg.append(torch.acos(cosang) * (180.0 / math.pi))

        canonical = self.r3_matcher.canonical_carrier_target_gfree(
            x_t=flow_xt_model, x0=flow_x0_model, x1=fused, t=t_int,
            boundary_eps=self.f01_hybrid_t_min,
        )
        consistent_carrier = torch.where(active, canonical, fused)
        pred_closed = pred_X.clone()
        pred_closed[paratope_mask] = fused

        with torch.no_grad():
            audit = {
                "state_closure_on": carrier.new_tensor(1.0),
                "state_endpoint_pre_rms": self._rms_tensor(
                    pred_design.detach() - implied.detach()
                ).to(carrier.dtype),
                "state_endpoint_post_rms": self._rms_tensor(
                    fused.detach() - implied.detach()
                ).to(carrier.dtype),
                "state_carrier_projection_rms": self._rms_tensor(
                    consistent_carrier.detach() - carrier.detach()
                ).to(carrier.dtype),
                "state_centroid_gap_A_model": (
                    torch.stack(centroid_gaps).mean().to(carrier.dtype)
                    if centroid_gaps else carrier.new_zeros(())
                ),
                "state_aligned_shape_rms_model": (
                    torch.stack(aligned_shape_rms).mean().to(carrier.dtype)
                    if aligned_shape_rms else carrier.new_zeros(())
                ),
                "state_rotation_deg": (
                    torch.stack(rotation_deg).mean().to(carrier.dtype)
                    if rotation_deg else carrier.new_zeros(())
                ),
            }
        return pred_closed, consistent_carrier, audit

    def _carrier_from_single_endpoint(
            self, pred_X, shadow_carrier, paratope_mask, interface_batch_id,
            flow_xt_model, flow_x0_model, flow_t):
        """Derive the recurrent Score--Flow carrier from the sole learned endpoint.

        ``pred_X[paratope_mask]`` is the only learned clean-coordinate authority.
        The shadow EGNN coordinate is retained only as an intra-round geometric
        workspace for message passing and is discarded before recurrent feedback.
        """
        if flow_xt_model is None or flow_x0_model is None or flow_t is None:
            raise RuntimeError(
                "scoreflow_single_endpoint requires flow Xt, source X0 and flow_t"
            )
        endpoint = pred_X[paratope_mask]
        if tuple(endpoint.shape) != tuple(shadow_carrier.shape):
            raise RuntimeError(
                f"single-endpoint shape mismatch: endpoint={tuple(endpoint.shape)} "
                f"shadow={tuple(shadow_carrier.shape)}"
            )
        t_int = self._time_for_interface(flow_t, interface_batch_id, endpoint)
        canonical = self.r3_matcher.canonical_carrier_target_gfree(
            x_t=flow_xt_model, x0=flow_x0_model, x1=endpoint, t=t_int,
            boundary_eps=self.f01_hybrid_t_min,
        )
        active = torch.as_tensor(
            t_int, device=endpoint.device, dtype=endpoint.dtype
        ) >= float(self.f01_hybrid_t_min)
        while active.dim() < endpoint.dim():
            active = active.unsqueeze(-1)
        carrier = torch.where(active, canonical, endpoint)

        with torch.no_grad():
            shadow_implied = self.r3_matcher.endpoint_from_canonical_carrier_gfree(
                x_t=flow_xt_model, x0=flow_x0_model, carrier=shadow_carrier, t=t_int,
                boundary_eps=self.f01_hybrid_t_min,
            )
            shadow_implied = torch.where(active, shadow_implied, shadow_carrier)
            audit = {
                "single_endpoint_on": endpoint.new_tensor(1.0),
                "shadow_endpoint_gap_rms": self._rms_tensor(
                    shadow_implied.detach() - endpoint.detach()
                ).to(endpoint.dtype),
            }
        return carrier, audit

    def _coarse_anchor_distance_loss(
            self, pred_X, true_X, true_S, paratope_mask, batch_id,
            is_antigen, xloss_mask):
        """Non-saturating CA-distance supervision for coarse H3 placement.

        Native DF and DA pairs within the configured support cutoff are supervised
        by absolute distance error.  Relations are equally averaged per complex,
        then scaled by the existing R3 coordinate scaling (0.1 in the formal run).
        """
        zero = pred_X.sum() * 0.0
        ca_valid = xloss_mask[:, 1].bool() if xloss_mask.dim() == 2 else xloss_mask.bool()
        biological = (true_S >= 0) & (true_S < 20) & ca_valid
        generation = paratope_mask.bool() & biological
        antigen = is_antigen.bool() & biological
        framework = biological & (~generation) & (~antigen)
        pred_ca = true_X[:, 1].clone()
        pred_ca[generation] = pred_X[generation, 1]
        true_ca = true_X[:, 1]

        graph_losses = []
        df_mae_sum = zero.detach().float()
        da_mae_sum = zero.detach().float()
        df_graphs = da_graphs = 0
        df_pairs = da_pairs = 0
        B = int(batch_id.max().item()) + 1 if batch_id.numel() else 0
        for gid in range(B):
            g = batch_id == gid
            gi = torch.nonzero(g & generation, as_tuple=False).flatten()
            if gi.numel() == 0:
                continue
            relation_losses = []
            for ctx_mask, relation in ((framework, 'DF'), (antigen, 'DA')):
                cj = torch.nonzero(g & ctx_mask, as_tuple=False).flatten()
                if cj.numel() == 0:
                    continue
                native_d = torch.cdist(true_ca[gi].float(), true_ca[cj].float())
                support = native_d <= float(self.coarse_anchor_cutoff_A)
                if not bool(support.any()):
                    continue
                pred_d = torch.cdist(pred_ca[gi].float(), pred_ca[cj].float())
                mae = (pred_d - native_d).abs()[support].mean()
                relation_losses.append(mae.to(pred_X.dtype))
                n = int(support.sum().item())
                if relation == 'DF':
                    df_mae_sum = df_mae_sum + mae.detach()
                    df_graphs += 1
                    df_pairs += n
                else:
                    da_mae_sum = da_mae_sum + mae.detach()
                    da_graphs += 1
                    da_pairs += n
            if relation_losses:
                graph_losses.append(torch.stack(relation_losses).mean())

        loss_A = torch.stack(graph_losses).mean() if graph_losses else zero
        loss = loss_A * float(self.flow_coordinate_scaling)
        audit = {
            'coarse_anchor_DF_mae_A': (df_mae_sum / max(df_graphs, 1)).to(pred_X.dtype),
            'coarse_anchor_DA_mae_A': (da_mae_sum / max(da_graphs, 1)).to(pred_X.dtype),
            'coarse_anchor_DF_pairs': pred_X.new_tensor(float(df_pairs)),
            'coarse_anchor_DA_pairs': pred_X.new_tensor(float(da_pairs)),
        }
        return loss, audit


    def _forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                 surface, residue_pos, template, lengths, init_noise=None,
                 interface_init=None, sequence_init=None, flow_t=None,
                 flow_source_init=None):
        """R05 predictor at one transport state."""
        batch_id = self.batch_constants['batch_id']
        # V10.1: `_forward` owns its diagnostic-capture scope.  The same flag is
        # also evaluated inside `message_passing`, but that local variable is not
        # visible here.  Keep the predicates identical so round-level EGNN
        # diagnostics are collected exactly when message-level diagnostics are.
        capture_diag = bool(
            getattr(self, '_diagnostic_capture', False)
            or getattr(self, 'geometry_forensics_enabled', False)
            or getattr(self, 'sample_forensics_enabled', False)
        )
        X, S, surface = X.clone(), S.clone(), surface.clone()
        X, S = self.init_mask(X, S, cmask, smask, template)
        X, S = self.replace_pep(X, S, paratope_mask, X_pep, S_pep)

        X = self.normalizer.centering(X, S, batch_id, self.aa_feature)
        X = self.normalizer.normalize(X)
        surface = self.normalizer.normalize(surface)
        X = self.aa_feature.update_global_coordinates(X, S)

        if interface_init is None:
            interface_X, interface_S = self.init_interface(
                X, S, paratope_mask, batch_id, init_noise)
            interface_X, interface_S = self._condition_initial_interface(
                interface_X, interface_S, X_pep, S_pep)
        else:
            interface_X = self._raw_interface_to_model_frame(
                interface_init, paratope_mask, batch_id)
            interface_S = (
                sequence_init.to(device=S.device, dtype=torch.long).clone()
                if sequence_init is not None else S[paratope_mask].clone())

        flow_xt_model = interface_X.clone() if interface_init is not None else None
        flow_x0_model = None
        if flow_source_init is not None:
            flow_x0_model = self._raw_interface_to_model_frame(
                flow_source_init, paratope_mask, batch_id
            )
        if self.coordinate_state_mode in {"scoreflow_endpoint_fused", "scoreflow_single_endpoint"}:
            if flow_xt_model is None or flow_x0_model is None or flow_t is None:
                raise RuntimeError(
                    "Score--Flow state mode requires explicit interface_init, "
                    "flow_source_init and flow_t."
                )
            if bool((paratope_mask & (~cmask)).any()):
                raise RuntimeError(
                    "Score--Flow state mode requires generated/paratope residues to have coordinate "
                    "authority: paratope_mask must be a subset of cmask."
                )

        pep_X_model, pep_coord_valid = None, None
        if X_pep is not None and X_pep.shape == interface_X.shape:
            pep_X_raw = X_pep.to(device=X.device, dtype=X.dtype)
            pep_coord_valid = self._valid_proposal_backbone(pep_X_raw)
            if bool(pep_coord_valid.any()):
                pep_X_model = self._raw_interface_to_model_frame(
                    pep_X_raw, paratope_mask, batch_id)

        seq_ref = interface_X.new_zeros(
            (paratope_mask.shape[0], self.seq_pep_condition_embedding.embedding_dim))
        seq_cond, seq_cond_mask = self._build_seq_pep_condition_for_residues(
            S_pep, paratope_mask, seq_ref)

        trunk_S = S
        biological = paratope_mask.bool() | ((trunk_S >= 0) & (trunk_S < self.num_classes))
        trunk_X = X.clone()
        trunk_X[paratope_mask] = interface_X.to(trunk_X.dtype)
        # R05 coordinates are normalized by 10 for physical refinement.  The
        # compact relational trunk keeps AbX/DiffAb geometry conventions in
        # Angstrom, so restore scale without undoing the harmless centering.
        relational_X = self.normalizer.unnormalize(trunk_X)
        trunk_state = self.native_trunk(
            X=relational_X, S=trunk_S,
            segment_ids=self.batch_constants['segment_ids'],
            residue_pos=residue_pos, batch_id=batch_id,
            valid_mask=biological, is_antigen=self.batch_constants['is_ag'],
            # Keep donor structural information barrier (cmask) separate from
            # task auxiliary support (paratope_mask).  For formal H3 generation
            # the latter is exactly the generated stochastic degree of freedom.
            design_mask=cmask, aux_task_mask=paratope_mask, flow_t=flow_t,
            cdr_type=self.cdr_type, residue_feature=self.aa_feature,
            round_idx=-1,
            atom_observed_mask=self.batch_constants.get('xloss_mask'))
        self._last_trunk_state = trunk_state

        r_logits, r_interface_X, r_edge_dist = [], [interface_X.clone()], []
        pred_S_dist, memory_H = None, None
        self._last_state_closure_audit = {}
        round_egnn_diagnostics = []
        for round_idx in range(self.round):
            if round_idx >= self.proposal_adapter_start_round:
                coord_cond, coord_mask = self._build_coord_pep_condition_for_residues(
                    pep_X_model, interface_X, paratope_mask,
                    pep_coord_valid=pep_coord_valid)
                seq_this, seq_mask_this = seq_cond, seq_cond_mask
            else:
                coord_cond = coord_mask = seq_this = seq_mask_this = None

            pred_logits, pred_X, interface_X, H, edge_dist = self.message_passing(
                X, S, residue_pos, interface_X, surface, paratope_mask,
                batch_id, memory_H=memory_H, smooth_prob=pred_S_dist,
                smooth_mask=smask, flow_t=flow_t,
                coord_pep_condition=coord_cond,
                coord_pep_condition_mask=coord_mask,
                seq_pep_condition=seq_this,
                seq_pep_condition_mask=seq_mask_this,
                trunk_state=trunk_state)

            if capture_diag:
                round_egnn_diagnostics.append({
                    'round_idx': int(round_idx),
                    'bridge': {
                        k: v.detach() if torch.is_tensor(v) else v
                        for k, v in (self._last_message_diagnostics or {}).items()
                    },
                    'coord': {
                        k: v.detach() if torch.is_tensor(v) else v
                        for k, v in (getattr(self.gnn, 'last_coord_diagnostics', {}) or {}).items()
                    },
                })

            memory_H = H

            if self.coordinate_state_mode == "scoreflow_endpoint_fused":
                pred_X, interface_X, closure_audit = self._close_scoreflow_endpoint_state(
                    pred_X=pred_X, carrier=interface_X,
                    paratope_mask=paratope_mask,
                    interface_batch_id=self.batch_constants['interface_batch_id'],
                    flow_xt_model=flow_xt_model, flow_x0_model=flow_x0_model,
                    flow_t=flow_t,
                )
                closure_audit = {
                    f"round{round_idx}_{k}": v for k, v in closure_audit.items()
                }
                self._last_state_closure_audit.update(closure_audit)
            elif self.coordinate_state_mode == "scoreflow_single_endpoint":
                interface_X, authority_audit = self._carrier_from_single_endpoint(
                    pred_X=pred_X, shadow_carrier=interface_X,
                    paratope_mask=paratope_mask,
                    interface_batch_id=self.batch_constants['interface_batch_id'],
                    flow_xt_model=flow_xt_model, flow_x0_model=flow_x0_model,
                    flow_t=flow_t,
                )
                authority_audit = {
                    f"round{round_idx}_{k}": v for k, v in authority_audit.items()
                }
                self._last_state_closure_audit.update(authority_audit)

            r_interface_X.append(interface_X.clone())
            r_logits.append((pred_logits, smask))
            r_edge_dist.append(edge_dist)
            X = X.clone()
            X[cmask] = pred_X[cmask]
            # Global generated coordinates always follow pred_X.  In the V224
            # single-endpoint mode the recurrent carrier above is analytically
            # derived from this same endpoint; the shadow EGNN coordinate is not
            # a recurrent authority.
            X = self.aa_feature.update_global_coordinates(X, S)

            if not self.struct_only:
                S = S.clone()
                if round_idx == self.round - 1:
                    S[smask] = torch.argmax(pred_logits[smask], dim=-1)
                else:
                    pred_S_dist = torch.softmax(pred_logits[smask], dim=-1)

        interface_batch_id = self.batch_constants['interface_batch_id']
        prmsd = self.prmsd_ffn(H[cmask]).squeeze() if self.struct_only else None
        pred_X = self.normalizer.uncentering(
            self.normalizer.unnormalize(pred_X), batch_id)
        for i, value in enumerate(r_interface_X):
            value = self.normalizer.unnormalize(value)
            r_interface_X[i] = self.normalizer.uncentering(
                value, interface_batch_id, _type=4)
        self.normalizer.clear_cache()
        self._last_round_egnn_diagnostics = round_egnn_diagnostics
        self._maybe_log_coordinate_controller_audit(round_egnn_diagnostics)
        return H, S, r_logits, pred_X, r_interface_X, r_edge_dist, prmsd


    def forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths, xloss_mask,
                context_ratio=0):
        """Train the fixed R05/U02 state path."""
        cmask, smask = cmask.bool(), smask.bool()
        if self.backbone_only:
            X, template, xloss_mask = X[:, :4], template[:, :4], xloss_mask[:, :4]
            if X_pep is not None:
                X_pep = X_pep[:, :4]

        true_X, true_S = X.clone(), S.clone()
        self._prepare_batch_constants(S, paratope_mask, lengths)
        self.batch_constants['xloss_mask'] = xloss_mask.bool()
        batch_id = self.batch_constants['batch_id']
        interface_batch_id = self.batch_constants['interface_batch_id']
        batch_size_raw = self.batch_constants['batch_size']
        batch_size = int(batch_size_raw.item()) if torch.is_tensor(batch_size_raw) else int(batch_size_raw)

        # Historical R05 sequence-context curriculum.
        sequence_loss_mask = smask.clone()
        if context_ratio > 0:
            keep = torch.rand_like(smask, dtype=torch.float) >= context_ratio
            smask = smask & keep
            sequence_loss_mask = smask
        sequence_path_mask = smask

        gt_interface_X = true_X[paratope_mask]
        interface_X, interface_S = self.init_interface(
            X, S, paratope_mask, batch_id)
        interface_X, interface_S = self._condition_initial_interface(
            interface_X, interface_S, X_pep, S_pep)

        t_graph = self._sample_flow_times(batch_size, X.device, X.dtype)
        t_int = self._time_for_interface(
            t_graph, interface_batch_id, interface_X)
        Xt, path_info = self._r05_primary_path(
            interface_X, gt_interface_X, t_graph, t_int,
            interface_batch_id)
        coord_target, target_info = self._r05_coordinate_target(
            Xt, interface_X, gt_interface_X, t_int)

        if self.struct_only:
            St, sequence_state = interface_S, None
        else:
            St = self._sample_categorical_path(
                true_S[paratope_mask], interface_S, t_graph,
                interface_batch_id,
                corrupt_mask=sequence_path_mask[paratope_mask])
            sequence_state = St

        self._last_message_diagnostics = {}
        H, pred_S, r_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(
            X, S, cmask, smask, paratope_mask, X_pep, S_pep,
            surface, residue_pos, template, lengths,
            interface_init=Xt, sequence_init=sequence_state,
            flow_t=t_graph, flow_source_init=interface_X)

        snll = X.new_zeros(())
        count = X.new_zeros(())
        if not self.struct_only:
            for logits, _ in r_logits:
                if bool(sequence_loss_mask.any()):
                    snll = snll + F.cross_entropy(
                        logits[sequence_loss_mask], true_S[sequence_loss_mask],
                        reduction='sum')
                    count = count + sequence_loss_mask.sum()
            snll = snll / count.clamp_min(1.0)

        structure_loss_mask = (
            paratope_mask if self.generated_region_only_loss else cmask
        )
        struct_loss, struct_details, bb_rmsd, _ = self.protein_feature.structure_loss(
            pred_X, true_X, true_S, structure_loss_mask, batch_id, xloss_mask,
            self.aa_feature)

        atom_pos = self.aa_feature._construct_atom_pos(true_S[paratope_mask])
        atom_mask = atom_pos != self.aa_feature.atom_pos_pad_idx
        interface_loss = self._coordinate_training_objective(
            r_interface_X[-1], coord_target, atom_mask, interface_batch_id)

        if self.pred_edge_dist:
            gt_edge_dist = self._get_inter_edge_dist(
                self.normalizer.normalize(true_X), true_S)
            r_ed_losses = [F.smooth_l1_loss(v, gt_edge_dist) for v in r_edge_dist]
            ed_loss = sum(r_ed_losses, X.new_zeros(()))
        else:
            r_ed_losses = [X.new_zeros(()) for _ in range(self.round)]
            ed_loss = X.new_zeros(())
        dock_loss = interface_loss + ed_loss

        distogram_loss = X.new_zeros(())
        distogram_audit = {}
        if self.loss_distogram_weight > 0.0:
            distogram_loss, distogram_audit = self.native_trunk.distogram_loss_from_native(
                self._last_trunk_state, true_X, true_S,
                collect_audit=bool(getattr(self, "_diagnostic_capture", False)),
            )

        coarse_anchor_loss = X.new_zeros(())
        coarse_anchor_audit = {}
        if self.loss_coarse_anchor_weight > 0.0:
            coarse_anchor_loss, coarse_anchor_audit = self._coarse_anchor_distance_loss(
                pred_X=pred_X, true_X=true_X, true_S=true_S,
                paratope_mask=paratope_mask, batch_id=batch_id,
                is_antigen=self.batch_constants['is_ag'], xloss_mask=xloss_mask,
            )

        smooth_lddt_loss = X.new_zeros(())
        smooth_lddt_audit = {}
        if self.loss_smooth_lddt_weight > 0.0:
            valid_atom_mask = self.batch_constants['xloss_mask'].bool()
            # Fixed context is observed at inference and must remain exact in this
            # auxiliary objective.  V213 changes only WHICH generated coordinate
            # object supplies the design residues.
            aux_pred_X = true_X.clone()
            smooth_design_mask = cmask

            if self.smooth_lddt_prediction_source == "pred_design_endpoint":
                # Historical R33 semantics.  V223 formal experiments do not use
                # this branch because cmask may exceed the generated H3 support.
                smooth_pred_design = pred_X[cmask]
            elif self.smooth_lddt_prediction_source == "terminal_fused_endpoint":
                if self.coordinate_state_mode not in {"scoreflow_endpoint_fused", "scoreflow_single_endpoint"}:
                    raise RuntimeError(
                        "terminal_fused_endpoint smooth-lDDT requires "
                        "coordinate_state.mode in {scoreflow_endpoint_fused, scoreflow_single_endpoint}"
                    )
                smooth_design_mask = paratope_mask
                smooth_pred_design = pred_X[paratope_mask]
            elif self.smooth_lddt_prediction_source == "carrier_implied_endpoint":
                # The shadow/interface carrier has one row per JSON-defined
                # paratope residue.  ``cmask`` is a different coordinate/template
                # authority: it may legitimately contain additional framework
                # rows.  The only required task contract is therefore
                #       paratope_mask <= cmask,
                # never cmask == paratope_mask.
                if cmask.shape != paratope_mask.shape:
                    raise RuntimeError(
                        "carrier_implied_endpoint smooth-lDDT requires cmask and "
                        "paratope_mask to have the same residue axis, got "
                        f"cmask={tuple(cmask.shape)} paratope={tuple(paratope_mask.shape)}."
                    )
                missing_coord = paratope_mask & (~cmask)
                if bool(missing_coord.any()):
                    raise RuntimeError(
                        "carrier_implied_endpoint smooth-lDDT requires every "
                        "paratope residue to have coordinate authority: "
                        f"missing={int(missing_coord.sum().item())}."
                    )

                carrier = r_interface_X[-1]
                expected_shape = true_X[paratope_mask].shape
                if tuple(carrier.shape) != tuple(expected_shape):
                    raise RuntimeError(
                        "carrier/paratope shape mismatch for smooth-lDDT: "
                        f"carrier={tuple(carrier.shape)} expected={tuple(expected_shape)}."
                    )
                canonical_x1 = self.r3_matcher.endpoint_from_canonical_carrier_gfree(
                    x_t=Xt,
                    x0=interface_X,
                    carrier=carrier,
                    t=t_int,
                    boundary_eps=self.f01_hybrid_t_min,
                )
                active = torch.as_tensor(
                    t_int, device=carrier.device, dtype=carrier.dtype
                ) >= float(self.f01_hybrid_t_min)
                while active.dim() < carrier.dim():
                    active = active.unsqueeze(-1)
                # Below t_min the U02 training carrier IS the endpoint. Above
                # t_min use the exact inverse of the canonical carrier mapping.
                # Scatter only onto the paratope rows.  All non-paratope context,
                # including any extra cmask framework rows, stays exactly native
                # in this auxiliary objective.
                smooth_pred_design = torch.where(active, canonical_x1, carrier)
                smooth_design_mask = paratope_mask
            else:  # guarded in __init__, retained as a local fail-fast boundary
                raise RuntimeError(
                    "unreachable smooth-lDDT prediction source: "
                    f"{self.smooth_lddt_prediction_source!r}"
                )

            aux_pred_X[smooth_design_mask] = smooth_pred_design
            smooth_lddt_loss, smooth_lddt_audit = design_region_smooth_lddt_loss(
                pred_X=aux_pred_X,
                true_X=true_X,
                valid_atom_mask=valid_atom_mask,
                design_residue_mask=smooth_design_mask,
                batch_id=batch_id,
                is_antigen_mask=self.batch_constants['is_ag'],
                cutoff=self.smooth_lddt_cutoff,
                collect_audit=bool(getattr(self, "_diagnostic_capture", False)),
            )
            with torch.no_grad():
                design_valid = valid_atom_mask[smooth_design_mask]
                if bool(design_valid.any()):
                    endpoint_err = (
                        smooth_pred_design.detach().float()
                        - true_X[smooth_design_mask].detach().float()
                    )[design_valid]
                    smooth_lddt_audit['endpoint_rms_A'] = endpoint_err.square().mean().sqrt().to(X.dtype)
                    smooth_lddt_audit['endpoint_absmax_A'] = endpoint_err.abs().max().to(X.dtype)
                else:
                    smooth_lddt_audit['endpoint_rms_A'] = X.new_zeros(())
                    smooth_lddt_audit['endpoint_absmax_A'] = X.new_zeros(())
                smooth_lddt_audit['source_carrier_implied_endpoint'] = X.new_tensor(
                    1.0 if self.smooth_lddt_prediction_source == "carrier_implied_endpoint" else 0.0
                )
                smooth_lddt_audit['design_rows'] = smooth_design_mask.sum().to(X.dtype)
                smooth_lddt_audit['coord_rows'] = cmask.sum().to(X.dtype)
                smooth_lddt_audit['coord_outside_design_rows'] = (
                    cmask & (~smooth_design_mask)
                ).sum().to(X.dtype)

        if self.struct_only:
            prmsd_loss = F.smooth_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss
        else:
            pdev_loss = prmsd_loss = None

        loss = (
            self.loss_sequence_weight * snll
            + self.loss_structure_weight * struct_loss
            + self.loss_interface_weight * interface_loss
            + self.loss_edge_weight * ed_loss
            + self.loss_distogram_weight * distogram_loss
            + self.loss_coarse_anchor_weight * coarse_anchor_loss
            + self.loss_smooth_lddt_weight * smooth_lddt_loss
        )
        if pdev_loss is not None:
            loss = loss + pdev_loss

        # Per-complex geometry diagnostics are computed only when explicitly
        # enabled.  They are detached and never enter the objective.
        if self.geometry_forensics_enabled:
            with torch.no_grad():
                forensic_mask = (
                    paratope_mask if self.generated_region_only_loss else cmask
                )
                design_gid = batch_id[forensic_mask]
                design_atom_mask = xloss_mask[forensic_mask].bool()
                pred_design_rms = self._per_graph_coord_rms(
                    pred_X[forensic_mask], true_X[forensic_mask], design_atom_mask,
                    design_gid, batch_size)
                pred_design_absmax = self._per_graph_absmax(
                    pred_X[forensic_mask], design_gid, batch_size)
                carrier_target_rms = self._per_graph_coord_rms(
                    r_interface_X[-1], coord_target, atom_mask,
                    interface_batch_id, batch_size)
                carrier_absmax = self._per_graph_absmax(
                    r_interface_X[-1], interface_batch_id, batch_size)

                # Physical refinement-round growth in Angstrom.  r_interface_X
                # is already unnormalized/uncentered at this point, so these
                # diagnostics directly reveal whether round 1/2/3 is the first
                # Cartesian amplification point.
                round_delta_rms = []
                round_absmax = []
                for ridx in range(1, len(r_interface_X)):
                    round_delta_rms.append(self._per_graph_coord_rms(
                        r_interface_X[ridx], r_interface_X[ridx - 1], atom_mask,
                        interface_batch_id, batch_size))
                    round_absmax.append(self._per_graph_absmax(
                        r_interface_X[ridx], interface_batch_id, batch_size))
                round_delta_rms = (
                    torch.stack(round_delta_rms, dim=0)
                    if round_delta_rms else pred_design_rms.new_zeros((0, batch_size))
                )
                round_absmax = (
                    torch.stack(round_absmax, dim=0)
                    if round_absmax else pred_design_rms.new_zeros((0, batch_size))
                )

                worst_graph = torch.argmax(pred_design_rms) if pred_design_rms.numel() else torch.zeros((), device=X.device, dtype=torch.long)
                self.last_geometry_forensics = {
                    'per_graph_pred_design_rms_A': pred_design_rms.detach(),
                    'per_graph_pred_design_absmax_A': pred_design_absmax.detach(),
                    'per_graph_carrier_target_rms_A': carrier_target_rms.detach(),
                    'per_graph_carrier_absmax_A': carrier_absmax.detach(),
                    'per_round_graph_delta_rms_A': round_delta_rms.detach(),
                    'per_round_graph_absmax_A': round_absmax.detach(),
                    'per_graph_t': t_graph.detach().float(),
                    'worst_graph_index': worst_graph.detach(),
                }
        else:
            self.last_geometry_forensics = {}

        with torch.no_grad():
            aar = ((pred_S[sequence_loss_mask] == true_S[sequence_loss_mask]).float().mean()
                   if bool(sequence_loss_mask.any()) else X.new_zeros(()))
            trunk_diag = (self._last_trunk_state.get('diag', {})
                          if isinstance(self._last_trunk_state, dict) else {})
            smooth_intra = smooth_lddt_audit.get('intra', X.new_zeros(()))
            smooth_scaffold = smooth_lddt_audit.get('scaffold', X.new_zeros(()))
            smooth_antigen = smooth_lddt_audit.get('antigen', X.new_zeros(()))

            self.last_scorefm_losses = {
                'scorefm_total': interface_loss.detach(),
                'scorefm_endpoint': interface_loss.detach(),
                'distogram_loss': distogram_loss.detach(),
                'distogram_weighted_loss': (
                    self.loss_distogram_weight * distogram_loss
                ).detach(),
                'coarse_anchor_loss': coarse_anchor_loss.detach(),
                'coarse_anchor_weighted_loss': (
                    self.loss_coarse_anchor_weight * coarse_anchor_loss
                ).detach(),
                **{k: v.detach() for k, v in coarse_anchor_audit.items()},
                'smooth_lddt_loss': smooth_lddt_loss.detach(),
                'smooth_lddt_weighted_loss': (
                    self.loss_smooth_lddt_weight * smooth_lddt_loss
                ).detach(),
                'smooth_lddt_DD': smooth_intra.detach(),
                'smooth_lddt_DF': smooth_scaffold.detach(),
                'smooth_lddt_DA': smooth_antigen.detach(),
                'smooth_lddt_support_weight': smooth_lddt_audit.get(
                    'support_weight', X.new_zeros(())
                ).detach(),
                'smooth_lddt_endpoint_rms_A': smooth_lddt_audit.get(
                    'endpoint_rms_A', X.new_zeros(())
                ).detach(),
                'smooth_lddt_endpoint_absmax_A': smooth_lddt_audit.get(
                    'endpoint_absmax_A', X.new_zeros(())
                ).detach(),
                'smooth_lddt_source_carrier_endpoint': smooth_lddt_audit.get(
                    'source_carrier_implied_endpoint', X.new_zeros(())
                ).detach(),
                'smooth_lddt_design_rows': smooth_lddt_audit.get(
                    'design_rows', X.new_zeros(())
                ).detach(),
                'smooth_lddt_coord_rows': smooth_lddt_audit.get(
                    'coord_rows', X.new_zeros(())
                ).detach(),
                'smooth_lddt_coord_outside_design_rows': smooth_lddt_audit.get(
                    'coord_outside_design_rows', X.new_zeros(())
                ).detach(),
                'relational_single_rms': trunk_diag.get(
                    'relational_single_rms', X.new_zeros(())
                ).detach(),
                'relational_pair_rms': trunk_diag.get(
                    'relational_pair_rms', X.new_zeros(())
                ).detach(),
                'relational_antigen_keep_fraction': trunk_diag.get(
                    'relational_antigen_keep_fraction', X.new_ones(())
                ).detach(),
                **{k: v.detach() for k, v in distogram_audit.items()},
                **{k: v.detach() for k, v in path_info.items()},
                **{k: v.detach() for k, v in target_info.items()},
            }

            self.last_abflow_diagnostics = {
                't_min': t_graph.min().detach(),
                't_mean': t_graph.mean().detach(),
                't_max': t_graph.max().detach(),
                'coordinate_state_closed': X.new_tensor(
                    1.0 if self.coordinate_state_mode == "scoreflow_endpoint_fused" else 0.0
                ),
                'coordinate_single_endpoint': X.new_tensor(
                    1.0 if self.coordinate_state_mode == "scoreflow_single_endpoint" else 0.0
                ),
                **{k: v.detach() if torch.is_tensor(v) else v
                   for k, v in self._last_message_diagnostics.items()},
                **{k: v.detach() if torch.is_tensor(v) else v
                   for k, v in self._last_state_closure_audit.items()},
            }

        self._clean_batch_constants()
        return (loss, (snll, aar), (struct_loss, *struct_details),
                (dock_loss, interface_loss, ed_loss, r_ed_losses),
                (pdev_loss, prmsd_loss))

    def _sampling_time_grid(self, n_steps, device, dtype):
        return torch.linspace(0.0, 1.0, max(1, int(n_steps)) + 1,
                              device=device, dtype=dtype)

    @torch.no_grad()
    def sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
               surface, residue_pos, template, lengths, n_steps=10,
               init_noise=None, return_hidden=False, show_progress=False,
               progress_desc=None, xloss_mask=None):
        """Generate with the matched R05/U02 canonical-carrier sampler."""
        n_steps = max(1, int(n_steps))
        if self.sample_forensics_enabled:
            self.reset_sample_forensics()
        cmask, smask = cmask.bool(), smask.bool()
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]
            if X_pep is not None:
                X_pep = X_pep[:, :4]
            if xloss_mask is not None:
                xloss_mask = xloss_mask[:, :4]

        gen_X, gen_S = X.clone(), S.clone()
        self._prepare_batch_constants(S, paratope_mask, lengths)
        self.batch_constants['xloss_mask'] = (
            xloss_mask.bool() if xloss_mask is not None
            else _abflow_ca_fill_observed_mask(
                S, X, tol2=self.native_trunk.ca_fill_tol2
            ).bool())

        batch_id = self.batch_constants['batch_id']
        batch_size_raw = self.batch_constants['batch_size']
        batch_size = int(batch_size_raw.item()) if torch.is_tensor(batch_size_raw) else int(batch_size_raw)
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]
        interface_cmask = paratope_mask[cmask]

        Xt, St = self.init_interface(
            X, S, paratope_mask, batch_id, init_noise=init_noise)
        Xt, St = self._condition_initial_interface(Xt, St, X_pep, S_pep)
        source_X0 = Xt.clone()
        time_grid = self._sampling_time_grid(n_steps, X.device, X.dtype)

        steps = range(n_steps)
        if show_progress:
            steps = tqdm(steps, total=n_steps,
                         desc=progress_desc or 'Sampling ODE',
                         leave=False, dynamic_ncols=True)

        for i in steps:
            t, t_next = time_grid[i], time_grid[i + 1]
            dt = t_next - t
            flow_t = t.reshape(1).expand(batch_size)
            if show_progress and hasattr(steps, 'set_postfix'):
                steps.set_postfix(t=f'{float(t):.2f}')

            prev_diag_capture = bool(getattr(self, '_diagnostic_capture', False))
            if self.sample_forensics_enabled:
                self._diagnostic_capture = True
            try:
                H, pred_S, r_logits, pred_X, r_interface_X, _, prmsd = self._forward(
                    X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                    surface, residue_pos, template, lengths,
                    interface_init=Xt,
                    sequence_init=None if self.struct_only else St,
                    flow_t=flow_t, flow_source_init=source_X0)
                bridge_diag = dict(self._last_message_diagnostics or {})
            finally:
                self._diagnostic_capture = prev_diag_capture

            carrier = r_interface_X[-1]
            Xt_before = Xt
            Xt_next, matcher_diag = self.r3_matcher.exact_carrier_scoreflow_step_gfree(
                x_t=Xt_before, x0=source_X0, carrier=carrier,
                t=t, t_next=t_next,
                canonical_t_min=self.f01_hybrid_t_min)

            if self.sample_forensics_enabled:
                t_value = float(t.detach().float().item())
                t_next_value = float(t_next.detach().float().item())
                if t_value < float(self.f01_hybrid_t_min):
                    implied_x1 = carrier
                    carrier_mode = 'endpoint_boundary'
                else:
                    implied_x1 = self.r3_matcher.endpoint_from_canonical_carrier_gfree(
                        x_t=Xt_before, x0=source_X0, carrier=carrier,
                        t=t, boundary_eps=self.f01_hybrid_t_min)
                    carrier_mode = 'canonical'

                xt_stats = self._forensic_graph_stats(
                    Xt_before, interface_batch_id, batch_size)
                carrier_stats = self._forensic_graph_stats(
                    carrier, interface_batch_id, batch_size)
                x1_stats = self._forensic_graph_stats(
                    implied_x1, interface_batch_id, batch_size)
                next_stats = self._forensic_graph_stats(
                    Xt_next, interface_batch_id, batch_size)
                carrier_residual_stats = self._forensic_graph_stats(
                    carrier - Xt_before, interface_batch_id, batch_size)
                step_stats = self._forensic_graph_stats(
                    Xt_next - Xt_before, interface_batch_id, batch_size)

                # Physical-round EGNN diagnostics from this exact sampler
                # forward.  Values are batch-level maxima, repeated in each
                # row only to keep every JSONL record self-contained.
                round_coord_update_absmax = []
                round_coord_coeff_absmax = []
                round_worst_stage = []
                for rec in (self._last_round_egnn_diagnostics or []):
                    coord_diag = rec.get('coord', {}) or {}
                    round_coord_update_absmax.append(self._diag_float(
                        coord_diag, 'coord_update_absmax_max'))
                    round_coord_coeff_absmax.append(self._diag_float(
                        coord_diag, 'coord_coeff_absmax_max'))
                    candidates = []
                    for key, value in coord_diag.items():
                        if key.endswith('.coord_update_absmax'):
                            try:
                                fv = float(value.detach().float().item()) if torch.is_tensor(value) else float(value)
                            except Exception:
                                continue
                            if math.isfinite(fv):
                                candidates.append((fv, key.rsplit('.', 1)[0]))
                    round_worst_stage.append(
                        max(candidates, default=(float('nan'), 'NA'))[1]
                    )

                for gid in range(batch_size):
                    row = {
                        **self._sample_forensic_identity(gid),
                        'stage': 'sampling_step',
                        'step': int(i),
                        't': t_value,
                        't_next': t_next_value,
                        'carrier_mode': carrier_mode,
                        'finite': bool(
                            xt_stats[gid]['finite'] and carrier_stats[gid]['finite']
                            and x1_stats[gid]['finite'] and next_stats[gid]['finite']
                        ),
                        'xt_absmax_A': xt_stats[gid]['absmax'],
                        'carrier_absmax_A': carrier_stats[gid]['absmax'],
                        'carrier_residual_rms_A': carrier_residual_stats[gid]['rms'],
                        'implied_x1_absmax_A': x1_stats[gid]['absmax'],
                        'implied_x1_rms_A': x1_stats[gid]['rms'],
                        'xnext_absmax_A': next_stats[gid]['absmax'],
                        'step_delta_rms_A': step_stats[gid]['rms'],
                        'canonical_residual_rms': self._diag_float(
                            matcher_diag, 'canonical_residual_rms'),
                        'canonical_ratio_mean': self._diag_float(
                            matcher_diag, 'canonical_ratio_mean'),
                        'bridge_single_ratio': self._diag_float(
                            bridge_diag, 'bridge_single_delta_to_base_ratio'),
                        'bridge_pair_ratio_mean': self._diag_float(
                            bridge_diag, 'bridge_pair_delta_to_base_ratio_mean'),
                        'bridge_pair_ratio_max': self._diag_float(
                            bridge_diag, 'bridge_pair_delta_to_base_ratio_max'),
                        'physical_round_coord_update_absmax': round_coord_update_absmax,
                        'physical_round_coord_coeff_absmax': round_coord_coeff_absmax,
                        'physical_round_worst_stage': round_worst_stage,
                    }
                    self._append_sample_forensic_record(row)

            Xt = Xt_next

            if not self.struct_only:
                logits = r_logits[-1][0][paratope_mask]
                probs = F.softmax(logits - logits.max(-1, keepdim=True).values, dim=-1)
                proposed = torch.multinomial(probs.clamp_min(1e-8), 1).squeeze(-1)
                refresh_prob = self.flow_matcher.categorical_refresh_probability(t, dt)
                refresh = (torch.rand(St.shape, device=St.device) < refresh_prob)
                refresh = refresh & smask[paratope_mask]
                St = torch.where(refresh, proposed, St)

        H_final, pred_X_final, prmsd_final = H, pred_X, prmsd
        interface_X_final = Xt
        if not self.struct_only:
            final_logits = r_logits[-1][0]
            pred_S_final = pred_S.clone()
            if bool(smask.any()):
                pred_S_final[smask] = torch.argmax(final_logits[smask], dim=-1)
            logits = final_logits[smask]
            if logits.shape[0]:
                probs = torch.softmax(logits, dim=-1).max(-1).values
                metric = scatter_mean(
                    -torch.log(probs.clamp_min(1e-8)), s_batch_id,
                    dim=0, dim_size=batch_size)
            else:
                metric = X.new_zeros(batch_size)
        else:
            metric = scatter_mean(
                prmsd_final[interface_cmask], interface_batch_id,
                dim=0, dim_size=batch_size)

        if self.coordinate_state_mode in {"scoreflow_endpoint_fused", "scoreflow_single_endpoint"}:
            # Formal Score--Flow readout: the 10-step integrated Score--Flow endpoint is
            # the only generated-coordinate authority.  Fixed framework/antigen
            # coordinates are inherited exactly from the input complex.
            gen_X[paratope_mask] = interface_X_final
            if self.sample_forensics_enabled:
                pred_design = pred_X_final[paratope_mask].detach().float()
                integrated = interface_X_final.detach().float()
                for b in range(batch_size):
                    idx = interface_batch_id == b
                    if not bool(idx.any()):
                        continue
                    delta = pred_design[idx] - integrated[idx]
                    self._append_sample_forensic_record({
                        **self._sample_forensic_identity(b),
                        'stage': 'terminal_integrated_endpoint',
                        'step': int(n_steps),
                        't': 1.0,
                        'finite': bool(torch.isfinite(integrated[idx]).all().item()),
                        'terminal_pred_vs_integrated_rms_A': float(
                            delta.square().mean().sqrt().item()
                        ),
                        'integrated_endpoint_absmax_A': float(
                            integrated[idx].abs().amax().item()
                        ),
                        'fixed_context_transform_applied': False,
                    })
        else:
            # Historical readout retained only for backward-compatible configs.
            gen_X[cmask] = pred_X_final[cmask]
            for b in range(batch_size):
                graph = batch_id == b
                design = graph & paratope_mask
                ori = gen_X[design][:, :4]
                pred = interface_X_final[interface_batch_id == b][:, :4]
                ab = graph & is_ab
                _, R, trans = kabsch_torch(ori.reshape(-1, 3), pred.reshape(-1, 3))
                gen_X[ab] = torch.matmul(gen_X[ab], R.T) + trans

        if not self.struct_only:
            gen_S[smask] = pred_S_final[smask]

        self._clean_batch_constants()
        if return_hidden:
            return gen_X, gen_S, metric, H_final
        return gen_X, gen_S, metric

    @torch.no_grad()
    def sample_many(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                    surface, residue_pos, template, lengths, n_samples=5,
                    n_steps=20, return_hidden=False, show_progress=False):
        outputs = []
        for i in range(int(n_samples)):
            n_atom = 4 if self.backbone_only else X.shape[1]
            noise = torch.randn(int(paratope_mask.sum()), n_atom, 3, device=X.device)
            outputs.append(self.sample(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface,
                residue_pos, template, lengths, n_steps=n_steps,
                init_noise=noise, return_hidden=return_hidden,
                show_progress=show_progress,
                progress_desc=f'Sample {i + 1}/{n_samples} ODE'))
        if return_hidden:
            xs, ss, ms, hs = zip(*outputs)
            return list(xs), list(ss), list(ms), list(hs)
        xs, ss, ms = zip(*outputs)
        return list(xs), list(ss), list(ms)
