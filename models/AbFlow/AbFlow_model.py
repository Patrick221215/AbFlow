#!/usr/bin/python
# -*- coding:utf-8 -*-
import math, time, os
from contextlib import nullcontext
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from data.pdb_utils import VOCAB
from utils.nn_utils import SeparatedAminoAcidFeature, ProteinFeature
from utils.nn_utils import GMEdgeConstructor, SeperatedCoordNormalizer
from utils.nn_utils import _knn_edges
from evaluation.rmsd import kabsch_torch

from ..modules.am_enc import AMEncoder
from ..modules.am_enc_pair_time import AMEncoderPairTime
from ..modules.am_egnn import AMEGNN
from .abflow_conditional_matcher import AbFlowConditionalMatcher


# v57 Structured-Primary: global and orthogonal-local full-atom stochastic conditional flow.


def _env_str(name, default):
    value = os.environ.get(name, None)
    if value is None or value == "":
        return default
    return value


def _env_float(name, default):
    value = os.environ.get(name, None)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name, default):
    value = os.environ.get(name, None)
    if value is None or value == "":
        return default
    return int(value)


def _env_flag(name, default=False):
    value = os.environ.get(name, None)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}



def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """Sinusoidal embedding for continuous flow time t in [0, 1].

    This is the same style of time embedding used in diffusion models and in
    the uploaded AbX Seqformer.  It lets AbFlow learn f_theta(X_t, t, c)
    instead of forcing one network to average over all noise/flow times.

    Args:
        timesteps: [B] tensor with values in [0, 1].
        embedding_dim: output channel dimension.
        max_positions: frequency scale.
    Returns:
        [B, embedding_dim] sinusoidal embeddings.
    """
    if timesteps.dim() == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.float() * max_positions
    half_dim = embedding_dim // 2
    if half_dim <= 1:
        emb = timesteps[:, None]
        return F.pad(emb, (0, max(0, embedding_dim - 1)))[:, :embedding_dim]
    freq = math.log(max_positions) / (half_dim - 1)
    freq = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -freq)
    emb = timesteps[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb

class AbFlowModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, num_classes, num_verts, 
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6,
                 n_layers=3, iter_round=3, dropout=0.1, 
                 pep_seq=True, pep_struct=True, struct_only=False,
                 backbone_only=False, fix_channel_weights=False, pred_edge_dist=True,
                 keep_memory=True, cdr_type='H3', paratope='H3', relative_position=False) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.bind_dist_cutoff = bind_dist_cutoff
        self.k_neighbors = k_neighbors
        self.round = iter_round
        
        self.pep_seq = pep_seq
        self.pep_struct = pep_struct
        self.struct_only = struct_only

        # options
        self.backbone_only = backbone_only
        self.fix_channel_weights = fix_channel_weights
        self.pred_edge_dist = pred_edge_dist
        self.keep_memory = keep_memory
        if self.backbone_only:
            n_channel = 4
        # Effective coordinate channel number after backbone_only.
        # Score-FM is defined directly on AbFlow full-atom Cartesian coordinates
        # X ∈ R^{N x n_channel x 3}, so all coordinate losses must use this value.
        self.n_channel = n_channel
        self.cdr_type = cdr_type
        self.paratope = paratope

        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            relative_position=relative_position,
            edge_constructor=GMEdgeConstructor,
            fix_atom_weights=fix_channel_weights,
            backbone_only=backbone_only
        )
        self.protein_feature = ProteinFeature(backbone_only=backbone_only)
        if keep_memory:
            self.memory_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, embed_size)
            )
        if self.pred_edge_dist:  # use predicted dist for KNN-graph at the interface
            if self.keep_memory:  # this ffn acts on the memory
                self.edge_H_ffn = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
            self.edge_dist_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
            # this GNN encodes the initial hidden states for initial edge distance prediction
            self.init_gnn = AMEGNN(
                embed_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, n_layers=n_layers, residual=True,
                dropout=dropout, dense=False)
        if not struct_only:
            self.ffn_residue = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.num_classes)
            )
        else:
            self.prmsd_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        # FoldFlow-inspired pair-level time conditioning.
        #
        # The scientific parent is now GT-SATC.  Pair-time is therefore a
        # strictly optional child module:
        #   off       : exact original AMEncoder architecture
        #   interface : time-condition only true antibody-antigen local edges
        #               and antigen-surface messages
        #   context   : time-condition context/global/local-context messages only
        #   all       : interface scope + context scope
        #
        # The pair-time encoder keeps all original GCL parameter shapes and adds
        # only zero-initialized channel-wise time scales.  This makes the initial
        # function exactly the base AMEncoder and avoids changing the RNG stream
        # for later model parameters.
        pair_scope = _env_str("ABFLOW_PAIR_TIME_SCOPE", "off").lower()
        # Backward compatibility with the previous boolean Stage-2 prototype.
        if (
            pair_scope == "off"
            and _env_flag("ABFLOW_PAIR_TIME_CONDITIONING", False)
        ):
            pair_scope = "all"
        if pair_scope not in {"off", "interface", "context", "all"}:
            raise ValueError(
                "ABFLOW_PAIR_TIME_SCOPE must be off, interface, context, or all."
            )
        self.pair_time_scope = pair_scope
        self.pair_time_conditioning = pair_scope != "off"

        if self.pair_time_conditioning:
            self.gnn = AMEncoderPairTime(
                embed_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, num_verts=num_verts, n_layers=n_layers,
                residual=True, dropout=dropout, dense=False,
                pair_time_scope=pair_scope,
            )
        else:
            self.gnn = AMEncoder(
                embed_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, num_verts=num_verts, n_layers=n_layers, residual=True,
                dropout=dropout, dense=False,
            )
        
        self.normalizer = SeperatedCoordNormalizer()

        # training related cache
        self.batch_constants = {}

        # =========================================================
        # Analytic-score-consistent endpoint Flow Matching
        # =========================================================
        # The network predicts a clean endpoint X_1^theta. Following the
        # analytic parameterization used in AbX, the coordinate score is
        # induced from the known forward kernel; no independent score head is
        # introduced.
        #
        # Core principles of this version:
        #   1. X_0/S_0 always come from the reference distribution.
        #   2. X_pep/S_pep are conditions only and never overwrite X_t/S_t.
        #   3. The endpoint objective is defined exactly once per complex.
        #   4. Endpoint and analytic DSM targets replace one another per complex;
        #      they are never stacked for the same sample.
        #   5. Analytic DSM is applied only to CA translation, whose reference
        #      kernel is exactly isotropic Gaussian around the antigen center.
        #   6. The real transport path reaches t=1; min_sigma protects only
        #      analytic score/bridge denominators.
        self.scorefm_eps = 1e-8
        self.scorefm_min_sigma = _env_float(
            "ABFLOW_SCOREFM_MIN_SIGMA", 1e-2
        )
        if not (0.0 < self.scorefm_min_sigma < 1.0):
            raise ValueError(
                "ABFLOW_SCOREFM_MIN_SIGMA must be in (0, 1)."
            )
        # Centralized deterministic conditional-flow algebra.  This object is
        # parameter-free and RNG-free, so the refactor does not add a trainable
        # module or consume random numbers.
        self.flow_matcher = AbFlowConditionalMatcher(
            min_sigma=self.scorefm_min_sigma,
            eps=self.scorefm_eps,
        )

        # Coordinate objective:
        #   endpoint:
        #       Unique per-complex clean-endpoint SmoothL1 at every t.
        #   analytic_core:
        #       Old reference-source analytic score diagnostic. It is retained
        #       only for historical ablation because PCS-RC uses a proposal-
        #       conditioned source rather than an antigen-centered Gaussian.
        #   velocity_core:
        #       Deprecated deterministic bridge-velocity replacement. It is
        #       retained only for backwards compatibility with v32 logs.
        #   si_score:
        #       PCS_RC_LC_R1 endpoint baseline plus a stochastic-interpolant
        #       analytic score regularizer. No independent score head is added.
        #   si_score_fm:
        #       si_score plus stochastic-interpolant velocity consistency.
        #       Retained for historical diagnostics.
        #   traj_consistency:
        #       PCS_RC_LC_R1 endpoint baseline plus trajectory endpoint
        #       consistency between two neighboring states along the model-
        #       induced flow. No independent score/velocity head is added.
        #   traj_consistency_fm:
        #       traj_consistency plus an induced velocity-field consistency
        #       term. This is retained as a full two-query diagnostic.
        #   score_aware_traj_lite:
        #       One-forward score-aware trajectory/tangent regularization.
        #       The training state is perturbed off the clean bridge by a known
        #       noise direction, so the induced endpoint velocity must contain
        #       an explicit score-like correction back toward the clean path.
        #   score_aware_traj_fm_lite:
        #       score_aware_traj_lite plus a small projected correction-magnitude
        #       consistency term.  No independent score or velocity head is added.
        #   score_aware_traj_if_lite:
        #       Interface-weighted SATC.  The same one-forward off-path score
        #       correction is retained, but the regularizer is weighted toward
        #       native paratope residues close to antigen, directly targeting
        #       DockQ/CAAR without an extra forward or a new prediction head.
        #   score_aware_traj_if_fm_lite:
        #       Interface-weighted SATC plus a very soft projected velocity
        #       magnitude term for controlled flow-field ablation.
        #   score_aware_traj_nt_lite:
        #       Normal--tangent decomposed SATC.  The endpoint-induced velocity
        #       is decomposed into a clean tangent transport component and an
        #       off-path normal correction component.  Only the normal score
        #       projection is constrained, so tangent transport remains governed
        #       by endpoint flow matching.
        #   score_aware_traj_nt_fm_lite:
        #       score_aware_traj_nt_lite plus a very weak normal-correction
        #       magnitude term.  This absorbs the useful AAR/CAAR signal of
        #       SATC_FM without constraining the full velocity vector.
        self.scorefm_loss_mode = _env_str(
            "ABFLOW_SCOREFM_LOSS_MODE", "endpoint"
        ).lower()
        if self.scorefm_loss_mode in {"off", "none", "base"}:
            self.scorefm_loss_mode = "endpoint"
        if self.scorefm_loss_mode in {"core", "dtm_core", "hybrid"}:
            self.scorefm_loss_mode = "analytic_core"
        if self.scorefm_loss_mode in {
            "velocity_core", "flow_velocity", "fm_velocity",
            "pcs_velocity", "pcs_velocity_core",
        }:
            self.scorefm_loss_mode = "velocity_core"
        if self.scorefm_loss_mode in {
            "si_score", "stochastic_score", "score_interpolant",
            "stochastic_interpolant_score",
        }:
            self.scorefm_loss_mode = "si_score"
        if self.scorefm_loss_mode in {
            "si_score_fm", "stochastic_score_fm", "score_fm",
            "stochastic_interpolant", "stochastic_interpolant_fm",
        }:
            self.scorefm_loss_mode = "si_score_fm"
        if self.scorefm_loss_mode in {
            "traj_consistency", "trajectory_consistency", "tc",
            "traj", "r1_traj",
        }:
            self.scorefm_loss_mode = "traj_consistency"
        if self.scorefm_loss_mode in {
            "traj_consistency_fm", "trajectory_consistency_fm",
            "tc_fm", "traj_fm", "r1_traj_fm",
        }:
            self.scorefm_loss_mode = "traj_consistency_fm"
        if self.scorefm_loss_mode in {
            "score_aware_traj_lite", "satc_lite",
            "score_aware_trajectory_lite", "r1_satc_lite",
        }:
            self.scorefm_loss_mode = "score_aware_traj_lite"
        if self.scorefm_loss_mode in {
            "score_aware_traj_fm_lite", "satc_fm_lite",
            "score_aware_trajectory_fm_lite", "r1_satc_fm_lite",
        }:
            self.scorefm_loss_mode = "score_aware_traj_fm_lite"
        if self.scorefm_loss_mode in {
            "score_aware_traj_if_lite", "satc_if_lite",
            "satc_if_main", "interface_satc",
            "score_aware_interface_traj_lite",
        }:
            self.scorefm_loss_mode = "score_aware_traj_if_lite"
        if self.scorefm_loss_mode in {
            "score_aware_traj_if_fm_lite", "satc_if_fm_lite",
            "satc_if_fm_soft", "interface_satc_fm",
            "score_aware_interface_traj_fm_lite",
        }:
            self.scorefm_loss_mode = "score_aware_traj_if_fm_lite"
        if self.scorefm_loss_mode in {
            "score_aware_traj_nt_lite", "satc_nt_lite",
            "satc_nt_main", "normal_tangent_satc",
            "score_aware_normal_tangent_lite",
        }:
            self.scorefm_loss_mode = "score_aware_traj_nt_lite"
        if self.scorefm_loss_mode in {
            "score_aware_traj_nt_fm_lite", "satc_nt_fm_lite",
            "satc_nt_fm_soft", "normal_tangent_satc_fm",
            "score_aware_normal_tangent_fm_lite",
        }:
            self.scorefm_loss_mode = "score_aware_traj_nt_fm_lite"
        if self.scorefm_loss_mode in {
            "score_aware_traj_if_nt_lite", "satc_if_nt_lite",
            "satc_if_nt_main", "interface_normal_tangent_satc",
            "target_aligned_satc", "target_aligned_nt_satc",
        }:
            self.scorefm_loss_mode = "score_aware_traj_if_nt_lite"
        if self.scorefm_loss_mode in {
            "score_aware_traj_if_nt_fm_lite", "satc_if_nt_fm_lite",
            "satc_if_nt_fm_soft", "interface_normal_tangent_satc_fm",
            "target_aligned_satc_fm", "target_aligned_nt_satc_fm",
        }:
            self.scorefm_loss_mode = "score_aware_traj_if_nt_fm_lite"
        if self.scorefm_loss_mode in {
            "score_aware_graph_translation_consistency",
            "graph_translation_satc", "gt_satc", "placement_satc",
            "h3_translation_satc",
        }:
            self.scorefm_loss_mode = "score_aware_graph_translation_consistency"
        if self.scorefm_loss_mode in {
            "structured_global_endpoint",
            "structured_bridge_global_endpoint",
            "ssf_global_endpoint",
        }:
            self.scorefm_loss_mode = "structured_global_endpoint"
        if self.scorefm_loss_mode in {
            "structured_global_cfm",
            "structured_bridge_global_cfm",
            "ssf_global_cfm",
        }:
            self.scorefm_loss_mode = "structured_global_cfm"
        if self.scorefm_loss_mode in {
            "structured_multiscale_cfm",
            "structured_global_local_cfm",
            "ssf_multiscale_cfm",
        }:
            self.scorefm_loss_mode = "structured_multiscale_cfm"
        if self.scorefm_loss_mode not in {
            "endpoint", "analytic_core", "velocity_core",
            "si_score", "si_score_fm",
            "traj_consistency", "traj_consistency_fm",
            "score_aware_traj_lite", "score_aware_traj_fm_lite",
            "score_aware_traj_if_lite", "score_aware_traj_if_fm_lite",
            "score_aware_traj_nt_lite", "score_aware_traj_nt_fm_lite",
            "score_aware_traj_if_nt_lite", "score_aware_traj_if_nt_fm_lite",
            "score_aware_graph_translation_consistency",
            "structured_global_endpoint", "structured_global_cfm",
            "structured_multiscale_cfm",
        }:
            raise ValueError(
                "Unknown ABFLOW_SCOREFM_LOSS_MODE="
                f"{self.scorefm_loss_mode}. Choose from endpoint, "
                "analytic_core, velocity_core, si_score, si_score_fm, "
                "traj_consistency, traj_consistency_fm, "
                "score_aware_traj_lite, score_aware_traj_fm_lite, "
                "score_aware_traj_if_lite, score_aware_traj_if_fm_lite, "
                "score_aware_traj_nt_lite, score_aware_traj_nt_fm_lite, "
                "score_aware_traj_if_nt_lite, score_aware_traj_if_nt_fm_lite, "
                "score_aware_graph_translation_consistency, structured_global_endpoint, "
                "structured_global_cfm, structured_multiscale_cfm."
            )

        # Stochastic-interpolant controls.  These regularizers keep the strong
        # PCS_RC_LC_R1 endpoint objective as the primary target and add a small
        # analytic score / velocity consistency term on noisy intermediate
        # states.  The score is induced by the endpoint head; no extra score
        # head or velocity head is introduced.
        self.si_gamma_scale = _env_float("ABFLOW_SI_GAMMA_SCALE", 0.25)
        if not (0.0 < self.si_gamma_scale <= 1.0):
            raise ValueError("ABFLOW_SI_GAMMA_SCALE must be in (0, 1].")
        self.si_score_weight = _env_float("ABFLOW_SI_SCORE_WEIGHT", 0.002)
        self.si_velocity_weight = _env_float("ABFLOW_SI_VELOCITY_WEIGHT", 0.01)
        if self.si_score_weight < 0.0 or self.si_velocity_weight < 0.0:
            raise ValueError("ABFLOW_SI_*_WEIGHT must be non-negative.")

        # =========================================================
        # Primary structured stochastic conditional path (v56)
        # =========================================================
        # This is the probability path itself, not an SATC/GT auxiliary.
        # H3 is translated as one Cartesian block around the PCS-RC -> native
        # interpolant, preserving every intra-H3 atom/residue distance.
        self.structured_gamma_scale = _env_float(
            "ABFLOW_STRUCTURED_GAMMA_SCALE", 0.05
        )
        self.structured_transport_max = _env_float(
            "ABFLOW_STRUCTURED_TRANSPORT_MAX", 20.0
        )
        self.structured_gamma_abs_max = _env_float(
            "ABFLOW_STRUCTURED_GAMMA_ABS_MAX", 1.0
        )
        if not (0.0 < self.structured_gamma_scale <= 1.0):
            raise ValueError("ABFLOW_STRUCTURED_GAMMA_SCALE must be in (0, 1].")
        if self.structured_transport_max <= 0.0:
            raise ValueError("ABFLOW_STRUCTURED_TRANSPORT_MAX must be positive.")
        if self.structured_gamma_abs_max <= 0.0:
            raise ValueError("ABFLOW_STRUCTURED_GAMMA_ABS_MAX must be positive.")

        # Orthogonal local path scale for S03.  This is dimensionless relative
        # to the actual PCS-RC -> native local CA deformation after removing the
        # graph-level translation component.  The default intentionally matches
        # the global 0.05 scale so S03 adds a new geometric subspace rather than
        # a new hand-tuned strength regime.
        self.structured_local_gamma_scale = _env_float(
            "ABFLOW_STRUCTURED_LOCAL_GAMMA_SCALE", 0.05
        )
        if not (0.0 < self.structured_local_gamma_scale <= 1.0):
            raise ValueError(
                "ABFLOW_STRUCTURED_LOCAL_GAMMA_SCALE must be in (0, 1]."
            )

        # Trajectory-consistency controls.
        # These terms do not introduce a new score head or velocity head.
        # The model is evaluated at Xt and at a neighboring model-induced
        # state Xt+dt, and the induced endpoint / velocity field is required
        # to be locally self-consistent.  Endpoint reconstruction remains the
        # main supervised objective.
        self.traj_consistency_weight = _env_float(
            "ABFLOW_TRAJ_CONSISTENCY_WEIGHT", 0.05
        )
        self.traj_velocity_weight = _env_float(
            "ABFLOW_TRAJ_VELOCITY_WEIGHT", 0.0
        )
        self.traj_delta_t = _env_float("ABFLOW_TRAJ_DELTA_T", 0.15)
        self.traj_t_min = _env_float("ABFLOW_TRAJ_T_MIN", 0.05)
        self.traj_t_max = _env_float("ABFLOW_TRAJ_T_MAX", 0.80)
        if self.traj_consistency_weight < 0.0 or self.traj_velocity_weight < 0.0:
            raise ValueError("ABFLOW_TRAJ_*_WEIGHT must be non-negative.")
        if not (0.0 < self.traj_delta_t < 1.0):
            raise ValueError("ABFLOW_TRAJ_DELTA_T must be in (0, 1).")
        if not (0.0 <= self.traj_t_min < self.traj_t_max <= 1.0):
            raise ValueError("Require 0 <= ABFLOW_TRAJ_T_MIN < ABFLOW_TRAJ_T_MAX <= 1.")

        # Lightweight score-aware trajectory controls.  Unlike full trajectory
        # consistency, these modes do not call _forward a second time.  They
        # perturb the current bridge state off the clean trajectory and then
        # require the endpoint-induced velocity to contain a correction component
        # aligned with the known analytic score direction (-epsilon).
        self.satc_apply_prob = _env_float("ABFLOW_SATC_APPLY_PROB", 0.50)
        self.satc_gamma_scale = _env_float("ABFLOW_SATC_GAMMA_SCALE", 0.08)
        self.satc_score_weight = _env_float("ABFLOW_SATC_SCORE_WEIGHT", 0.02)
        self.satc_velocity_weight = _env_float("ABFLOW_SATC_VELOCITY_WEIGHT", 0.003)
        self.satc_t_min = _env_float("ABFLOW_SATC_T_MIN", 0.10)
        self.satc_t_max = _env_float("ABFLOW_SATC_T_MAX", 0.80)

        # Tube calibration keeps the existing full-atom Cartesian AbFlow state.
        # legacy_absolute reproduces v45 exactly:
        #     gamma(t) = gamma_scale * t * (1-t).
        # transport_calibrated removes the arbitrary Angstrom coefficient and
        # measures tube width relative to the current PCS source-to-native path:
        #     gamma_g(t) = gamma_scale * RMS_g(X1-X0) * 4t(1-t).
        # Here gamma_scale is dimensionless and gamma_g reaches the configured
        # fraction of the graph-level transport RMS at t=0.5.  The absolute cap
        # is only a safety guard against malformed proposals; it is not the main
        # scale definition.  Noise remains iid in the actual AbFlow state space,
        # so the analytic isotropic score direction used by SATC is unchanged.
        self.satc_tube_mode = _env_str(
            "ABFLOW_SATC_TUBE_MODE", "legacy_absolute"
        ).lower()
        if self.satc_tube_mode not in {
            "legacy_absolute", "transport_calibrated",
            "graph_translation_calibrated",
        }:
            raise ValueError(
                "ABFLOW_SATC_TUBE_MODE must be legacy_absolute, "
                "transport_calibrated or graph_translation_calibrated."
            )
        self.satc_transport_rms_min = _env_float(
            "ABFLOW_SATC_TRANSPORT_RMS_MIN", 0.25
        )
        self.satc_transport_rms_max = _env_float(
            "ABFLOW_SATC_TRANSPORT_RMS_MAX", 20.0
        )
        self.satc_gamma_abs_max = _env_float(
            "ABFLOW_SATC_GAMMA_ABS_MAX", 0.50
        )
        if not (
            0.0 < self.satc_transport_rms_min
            < self.satc_transport_rms_max
        ):
            raise ValueError(
                "Require 0 < ABFLOW_SATC_TRANSPORT_RMS_MIN < "
                "ABFLOW_SATC_TRANSPORT_RMS_MAX."
            )
        if self.satc_gamma_abs_max <= 0.0:
            raise ValueError("ABFLOW_SATC_GAMMA_ABS_MAX must be positive.")

        # Projection bounding is separated from the NT semantics.  hard_clip
        # preserves the raw projection coefficient and therefore keeps every
        # target (including magnitude target 1) at its exact theoretical value.
        # legacy_tanh is retained only to reproduce v45.
        self.satc_projection_bound_mode = _env_str(
            "ABFLOW_SATC_PROJECTION_BOUND_MODE", "legacy_tanh"
        ).lower()
        if self.satc_projection_bound_mode not in {
            "legacy_tanh", "hard_clip"
        }:
            raise ValueError(
                "ABFLOW_SATC_PROJECTION_BOUND_MODE must be legacy_tanh "
                "or hard_clip."
            )
        self.satc_magnitude_loss_mode = _env_str(
            "ABFLOW_SATC_MAGNITUDE_LOSS_MODE", "legacy_tanh"
        ).lower()
        if self.satc_magnitude_loss_mode not in {
            "legacy_tanh", "unbiased_ratio_huber"
        }:
            raise ValueError(
                "ABFLOW_SATC_MAGNITUDE_LOSS_MODE must be legacy_tanh "
                "or unbiased_ratio_huber."
            )
        if not (0.0 <= self.satc_apply_prob <= 1.0):
            raise ValueError("ABFLOW_SATC_APPLY_PROB must be in [0, 1].")
        if not (0.0 < self.satc_gamma_scale <= 1.0):
            raise ValueError("ABFLOW_SATC_GAMMA_SCALE must be in (0, 1].")
        if self.satc_score_weight < 0.0 or self.satc_velocity_weight < 0.0:
            raise ValueError("ABFLOW_SATC_*_WEIGHT must be non-negative.")
        if not (0.0 <= self.satc_t_min < self.satc_t_max <= 1.0):
            raise ValueError("Require 0 <= ABFLOW_SATC_T_MIN < ABFLOW_SATC_T_MAX <= 1.")

        # Normal--tangent SATC controls.  Direction-only NT uses a lower-bound
        # pull along the analytic normal score direction instead of forcing the
        # whole correction vector to align with score.  This leaves orthogonal
        # endpoint/transport errors to the main endpoint objective.
        self.satc_nt_min_pull = _env_float("ABFLOW_SATC_NT_MIN_PULL", 0.15)
        self.satc_nt_pull_clip = _env_float("ABFLOW_SATC_NT_PULL_CLIP", 2.0)
        if not (0.0 <= self.satc_nt_min_pull <= 1.5):
            raise ValueError("ABFLOW_SATC_NT_MIN_PULL must be in [0, 1.5].")
        if not (0.25 <= self.satc_nt_pull_clip <= 10.0):
            raise ValueError("ABFLOW_SATC_NT_PULL_CLIP must be in [0.25, 10.0].")

        # Interface-weighted SATC controls.  These do not change the endpoint
        # objective and do not add a second forward pass.  They only reweight the
        # score-aware correction regularizer toward paratope residues that are
        # close to antigen in the native complex, so the added signal is targeted
        # at DockQ/CAAR rather than spread uniformly over all CDR residues.
        self.satc_interface_weight_alpha = _env_float(
            "ABFLOW_SATC_INTERFACE_WEIGHT_ALPHA", 1.0
        )
        self.satc_interface_cutoff = _env_float(
            "ABFLOW_SATC_INTERFACE_CUTOFF", 8.0
        )
        self.satc_interface_temperature = _env_float(
            "ABFLOW_SATC_INTERFACE_TEMPERATURE", 1.0
        )
        self.satc_interface_normalize = _env_flag(
            "ABFLOW_SATC_INTERFACE_NORMALIZE", True
        )
        if self.satc_interface_weight_alpha < 0.0:
            raise ValueError("ABFLOW_SATC_INTERFACE_WEIGHT_ALPHA must be non-negative.")
        if self.satc_interface_cutoff <= 0.0:
            raise ValueError("ABFLOW_SATC_INTERFACE_CUTOFF must be positive.")
        if self.satc_interface_temperature <= 0.0:
            raise ValueError("ABFLOW_SATC_INTERFACE_TEMPERATURE must be positive.")

        # SATC schedule controls.  Current experiments show that the strongest
        # mechanism is SATC_MAIN (score-aware correction direction), whereas the
        # FM/velocity-magnitude component can help AAR/CAAR but becomes harmful
        # if kept at full strength late in training.  We therefore decouple the
        # late-stage schedules for off-path perturbation, score-direction loss
        # and velocity-magnitude loss.
        self.satc_schedule = _env_str("ABFLOW_SATC_SCHEDULE", "constant").lower()
        if self.satc_schedule in {"none", "off"}:
            self.satc_schedule = "constant"
        if self.satc_schedule not in {"constant", "linear_decay", "cosine_decay"}:
            raise ValueError(
                "ABFLOW_SATC_SCHEDULE must be constant, linear_decay or cosine_decay."
            )
        self.satc_steps_per_epoch = max(1, _env_int("ABFLOW_SATC_STEPS_PER_EPOCH", 52))
        self.satc_decay_start_epoch = _env_float("ABFLOW_SATC_DECAY_START_EPOCH", 100.0)
        self.satc_decay_end_epoch = _env_float("ABFLOW_SATC_DECAY_END_EPOCH", 130.0)
        self.satc_perturb_final_scale = _env_float("ABFLOW_SATC_PERTURB_FINAL_SCALE", 1.0)
        self.satc_score_final_scale = _env_float("ABFLOW_SATC_SCORE_FINAL_SCALE", 1.0)
        self.satc_velocity_final_scale = _env_float("ABFLOW_SATC_VELOCITY_FINAL_SCALE", 1.0)
        if self.satc_decay_end_epoch <= self.satc_decay_start_epoch:
            raise ValueError("ABFLOW_SATC_DECAY_END_EPOCH must be > ABFLOW_SATC_DECAY_START_EPOCH.")
        for _name, _value in {
            "ABFLOW_SATC_PERTURB_FINAL_SCALE": self.satc_perturb_final_scale,
            "ABFLOW_SATC_SCORE_FINAL_SCALE": self.satc_score_final_scale,
            "ABFLOW_SATC_VELOCITY_FINAL_SCALE": self.satc_velocity_final_scale,
        }.items():
            if not (0.0 <= float(_value) <= 1.0):
                raise ValueError(f"{_name} must be in [0, 1].")
        self.register_buffer(
            "satc_train_step", torch.zeros((), dtype=torch.long), persistent=True
        )

        # v52 graph-translation SATC scheduling.  The extra teacher query is
        # executed at a deterministic interval shared by every DDP rank.  This
        # avoids rank-dependent control flow while limiting the average cost.
        # The auxiliary is activated only after the endpoint field has learned a
        # usable clean bridge.  Validation never applies the stochastic branch.
        self.satc_gt_interval = max(
            1, _env_int("ABFLOW_SATC_GT_INTERVAL", 4)
        )
        self.satc_gt_start_epoch = _env_float(
            "ABFLOW_SATC_GT_START_EPOCH", 5.0
        )
        if self.satc_gt_start_epoch < 0.0:
            raise ValueError("ABFLOW_SATC_GT_START_EPOCH must be non-negative.")

        self.scorefm_dsm_t_min = _env_float(
            "ABFLOW_SCOREFM_DSM_T_MIN", 0.2
        )
        self.scorefm_dsm_t_max = _env_float(
            "ABFLOW_SCOREFM_DSM_T_MAX", 0.8
        )
        if not (
            0.0 <= self.scorefm_dsm_t_min
            < self.scorefm_dsm_t_max <= 1.0
        ):
            raise ValueError(
                "Require 0 <= ABFLOW_SCOREFM_DSM_T_MIN < "
                "ABFLOW_SCOREFM_DSM_T_MAX <= 1."
            )

        # State-path controls.
        self.scorefm_per_sample_t = _env_flag(
            "ABFLOW_SCOREFM_PER_SAMPLE_T", True
        )
        self.scorefm_t_sampling = _env_str(
            "ABFLOW_SCOREFM_T_SAMPLING", "uniform"
        ).lower()
        self.scorefm_state_path = _env_flag(
            "ABFLOW_SCOREFM_STATE_PATH", True
        )

        # Node-level time conditioning. Pair-time conditioning is deliberately
        # deferred so that the present experiments remain attributable.
        self.scorefm_time_embed = _env_flag(
            "ABFLOW_SCOREFM_TIME_EMBED", True
        )
        self.flow_time_mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.SiLU(),
            nn.Linear(embed_size, embed_size),
        )

        # Minimal sampler set: residual is a conservative ablation; bridge is
        # the endpoint-parameterized FM sampler used by default.
        self.scorefm_sampler_mode = _env_str(
            "ABFLOW_SCOREFM_SAMPLER_MODE", "bridge"
        ).lower()
        if self.scorefm_sampler_mode not in {"residual", "bridge"}:
            raise ValueError(
                "Unknown ABFLOW_SCOREFM_SAMPLER_MODE="
                f"{self.scorefm_sampler_mode}. Choose from residual, bridge."
            )

        # =========================================================
        # Proposal-conditioned source and recurrent proposal context
        # =========================================================
        # Source modes:
        #   reference:
        #       X_0/S_0 are sampled from the antigen-centered reference source.
        #   pcs:
        #       Proposal-conditioned source. X_pep/S_pep define or bias X_0/S_0,
        #       but the recurrent global context remains the current generated
        #       state. This tests the source role alone.
        #   pcs_rc:
        #       Proposal-conditioned source + recurrent proposal context. X_pep/S_pep
        #       define or bias X_0/S_0 and are also used to build the global
        #       proposal context at every _forward call, while interface_X/St remain
        #       the explicit generated state. This recovers the original AbFlow
        #       information strength without overwriting the generated state.
        self.abflow_source_mode = _env_str(
            "ABFLOW_SOURCE_MODE", "reference"
        ).lower()
        if self.abflow_source_mode in {"ref", "reference"}:
            self.abflow_source_mode = "reference"
        elif self.abflow_source_mode in {"cond", "conditional", "proposal", "pcs"}:
            self.abflow_source_mode = "pcs"
        elif self.abflow_source_mode in {"pcs_rc", "proposal_context", "proposal_recurrent_context"}:
            self.abflow_source_mode = "pcs_rc"
        else:
            raise ValueError(
                "Unknown ABFLOW_SOURCE_MODE="
                f"{self.abflow_source_mode}. Choose reference, pcs, or pcs_rc."
            )

        self.abflow_recurrent_proposal_context = _env_flag(
            "ABFLOW_RECURRENT_PROPOSAL_CONTEXT",
            self.abflow_source_mode == "pcs_rc",
        )

        # Deterministic proposal-conditioned source.
        #
        # We intentionally remove continuous peptide-source weights from the
        # formal method.  In PCS/PCS-RC, a valid proposal defines the source
        # state; invalid proposal residues fall back to the reference source.
        # This avoids heuristic mixtures such as 0.5 * reference + 0.5 * proposal
        # and makes the base distribution easy to state and reproduce.
        self.coord_pep_source_weight = 1.0
        self.seq_pep_source_weight = 1.0

        # =========================================================
        # Peptide information as condition, never as source-state injection
        # =========================================================
        # Coordinate proposal conditioning is represented only in scalar hidden
        # space. The directional CA displacement is expressed in the proposal's
        # local N-CA-C frame, which is stable even when the current flow state is
        # highly noisy at low t. Distance statistics use log1p compression to
        # limit dynamic range. The resulting scalar features are invariant to global
        # SE(3) transformations. Crucially, X_pep never directly updates
        # interface_X; it conditions f_theta instead of acting as a post-hoc
        # coordinate correction.
        self.coord_pep_as_condition = _env_flag(
            "ABFLOW_COORD_PEP_AS_CONDITION", False
        )
        self.coord_pep_condition_dim = 6
        if self.coord_pep_as_condition:
            # Input: current hidden state H_0 plus six E(3)-invariant features:
            #   local-frame CA displacement (3),
            #   CA distance, mean backbone distance, RMS backbone distance (3).
            self.coord_pep_condition_adapter = nn.Sequential(
                nn.Linear(embed_size + self.coord_pep_condition_dim, embed_size),
                nn.SiLU(),
                nn.Linear(embed_size, embed_size),
            )
            # Zero-start residual adapter: the initial model is exactly REF.
            nn.init.zeros_(self.coord_pep_condition_adapter[-1].weight)
            nn.init.zeros_(self.coord_pep_condition_adapter[-1].bias)
        else:
            self.coord_pep_condition_adapter = None

        # S_pep is a proposal-token condition, not an ESM representation and
        # never a replacement for S_t.  Fusion is residue- and state-dependent:
        # H_0 already contains the current state and time embedding, so the
        # adapter can learn when the proposal token is useful instead of applying
        # one global scalar to every residue and every time.
        self.seq_input_mode = _env_str(
            "ABFLOW_SEQ_INPUT_MODE", "state"
        ).lower()
        if self.seq_input_mode not in {"state", "pep_condition"}:
            raise ValueError(
                "Unknown ABFLOW_SEQ_INPUT_MODE="
                f"{self.seq_input_mode}. Choose from state, pep_condition."
            )
        if self.seq_input_mode == "pep_condition":
            self.seq_pep_condition_embedding = nn.Embedding(
                num_classes, embed_size
            )
            self.seq_pep_condition_adapter = nn.Sequential(
                nn.Linear(2 * embed_size, embed_size),
                nn.SiLU(),
                nn.Linear(embed_size, embed_size),
            )
            nn.init.zeros_(self.seq_pep_condition_adapter[-1].weight)
            nn.init.zeros_(self.seq_pep_condition_adapter[-1].bias)
        else:
            self.seq_pep_condition_embedding = None
            self.seq_pep_condition_adapter = None

        # Dual-role sequence state/context for PCS-RC.
        #
        # S_t is the generated categorical state.  S_pep remains a proposal
        # condition/context and must never be blended into a fractional amino
        # acid.  v52 therefore introduces ``hard_exact``: the residue-identity
        # contribution, atom identities, atom masks and atom weights on the H3
        # state are replaced exactly by those implied by S_t.  No learned state
        # adapter and no second multiplication by t are used.
        legacy_shadow = _env_flag("ABFLOW_SHADOW_SEQ_STATE", False)
        self.dual_sequence_state = _env_flag(
            "ABFLOW_DUAL_SEQUENCE_STATE", legacy_shadow
        )
        self.shadow_seq_state = self.dual_sequence_state
        self.dual_sequence_atom_mode = _env_str(
            "ABFLOW_DUAL_SEQUENCE_ATOM_MODE", "hard_exact"
        ).lower()
        if self.dual_sequence_atom_mode not in {
            "hard", "hard_exact", "hidden_only", "time_gated_union"
        }:
            raise ValueError(
                "ABFLOW_DUAL_SEQUENCE_ATOM_MODE must be hard, hard_exact, "
                "hidden_only or time_gated_union."
            )

        # Legacy learned adapters are retained only for historical modes.  The
        # formal v52 hard_exact path is parameter-free and cannot grow until it
        # dominates the PCS-RC hidden state.
        if (
            self.dual_sequence_state
            and not self.struct_only
            and self.dual_sequence_atom_mode != "hard_exact"
        ):
            self.seq_state_adapter = nn.Sequential(
                nn.Linear(2 * embed_size, embed_size),
                nn.SiLU(),
                nn.Linear(embed_size, embed_size),
            )
            nn.init.zeros_(self.seq_state_adapter[-1].weight)
            nn.init.zeros_(self.seq_state_adapter[-1].bias)
        else:
            self.seq_state_adapter = None
        self.seq_state_embedding = None

        # =========================================================
        # Joint path/sampler consistency controls (v52)
        # =========================================================
        # ``legacy`` reproduces the original context curriculum, where most
        # designed residues are replaced by native context early in training.
        # ``loss_only`` keeps the complete categorical path state for every
        # designed residue, but may subsample the CE supervision mask.
        # ``off`` disables native-context curriculum entirely and is the clean
        # train/inference-matched setting used by the formal joint-path runs.
        self.sequence_context_mode = _env_str(
            "ABFLOW_SEQUENCE_CONTEXT_MODE", "legacy"
        ).lower()
        if self.sequence_context_mode not in {"legacy", "loss_only", "off"}:
            raise ValueError(
                "ABFLOW_SEQUENCE_CONTEXT_MODE must be legacy, loss_only or off."
            )

        # The bridge Euler/CTMC step already reaches the terminal state.  The
        # old sampler queried the network once more at exactly t=1, a boundary
        # never sampled during continuous-time training, and replaced the
        # integrated state with that extra prediction.  Formal runs use the
        # integrated endpoint and the last left-endpoint logits instead.
        self.final_readout_mode = _env_str(
            "ABFLOW_FINAL_READOUT_MODE", "integrated_endpoint"
        ).lower()
        if self.final_readout_mode not in {
            "integrated_endpoint", "legacy_t1_query"
        }:
            raise ValueError(
                "ABFLOW_FINAL_READOUT_MODE must be integrated_endpoint or "
                "legacy_t1_query."
            )
        self.sequence_decode_mode = _env_str(
            "ABFLOW_SEQUENCE_DECODE_MODE", "argmax"
        ).lower()
        if self.sequence_decode_mode not in {"argmax", "ctmc_sample"}:
            raise ValueError(
                "ABFLOW_SEQUENCE_DECODE_MODE must be argmax or ctmc_sample."
            )

        # Fixed validation paths make checkpoint ranking comparable across
        # epochs.  Training remains stochastic.
        self.deterministic_validation = _env_flag(
            "ABFLOW_DETERMINISTIC_VALIDATION", True
        )

        # Diagnostics are observational only. Gradient-conflict probing is
        # explicitly periodic because autograd.grad adds cost.
        self.grad_conflict_diagnostics = _env_flag(
            "ABFLOW_GRAD_CONFLICT_DIAGNOSTICS", False
        )
        self.last_gradient_diagnostics = {}
        self._diagnostic_objective_tensors = {}
        # AMP/DDP-safe gradient-conflict probe. Differentiate objectives with
        # respect to a shared activation rather than a DDP parameter.
        self._diagnostic_probe_tensor = None
        self._last_gradient_diagnostic_error = ""
        self._diagnostic_validation_mode = False
        # Set by the trainer only on recorded/probed steps so diagnostics do not
        # turn every expensive training batch into a synchronization point.
        self._diagnostic_capture = False

        self.seq_ce_weight = _env_float("ABFLOW_SEQ_CE_WEIGHT", 1.0)

        # Local-correction schedule for proposal adapters.
        #
        # start_round=0 reproduces PCS_RC_COND: proposal-relative adapters are
        # active before the first refinement round and can influence placement.
        # start_round=1 is the recommended PCS_RC_LC setting: the first round
        # establishes the interface placement using the PCS-RC backbone, while
        # later rounds use proposal-relative features for local geometry and
        # sequence correction.  This directly targets the observed trade-off:
        # preserve PCS_RC raw H3 placement/DockQ while absorbing the local
        # structural benefit of PCS_RC_COND.
        self.proposal_adapter_start_round = max(
            0, _env_int("ABFLOW_PROPOSAL_ADAPTER_START_ROUND", 0)
        )

        self.last_scorefm_losses = {}
        self.last_abflow_diagnostics = {}
        # Detached condition-strength diagnostics. They are useful for debugging
        # but require GPU reductions and occasional synchronizations, so they are
        # disabled by default for expensive training runs.
        self.condition_diagnostics_enabled = _env_flag(
            "ABFLOW_CONDITION_DIAGNOSTICS", False
        )
        # Expensive safety checks that force GPU->CPU synchronization are off in
        # normal training. Enable only when debugging malformed edge tensors.
        self.runtime_checks = _env_flag("ABFLOW_RUNTIME_CHECKS", False)
        self._last_condition_diagnostics = {}
        self._latest_condition_diagnostics = {}


    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return X, S
    
    def replace_pep(self, X, S, paratope_mask, X_pep, S_pep,
                    replace_seq=True, replace_struct=True):
        """Build a proposal-conditioned global context.

        This function is not used to overwrite the explicit generated state
        Xt/St.  In PCS-RC mode it creates the recurrent proposal context that
        the original AbFlow effectively used through hard replacement, while
        the shadow interface still receives the actual generated state.
        """
        if (
            replace_seq
            and getattr(self, 'pep_seq', True)
            and S_pep is not None
            and S_pep.numel() == int(paratope_mask.sum().item())
        ):
            pep_S = S_pep.to(device=S.device, dtype=torch.long)
            valid = (pep_S >= 0) & (pep_S < self.num_classes)
            if valid.any():
                local_S = S[paratope_mask].clone()
                local_S = torch.where(valid, pep_S, local_S)
                S[paratope_mask] = local_S

        if (
            replace_struct
            and getattr(self, 'pep_struct', True)
            and X_pep is not None
            and X_pep.shape == X[paratope_mask].shape
        ):
            pep_X = X_pep.to(device=X.device, dtype=X.dtype)
            proposal_backbone = pep_X[:, :min(3, pep_X.shape[1])]
            valid = (
                torch.isfinite(proposal_backbone).all(dim=-1).all(dim=-1)
                & (proposal_backbone.abs().sum(dim=-1).sum(dim=-1) > self.scorefm_eps)
            )
            if valid.any():
                local_X = X[paratope_mask].clone()
                local_X = torch.where(valid.view(-1, 1, 1), pep_X, local_X)
                X[paratope_mask] = local_X
        return X, S

    @torch.no_grad()
    def _condition_initial_interface(self, interface_X, interface_S, X_pep, S_pep):
        """Sample a deterministic proposal-conditioned source state.

        reference mode keeps the antigen-centered random source.  PCS/PCS-RC
        mode uses X_pep/S_pep as the declared source whenever the corresponding
        proposal residue is valid; invalid proposal residues fall back to the
        reference source.

        No continuous source mixing weight is used here.  This is deliberate:
        the formal base should not depend on an unexplained heuristic coefficient.
        """
        if self.abflow_source_mode not in {"pcs", "pcs_rc"}:
            return interface_X, interface_S

        if (
            getattr(self, 'pep_struct', True)
            and X_pep is not None
            and X_pep.shape == interface_X.shape
        ):
            pep_X = X_pep.to(device=interface_X.device, dtype=interface_X.dtype)
            proposal_backbone = pep_X[:, :min(3, pep_X.shape[1])]
            valid = (
                torch.isfinite(proposal_backbone).all(dim=-1).all(dim=-1)
                & (proposal_backbone.abs().sum(dim=-1).sum(dim=-1) > self.scorefm_eps)
            )
            if valid.any():
                interface_X = torch.where(valid.view(-1, 1, 1), pep_X, interface_X)

        if (
            not self.struct_only
            and getattr(self, 'pep_seq', True)
            and S_pep is not None
            and S_pep.shape == interface_S.shape
        ):
            pep_S = S_pep.to(device=interface_S.device, dtype=torch.long)
            valid = (pep_S >= 0) & (pep_S < self.num_classes)
            if valid.any():
                interface_S = torch.where(valid, pep_S, interface_S)

        return interface_X, interface_S

    @torch.no_grad()
    def _sample_categorical_path(self, clean_S, base_S, t_graph,
                                 interface_batch_id, corrupt_mask=None):
        """Sample the linear categorical bridge q_t(S_t | S_1, S_0).

        Conditional on a source/target pair, each residue is at the target token
        with probability t and at the source token with probability 1-t.  This
        is the same convex categorical path whose endpoint-prediction CTMC has
        jump hazard 1/(1-t).  Training therefore remains stochastic, whereas
        validation uses a fixed pseudo-random draw so checkpoint losses are
        comparable across epochs.
        """
        t_graph = torch.as_tensor(
            t_graph, device=clean_S.device, dtype=torch.float32
        )
        if t_graph.dim() == 0 or t_graph.numel() == 1:
            keep_prob = t_graph.reshape(1).expand_as(clean_S)
        else:
            keep_prob = t_graph[interface_batch_id]

        if self.deterministic_validation and not self.training:
            # Stateless deterministic uniforms in [0, 1).  The construction is
            # independent of global RNG state and therefore identical at every
            # validation epoch for the same residue ordering.
            idx = torch.arange(
                clean_S.numel(), device=clean_S.device, dtype=torch.float32
            )
            uniforms = torch.frac(
                torch.sin((idx + 1.0) * 12.9898) * 43758.5453
            ).abs()
        else:
            uniforms = torch.rand(
                clean_S.shape, device=clean_S.device
            )

        keep_clean = uniforms < keep_prob.clamp(0.0, 1.0)
        sampled = torch.where(keep_clean, clean_S, base_S).long()
        if corrupt_mask is None:
            return sampled
        corrupt_mask = corrupt_mask.to(device=clean_S.device, dtype=torch.bool)
        return torch.where(corrupt_mask, sampled, clean_S).long()

    def align_epi_ab(self, local_inter_edges, local_is_ab):
        """Orient every cross-interface edge as antigen -> antibody.

        Previous versions used a Python loop over edges. That forced thousands
        of small CPU-controlled tensor writes per batch and easily lowered GPU
        utilization.  This vectorized version performs the same orientation with
        boolean masks on the current device.

        Input:
            local_inter_edges: [2, E] local edges after KNN selection.
            local_is_ab:       [N_local] True for antibody/paratope nodes.

        Output:
            aligned:   [2, E], every edge is epitope/antigen -> antibody.
            epi_index: local indices of antigen/epitope nodes.
        """
        if local_inter_edges.dim() != 2 or local_inter_edges.shape[0] != 2:
            raise ValueError(
                "local_inter_edges must have shape [2, E], got "
                f"{tuple(local_inter_edges.shape)}."
            )

        row, col = local_inter_edges[0], local_inter_edges[1]
        row_is_ab = local_is_ab[row]
        col_is_ab = local_is_ab[col]

        if self.runtime_checks:
            valid_cross = torch.logical_xor(row_is_ab, col_is_ab)
            if not bool(valid_cross.all()):
                bad = int((~valid_cross).sum().detach().cpu().item())
                raise RuntimeError(
                    f"Found {bad} non-cross edges in local_inter_edges."
                )

        # If row is antibody and col is antigen, swap so row=antigen, col=antibody.
        swap = row_is_ab & (~col_is_ab)
        aligned = local_inter_edges.clone()
        aligned[0, swap] = col[swap]
        aligned[1, swap] = row[swap]

        epi_index = torch.nonzero(~local_is_ab, as_tuple=False).reshape(-1)
        return aligned, epi_index

    def optimal_alignment(self, X0, target_X):
        """
        计算X0到target_X的最优旋转和排序
        Args:
            X0: [N, n_channel, 3] 初始构象
            target_X: [N, n_channel, 3] 目标构象
        Returns:
            R: [3, 3] 最优旋转矩阵
            perm: [N] 最优排序
            X0_aligned: [N, n_channel, 3] 经过旋转和排序后的X0
        """
        from scipy.optimize import linear_sum_assignment
        # 1. 先计算最优旋转
        X0_flat = X0.reshape(-1, 3)
        target_X_flat = target_X.reshape(-1, 3)
        _, R, t = kabsch_torch(X0_flat, target_X_flat)
        X0_rotated = torch.matmul(X0, R.T) + t
        
        # 2. 计算最优排序 (使用匈牙利算法)
        cost_matrix = torch.cdist(X0_rotated.reshape(-1, 3), target_X.reshape(-1, 3))
        cost_matrix = cost_matrix.reshape(X0.shape[0], X0.shape[1], -1)  # [N, n_channel, N*n_channel]
        cost_matrix = cost_matrix.reshape(X0.shape[0]*X0.shape[1], -1)  # [N*n_channel, N*n_channel]
        
        # 使用匈牙利算法找最优匹配
        perm = linear_sum_assignment(cost_matrix.cpu().numpy())[1]
        perm = torch.from_numpy(perm).to(X0.device)
        
        # 应用旋转和排序
        X0_aligned = X0_rotated.reshape(-1, 3)[perm].reshape(X0.shape)
        
        return R, perm, X0_aligned
        

    def _sample_flow_times(self, batch_size, device, dtype=torch.float32):
        """Sample continuous flow times.

        Training remains stochastic.  Validation uses a fixed midpoint-stratified
        grid so validation loss changes reflect model changes rather than a new
        random set of path times.
        """
        n = int(batch_size) if getattr(self, 'scorefm_per_sample_t', False) else 1
        mode = getattr(self, 'scorefm_t_sampling', 'uniform')

        if self.deterministic_validation and not self.training:
            if n <= 1:
                t = torch.full((n,), 0.5, device=device, dtype=dtype)
            else:
                t = (
                    torch.arange(n, device=device, dtype=dtype) + 0.5
                ) / float(n)
            if mode in {'low_t', 'low', 'square'}:
                t = t ** 2
            elif mode in {'mid_t', 'mid'}:
                t = 0.2 + 0.6 * t
            elif mode in {'late_t', 'late'}:
                t = 0.55 + 0.35 * t
            elif mode not in {'uniform', 'stratified', 'strat'}:
                raise ValueError(
                    f"Unknown ABFLOW_SCOREFM_T_SAMPLING={mode}. "
                    "Choose from uniform, low_t, stratified, mid_t, late_t."
                )
            return t.clamp(min=0.0, max=1.0)

        if mode == 'uniform':
            t = torch.rand(n, device=device, dtype=dtype)

        elif mode in {'low_t', 'low', 'square'}:
            u = torch.rand(n, device=device, dtype=dtype)
            t = u ** 2

        elif mode in {'stratified', 'strat'}:
            if n <= 1:
                t = torch.rand(n, device=device, dtype=dtype)
            else:
                base = (torch.arange(n, device=device, dtype=dtype) +
                        torch.rand(n, device=device, dtype=dtype)) / float(n)
                perm = torch.randperm(n, device=device)
                t = base[perm]

        elif mode in {'mid_t', 'mid'}:
            t = 0.2 + 0.6 * torch.rand(n, device=device, dtype=dtype)

        elif mode in {'late_t', 'late'}:
            t = 0.55 + 0.35 * torch.rand(n, device=device, dtype=dtype)

        else:
            raise ValueError(
                f"Unknown ABFLOW_SCOREFM_T_SAMPLING={mode}. "
                "Choose from uniform, low_t, stratified, mid_t, late_t."
            )

        return t.clamp(min=0.0, max=1.0)

    def _time_for_interface(self, t_graph, interface_batch_id, ref_tensor):
        """Broadcast graph-level time to [N_interface, 1, 1]."""
        if t_graph is None:
            return None
        t_graph = torch.as_tensor(t_graph, device=ref_tensor.device, dtype=ref_tensor.dtype)
        if t_graph.dim() == 0 or t_graph.numel() == 1:
            return t_graph.reshape(1, 1, 1)
        return t_graph[interface_batch_id].reshape(-1, 1, 1)

    def _flow_time_embedding_for_residues(self, flow_t, batch_id, H_0):
        """Create residue-wise time embeddings aligned with H_0."""
        if flow_t is None or not getattr(self, 'scorefm_time_embed', False):
            return None
        flow_t = torch.as_tensor(flow_t, device=H_0.device, dtype=H_0.dtype)
        if flow_t.dim() == 0 or flow_t.numel() == 1:
            n_graph = int(batch_id.max().item()) + 1 if batch_id.numel() > 0 else 1
            flow_t = flow_t.reshape(1).expand(n_graph)

        t_emb = get_timestep_embedding(flow_t, H_0.shape[-1]).to(dtype=H_0.dtype, device=H_0.device)
        t_emb = self.flow_time_mlp(t_emb)
        return t_emb[batch_id]


    def _flow_time_values_for_residues(self, flow_t, batch_id, ref_tensor):
        """Broadcast graph-level t to one scalar per residue."""
        if flow_t is None:
            return torch.ones(
                ref_tensor.shape[0], device=ref_tensor.device,
                dtype=ref_tensor.dtype
            )
        t = torch.as_tensor(
            flow_t, device=ref_tensor.device, dtype=ref_tensor.dtype
        )
        if t.dim() == 0 or t.numel() == 1:
            return t.reshape(1).expand(ref_tensor.shape[0]).clamp(0.0, 1.0)
        t = t.reshape(-1)
        return t[batch_id].clamp(0.0, 1.0)

    def _pair_time_edge_attributes(
            self, flow_t, batch_id, local_mask, ctx_edges,
            local_ctx_edges, local_inter_edges, aligned_local_inter_edges,
            ref_tensor):
        """Build [t, enabled_mask] for each message edge.

        interface scope:
            ctx edges            -> disabled
            local context edges  -> disabled
            true Ab-Ag edges     -> enabled
            surface Ab-Ag edges  -> enabled

        context scope:
            ctx edges            -> enabled
            local context edges  -> enabled
            true Ab-Ag edges     -> disabled
            surface Ab-Ag edges  -> disabled

        all scope:
            every edge family above is enabled.

        The explicit mask is important.  A zero time value is a valid flow time
        and must not be overloaded to mean "this edge family is disabled".
        """
        scope = str(getattr(self, "pair_time_scope", "off")).lower()
        if scope == "off":
            return None, None, None

        t_res = self._flow_time_values_for_residues(
            flow_t, batch_id, ref_tensor
        ).to(dtype=ref_tensor.dtype)
        local_t = t_res[local_mask]

        def pack(time_values, enabled):
            time_values = time_values.reshape(-1, 1)
            if isinstance(enabled, bool):
                mask = torch.full_like(
                    time_values, 1.0 if enabled else 0.0
                )
            else:
                mask = enabled.to(
                    device=time_values.device, dtype=time_values.dtype
                ).reshape(-1, 1)
            return torch.cat([time_values, mask], dim=-1)

        # Global/context edges.
        ctx_attr = pack(
            t_res[ctx_edges[0]],
            scope in {"context", "all"},
        )

        # local_edges in message_passing is exactly
        # cat([local_ctx_edges, local_inter_edges], dim=1).
        local_ctx_t = local_t[local_ctx_edges[0]]
        local_inter_t = local_t[local_inter_edges[0]]
        local_time = torch.cat([local_ctx_t, local_inter_t], dim=0)

        if scope == "interface":
            local_mask_attr = torch.cat(
                [
                    torch.zeros_like(local_ctx_t),
                    torch.ones_like(local_inter_t),
                ],
                dim=0,
            )
        elif scope == "context":
            local_mask_attr = torch.cat(
                [
                    torch.ones_like(local_ctx_t),
                    torch.zeros_like(local_inter_t),
                ],
                dim=0,
            )
        else:
            local_mask_attr = torch.ones_like(local_time)
        local_attr = pack(local_time, local_mask_attr)

        # aligned_local_inter_edges contains only true antigen-antibody edges.
        surf_attr = pack(
            local_t[aligned_local_inter_edges[0]],
            scope in {"interface", "all"},
        )
        return ctx_attr, local_attr, surf_attr

    def _build_coord_pep_condition_for_residues(
            self, pep_X_model, interface_X, paratope_mask,
            pep_coord_valid=None):
        """Build dynamic proposal-coordinate condition features.

        X_pep is condition only: this function never modifies interface_X.

        Direction:
            The displacement from the current CA to the proposal CA is projected
            into the proposal N-CA-C local frame. Under any global proper
            rotation/translation, the frame and displacement transform together,
            so the projected components are SE(3)-invariant.

        Magnitude:
            The signed local displacement is compressed radially so its norm is
            log1p(CA distance). CA, mean-backbone and RMS-backbone distances are
            also compressed with log1p. This preserves direction and near-range
            sensitivity while preventing a poor proposal from dominating the
            hidden-state adapter through extreme raw distances.

        Robustness:
            Invalid proposal residues are masked. Degenerate proposal frames use
            distance-only conditioning by setting directional components to zero.

        Features per paratope residue:
            1-3) radially log-compressed proposal-local CA displacement;
            4)   log1p(CA distance);
            5)   log1p(mean backbone distance);
            6)   log1p(RMS backbone distance).
        """
        if (
            not self.coord_pep_as_condition
            or self.coord_pep_condition_adapter is None
            or pep_X_model is None
        ):
            return None, None

        if pep_X_model.shape != interface_X.shape:
            raise ValueError(
                "pep_X_model/interface_X shape mismatch: "
                f"{tuple(pep_X_model.shape)} vs {tuple(interface_X.shape)}"
            )
        if interface_X.shape[1] < 3:
            raise ValueError(
                "Coordinate conditioning requires N/CA/C channels."
            )

        n_int = int(interface_X.shape[0])
        if int(paratope_mask.sum().item()) != n_int:
            raise ValueError(
                "paratope/interface size mismatch: "
                f"{int(paratope_mask.sum().item())} vs {n_int}."
            )
        if pep_coord_valid is None:
            valid_int = torch.ones(
                n_int, device=interface_X.device, dtype=torch.bool
            )
        else:
            valid_int = torch.as_tensor(
                pep_coord_valid,
                device=interface_X.device,
                dtype=torch.bool,
            ).reshape(-1)
            if valid_int.numel() != n_int:
                raise ValueError(
                    "pep_coord_valid length mismatch: "
                    f"expected {n_int}, got {valid_int.numel()}."
                )

        delta = pep_X_model - interface_X
        ca_delta = delta[:, 1]
        ca_dist = torch.norm(ca_delta, dim=-1, keepdim=True)

        # Stable proposal-local N-CA-C frame.
        n_vec = pep_X_model[:, 0] - pep_X_model[:, 1]
        c_vec = pep_X_model[:, 2] - pep_X_model[:, 1]

        c_norm = torch.norm(c_vec, dim=-1, keepdim=True)
        e1 = F.normalize(c_vec, dim=-1, eps=self.scorefm_eps)

        n_orth = (
            n_vec
            - (n_vec * e1).sum(dim=-1, keepdim=True) * e1
        )
        n_orth_norm = torch.norm(n_orth, dim=-1, keepdim=True)
        e2 = F.normalize(n_orth, dim=-1, eps=self.scorefm_eps)
        e3 = F.normalize(
            torch.cross(e1, e2, dim=-1),
            dim=-1,
            eps=self.scorefm_eps,
        )

        local_delta = torch.stack(
            [
                (ca_delta * e1).sum(dim=-1),
                (ca_delta * e2).sum(dim=-1),
                (ca_delta * e3).sum(dim=-1),
            ],
            dim=-1,
        )

        # Parameter-free radial dynamic-range compression.  The three signed
        # proposal-local components retain their direction, while their joint
        # magnitude changes from d to log(1+d).  This avoids letting a very poor
        # proposal dominate the residual adapter through an arbitrarily large
        # raw displacement, without introducing a peptide-prior weight.
        ca_dist_safe = ca_dist.clamp_min(self.scorefm_eps)
        local_delta = (
            local_delta
            * (torch.log1p(ca_dist) / ca_dist_safe)
        )

        frame_valid = (
            torch.isfinite(c_norm.squeeze(-1))
            & torch.isfinite(n_orth_norm.squeeze(-1))
            & (c_norm.squeeze(-1) > 1e-4)
            & (n_orth_norm.squeeze(-1) > 1e-4)
        )
        direction_valid = valid_int & frame_valid
        local_delta = torch.where(
            direction_valid.unsqueeze(-1),
            local_delta,
            torch.zeros_like(local_delta),
        )

        # N/CA/C/O are available independently of the sampled side-chain token.
        n_bb = min(4, interface_X.shape[1])
        bb_dist = torch.norm(delta[:, :n_bb], dim=-1)
        mean_bb_dist = bb_dist.mean(dim=-1, keepdim=True)
        rms_bb_dist = torch.sqrt(
            (bb_dist ** 2).mean(dim=-1, keepdim=True)
            + self.scorefm_eps
        )

        distance_feat = torch.cat(
            [ca_dist, mean_bb_dist, rms_bb_dist],
            dim=-1,
        )
        distance_feat = torch.log1p(
            distance_feat.clamp_min(0.0)
        )

        feat_int = torch.cat(
            [local_delta, distance_feat],
            dim=-1,
        )
        feat_int = torch.nan_to_num(
            feat_int, nan=0.0, posinf=0.0, neginf=0.0
        )
        feat_int = (
            feat_int
            * valid_int.unsqueeze(-1).to(feat_int.dtype)
        )

        n_res = int(paratope_mask.shape[0])
        feat_full = interface_X.new_zeros(
            (n_res, self.coord_pep_condition_dim)
        )
        mask_full = torch.zeros(
            n_res, device=interface_X.device, dtype=torch.bool
        )
        feat_full[paratope_mask] = feat_int
        mask_full[paratope_mask] = valid_int
        return feat_full, mask_full

    def _build_seq_pep_condition_for_residues(
            self, S_pep, paratope_mask, ref_tensor):
        """Build residue-level sequence proposal conditions.

        S_pep is never written into S_t. Valid proposal tokens are embedded and
        placed on the corresponding paratope residues through functional tensor
        construction. Invalid or missing tokens contribute exactly zero.

        The returned tensor has the same hidden width and dtype as ref_tensor.
        """
        if self.seq_pep_condition_embedding is None or S_pep is None:
            return None, None

        n_int = int(paratope_mask.sum().item())
        if S_pep.numel() != n_int:
            raise ValueError(
                "S_pep/paratope size mismatch: "
                f"expected {n_int}, got {S_pep.numel()}."
            )

        pep_S = S_pep.to(
            device=ref_tensor.device, dtype=torch.long
        ).reshape(-1)
        valid_int = torch.logical_and(
            pep_S >= 0, pep_S < self.num_classes
        )
        if not valid_int.any():
            return None, None

        n_res = int(paratope_mask.shape[0])
        par_idx = paratope_mask.nonzero(
            as_tuple=False
        ).reshape(-1)
        valid_idx = par_idx[valid_int]

        # Build full residue-aligned token/mask tensors without modifying an
        # embedding output in place. Invalid positions use token 0 but are
        # multiplied by a zero mask, so they contribute no forward value or
        # embedding gradient.
        full_tokens = torch.zeros(
            n_res, device=ref_tensor.device, dtype=torch.long
        )
        full_tokens = full_tokens.index_copy(
            0, valid_idx, pep_S[valid_int]
        )

        cond_mask = torch.zeros(
            n_res, device=ref_tensor.device, dtype=torch.bool
        )
        cond_mask = cond_mask.index_fill(0, valid_idx, True)

        cond_emb = self.seq_pep_condition_embedding(
            full_tokens
        ).to(
            device=ref_tensor.device,
            dtype=ref_tensor.dtype,
        )
        cond_emb = (
            cond_emb
            * cond_mask.unsqueeze(-1).to(cond_emb.dtype)
        )
        return cond_emb, cond_mask

    def _build_dual_sequence_state_features(
            self, sequence_state_full, residue_pos, ref_tensor):
        """Construct residue/atom features from the explicit categorical state.

        The proposal sequence remains the recurrent context used by the original
        PCS-RC graph.  This helper computes the *state* features with the same
        embedding tables as ``aa_feature`` so S_t has the same semantics as an
        ordinary graph sequence: residue identity, atom identity, atom-position
        mask and atom weights all change together.
        """
        if not self.dual_sequence_state or sequence_state_full is None:
            return None
        state = torch.as_tensor(
            sequence_state_full, device=ref_tensor.device, dtype=torch.long
        ).reshape(-1)
        if state.numel() != ref_tensor.shape[0]:
            raise ValueError(
                "sequence_state_full length mismatch: "
                f"expected {ref_tensor.shape[0]}, got {state.numel()}."
            )
        valid = (state >= 0) & (state < self.num_classes)
        safe_state = state.clamp(min=0, max=self.num_classes - 1)
        if residue_pos is None:
            residue_pos = self.aa_feature._construct_residue_pos(safe_state)
        pos_embedding = self.aa_feature.aa_embedding.res_pos_embedding(residue_pos)
        residue_hidden = self.aa_feature.aa_embedding.residue_embedding(safe_state)
        residue_hidden = residue_hidden + pos_embedding
        atom_type = self.aa_feature.residue_atom_type[safe_state]
        atom_pos = self.aa_feature.residue_atom_pos[safe_state]
        atom_embedding = (
            self.aa_feature.aa_embedding.atom_embedding(atom_type)
            + self.aa_feature.aa_embedding.atom_pos_embedding(atom_pos)
        )
        atom_weights = self.aa_feature.get_atom_weights(safe_state)
        return {
            "residue_hidden": residue_hidden.to(dtype=ref_tensor.dtype),
            "atom_embedding": atom_embedding.to(dtype=ref_tensor.dtype),
            "atom_weights": atom_weights.to(dtype=ref_tensor.dtype),
            "atom_pos": atom_pos,
            "valid": valid,
            "tokens": safe_state,
        }

    def message_passing(self, X, S, residue_pos, interface_X, surf, paratope_mask,
                        batch_id, round_idx, memory_H=None, smooth_prob=None,
                        smooth_mask=None, flow_t=None,
                        coord_pep_condition=None,
                        coord_pep_condition_mask=None,
                        seq_pep_condition=None,
                        seq_pep_condition_mask=None,
                        sequence_state_full=None):
        # embeddings, hidden state, (internal edges, external edges),
        # (A : c*d, w : c*1)
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(
            X, S, batch_id, self.k_neighbors, residue_pos,
            smooth_prob=smooth_prob, smooth_mask=smooth_mask
        )

        time_emb = self._flow_time_embedding_for_residues(
            flow_t, batch_id, H_0
        )
        if time_emb is not None:
            H_0 = H_0 + time_emb

        # Reset detached condition-strength diagnostics only on sampled steps.
        # The state/atom computation itself is always active; only reductions and
        # GPU synchronizations used for observability are gated.
        diagnostics_active = bool(
            self.condition_diagnostics_enabled
            and getattr(self, "_diagnostic_capture", False)
        )
        zero_diag = H_0.detach().new_tensor(0.0)
        if diagnostics_active:
            self._last_condition_diagnostics = {
                "coord_condition_residual_ratio": zero_diag,
                "seq_condition_residual_ratio": zero_diag,
                "seq_state_residual_ratio": zero_diag,
                "coord_condition_valid_rate": zero_diag,
                "seq_condition_valid_rate": zero_diag,
                "seq_state_valid_rate": zero_diag,
                "seq_state_token_disagreement_rate": zero_diag,
                "seq_state_atom_mask_disagreement_rate": zero_diag,
                "seq_state_atom_weight_delta_ratio": zero_diag,
                "dual_sequence_state_enabled": H_0.detach().new_tensor(
                    1.0 if self.dual_sequence_state else 0.0
                ),
            }
        else:
            self._last_condition_diagnostics = {}

        # Coordinate proposal enters only through a zero-start residual feature
        # adapter.  H_0 already contains the current state and time embedding,
        # making the fusion residue-, context-, and time-dependent.
        if self.coord_pep_condition_adapter is not None:
            if coord_pep_condition is not None and coord_pep_condition_mask is not None:
                cond_feat = coord_pep_condition.to(
                    device=H_0.device, dtype=H_0.dtype
                )
                if cond_feat.shape != (
                    H_0.shape[0], self.coord_pep_condition_dim
                ):
                    raise ValueError(
                        "coordinate condition shape mismatch: expected "
                        f"{(H_0.shape[0], self.coord_pep_condition_dim)}, "
                        f"got {tuple(cond_feat.shape)}."
                    )
                cond_mask = coord_pep_condition_mask.to(
                    device=H_0.device, dtype=torch.bool
                )
                coord_residual = self.coord_pep_condition_adapter(
                    torch.cat([H_0, cond_feat], dim=-1)
                )
                H_0 = H_0 + (
                    coord_residual
                    * cond_mask.unsqueeze(-1).to(H_0.dtype)
                )

                if diagnostics_active:
                    with torch.no_grad():
                        # This branch is intentionally optional because the
                        # reductions below can synchronize GPU work.
                        if bool(cond_mask.any()):
                            base_rms = torch.sqrt(
                                (H_0[cond_mask].detach() ** 2).mean()
                                + self.scorefm_eps
                            )
                            residual_rms = torch.sqrt(
                                (coord_residual[cond_mask].detach() ** 2).mean()
                                + self.scorefm_eps
                            )
                            self._last_condition_diagnostics[
                                "coord_condition_residual_ratio"
                            ] = residual_rms / base_rms.clamp_min(self.scorefm_eps)
                            self._last_condition_diagnostics[
                                "coord_condition_valid_rate"
                            ] = cond_mask.float().mean()
            else:
                # DDP safety when the adapter exists but this batch has no valid
                # coordinate proposal.  The zero-valued term marks the parameters
                # as used without changing the forward value.
                dummy = sum(p.sum() for p in self.coord_pep_condition_adapter.parameters())
                H_0 = H_0 + 0.0 * dummy

        # Exact categorical state semantics (v52).
        #
        # The recurrent graph still carries proposal context, but the H3 state
        # representation is discrete and uniquely determined by S_t.  In
        # hard_exact mode we subtract the proposal-token residue embedding and
        # add the state-token embedding, while keeping time, coordinate condition
        # and memory channels untouched.  Atom identities/masks/weights are
        # switched exactly to S_t.  This avoids q_t -> q_{t^2} double gating and
        # avoids the non-physical union of two amino-acid atom topologies.
        state_atom_pos_full = None
        if self.dual_sequence_state:
            state_features = self._build_dual_sequence_state_features(
                sequence_state_full, residue_pos, H_0
            )
            if state_features is not None:
                state_valid = state_features["valid"]
                state_mask = paratope_mask & state_valid
                state_hidden = state_features["residue_hidden"]
                base_before_state = H_0

                state_t = self._flow_time_values_for_residues(
                    flow_t, batch_id, H_0
                )
                context_atom_pos = self.aa_feature._construct_atom_pos(S)
                context_atom_embeddings = atom_embeddings
                context_atom_weights = atom_weights

                if self.dual_sequence_atom_mode == "hard_exact":
                    context_features = self._build_dual_sequence_state_features(
                        S, residue_pos, H_0
                    )
                    context_hidden = context_features["residue_hidden"]
                    state_residual = state_hidden - context_hidden
                    H_0 = H_0 + (
                        state_residual
                        * state_mask.unsqueeze(-1).to(H_0.dtype)
                    )
                else:
                    state_residual = self.seq_state_adapter(
                        torch.cat([H_0, state_hidden], dim=-1)
                    )
                    hidden_gate = (
                        state_t
                        if self.dual_sequence_atom_mode == "time_gated_union"
                        else torch.ones_like(state_t)
                    )
                    H_0 = H_0 + (
                        state_residual
                        * state_mask.unsqueeze(-1).to(H_0.dtype)
                        * hidden_gate.unsqueeze(-1)
                    )

                if self.dual_sequence_atom_mode in {"hard", "hard_exact"}:
                    atom_embeddings = torch.where(
                        state_mask.view(-1, 1, 1),
                        state_features["atom_embedding"],
                        atom_embeddings,
                    )
                    atom_weights = torch.where(
                        state_mask.view(-1, 1),
                        state_features["atom_weights"],
                        atom_weights,
                    )
                    state_atom_pos_full = torch.where(
                        state_mask.view(-1, 1),
                        state_features["atom_pos"],
                        context_atom_pos,
                    )

                elif self.dual_sequence_atom_mode == "time_gated_union":
                    # Historical diagnostic only.  Formal v52 runs never use it.
                    atom_gate = (
                        state_t.view(-1, 1, 1)
                        * state_mask.view(-1, 1, 1).to(H_0.dtype)
                    )
                    atom_embeddings = (
                        context_atom_embeddings
                        + atom_gate * (
                            state_features["atom_embedding"]
                            - context_atom_embeddings
                        )
                    )
                    weight_gate = atom_gate.squeeze(-1)
                    atom_weights = (
                        context_atom_weights
                        + weight_gate * (
                            state_features["atom_weights"]
                            - context_atom_weights
                        )
                    )
                    context_valid_atom = (
                        context_atom_pos != self.aa_feature.atom_pos_pad_idx
                    )
                    state_valid_atom = (
                        state_features["atom_pos"]
                        != self.aa_feature.atom_pos_pad_idx
                    )
                    union_valid_atom = context_valid_atom | (
                        state_valid_atom & state_mask.view(-1, 1)
                    )
                    preferred_pos = torch.where(
                        state_valid_atom,
                        state_features["atom_pos"],
                        context_atom_pos,
                    )
                    state_atom_pos_full = torch.where(
                        union_valid_atom,
                        preferred_pos,
                        torch.full_like(
                            preferred_pos, self.aa_feature.atom_pos_pad_idx
                        ),
                    )

                elif self.dual_sequence_atom_mode == "hidden_only":
                    state_atom_pos_full = None

                if state_atom_pos_full is not None:
                    ctx_edges, inter_edges = self.aa_feature.construct_edges(
                        X, S, batch_id, self.k_neighbors,
                        atom_pos=state_atom_pos_full,
                        segment_ids=self.batch_constants["segment_ids"],
                    )

                if diagnostics_active:
                    with torch.no_grad():
                        if bool(state_mask.any()):
                            context_tokens = S.to(
                                device=H_0.device, dtype=torch.long
                            )
                            token_disagreement = (
                                state_features["tokens"][state_mask]
                                != context_tokens[state_mask]
                            ).float().mean()
                            context_pos = context_atom_pos[state_mask]
                            state_pos = state_features["atom_pos"][state_mask]
                            context_pad = (
                                context_pos == self.aa_feature.atom_pos_pad_idx
                            )
                            state_pad = (
                                state_pos == self.aa_feature.atom_pos_pad_idx
                            )
                            atom_mask_disagreement = (
                                context_pad != state_pad
                            ).float().mean()
                            context_weights = self.aa_feature.get_atom_weights(
                                context_tokens[state_mask]
                            ).to(H_0.dtype)
                            state_weights = state_features["atom_weights"][state_mask]
                            weight_delta = torch.sqrt(
                                (state_weights - context_weights)
                                .detach().pow(2).mean()
                                + self.scorefm_eps
                            )
                            weight_base = torch.sqrt(
                                context_weights.detach().pow(2).mean()
                                + self.scorefm_eps
                            )
                            base_rms = torch.sqrt(
                                base_before_state[state_mask]
                                .detach().pow(2).mean()
                                + self.scorefm_eps
                            )
                            residual_rms = torch.sqrt(
                                state_residual[state_mask]
                                .detach().pow(2).mean()
                                + self.scorefm_eps
                            )
                            self._last_condition_diagnostics[
                                "seq_state_residual_ratio"
                            ] = residual_rms / base_rms.clamp_min(
                                self.scorefm_eps
                            )
                            self._last_condition_diagnostics[
                                "seq_state_valid_rate"
                            ] = state_mask.float().mean()
                            self._last_condition_diagnostics[
                                "seq_state_token_disagreement_rate"
                            ] = token_disagreement
                            self._last_condition_diagnostics[
                                "seq_state_atom_mask_disagreement_rate"
                            ] = atom_mask_disagreement
                            self._last_condition_diagnostics[
                                "seq_state_atom_weight_delta_ratio"
                            ] = weight_delta / weight_base.clamp_min(
                                self.scorefm_eps
                            )
                            self._last_condition_diagnostics[
                                "seq_state_time_gate_mean"
                            ] = state_t[state_mask].mean()
                            self._last_condition_diagnostics[
                                "seq_state_atom_mode_hard"
                            ] = H_0.detach().new_tensor(
                                1.0 if self.dual_sequence_atom_mode == "hard"
                                else 0.0
                            )
                            self._last_condition_diagnostics[
                                "seq_state_atom_mode_hard_exact"
                            ] = H_0.detach().new_tensor(
                                1.0 if self.dual_sequence_atom_mode == "hard_exact"
                                else 0.0
                            )
                            self._last_condition_diagnostics[
                                "seq_state_atom_mode_hidden_only"
                            ] = H_0.detach().new_tensor(
                                1.0 if self.dual_sequence_atom_mode == "hidden_only"
                                else 0.0
                            )
                            self._last_condition_diagnostics[
                                "seq_state_atom_mode_time_gated_union"
                            ] = H_0.detach().new_tensor(
                                1.0 if self.dual_sequence_atom_mode
                                == "time_gated_union" else 0.0
                            )
            elif self.seq_state_adapter is not None:
                dummy = sum(p.sum() for p in self.seq_state_adapter.parameters())
                H_0 = H_0 + 0.0 * dummy

        # Sequence proposal is fused through a residue- and time-dependent
        # zero-start adapter, rather than a single global scalar shared by all
        # samples, residues and times.
        if self.seq_pep_condition_adapter is not None:
            if seq_pep_condition is not None and seq_pep_condition_mask is not None:
                seq_cond = seq_pep_condition.to(
                    device=H_0.device, dtype=H_0.dtype
                )
                if seq_cond.shape != H_0.shape:
                    raise ValueError(
                        "sequence condition shape mismatch: expected "
                        f"{tuple(H_0.shape)}, got {tuple(seq_cond.shape)}."
                    )
                seq_mask = seq_pep_condition_mask.to(
                    device=H_0.device, dtype=torch.bool
                )
                seq_residual = self.seq_pep_condition_adapter(
                    torch.cat([H_0, seq_cond], dim=-1)
                )
                H_0 = H_0 + (
                    seq_residual
                    * seq_mask.unsqueeze(-1).to(H_0.dtype)
                )

                if diagnostics_active:
                    with torch.no_grad():
                        if bool(seq_mask.any()):
                            base_rms = torch.sqrt(
                                (H_0[seq_mask].detach() ** 2).mean()
                                + self.scorefm_eps
                            )
                            residual_rms = torch.sqrt(
                                (seq_residual[seq_mask].detach() ** 2).mean()
                                + self.scorefm_eps
                            )
                            self._last_condition_diagnostics[
                                "seq_condition_residual_ratio"
                            ] = residual_rms / base_rms.clamp_min(self.scorefm_eps)
                            self._last_condition_diagnostics[
                                "seq_condition_valid_rate"
                            ] = seq_mask.float().mean()
            else:
                dummy = sum(p.sum() for p in self.seq_pep_condition_adapter.parameters())
                if self.seq_pep_condition_embedding is not None:
                    dummy = dummy + sum(
                        p.sum() for p in self.seq_pep_condition_embedding.parameters()
                    )
                H_0 = H_0 + 0.0 * dummy

        if not self.keep_memory:
            memory_H = None

        if memory_H is not None:
            H_0 = H_0 + self.memory_ffn(memory_H)

        if self.pred_edge_dist:
            if memory_H is not None:
                edge_H = self.edge_H_ffn(memory_H)
            else:
                # replace the MLP with gnn for initial edge distance prediction
                edge_H, dumb_X = self.init_gnn(H_0, X, ctx_edges,
                                       channel_attr=atom_embeddings,
                                       channel_weights=atom_weights)
                X = X + dumb_X * 0  # to cheat the autograd check

        # update coordination of the global node
        X = self.aa_feature.update_global_coordinates(X, S)

        # prepare local complex
        local_mask = self.batch_constants['local_mask']
        local_is_ab = self.batch_constants['local_is_ab']
        local_batch_id = self.batch_constants['local_batch_id']
        local_X = X[local_mask].clone()
        
        local_X[local_is_ab] = interface_X
        # prepare local complex edges
        local_ctx_edges = self.batch_constants['local_ctx_edges']  # [2, Ec]
        local_inter_edges = self.batch_constants['local_inter_edges']  # [2, Ei]
        atom_pos = (
            state_atom_pos_full[local_mask]
            if state_atom_pos_full is not None
            else self.aa_feature._construct_atom_pos(S[local_mask])
        )
        offsets, max_n, gni2lni = self.batch_constants['local_edge_infos']
        # Context and interaction edges are both derived from the current state.
        local_ctx_edges = _knn_edges(
            local_X, atom_pos, local_ctx_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni))
        # For interaction edges, optionally use the learned distance predictor.
        if self.pred_edge_dist:
            local_H = edge_H[local_mask]
            src_H, dst_H = local_H[local_inter_edges[0]], local_H[local_inter_edges[1]]
            p_edge_dist = self.edge_dist_ffn(torch.cat([src_H, dst_H], dim=-1)) +\
                          self.edge_dist_ffn(torch.cat([dst_H, src_H], dim=-1))  # perm-invariant
            p_edge_dist = p_edge_dist.squeeze(-1)
        else:
            p_edge_dist = None
        local_inter_edges = _knn_edges(
            local_X, atom_pos, local_inter_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni), given_dist=p_edge_dist)
        local_edges = torch.cat([local_ctx_edges, local_inter_edges], dim=1)
        
        #prepare surface
        # surf_start = time.time()
        aligned_local_inter_edges, epi_index = self.align_epi_ab(local_inter_edges, local_is_ab)
        # self.timing_stats['surface_processing'] += time.time() - surf_start

        # Capture the final-round shared activation only on diagnostic probe
        # steps. Sequence logits and coordinate outputs both depend on H_0.
        if (
            bool(getattr(self, "_diagnostic_capture", False))
            and int(round_idx) == int(self.round) - 1
        ):
            self._diagnostic_probe_tensor = H_0

        # Explicit pair/edge-level time conditioning (F02 only).  The original
        # model already conditions nodes on t.  Here the same scalar t is made
        # directly available to the edge MLPs so an identical geometric pair
        # can be interpreted differently at early vs late transport time.
        ctx_time_attr, local_time_attr, surf_time_attr = (
            self._pair_time_edge_attributes(
                flow_t, batch_id, local_mask, ctx_edges,
                local_ctx_edges, local_inter_edges,
                aligned_local_inter_edges, H_0,
            )
        )

        # message passing
        # sme_start = time.time()
        if self.pair_time_conditioning:
            H, pred_X, pred_local_X = self.gnn(
                H_0, X, ctx_edges, local_mask, local_X, surf, local_edges,
                paratope_mask, local_is_ab, aligned_local_inter_edges, epi_index,
                channel_attr=atom_embeddings, channel_weights=atom_weights,
                ctx_edge_attr=ctx_time_attr,
                inter_edge_attr=local_time_attr,
                surf_edge_attr=surf_time_attr,
            )
        else:
            H, pred_X, pred_local_X = self.gnn(
                H_0, X, ctx_edges, local_mask, local_X, surf, local_edges,
                paratope_mask, local_is_ab, aligned_local_inter_edges, epi_index,
                channel_attr=atom_embeddings, channel_weights=atom_weights,
            )
        # self.timing_stats['sme_encoding'] += time.time() - sme_start

        interface_X = pred_local_X[local_is_ab]
        pred_logits = None if self.struct_only else self.ffn_residue(H)

        return pred_logits, pred_X, interface_X, H, p_edge_dist  # [N, num_classes], [N, n_channel, 3], [Ncdr, n_channel, 3], [N, hidden_size]
    
    @torch.no_grad()
    def init_interface(self, X, S, paratope_mask, batch_id, init_noise=None):
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]  # [bs, 3]
        init_local_X = torch.zeros_like(X[paratope_mask])
        init_local_X = init_local_X + ag_centers[batch_id[paratope_mask]].unsqueeze(1)
        noise = torch.randn_like(init_local_X) if init_noise is None else init_noise
        ca_noise = noise[:, 1]
        noise = noise / 10  + ca_noise.unsqueeze(1) # scale other atoms
        noise[:, 1] = ca_noise
        init_local_X = init_local_X + noise

        init_local_S = torch.randint(0, self.num_classes, 
                                   (paratope_mask.sum(),), 
                                   device=X.device,
                                   dtype=torch.long)
        return init_local_X, init_local_S

    @torch.no_grad()
    def _prepare_batch_constants(self, S, paratope_mask, lengths):
        # generate batch id
        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        self.batch_constants['batch_id'] = batch_id
        self.batch_constants['batch_size'] = torch.max(batch_id) + 1

        segment_ids = self.aa_feature._construct_segment_ids(S)
        self.batch_constants['segment_ids'] = segment_ids

        # interface relatd
        is_ag = segment_ids == self.aa_feature.ag_seg_id
        not_ag_global = S != self.aa_feature.boa_idx
        local_mask = torch.logical_or(
            paratope_mask, torch.logical_and(is_ag, not_ag_global)
        )
        local_segment_ids = segment_ids[local_mask]
        local_is_ab = local_segment_ids != self.aa_feature.ag_seg_id
        local_batch_id = batch_id[local_mask]
        self.batch_constants['is_ag'] = is_ag
        self.batch_constants['local_mask'] = local_mask
        self.batch_constants['local_is_ab'] = local_is_ab
        self.batch_constants['local_batch_id'] = local_batch_id
        self.batch_constants['local_segment_ids'] = local_segment_ids
        # interface local edges
        (row, col), (offsets, max_n, gni2lni) = self.aa_feature.edge_constructor.get_batch_edges(local_batch_id)
        row_segment_ids, col_segment_ids = local_segment_ids[row], local_segment_ids[col]
        # is_ctx = row_segment_ids == col_segment_ids
        # is_inter = torch.logical_not(is_ctx)
        
        row_is_ag = row_segment_ids == self.aa_feature.ag_seg_id
        col_is_ag = col_segment_ids == self.aa_feature.ag_seg_id
        is_inter = torch.logical_xor(row_is_ag, col_is_ag)
        is_ctx = torch.logical_not(is_inter)

        self.batch_constants['local_ctx_edges'] = torch.stack([row[is_ctx], col[is_ctx]])  # [2, Ec]
        self.batch_constants['local_inter_edges'] = torch.stack([row[is_inter], col[is_inter]])  # [2, Ei]
        self.batch_constants['local_edge_infos'] = (offsets, max_n, gni2lni)

        interface_batch_id = batch_id[paratope_mask]
        self.batch_constants['interface_batch_id'] = interface_batch_id
    
    def _clean_batch_constants(self):
        self.batch_constants = {}

    @torch.no_grad()
    def _get_inter_edge_dist(self, X, S):
        """Ground-truth inter-edge distance using the supplied sequence.

        This target is computed from native coordinates and ``true_S`` in
        ``forward``.  It must not depend on the transient dual sequence state
        used inside ``message_passing``.
        """
        local_mask = self.batch_constants['local_mask']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        src_dst = self.batch_constants['local_inter_edges'].T
        dist = X[local_mask][src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = atom_pos[src_dst] == self.aa_feature.atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * 1e10  # [Ef, n_channel, n_channel]
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
        return dist
    
    def _raw_interface_to_model_frame(self, interface_X, paratope_mask, batch_id):
        """Convert raw paratope coordinates into internal shadow frame.

        `_forward` centers the antigen/antibody and normalizes coordinates before
        message passing. Shadow paratope coordinates are later uncentered with
        `_type=4`, i.e. by adding the antigen center. Therefore an externally
        supplied raw X_t must be represented internally as:

            X_t_model = (X_t_raw - antigen_center) / std.
        """
        interface_batch_id = batch_id[paratope_mask]
        ag_centers = self.normalizer.ag_centers[interface_batch_id]
        return self.normalizer.normalize(interface_X - ag_centers.unsqueeze(1))

    def _reference_ca_mean(self, X, S, paratope_mask, batch_id):
        """Reference mean for CA translation under init_interface().

        init_interface samples each paratope CA as:
            antigen_center + N(0, I_3).
        Therefore the conditional CA score is analytically available with
        mean=antigen_center and identity covariance.
        """
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]
        return ag_centers[batch_id[paratope_mask]]

    def _analytic_ca_score_from_clean(
            self, Xt, clean_X, t, sigma_t, source_ca_mean):
        """Analytic CA translation score for the linear reference bridge.

        Path:
            X_t^CA = sigma_t X_0^CA + t X_1^CA,
            X_0^CA ~ N(source_ca_mean, I).

        Hence:
            p_t(X_t^CA | X_1^CA, c)
              = N(sigma_t * source_ca_mean + t * X_1^CA,
                  sigma_t^2 I)

            score = -(X_t^CA - sigma_t*mu_0 - t*X_1^CA) / sigma_t^2.

        Only CA translation is used here. The full-atom initialization is
        atom-correlated, so pretending that all atom channels are isotropic
        Gaussian would be mathematically inconsistent.
        """
        t = torch.as_tensor(t, device=Xt.device, dtype=Xt.dtype)
        sigma_t = torch.as_tensor(
            sigma_t, device=Xt.device, dtype=Xt.dtype
        )

        if t.dim() == 3:
            t_ca = t[:, 0, :]
        else:
            t_ca = t.reshape(-1, 1)

        if sigma_t.dim() == 3:
            sigma_ca = sigma_t[:, 0, :]
        else:
            sigma_ca = sigma_t.reshape(-1, 1)

        ca_idx = 1 if Xt.shape[1] > 1 else 0
        Xt_ca = Xt[:, ca_idx]
        clean_ca = clean_X[:, ca_idx]
        mean_t = (
            sigma_ca * source_ca_mean
            + t_ca * clean_ca
        )
        return -(
            Xt_ca - mean_t
        ) / (sigma_ca.pow(2) + self.scorefm_eps)

    def _interface_valid_graph_mask(
            self, interface_batch_id, n_graph, device):
        valid = torch.zeros(
            n_graph, device=device, dtype=torch.bool
        )
        if interface_batch_id.numel() > 0:
            valid[torch.unique(interface_batch_id)] = True
        return valid

    def _masked_residue_mse_per_graph(
            self, diff, atom_mask, interface_batch_id):
        """Per-complex normalized vector MSE.

        Reduction order:
            xyz -> atom channels -> residues -> complex.
        """
        if interface_batch_id.numel() == 0:
            return (
                diff.new_zeros(1),
                torch.zeros(1, device=diff.device, dtype=torch.bool),
            )

        n_graph = int(interface_batch_id.max().item()) + 1
        atom_mask_f = atom_mask.to(diff.dtype)

        atom_sq = (diff ** 2).sum(dim=-1) * atom_mask_f
        per_res = atom_sq.sum(dim=-1) / (
            3.0 * atom_mask_f.sum(dim=-1).clamp_min(1.0)
        )

        per_graph = scatter_mean(
            per_res,
            interface_batch_id,
            dim=0,
            dim_size=n_graph,
        )
        valid_graph = self._interface_valid_graph_mask(
            interface_batch_id, n_graph, diff.device
        )
        return per_graph, valid_graph

    def _masked_residue_smooth_l1_per_graph(
            self, pred, target, atom_mask, interface_batch_id):
        """Unique per-complex endpoint objective.

        This replaces both the old global interface loss and the duplicated
        auxiliary x1 loss. Each complex contributes one normalized value,
        independent of CDR length.
        """
        if interface_batch_id.numel() == 0:
            return (
                pred.new_zeros(1),
                torch.zeros(1, device=pred.device, dtype=torch.bool),
            )

        n_graph = int(interface_batch_id.max().item()) + 1
        atom_mask_f = atom_mask.to(pred.dtype)

        err = F.smooth_l1_loss(
            pred, target, reduction="none"
        ).sum(dim=-1)
        err = err * atom_mask_f
        per_res = err.sum(dim=-1) / (
            3.0 * atom_mask_f.sum(dim=-1).clamp_min(1.0)
        )

        per_graph = scatter_mean(
            per_res,
            interface_batch_id,
            dim=0,
            dim_size=n_graph,
        )
        valid_graph = self._interface_valid_graph_mask(
            interface_batch_id, n_graph, pred.device
        )
        return per_graph, valid_graph

    def _scorefm_time_per_graph(
            self, t, interface_batch_id, ref_tensor):
        """Convert scalar/graph/interface time to one value per complex."""
        if interface_batch_id.numel() == 0:
            return (
                ref_tensor.new_zeros(1),
                torch.zeros(
                    1, device=ref_tensor.device, dtype=torch.bool
                ),
            )

        n_graph = int(interface_batch_id.max().item()) + 1
        valid_graph = self._interface_valid_graph_mask(
            interface_batch_id, n_graph, ref_tensor.device
        )

        t_tensor = torch.as_tensor(
            t, device=ref_tensor.device, dtype=ref_tensor.dtype
        )
        if t_tensor.dim() == 0 or t_tensor.numel() == 1:
            t_graph = t_tensor.reshape(1).expand(n_graph)
        else:
            t_flat = t_tensor.reshape(-1)
            if t_flat.numel() == n_graph:
                t_graph = t_flat
            elif t_flat.numel() == interface_batch_id.numel():
                t_graph = scatter_mean(
                    t_flat,
                    interface_batch_id,
                    dim=0,
                    dim_size=n_graph,
                )
            else:
                raise ValueError(
                    "t must be scalar, graph-level [B], or "
                    "interface-level [N_int]. "
                    f"Got {t_flat.numel()} values for "
                    f"{n_graph} graphs and "
                    f"{interface_batch_id.numel()} residues."
                )

        return t_graph.clamp(0.0, 1.0), valid_graph

    def _deterministic_standard_normal(self, shape, device, dtype):
        """Stateless pseudo-Gaussian tensor for deterministic validation."""
        n = 1
        for dim in shape:
            n *= int(dim)
        idx = torch.arange(n, device=device, dtype=torch.float32) + 1.0
        u = torch.frac(torch.sin(idx * 12.9898 + 78.233) * 43758.5453).abs()
        u = u.clamp(1e-4, 1.0 - 1e-4)
        z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        return z.reshape(*shape).to(dtype=dtype)

    @torch.no_grad()
    def _structured_global_primary_path(
            self, mu_t, source_X0, target_X1, t_graph, interface_batch_id):
        """Primary geometry-preserving H3 graph-translation stochastic path.

        mu_t = (1-t)X0 + tX1
        xi_g ~ N(0, a_g^2 I3), shared by every H3 atom in graph g
        beta(t) = 4t(1-t)
        Z_t = mu_t + beta(t) xi_g
        u*_t = X1-X0 + beta'(t) xi_g

        Existing AbFlow bridge sampling uses
            v_theta = (Y_theta - Z_t)/(1-t).
        Hence the exact endpoint-like target for conditional flow matching is
            Y*_t = Z_t + (1-t)u*_t.
        """
        if interface_batch_id.numel() == 0:
            zero = target_X1.new_zeros(1)
            return mu_t, target_X1, {
                "transport_mean": zero, "path_rms": zero, "target_shift_rms": zero
            }

        n_graph = int(interface_batch_id.max().item()) + 1
        ca_idx = 1 if source_X0.shape[1] > 1 else 0
        src_centroid = scatter_mean(source_X0[:, ca_idx].float(), interface_batch_id, dim=0, dim_size=n_graph)
        tgt_centroid = scatter_mean(target_X1[:, ca_idx].float(), interface_batch_id, dim=0, dim_size=n_graph)
        transport = torch.linalg.norm(tgt_centroid - src_centroid, dim=-1).clamp(
            min=0.0, max=float(self.structured_transport_max)
        )
        amplitude = (
            float(self.structured_gamma_scale) * transport / math.sqrt(3.0)
        ).clamp(min=0.0, max=float(self.structured_gamma_abs_max))

        if self.deterministic_validation and not self.training:
            eps_graph = self._deterministic_standard_normal(
                (n_graph, 3), target_X1.device, torch.float32
            )
        else:
            eps_graph = torch.randn((n_graph, 3), device=target_X1.device, dtype=torch.float32)
        xi_graph = amplitude[:, None] * eps_graph

        t = torch.as_tensor(t_graph, device=target_X1.device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1 and n_graph > 1:
            t = t.expand(n_graph)
        if t.numel() != n_graph:
            raise ValueError(f"structured global path expects {n_graph} graph times, got {t.numel()}.")

        beta = 4.0 * t * (1.0 - t)
        beta_prime = 4.0 * (1.0 - 2.0 * t)
        path_shift_graph = beta[:, None] * xi_graph
        velocity_noise_graph = beta_prime[:, None] * xi_graph

        path_shift_int = path_shift_graph[interface_batch_id].to(mu_t.dtype)
        Xt = mu_t + path_shift_int[:, None, :]

        endpoint_shift_graph = path_shift_graph + (1.0 - t)[:, None] * velocity_noise_graph
        endpoint_shift_int = endpoint_shift_graph[interface_batch_id].to(target_X1.dtype)
        flow_endpoint_target = target_X1 + endpoint_shift_int[:, None, :]

        return Xt, flow_endpoint_target, {
            "transport_mean": transport.mean().to(target_X1.dtype),
            "path_rms": torch.sqrt(path_shift_graph.pow(2).mean().clamp_min(0.0)).to(target_X1.dtype),
            "target_shift_rms": torch.sqrt(endpoint_shift_graph.pow(2).mean().clamp_min(0.0)).to(target_X1.dtype),
        }

    @torch.no_grad()
    def _structured_multiscale_primary_path(
            self, mu_t, source_X0, target_X1, t_graph, interface_batch_id):
        """Primary global + orthogonal local structured stochastic path.

        Global mode (same as S02):
            xi_g : one 3D translation shared by the complete H3 loop.

        Local mode:
            d_i = CA_i(X1) - CA_i(X0)
            d_g = mean_{i in g} d_i
            r_i = d_i - d_g

        Hence mean_{i in g} r_i = 0 exactly.  We sample one scalar a_g per
        complex and define
            xi_i^local = eta_local * a_g * r_i.

        Every atom in residue i receives the same xi_i^local, so the residue's
        internal atom geometry is preserved.  Because the residual field has
        zero graph centroid, local deformation cannot duplicate the global H3
        placement mode.

        The total path is
            Z_t = mu_t + beta(t) (xi_g + xi_i^local),
            beta(t)=4t(1-t).

        Its exact conditional velocity is
            u*_t = X1-X0 + beta'(t)(xi_g + xi_i^local),

        and the current AbFlow endpoint-parameterized bridge sampler is matched
        by the analytic target
            Y*_t = Z_t + (1-t) u*_t.
        """
        if interface_batch_id.numel() == 0:
            zero = target_X1.new_zeros(1)
            return mu_t, target_X1, {
                "transport_mean": zero,
                "path_rms": zero,
                "target_shift_rms": zero,
                "local_transport_rms": zero,
                "local_path_rms": zero,
                "local_centroid_rms": zero,
            }

        n_graph = int(interface_batch_id.max().item()) + 1
        ca_idx = 1 if source_X0.shape[1] > 1 else 0

        src_ca = source_X0[:, ca_idx].float()
        tgt_ca = target_X1[:, ca_idx].float()
        ca_transport = tgt_ca - src_ca

        global_vec = scatter_mean(
            ca_transport, interface_batch_id, dim=0, dim_size=n_graph
        )
        global_dist = torch.linalg.norm(global_vec, dim=-1).clamp(
            min=0.0, max=float(self.structured_transport_max)
        )
        global_amp = (
            float(self.structured_gamma_scale)
            * global_dist
            / math.sqrt(3.0)
        ).clamp(
            min=0.0, max=float(self.structured_gamma_abs_max)
        )

        if self.deterministic_validation and not self.training:
            eps_global = self._deterministic_standard_normal(
                (n_graph, 3), target_X1.device, torch.float32
            )
            # Use a different deterministic stream from the 3D global draw.
            eps_local_scalar = self._deterministic_standard_normal(
                (n_graph, 2), target_X1.device, torch.float32
            )[:, 1]
        else:
            eps_global = torch.randn(
                (n_graph, 3), device=target_X1.device, dtype=torch.float32
            )
            eps_local_scalar = torch.randn(
                (n_graph,), device=target_X1.device, dtype=torch.float32
            )

        xi_global_graph = global_amp[:, None] * eps_global

        # Target-aligned local deformation mode after removing graph translation.
        local_residual = ca_transport - global_vec[interface_batch_id]
        local_residual_sq = local_residual.pow(2).sum(dim=-1) / 3.0
        local_rms_graph = torch.sqrt(
            scatter_mean(
                local_residual_sq,
                interface_batch_id,
                dim=0,
                dim_size=n_graph,
            ).clamp_min(0.0)
        )
        xi_local_res = (
            float(self.structured_local_gamma_scale)
            * eps_local_scalar[interface_batch_id, None]
            * local_residual
        )

        # Numerical zero-centroid projection.  Analytically local_residual is
        # already centered; re-projecting prevents float accumulation from
        # leaking local deformation into the global placement subspace.
        local_mean = scatter_mean(
            xi_local_res, interface_batch_id, dim=0, dim_size=n_graph
        )
        xi_local_res = (
            xi_local_res - local_mean[interface_batch_id]
        )

        xi_total_res = (
            xi_global_graph[interface_batch_id] + xi_local_res
        )

        t = torch.as_tensor(
            t_graph, device=target_X1.device, dtype=torch.float32
        ).reshape(-1)
        if t.numel() == 1 and n_graph > 1:
            t = t.expand(n_graph)
        if t.numel() != n_graph:
            raise ValueError(
                f"structured multiscale path expects {n_graph} graph times, "
                f"got {t.numel()}."
            )

        beta = 4.0 * t * (1.0 - t)
        beta_prime = 4.0 * (1.0 - 2.0 * t)

        beta_res = beta[interface_batch_id, None]
        beta_prime_res = beta_prime[interface_batch_id, None]

        path_shift_res = beta_res * xi_total_res
        velocity_noise_res = beta_prime_res * xi_total_res

        Xt = mu_t + path_shift_res.to(mu_t.dtype)[:, None, :]

        endpoint_shift_res = (
            path_shift_res
            + (1.0 - t[interface_batch_id])[:, None] * velocity_noise_res
        )
        flow_endpoint_target = (
            target_X1
            + endpoint_shift_res.to(target_X1.dtype)[:, None, :]
        )

        with torch.no_grad():
            local_path_shift_res = beta_res * xi_local_res
            local_centroid = scatter_mean(
                local_path_shift_res,
                interface_batch_id,
                dim=0,
                dim_size=n_graph,
            )
            return Xt, flow_endpoint_target, {
                "transport_mean": global_dist.mean().to(target_X1.dtype),
                "path_rms": torch.sqrt(
                    path_shift_res.pow(2).mean().clamp_min(0.0)
                ).to(target_X1.dtype),
                "target_shift_rms": torch.sqrt(
                    endpoint_shift_res.pow(2).mean().clamp_min(0.0)
                ).to(target_X1.dtype),
                "local_transport_rms": local_rms_graph.mean().to(
                    target_X1.dtype
                ),
                "local_path_rms": torch.sqrt(
                    local_path_shift_res.pow(2).mean().clamp_min(0.0)
                ).to(target_X1.dtype),
                "local_centroid_rms": torch.sqrt(
                    local_centroid.pow(2).mean().clamp_min(0.0)
                ).to(target_X1.dtype),
            }

    @torch.no_grad()
    def _satc_transport_calibrated_gamma(
            self, source_X0, target_X1, atom_mask, t_graph,
            interface_batch_id, gamma_scale):
        """Build a path-relative SATC tube without changing AbFlow's state.

        The original v45 coefficient had units of Angstrom but was selected as
        a fixed number independent of the actual PCS source-to-native distance.
        This helper instead computes one graph-level transport RMS in the same
        full-atom Cartesian state used by AbFlow and defines a dimensionless
        relative tube width:

            s_g = RMS_valid_atoms(X1 - X0)
            gamma_g(t) = eta * s_g * 4 t (1-t).

        Therefore eta is directly interpretable: at t=0.5 the stochastic tube
        standard deviation is eta times the clean transport RMS.  We deliberately
        keep iid Gaussian noise in the existing Cartesian state; changing its
        covariance would change the analytic score from -epsilon/gamma and would
        require a different score target.
        """
        if interface_batch_id.numel() == 0:
            zero_graph = target_X1.new_zeros(1)
            return target_X1.new_zeros((0, 1, 1)), zero_graph, zero_graph

        transport = (target_X1 - source_X0).detach().float()
        mask = atom_mask.to(device=transport.device, dtype=transport.dtype)
        valid_atoms = mask.sum(dim=-1).clamp_min(1.0)
        per_res_mse = (
            transport.pow(2).sum(dim=-1) * mask
        ).sum(dim=-1) / (3.0 * valid_atoms)

        n_graph = int(interface_batch_id.max().item()) + 1
        graph_mse = scatter_mean(
            per_res_mse, interface_batch_id, dim=0, dim_size=n_graph
        )
        graph_rms = torch.sqrt(graph_mse.clamp_min(self.scorefm_eps))
        graph_rms = graph_rms.clamp(
            min=float(self.satc_transport_rms_min),
            max=float(self.satc_transport_rms_max),
        )

        t_graph = torch.as_tensor(
            t_graph, device=transport.device, dtype=transport.dtype
        ).reshape(-1)
        if t_graph.numel() == 1 and n_graph > 1:
            t_graph = t_graph.expand(n_graph)
        if t_graph.numel() != n_graph:
            raise ValueError(
                "transport-calibrated SATC expects graph-level time with "
                f"{n_graph} values, got {t_graph.numel()}."
            )

        bridge_shape = 4.0 * t_graph * (1.0 - t_graph)
        gamma_graph = (
            float(gamma_scale) * graph_rms * bridge_shape
        ).clamp(
            min=0.0,
            max=float(self.satc_gamma_abs_max),
        )
        gamma_int = gamma_graph[interface_batch_id].reshape(-1, 1, 1)
        return (
            gamma_int.to(dtype=target_X1.dtype),
            graph_rms.to(dtype=target_X1.dtype),
            gamma_graph.to(dtype=target_X1.dtype),
        )

    @torch.no_grad()
    def _satc_interface_residue_weights(self, X, paratope_mask):
        """Native-interface weights for SATC regularization.

        The returned vector has one value per paratope residue in the same order
        as X[paratope_mask].  A residue receives a larger weight when its native
        CA atom is close to any antigen residue in the local antibody-antigen
        graph.  The weights are normalized to graph mean one by default, so this
        focuses the SATC signal spatially without changing the total regularizer
        scale across complexes.
        """
        n_int = int(paratope_mask.sum().item())
        if n_int == 0:
            return X.new_zeros(0)
        if float(getattr(self, "satc_interface_weight_alpha", 0.0)) <= 0.0:
            return X.new_ones(n_int)

        local_mask = self.batch_constants.get('local_mask', None)
        local_is_ab = self.batch_constants.get('local_is_ab', None)
        local_inter_edges = self.batch_constants.get('local_inter_edges', None)
        interface_batch_id = self.batch_constants.get('interface_batch_id', None)
        if (
            local_mask is None or local_is_ab is None
            or local_inter_edges is None or interface_batch_id is None
            or local_inter_edges.numel() == 0
        ):
            return X.new_ones(n_int)

        local_X = X[local_mask]
        ca_idx = 1 if local_X.shape[1] > 1 else 0
        ca = local_X[:, ca_idx]
        row, col = local_inter_edges[0], local_inter_edges[1]
        row_is_ab = local_is_ab[row]
        col_is_ab = local_is_ab[col]
        valid_cross = torch.logical_xor(row_is_ab, col_is_ab)
        if not bool(valid_cross.any()):
            return X.new_ones(n_int)

        row = row[valid_cross]
        col = col[valid_cross]
        row_is_ab = row_is_ab[valid_cross]
        ab_local = torch.where(row_is_ab, row, col)
        ag_local = torch.where(row_is_ab, col, row)

        ab_local_order = torch.nonzero(local_is_ab, as_tuple=False).reshape(-1)
        if ab_local_order.numel() != n_int:
            return X.new_ones(n_int)
        local_to_int = torch.full(
            (local_is_ab.numel(),), -1,
            device=X.device, dtype=torch.long,
        )
        local_to_int[ab_local_order] = torch.arange(n_int, device=X.device)
        ab_int = local_to_int[ab_local]
        valid_ab = ab_int >= 0
        if not bool(valid_ab.any()):
            return X.new_ones(n_int)

        ab_int = ab_int[valid_ab]
        ab_local = ab_local[valid_ab]
        ag_local = ag_local[valid_ab]
        dist = torch.linalg.norm(ca[ab_local] - ca[ag_local], dim=-1)
        cutoff = float(self.satc_interface_cutoff)
        temperature = float(self.satc_interface_temperature)
        edge_contact = torch.sigmoid((cutoff - dist) / temperature).to(X.dtype)

        contact_strength = X.new_zeros(n_int)
        # Compatibility note:
        #   torch.Tensor.scatter_reduce_ is unavailable in the PyTorch version used
        #   by the current AbFlow environment.  We keep the same "amax over antigen
        #   neighbors" semantics with a small per-interface loop.  n_int is the
        #   number of antibody interface residues, so this is negligible compared
        #   with the model forward and does not change the SATC objective.
        for ridx in range(n_int):
            ridx_mask = (ab_int == ridx)
            if bool(ridx_mask.any()):
                contact_strength[ridx] = edge_contact[ridx_mask].max()
        weights = 1.0 + float(self.satc_interface_weight_alpha) * contact_strength

        if bool(getattr(self, "satc_interface_normalize", True)):
            n_graph = int(interface_batch_id.max().item()) + 1 if interface_batch_id.numel() > 0 else 1
            mean_w = scatter_mean(
                weights, interface_batch_id, dim=0, dim_size=n_graph
            )
            weights = weights / mean_w[interface_batch_id].clamp_min(self.scorefm_eps)

        return weights.clamp_min(0.25)

    def _satc_schedule_phase(self):
        """Return the current SATC schedule phase in [0, 1]."""
        if getattr(self, "satc_schedule", "constant") == "constant":
            return 1.0
        step = float(getattr(self, "satc_train_step", torch.zeros(())).detach().cpu().item())
        epoch = step / float(max(1, getattr(self, "satc_steps_per_epoch", 1)))
        start = float(getattr(self, "satc_decay_start_epoch", 100.0))
        end = float(getattr(self, "satc_decay_end_epoch", 130.0))
        if epoch <= start:
            return 1.0
        if epoch >= end:
            return 0.0
        progress = (epoch - start) / max(end - start, self.scorefm_eps)
        progress = min(1.0, max(0.0, progress))
        if getattr(self, "satc_schedule", "constant") == "linear_decay":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _satc_effective_runtime(self, increment_step=False):
        """Effective one-forward SATC coefficients for this forward pass.

        The best configuration keeps score-aware direction alignment as the
        primary signal.  The hybrid configuration uses velocity magnitude only
        as a transient weak auxiliary signal and decays it much faster late in
        training.
        """
        if bool(increment_step) and bool(self.training):
            with torch.no_grad():
                self.satc_train_step.add_(1)
        phase = self._satc_schedule_phase()

        def interp(final_scale):
            final_scale = float(final_scale)
            return final_scale + (1.0 - final_scale) * phase

        perturb_scale = interp(getattr(self, "satc_perturb_final_scale", 1.0))
        score_scale = interp(getattr(self, "satc_score_final_scale", 1.0))
        velocity_scale = interp(getattr(self, "satc_velocity_final_scale", 1.0))
        return {
            "phase": phase,
            "apply_prob": float(self.satc_apply_prob) * perturb_scale,
            "gamma_scale": float(self.satc_gamma_scale) * perturb_scale,
            "score_weight": float(self.satc_score_weight) * score_scale,
            "velocity_weight": float(self.satc_velocity_weight) * velocity_scale,
            "perturb_scale": perturb_scale,
            "score_scale": score_scale,
            "velocity_scale": velocity_scale,
        }

    def _satc_gt_runtime(self, increment_step=False):
        """Deterministic DDP-safe schedule for the v52 extra teacher query."""
        step = int(self.satc_train_step.detach().item())
        epoch = step / float(max(1, self.satc_steps_per_epoch))
        active = bool(
            self.training
            and epoch >= float(self.satc_gt_start_epoch)
            and (step % int(self.satc_gt_interval) == 0)
        )
        if bool(increment_step) and bool(self.training):
            with torch.no_grad():
                self.satc_train_step.add_(1)
        return {
            "step": step,
            "epoch": epoch,
            "active_batch": active,
            "interval": int(self.satc_gt_interval),
            "start_epoch": float(self.satc_gt_start_epoch),
        }

    @torch.no_grad()
    def _satc_graph_translation_state(
            self, clean_Xt, source_X0, target_X1, t_graph,
            interface_batch_id):
        """Perturb only the H3 graph-translation subspace.

        Let c_t be the H3 CA centroid.  For each complex,

            z_t = c_t + gamma_g(t) eps_g,  eps_g ~ N(0, I_3),
            gamma_g(t) = eta * ||c_1-c_0|| / sqrt(3) * 4t(1-t).

        The same translation is broadcast to every atom in the H3 loop, so all
        intra-H3 distances, bond lengths and atom-relative geometry are exactly
        preserved.  The conditional score in this three-dimensional subspace is
        -eps_g / gamma_g.  eta is dimensionless: the expected RMS translation at
        t=0.5 is eta times the source-to-target centroid transport.
        """
        if interface_batch_id.numel() == 0:
            return clean_Xt, clean_Xt.new_zeros(1, 3), clean_Xt.new_zeros(1), clean_Xt.new_zeros(1, dtype=torch.bool), clean_Xt.new_zeros(1)

        n_graph = int(interface_batch_id.max().item()) + 1
        ca_idx = 1 if clean_Xt.shape[1] > 1 else 0
        source_centroid = scatter_mean(
            source_X0[:, ca_idx].float(), interface_batch_id,
            dim=0, dim_size=n_graph,
        )
        target_centroid = scatter_mean(
            target_X1[:, ca_idx].float(), interface_batch_id,
            dim=0, dim_size=n_graph,
        )
        transport = torch.linalg.norm(
            target_centroid - source_centroid, dim=-1
        ).clamp(
            min=float(self.satc_transport_rms_min),
            max=float(self.satc_transport_rms_max),
        )
        t_graph_f = torch.as_tensor(
            t_graph, device=clean_Xt.device, dtype=torch.float32
        ).reshape(-1)
        if t_graph_f.numel() == 1 and n_graph > 1:
            t_graph_f = t_graph_f.expand(n_graph)
        if t_graph_f.numel() != n_graph:
            raise ValueError(
                f"Expected {n_graph} graph times, got {t_graph_f.numel()}."
            )
        bridge_shape = 4.0 * t_graph_f * (1.0 - t_graph_f)
        gamma_graph = (
            float(self.satc_gamma_scale)
            * transport
            * bridge_shape
            / math.sqrt(3.0)
        ).clamp(min=0.0, max=float(self.satc_gamma_abs_max))
        active_graph = (
            (t_graph_f >= float(self.satc_t_min))
            & (t_graph_f <= float(self.satc_t_max))
            & (gamma_graph > self.scorefm_eps)
        )
        eps_graph = torch.randn(
            (n_graph, 3), device=clean_Xt.device, dtype=torch.float32
        )
        delta_graph = gamma_graph[:, None] * eps_graph
        delta_graph = delta_graph * active_graph[:, None].to(delta_graph.dtype)
        delta_int = delta_graph[interface_batch_id].to(clean_Xt.dtype)
        perturbed = clean_Xt + delta_int[:, None, :]
        return (
            perturbed,
            delta_graph.to(clean_Xt.dtype),
            gamma_graph.to(clean_Xt.dtype),
            active_graph,
            transport.to(clean_Xt.dtype),
        )

    def _graph_translation_satc_objective(
            self, *, clean_Xt, perturbed_Xt, clean_pred_X1,
            perturbed_pred_X1, delta_graph, active_graph, t_graph,
            interface_batch_id, endpoint_loss):
        """Stable score-aware pull-back in the H3 translation subspace.

        The clean endpoint prediction acts as a stop-gradient teacher.  Requiring
        the perturbed state to predict the same H3 endpoint makes the induced
        endpoint-parameterized velocity change by exactly -delta/(1-t) when the
        consistency optimum is reached.  This is aligned with the analytic score
        -eps/gamma, but avoids the unstable division by ||correction_true||^2 that
        caused the v51 projection ratio to clip on nearly every residue.
        """
        zero = endpoint_loss * 0.0
        if not bool(active_graph.any()):
            return zero, {
                "scorefm_gt_satc_consistency": zero.detach(),
                "scorefm_gt_satc_rate": zero.detach(),
                "scorefm_gt_satc_perturb_rms": zero.detach(),
                "scorefm_gt_satc_endpoint_shift_rms": zero.detach(),
                "scorefm_gt_satc_velocity_cos": zero.detach(),
                "scorefm_gt_satc_response_ratio": zero.detach(),
                "scorefm_gt_satc_aux_to_endpoint": zero.detach(),
            }

        n_graph = int(interface_batch_id.max().item()) + 1
        ca_idx = 1 if clean_Xt.shape[1] > 1 else 0
        clean_pred_centroid = scatter_mean(
            clean_pred_X1[:, ca_idx], interface_batch_id,
            dim=0, dim_size=n_graph,
        )
        pert_pred_centroid = scatter_mean(
            perturbed_pred_X1[:, ca_idx], interface_batch_id,
            dim=0, dim_size=n_graph,
        )
        clean_state_centroid = scatter_mean(
            clean_Xt[:, ca_idx], interface_batch_id,
            dim=0, dim_size=n_graph,
        )
        pert_state_centroid = scatter_mean(
            perturbed_Xt[:, ca_idx], interface_batch_id,
            dim=0, dim_size=n_graph,
        )

        teacher = clean_pred_centroid.detach()
        per_graph = F.smooth_l1_loss(
            pert_pred_centroid, teacher, reduction="none"
        ).mean(dim=-1)
        consistency = per_graph[active_graph].mean()
        weighted = float(self.satc_score_weight) * consistency

        with torch.no_grad():
            endpoint_shift = pert_pred_centroid - clean_pred_centroid
            perturb = delta_graph.to(endpoint_shift.dtype)
            t = torch.as_tensor(
                t_graph, device=endpoint_shift.device,
                dtype=endpoint_shift.dtype,
            ).reshape(-1)
            sigma = (1.0 - t).clamp_min(self.scorefm_min_sigma)
            v_clean = self.flow_matcher.endpoint_velocity(
                clean_state_centroid, clean_pred_centroid, t[:, None]
            )
            v_pert = self.flow_matcher.endpoint_velocity(
                pert_state_centroid, pert_pred_centroid, t[:, None]
            )
            response = v_pert - v_clean
            target = -perturb / sigma[:, None]
            dot = (response * target).sum(dim=-1)
            response_norm = torch.linalg.norm(response, dim=-1)
            target_norm = torch.linalg.norm(target, dim=-1)
            cos = dot / (
                response_norm * target_norm + self.scorefm_eps
            )
            ratio = dot / (target_norm.pow(2) + self.scorefm_eps)
            perturb_rms = torch.sqrt(
                perturb[active_graph].pow(2).mean().clamp_min(0.0)
            )
            endpoint_shift_rms = torch.sqrt(
                endpoint_shift[active_graph].pow(2).mean().clamp_min(0.0)
            )
            aux_ratio = weighted.detach() / (
                endpoint_loss.detach().abs() + self.scorefm_eps
            )

        return weighted, {
            "scorefm_gt_satc_consistency": consistency.detach(),
            "scorefm_gt_satc_rate": active_graph.float().mean().detach(),
            "scorefm_gt_satc_perturb_rms": perturb_rms.detach(),
            "scorefm_gt_satc_endpoint_shift_rms": endpoint_shift_rms.detach(),
            "scorefm_gt_satc_velocity_cos": cos[active_graph].mean().detach(),
            "scorefm_gt_satc_response_ratio": ratio[active_graph].mean().detach(),
            "scorefm_gt_satc_aux_to_endpoint": aux_ratio.detach(),
        }

    def _coordinate_training_objective(
            self, *, Xt, X1, pred_clean_X, atom_mask,
            interface_batch_id, t, sigma_t, source_ca_mean,
            source_X0=None, si_gamma_t=None, si_gamma_prime_t=None,
            sat_eps_t=None, sat_gamma_t=None, sat_active_t=None,
            satc_residue_weight=None, satc_score_weight_eff=None,
            satc_velocity_weight_eff=None, satc_schedule_info=None,
            satc_transport_rms_graph=None, satc_gamma_graph=None,
            structured_endpoint_target=None, structured_path_details=None):
        """Coordinate objective for the shadow paratope.

        endpoint mode:
            Per-complex endpoint SmoothL1 for every sample.

        si_score / si_score_fm modes:
            Keep endpoint reconstruction as the main target and add small
            stochastic-interpolant analytic score / velocity regularizers.
            These terms are induced by the endpoint prediction and the known
            injected noise, so no independent score or velocity head is added.

        analytic_core mode:
            Historical reference-source score diagnostic.
        """
        primary_target = X1
        if (
            self.scorefm_loss_mode in {"structured_global_cfm", "structured_multiscale_cfm"}
            and structured_endpoint_target is not None
        ):
            primary_target = structured_endpoint_target

        endpoint_per_graph, endpoint_valid = (
            self._masked_residue_smooth_l1_per_graph(
                pred_clean_X, primary_target, atom_mask, interface_batch_id
            )
        )
        if endpoint_valid.any():
            endpoint_loss = endpoint_per_graph[endpoint_valid].mean()
        else:
            endpoint_loss = pred_clean_X.new_tensor(0.0)

        clean_endpoint_per_graph, clean_endpoint_valid = (
            self._masked_residue_smooth_l1_per_graph(
                pred_clean_X, X1, atom_mask, interface_batch_id
            )
        )
        if clean_endpoint_valid.any():
            clean_endpoint_loss = clean_endpoint_per_graph[clean_endpoint_valid].mean()
        else:
            clean_endpoint_loss = pred_clean_X.new_tensor(0.0)

        zero = endpoint_loss.detach() * 0.0
        # Keep differentiable objective components only until the trainer's
        # optional gradient-conflict probe has run.
        self._last_endpoint_objective_tensor = endpoint_loss
        self._last_satc_objective_tensor = endpoint_loss * 0.0

        if self.scorefm_loss_mode in {
            "endpoint", "traj_consistency", "traj_consistency_fm",
            "score_aware_graph_translation_consistency",
            "structured_global_endpoint", "structured_global_cfm",
            "structured_multiscale_cfm",
        }:
            _spd = structured_path_details or {}
            details = {
                "scorefm_total": endpoint_loss.detach(),
                "scorefm_endpoint": endpoint_loss.detach(),
                "scorefm_clean_endpoint": clean_endpoint_loss.detach(),
                "scorefm_structured_primary": endpoint_loss.detach(),
                "scorefm_structured_transport_mean": torch.as_tensor(
                    _spd.get("transport_mean", 0.0), device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_structured_path_rms": torch.as_tensor(
                    _spd.get("path_rms", 0.0), device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_structured_target_shift_rms": torch.as_tensor(
                    _spd.get("target_shift_rms", 0.0), device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_structured_local_transport_rms": torch.as_tensor(
                    _spd.get("local_transport_rms", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_structured_local_path_rms": torch.as_tensor(
                    _spd.get("local_path_rms", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_structured_local_centroid_rms": torch.as_tensor(
                    _spd.get("local_centroid_rms", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_structured_cfm": pred_clean_X.new_tensor(
                    1.0 if self.scorefm_loss_mode in {"structured_global_cfm", "structured_multiscale_cfm"} else 0.0
                ).detach(),
                "scorefm_dsm": zero,
                "scorefm_dsm_rate": zero,
                "scorefm_velocity": zero,
                "scorefm_velocity_rate": zero,
                "scorefm_traj_consistency": zero,
                "scorefm_traj_velocity": zero,
                "scorefm_traj_rate": zero,
            }
            return endpoint_loss, details

        if self.scorefm_loss_mode in {
            "score_aware_traj_lite", "score_aware_traj_fm_lite",
            "score_aware_traj_if_lite", "score_aware_traj_if_fm_lite",
            "score_aware_traj_nt_lite", "score_aware_traj_nt_fm_lite",
            "score_aware_traj_if_nt_lite", "score_aware_traj_if_nt_fm_lite"
        }:
            if (
                source_X0 is None or sat_eps_t is None
                or sat_gamma_t is None or sat_active_t is None
            ):
                details = {
                    "scorefm_total": endpoint_loss.detach(),
                    "scorefm_endpoint": endpoint_loss.detach(),
                    "scorefm_dsm": zero,
                    "scorefm_dsm_rate": zero,
                    "scorefm_velocity": zero,
                    "scorefm_velocity_rate": zero,
                    "scorefm_traj_consistency": zero,
                    "scorefm_traj_velocity": zero,
                    "scorefm_traj_rate": zero,
                    "scorefm_satc_score": zero,
                    "scorefm_satc_velocity": zero,
                    "scorefm_satc_rate": zero,
                }
                return endpoint_loss, details

            sigma_safe = torch.as_tensor(
                sigma_t, device=pred_clean_X.device, dtype=pred_clean_X.dtype
            ).clamp_min(self.scorefm_min_sigma)
            gamma = torch.as_tensor(
                sat_gamma_t, device=pred_clean_X.device, dtype=pred_clean_X.dtype
            ).clamp_min(self.scorefm_eps)
            eps = torch.as_tensor(
                sat_eps_t, device=pred_clean_X.device, dtype=pred_clean_X.dtype
            )
            active_res = torch.as_tensor(
                sat_active_t, device=pred_clean_X.device
            ).reshape(-1).bool()

            # Endpoint-induced velocity at the off-path state Z_t.
            # Clean bridge velocity is X1 - X0.  The remaining component should
            # point back along the analytic score direction -epsilon because
            # Z_t = X_t^clean + gamma(t) epsilon.
            pred_velocity = self.flow_matcher.endpoint_velocity(
                Xt, pred_clean_X, t
            )
            clean_velocity = self.flow_matcher.clean_velocity(
                source_X0, X1
            )
            correction_pred = pred_velocity - clean_velocity
            correction_true = self.flow_matcher.correction_target(
                gamma, t, eps
            )

            valid_atom = atom_mask.bool() & active_res[:, None]
            valid_res = valid_atom.any(dim=-1)

            score_loss = zero
            velocity_loss = zero
            normal_ratio_mean = zero
            normal_ratio_negative_rate = zero
            normal_ratio_satisfied_rate = zero
            normal_ratio_clipped_rate = zero
            perturb_total_rms = zero
            perturb_translation_rms = zero
            perturb_internal_rms = zero
            perturb_internal_energy_fraction = zero
            perturb_to_transport_rms = zero
            satc_rate = active_res.float().mean() if active_res.numel() > 0 else zero

            # Observe what the stochastic tube actually perturbs.  This does not
            # change the objective.  For each residue we decompose the Cartesian
            # displacement into a shared translation and atom-relative internal
            # deformation.  The decomposition directly tests whether the legacy
            # iid tube represents H3 placement recovery or mostly local atom noise.
            if bool(valid_res.any()):
                with torch.no_grad():
                    delta = (gamma * eps).float()
                    vm_full = valid_atom.to(delta.dtype)
                    n_atom = vm_full.sum(dim=-1).clamp_min(1.0)
                    translation = (
                        delta * vm_full.unsqueeze(-1)
                    ).sum(dim=1) / n_atom.unsqueeze(-1)
                    internal = delta - translation.unsqueeze(1)
                    total_energy_res = (
                        delta.pow(2).sum(dim=-1) * vm_full
                    ).sum(dim=-1) / (3.0 * n_atom)
                    translation_energy_res = translation.pow(2).sum(dim=-1) / 3.0
                    internal_energy_res = (
                        internal.pow(2).sum(dim=-1) * vm_full
                    ).sum(dim=-1) / (3.0 * n_atom)
                    active_valid = valid_res
                    total_energy = total_energy_res[active_valid].mean()
                    translation_energy = translation_energy_res[active_valid].mean()
                    internal_energy = internal_energy_res[active_valid].mean()
                    perturb_total_rms = torch.sqrt(total_energy.clamp_min(0.0))
                    perturb_translation_rms = torch.sqrt(translation_energy.clamp_min(0.0))
                    perturb_internal_rms = torch.sqrt(internal_energy.clamp_min(0.0))
                    perturb_internal_energy_fraction = internal_energy / (
                        total_energy + self.scorefm_eps
                    )
                    transport = (X1 - source_X0).detach().float()
                    transport_energy_res = (
                        transport.pow(2).sum(dim=-1) * vm_full
                    ).sum(dim=-1) / (3.0 * n_atom)
                    transport_rms = torch.sqrt(
                        transport_energy_res[active_valid].mean().clamp_min(0.0)
                    )
                    perturb_to_transport_rms = perturb_total_rms / (
                        transport_rms + self.scorefm_eps
                    )

            if bool(valid_res.any()):
                cp = correction_pred[valid_res]
                ct = correction_true[valid_res].detach()
                vm = valid_atom[valid_res]

                dot = (cp * ct).sum(dim=-1)
                cp_norm = cp.pow(2).sum(dim=-1).sqrt()
                ct_norm = ct.pow(2).sum(dim=-1).sqrt()

                if self.scorefm_loss_mode in {
                    "score_aware_traj_nt_lite", "score_aware_traj_nt_fm_lite",
                    "score_aware_traj_if_nt_lite", "score_aware_traj_if_nt_fm_lite"
                }:
                    # Normal--tangent decomposed SATC.  The tangent transport
                    # component is X1-X0 and is already handled by endpoint
                    # flow matching.  For the off-path perturbation, constrain
                    # only the scalar projection of the residual correction on
                    # the analytic normal score direction.  Orthogonal/tangent
                    # residuals are intentionally not penalized here; otherwise
                    # the regularizer can suppress useful endpoint transport and
                    # damage H3 placement/DockQ late in training.
                    # Stable normal pull objective.  The previous raw ratio
                    #     (cp · ct) / ||ct||^2
                    # is mathematically interpretable but numerically unsafe at
                    # initialization because ||ct|| is deliberately small
                    # (gamma/sigma times Gaussian noise).  A bad early prediction
                    # can make the raw ratio very negative and the squared hinge
                    # can dominate the whole AbFlow loss.  We therefore keep the
                    # same normal--tangent semantics, but evaluate the hinge on a
                    # smoothly bounded ratio.  This constrains only whether the
                    # residual correction has a positive pull-back component along
                    # the analytic score normal; it does not penalize tangent or
                    # orthogonal transport residuals.
                    normal_ratio_raw = dot / (ct_norm.pow(2) + self.scorefm_eps)
                    pull_clip = float(getattr(self, "satc_nt_pull_clip", 2.0))
                    if self.satc_projection_bound_mode == "hard_clip":
                        # Exact coefficient inside the safe interval.  In
                        # particular, a theoretical ratio of one remains one.
                        normal_ratio = normal_ratio_raw.clamp(
                            min=-pull_clip, max=pull_clip
                        )
                    else:
                        # Backward-compatible v45 behavior.
                        normal_ratio = (
                            torch.tanh(normal_ratio_raw / pull_clip)
                            * pull_clip
                        )
                    min_pull = float(getattr(self, "satc_nt_min_pull", 0.15))
                    score_atom_loss = F.relu(min_pull - normal_ratio).pow(2)
                    valid_ratio = normal_ratio_raw.masked_select(vm)
                    if valid_ratio.numel() > 0:
                        normal_ratio_mean = valid_ratio.mean()
                        normal_ratio_negative_rate = (valid_ratio < 0).float().mean()
                        normal_ratio_satisfied_rate = (
                            valid_ratio >= min_pull
                        ).float().mean()
                        normal_ratio_clipped_rate = (
                            valid_ratio.abs() >= pull_clip
                        ).float().mean()
                else:
                    # Legacy SATC: constrain the full residual correction vector
                    # to align with the analytic score direction.  Kept for
                    # ablations, but NT modes are preferred for the main method.
                    cos = dot / (cp_norm * ct_norm + self.scorefm_eps)
                    score_atom_loss = 1.0 - cos.clamp(-1.0, 1.0)

                score_atom_loss = score_atom_loss.masked_fill(~vm, 0.0)
                score_res_loss = score_atom_loss.sum(dim=-1) / vm.float().sum(dim=-1).clamp_min(1.0)

                graph_ids = interface_batch_id[valid_res]
                n_graph = int(interface_batch_id.max().item()) + 1
                if satc_residue_weight is not None and self.scorefm_loss_mode in {
                    "score_aware_traj_if_lite", "score_aware_traj_if_fm_lite",
                    "score_aware_traj_if_nt_lite", "score_aware_traj_if_nt_fm_lite"
                }:
                    res_w_full = torch.as_tensor(
                        satc_residue_weight, device=pred_clean_X.device,
                        dtype=pred_clean_X.dtype,
                    ).reshape(-1)
                    res_w = res_w_full[valid_res].clamp_min(self.scorefm_eps)
                    score_num = pred_clean_X.new_zeros(n_graph)
                    score_den = pred_clean_X.new_zeros(n_graph)
                    score_num.scatter_add_(0, graph_ids, score_res_loss * res_w)
                    score_den.scatter_add_(0, graph_ids, res_w)
                    per_graph = score_num / score_den.clamp_min(self.scorefm_eps)
                    score_loss = per_graph[score_den > self.scorefm_eps].mean()
                else:
                    res_w = None
                    per_graph = scatter_mean(score_res_loss, graph_ids, dim=0, dim_size=n_graph)
                    score_loss = per_graph.mean()

                if self.scorefm_loss_mode in {
                    "score_aware_traj_fm_lite", "score_aware_traj_if_fm_lite",
                    "score_aware_traj_nt_fm_lite", "score_aware_traj_if_nt_fm_lite"
                }:
                    # Project the learned correction onto the analytic score
                    # direction and softly match the target correction magnitude.
                    # This is a one-forward velocity-field constraint, not a
                    # second endpoint target and not an independent velocity head.
                    direction = ct / (ct_norm.unsqueeze(-1) + self.scorefm_eps)
                    proj = (cp * direction).sum(dim=-1)
                    target_mag = ct_norm.detach()
                    if self.scorefm_loss_mode in {"score_aware_traj_nt_fm_lite", "score_aware_traj_if_nt_fm_lite"}:
                        # NT-FM softly matches only the normal correction
                        # magnitude.  This absorbs the useful velocity signal
                        # from SATC_FM without constraining the full velocity
                        # vector or its orthogonal/tangent residuals.
                        # Match only the bounded normal-projection ratio.
                        # This preserves the useful velocity signal while
                        # preventing rare early outliers from dominating training.
                        pull_clip = float(getattr(self, "satc_nt_pull_clip", 2.0))
                        proj_ratio_raw = proj / (target_mag + self.scorefm_eps)
                        if self.satc_magnitude_loss_mode == "unbiased_ratio_huber":
                            # Hard clipping limits outliers but does not move the
                            # optimum: exact analytic magnitude has raw ratio 1
                            # and therefore zero SmoothL1 loss.
                            proj_ratio = proj_ratio_raw.clamp(
                                min=-pull_clip, max=pull_clip
                            )
                        else:
                            # Backward-compatible v45 mapping whose optimum is
                            # pull_clip*atanh(1/pull_clip), not exactly one.
                            proj_ratio = (
                                torch.tanh(proj_ratio_raw / pull_clip)
                                * pull_clip
                            )
                        vel_atom_loss = F.smooth_l1_loss(
                            proj_ratio,
                            torch.ones_like(proj_ratio),
                            reduction="none",
                        )
                        valid_ratio = proj_ratio_raw.masked_select(vm)
                        if valid_ratio.numel() > 0:
                            normal_ratio_mean = valid_ratio.mean()
                            normal_ratio_negative_rate = (valid_ratio < 0).float().mean()
                            normal_ratio_satisfied_rate = (valid_ratio >= 1.0).float().mean()
                            normal_ratio_clipped_rate = (
                                valid_ratio.abs() >= pull_clip
                            ).float().mean()
                    else:
                        vel_atom_loss = F.smooth_l1_loss(
                            proj, target_mag, reduction="none"
                        )
                    vel_atom_loss = vel_atom_loss.masked_fill(~vm, 0.0)
                    vel_res_loss = vel_atom_loss.sum(dim=-1) / vm.float().sum(dim=-1).clamp_min(1.0)
                    if res_w is not None:
                        vel_num = pred_clean_X.new_zeros(n_graph)
                        vel_den = pred_clean_X.new_zeros(n_graph)
                        vel_num.scatter_add_(0, graph_ids, vel_res_loss * res_w)
                        vel_den.scatter_add_(0, graph_ids, res_w)
                        per_graph_v = vel_num / vel_den.clamp_min(self.scorefm_eps)
                        velocity_loss = per_graph_v[vel_den > self.scorefm_eps].mean()
                    else:
                        per_graph_v = scatter_mean(vel_res_loss, graph_ids, dim=0, dim_size=n_graph)
                        velocity_loss = per_graph_v.mean()

            score_weight_eff = (
                float(self.satc_score_weight)
                if satc_score_weight_eff is None else float(satc_score_weight_eff)
            )
            velocity_weight_eff = (
                float(self.satc_velocity_weight)
                if satc_velocity_weight_eff is None else float(satc_velocity_weight_eff)
            )
            weighted_score = score_weight_eff * score_loss
            weighted_velocity = velocity_weight_eff * velocity_loss
            weighted_aux = weighted_score + weighted_velocity
            self._last_satc_objective_tensor = weighted_aux
            total = endpoint_loss + weighted_aux
            aux_to_endpoint = weighted_aux.detach() / (
                endpoint_loss.detach().abs() + self.scorefm_eps
            )
            transport_rms_mean = (
                zero if satc_transport_rms_graph is None
                else torch.as_tensor(
                    satc_transport_rms_graph,
                    device=pred_clean_X.device,
                    dtype=pred_clean_X.dtype,
                ).mean().detach()
            )
            gamma_mean = (
                zero if satc_gamma_graph is None
                else torch.as_tensor(
                    satc_gamma_graph,
                    device=pred_clean_X.device,
                    dtype=pred_clean_X.dtype,
                ).mean().detach()
            )
            if satc_residue_weight is not None:
                iw = torch.as_tensor(
                    satc_residue_weight, device=pred_clean_X.device,
                    dtype=pred_clean_X.dtype
                ).reshape(-1)
                interface_weight_mean = iw.mean().detach()
                interface_weight_std = iw.std(unbiased=False).detach()
                interface_weight_max = iw.max().detach()
                interface_weight_ess = (
                    iw.sum().pow(2)
                    / (iw.pow(2).sum() * max(1, iw.numel()) + self.scorefm_eps)
                ).detach()
            else:
                interface_weight_mean = zero
                interface_weight_std = zero
                interface_weight_max = zero
                interface_weight_ess = zero
            details = {
                "scorefm_total": total.detach(),
                "scorefm_endpoint": endpoint_loss.detach(),
                "scorefm_dsm": zero,
                "scorefm_dsm_rate": zero,
                "scorefm_velocity": zero,
                "scorefm_velocity_rate": zero,
                "scorefm_traj_consistency": zero,
                "scorefm_traj_velocity": zero,
                "scorefm_traj_rate": zero,
                "scorefm_satc_score": score_loss.detach(),
                "scorefm_satc_velocity": velocity_loss.detach(),
                "scorefm_satc_rate": satc_rate.detach(),
                "scorefm_satc_score_weight_eff": pred_clean_X.new_tensor(score_weight_eff),
                "scorefm_satc_velocity_weight_eff": pred_clean_X.new_tensor(velocity_weight_eff),
                "scorefm_satc_nt_min_pull": pred_clean_X.new_tensor(float(getattr(self, "satc_nt_min_pull", 0.15))),
                "scorefm_satc_nt_pull_clip": pred_clean_X.new_tensor(float(getattr(self, "satc_nt_pull_clip", 2.0))),
                "scorefm_satc_normal_ratio_mean": normal_ratio_mean.detach(),
                "scorefm_satc_normal_ratio_negative_rate": normal_ratio_negative_rate.detach(),
                "scorefm_satc_normal_ratio_satisfied_rate": normal_ratio_satisfied_rate.detach(),
                "scorefm_satc_normal_ratio_clipped_rate": normal_ratio_clipped_rate.detach(),
                "scorefm_satc_perturb_total_rms": perturb_total_rms.detach(),
                "scorefm_satc_perturb_translation_rms": perturb_translation_rms.detach(),
                "scorefm_satc_perturb_internal_rms": perturb_internal_rms.detach(),
                "scorefm_satc_perturb_internal_energy_fraction": perturb_internal_energy_fraction.detach(),
                "scorefm_satc_perturb_to_transport_rms": perturb_to_transport_rms.detach(),
                "scorefm_satc_interface_weight_mean": interface_weight_mean,
                "scorefm_satc_interface_weight_std": interface_weight_std,
                "scorefm_satc_interface_weight_max": interface_weight_max,
                "scorefm_satc_interface_weight_ess": interface_weight_ess,
                "scorefm_satc_aux_to_endpoint": aux_to_endpoint.detach(),
                "scorefm_satc_transport_rms_mean": transport_rms_mean,
                "scorefm_satc_gamma_mean": gamma_mean,
                "scorefm_satc_schedule_phase": pred_clean_X.new_tensor(
                    1.0 if satc_schedule_info is None else float(satc_schedule_info.get("phase", 1.0))
                ),
            }
            return total, details

        if self.scorefm_loss_mode in {"si_score", "si_score_fm"}:
            if source_X0 is None or si_gamma_t is None or si_gamma_prime_t is None:
                raise ValueError(
                    "si_score/si_score_fm require source_X0, si_gamma_t and "
                    "si_gamma_prime_t. These are created only in state_path mode."
                )

            t_tensor = torch.as_tensor(
                t, device=pred_clean_X.device, dtype=pred_clean_X.dtype
            )
            if t_tensor.dim() == 0 or t_tensor.numel() == 1:
                t_int = t_tensor.reshape(1, 1, 1)
            else:
                t_int = t_tensor.reshape(-1, 1, 1)

            gamma = torch.as_tensor(
                si_gamma_t, device=pred_clean_X.device, dtype=pred_clean_X.dtype
            ).clamp_min(self.scorefm_min_sigma)
            gamma_prime = torch.as_tensor(
                si_gamma_prime_t, device=pred_clean_X.device, dtype=pred_clean_X.dtype
            )

            # True and predicted means of the noisy stochastic interpolant:
            #   Z_t = (1-t) X0 + t X1 + gamma(t) eps.
            # The model still predicts X1; the score/velocity regularizers are
            # analytically induced by this endpoint prediction.
            mu_true = (1.0 - t_int) * source_X0 + t_int * X1
            mu_pred = (1.0 - t_int) * source_X0 + t_int * pred_clean_X

            # Analytic Gaussian score: s(z_t) = -(z_t - mu_t) / gamma(t)^2.
            # We compare gamma * score residual, following the AbX-style
            # scaled-score convention.  This keeps the target analytic while
            # avoiding an independent score head.
            pred_score = -(Xt - mu_pred) / (gamma ** 2)
            true_score = -(Xt - mu_true) / (gamma ** 2)
            scaled_score_diff = gamma * (pred_score - true_score)
            score_zero = torch.zeros_like(scaled_score_diff)
            score_per_graph, score_valid = (
                self._masked_residue_smooth_l1_per_graph(
                    scaled_score_diff, score_zero, atom_mask, interface_batch_id
                )
            )
            if score_valid.any():
                si_score_loss = score_per_graph[score_valid].mean()
            else:
                si_score_loss = pred_clean_X.new_tensor(0.0)

            si_velocity_loss = pred_clean_X.new_tensor(0.0)
            velocity_per_graph = endpoint_per_graph.new_zeros(endpoint_per_graph.shape)
            velocity_valid = endpoint_valid.clone()

            if self.scorefm_loss_mode == "si_score_fm":
                # Stochastic-interpolant velocity consistency.
                # True velocity:      X1 - X0 + gamma'(t) eps.
                # Predicted velocity: X1_pred - X0 + gamma'(t) eps_pred.
                # eps_pred is induced by the endpoint-predicted mean, not by
                # an extra head.
                eps_true = (Xt - mu_true) / gamma
                eps_pred = (Xt - mu_pred) / gamma
                true_velocity = X1 - source_X0 + gamma_prime * eps_true
                pred_velocity = pred_clean_X - source_X0 + gamma_prime * eps_pred
                velocity_per_graph, velocity_valid = (
                    self._masked_residue_smooth_l1_per_graph(
                        pred_velocity, true_velocity, atom_mask, interface_batch_id
                    )
                )
                if velocity_valid.any():
                    si_velocity_loss = velocity_per_graph[velocity_valid].mean()

            t_graph, t_valid = self._scorefm_time_per_graph(
                t, interface_batch_id, pred_clean_X
            )
            valid = endpoint_valid & score_valid & t_valid
            if self.scorefm_loss_mode == "si_score_fm":
                valid = valid & velocity_valid

            if not valid.any():
                details = {
                    "scorefm_total": endpoint_loss.detach(),
                    "scorefm_endpoint": endpoint_loss.detach(),
                    "scorefm_dsm": si_score_loss.detach(),
                    "scorefm_dsm_rate": zero,
                    "scorefm_velocity": si_velocity_loss.detach(),
                    "scorefm_velocity_rate": zero,
                    "scorefm_si_score": si_score_loss.detach(),
                    "scorefm_si_velocity": si_velocity_loss.detach(),
                }
                return endpoint_loss, details

            use_si = (
                (t_graph >= self.scorefm_dsm_t_min)
                & (t_graph <= self.scorefm_dsm_t_max)
            )

            total_per_graph = endpoint_per_graph.clone()
            total_per_graph = total_per_graph + torch.where(
                use_si,
                float(self.si_score_weight) * score_per_graph,
                torch.zeros_like(score_per_graph),
            )
            if self.scorefm_loss_mode == "si_score_fm":
                total_per_graph = total_per_graph + torch.where(
                    use_si,
                    float(self.si_velocity_weight) * velocity_per_graph,
                    torch.zeros_like(velocity_per_graph),
                )

            total = total_per_graph[valid].mean()
            details = {
                "scorefm_total": total.detach(),
                "scorefm_endpoint": endpoint_loss.detach(),
                "scorefm_dsm": si_score_loss.detach(),
                "scorefm_dsm_rate": use_si[valid].float().mean().detach(),
                "scorefm_velocity": si_velocity_loss.detach(),
                "scorefm_velocity_rate": (
                    use_si[valid].float().mean().detach()
                    if self.scorefm_loss_mode == "si_score_fm" else zero
                ),
                "scorefm_si_score": si_score_loss.detach(),
                "scorefm_si_velocity": si_velocity_loss.detach(),
            }
            return total, details

        if self.scorefm_loss_mode == "velocity_core":
            # PCS-consistent deterministic bridge Flow Matching.
            #
            # For PCS-RC-LC, X0 is a proposal-conditioned source rather than a
            # reference Gaussian. Therefore the mathematically consistent
            # dynamic target is the exact deterministic bridge velocity:
            #
            #     Xt = (1 - t) X0 + t X1,
            #     u*_t = (X1 - Xt) / (1 - t).
            #
            # The model predicts a clean endpoint X1^theta; its induced
            # velocity is
            #
            #     u^theta_t = (X1^theta - Xt) / (1 - t).
            #
            # Inside the configured time interval we replace endpoint
            # reconstruction by bridge-velocity supervision. Outside that
            # interval we keep endpoint reconstruction. This avoids stacking
            # redundant losses while testing whether dynamic trajectory
            # supervision improves the strong PCS_RC_LC_R1 baseline.
            sigma_safe = torch.as_tensor(
                sigma_t, device=pred_clean_X.device, dtype=pred_clean_X.dtype
            ).clamp_min(self.scorefm_min_sigma)

            pred_velocity = self.flow_matcher.endpoint_velocity(
                Xt, pred_clean_X, t
            )
            true_velocity = self.flow_matcher.endpoint_velocity(
                Xt, X1, t
            )

            velocity_per_graph, velocity_valid = (
                self._masked_residue_smooth_l1_per_graph(
                    pred_velocity, true_velocity, atom_mask, interface_batch_id
                )
            )

            if velocity_valid.any():
                velocity_loss = velocity_per_graph[velocity_valid].mean()
            else:
                velocity_loss = pred_clean_X.new_tensor(0.0)

            t_graph, t_valid = self._scorefm_time_per_graph(
                t, interface_batch_id, pred_clean_X
            )
            valid = endpoint_valid & velocity_valid & t_valid

            if not valid.any():
                details = {
                    "scorefm_total": endpoint_loss.detach(),
                    "scorefm_endpoint": endpoint_loss.detach(),
                    "scorefm_dsm": zero,
                    "scorefm_dsm_rate": zero,
                    "scorefm_velocity": velocity_loss.detach(),
                    "scorefm_velocity_rate": zero,
                }
                return endpoint_loss, details

            use_velocity = (
                (t_graph >= self.scorefm_dsm_t_min)
                & (t_graph <= self.scorefm_dsm_t_max)
            )

            per_graph = torch.where(
                use_velocity,
                velocity_per_graph,
                endpoint_per_graph,
            )
            total = per_graph[valid].mean()

            details = {
                "scorefm_total": total.detach(),
                "scorefm_endpoint": endpoint_loss.detach(),
                "scorefm_dsm": zero,
                "scorefm_dsm_rate": zero,
                "scorefm_velocity": velocity_loss.detach(),
                "scorefm_velocity_rate": use_velocity[valid].float().mean().detach(),
            }
            return total, details

        gt_score_ca = self._analytic_ca_score_from_clean(
            Xt, X1, t, sigma_t, source_ca_mean
        ).detach()
        pred_score_ca = self._analytic_ca_score_from_clean(
            Xt, pred_clean_X, t, sigma_t, source_ca_mean
        )

        if sigma_t.dim() == 3:
            sigma_ca = sigma_t[:, 0, :]
        else:
            sigma_ca = sigma_t.reshape(-1, 1)

        # AbX-style score scaling: sigma_t * score residual. DSM is used
        # only where t/(1-t) is bounded by the configured interval.
        scaled_score_diff = (
            sigma_ca * (pred_score_ca - gt_score_ca)
        ).unsqueeze(1)

        ca_idx = 1 if atom_mask.shape[1] > 1 else 0
        ca_mask = atom_mask[:, ca_idx:ca_idx + 1]

        dsm_per_graph, dsm_valid = self._masked_residue_mse_per_graph(
            scaled_score_diff, ca_mask, interface_batch_id
        )

        t_graph, t_valid = self._scorefm_time_per_graph(
            t, interface_batch_id, pred_clean_X
        )
        valid = endpoint_valid & dsm_valid & t_valid

        if dsm_valid.any():
            dsm_loss = dsm_per_graph[dsm_valid].mean()
        else:
            dsm_loss = pred_clean_X.new_tensor(0.0)

        if not valid.any():
            details = {
                "scorefm_total": endpoint_loss.detach(),
                "scorefm_endpoint": endpoint_loss.detach(),
                "scorefm_dsm": dsm_loss.detach(),
                "scorefm_dsm_rate": zero,
            }
            return endpoint_loss, details

        use_dsm = (
            (t_graph >= self.scorefm_dsm_t_min)
            & (t_graph <= self.scorefm_dsm_t_max)
        )

        per_graph = torch.where(
            use_dsm,
            dsm_per_graph,
            endpoint_per_graph,
        )
        total = per_graph[valid].mean()

        details = {
            "scorefm_total": total.detach(),
            "scorefm_endpoint": endpoint_loss.detach(),
            "scorefm_dsm": dsm_loss.detach(),
            "scorefm_dsm_rate": use_dsm[valid].float().mean().detach(),
        }
        return total, details

    def _forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                 surface, residue_pos, template, lengths, init_noise=None,
                 interface_init=None, sequence_init=None, flow_t=None):
        """Evaluate f_theta(X_t, S_t, t, proposal context).

        interface_init/sequence_init are the explicit generated state Xt/St.
        In PCS-RC mode X_pep/S_pep are also used to build a recurrent global
        proposal context, but they never overwrite the explicit shadow state.
        """
        batch_id = self.batch_constants['batch_id']

        X = X.clone()
        S = S.clone()
        surface = surface.clone()

        has_interface_state = interface_init is not None
        has_sequence_state = sequence_init is not None

        X, S = self.init_mask(X, S, cmask, smask, template)

        if has_interface_state:
            expected_shape = X[paratope_mask].shape
            if interface_init.shape != expected_shape:
                raise ValueError(
                    f"interface_init shape mismatch: expected {tuple(expected_shape)}, "
                    f"got {tuple(interface_init.shape)}."
                )

        if has_sequence_state:
            expected_shape = S[paratope_mask].shape
            if sequence_init.shape != expected_shape:
                raise ValueError(
                    f"sequence_init shape mismatch: expected {tuple(expected_shape)}, "
                    f"got {tuple(sequence_init.shape)}."
                )

        # Build the graph context.
        #
        # reference/PCS:
        #     the global graph carries the current generated state Xt/St.
        # PCS-RC:
        #     the global graph carries the recurrent proposal context X_pep/S_pep,
        #     while the shadow interface below carries the true generated Xt/St.
        #
        # This recovers the original AbFlow information channel, but avoids
        # erasing the explicit generated state at every flow step.
        use_recurrent_proposal_context = bool(
            getattr(self, "abflow_recurrent_proposal_context", False)
        )

        if use_recurrent_proposal_context:
            X, S = self.replace_pep(
                X, S, paratope_mask, X_pep, S_pep,
                replace_seq=True, replace_struct=True,
            )
        else:
            if has_interface_state:
                X[paratope_mask] = interface_init.to(
                    device=X.device, dtype=X.dtype
                )
            if has_sequence_state:
                S[paratope_mask] = sequence_init.to(
                    device=S.device, dtype=torch.long
                )


        X = self.normalizer.centering(X, S, batch_id, self.aa_feature)
        X = self.normalizer.normalize(X)
        surface = self.normalizer.normalize(surface)
        X = self.aa_feature.update_global_coordinates(X, S)

        if has_interface_state:
            interface_X = self._raw_interface_to_model_frame(
                interface_init, paratope_mask, batch_id
            )
            if has_sequence_state:
                interface_S = sequence_init.to(
                    device=S.device, dtype=torch.long
                ).clone()
            else:
                interface_S = S[paratope_mask].clone()
        else:
            interface_X, interface_S = self.init_interface(
                X, S, paratope_mask, batch_id, init_noise
            )
            interface_X, interface_S = self._condition_initial_interface(
                interface_X, interface_S, X_pep, S_pep
            )

        # Convert X_pep once to the internal frame. Its relation to the current
        # interface state is recomputed after every refinement round. Proposal
        # validity is tracked per residue so missing/invalid proposal coordinates
        # cannot silently become a condition.
        pep_X_model = None
        pep_coord_valid = None
        if (
            self.coord_pep_as_condition
            and X_pep is not None
            and X_pep.shape == interface_X.shape
        ):
            pep_X_raw = X_pep.to(device=X.device, dtype=X.dtype)
            proposal_backbone = pep_X_raw[:, :3]
            pep_coord_valid = (
                torch.isfinite(proposal_backbone).all(dim=-1).all(dim=-1)
                & (
                    proposal_backbone.abs()
                    .sum(dim=-1)
                    .sum(dim=-1)
                    > self.scorefm_eps
                )
            )
            if pep_coord_valid.any():
                pep_X_model = self._raw_interface_to_model_frame(
                    pep_X_raw, paratope_mask, batch_id
                )

        if self.seq_pep_condition_embedding is not None:
            seq_ref_tensor = interface_X.new_zeros(
                (paratope_mask.shape[0],
                 self.seq_pep_condition_embedding.embedding_dim)
            )
        else:
            seq_ref_tensor = interface_X.new_zeros(
                (paratope_mask.shape[0], 1)
            )
        seq_pep_condition, seq_pep_condition_mask = (
            self._build_seq_pep_condition_for_residues(
                S_pep, paratope_mask, seq_ref_tensor
            )
        )
        sequence_state_full = None
        if has_sequence_state and self.dual_sequence_state:
            sequence_state_full = S.clone()
            sequence_state_full[paratope_mask] = interface_S

        r_pred_S_logits, pred_S_dist = [], None
        r_interface_X = [interface_X.clone()]
        r_edge_dist = []
        memory_H = None
        diagnostics_active = bool(
            self.condition_diagnostics_enabled
            and getattr(self, "_diagnostic_capture", False)
        )
        condition_diag_rounds = [] if diagnostics_active else None

        for round_idx in range(self.round):
            # Role-separated local correction.  The recurrent proposal context
            # is present in every round through X/S.  The proposal-relative
            # adapters are optionally delayed so the first refinement round can
            # establish H3 placement before local proposal correction is applied.
            use_local_correction = (
                round_idx >= int(getattr(self, "proposal_adapter_start_round", 0))
            )

            if use_local_correction:
                (
                    coord_pep_condition,
                    coord_pep_condition_mask,
                ) = self._build_coord_pep_condition_for_residues(
                    pep_X_model,
                    interface_X,
                    paratope_mask,
                    pep_coord_valid=pep_coord_valid,
                )
                seq_pep_condition_this = seq_pep_condition
                seq_pep_condition_mask_this = seq_pep_condition_mask
            else:
                coord_pep_condition = None
                coord_pep_condition_mask = None
                seq_pep_condition_this = None
                seq_pep_condition_mask_this = None

            pred_S_logits, pred_X, interface_X, H, edge_dist = self.message_passing(
                X, S, residue_pos, interface_X, surface, paratope_mask,
                batch_id, round_idx, memory_H, pred_S_dist, smask,
                flow_t=flow_t,
                coord_pep_condition=coord_pep_condition,
                coord_pep_condition_mask=coord_pep_condition_mask,
                seq_pep_condition=seq_pep_condition_this,
                seq_pep_condition_mask=seq_pep_condition_mask_this,
                sequence_state_full=sequence_state_full,
            )

            if condition_diag_rounds is not None:
                condition_diag_rounds.append({
                    key: value.detach()
                    for key, value in self._last_condition_diagnostics.items()
                })

            memory_H = H
            r_interface_X.append(interface_X.clone())
            r_pred_S_logits.append((pred_S_logits, smask))
            r_edge_dist.append(edge_dist)

            X = X.clone()
            X[cmask] = pred_X[cmask]
            X = self.aa_feature.update_global_coordinates(X, S)

            if not self.struct_only:
                S = S.clone()
                if round_idx == self.round - 1:
                    S[smask] = torch.argmax(
                        pred_S_logits[smask], dim=-1
                    )
                else:
                    pred_S_dist = torch.softmax(
                        pred_S_logits[smask], dim=-1
                    )

        if condition_diag_rounds:
            keys = condition_diag_rounds[0].keys()
            self._latest_condition_diagnostics = {
                key: torch.stack(
                    [round_diag[key] for round_diag in condition_diag_rounds]
                ).mean()
                for key in keys
            }
        else:
            self._latest_condition_diagnostics = {}

        interface_batch_id = self.batch_constants['interface_batch_id']
        if self.struct_only:
            prmsd = self.prmsd_ffn(H[cmask]).squeeze()
        else:
            prmsd = None

        pred_X = self.normalizer.unnormalize(pred_X)
        pred_X = self.normalizer.uncentering(pred_X, batch_id)
        for i, interface_X_i in enumerate(r_interface_X):
            interface_X_i = self.normalizer.unnormalize(interface_X_i)
            interface_X_i = self.normalizer.uncentering(
                interface_X_i, interface_batch_id, _type=4
            )
            r_interface_X[i] = interface_X_i

        self.normalizer.clear_cache()
        return H, S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd


    @torch.no_grad()
    def _validation_proxy_diagnostics(
            self, *, true_X, true_S, pred_S, r_pred_S_logits, r_interface_X,
            paratope_mask, smask, batch_id, interface_batch_id):
        """Cheap validation proxies aligned with the final evaluation axes.

        These are not substitutes for TM-score/lDDT/DockQ and are never used as
        test-set checkpoint selection.  They answer where the refinement process
        changes: raw placement, aligned local geometry, native contacts and
        contact-residue sequence recovery.
        """
        out = {}
        if interface_batch_id.numel() == 0:
            return out
        true_int = true_X[paratope_mask]
        ca_idx = 1 if true_int.shape[1] > 1 else 0
        true_ca = true_int[:, ca_idx].float()
        n_graph = int(interface_batch_id.max().item()) + 1

        round_raw = []
        round_aligned = []
        for ridx, pred_int in enumerate(r_interface_X[1:]):
            pred_ca = pred_int[:, ca_idx].float()
            raw_values, aligned_values = [], []
            for g in range(n_graph):
                m = interface_batch_id == g
                if not bool(m.any()):
                    continue
                p, q = pred_ca[m], true_ca[m]
                raw_values.append(torch.sqrt(((p - q) ** 2).sum(-1).mean()))
                if p.shape[0] >= 3:
                    try:
                        _, rot, trans = kabsch_torch(p, q)
                        p_aligned = torch.matmul(p, rot.T) + trans
                        aligned_values.append(
                            torch.sqrt(((p_aligned - q) ** 2).sum(-1).mean())
                        )
                    except Exception:
                        pass
            if raw_values:
                rv = torch.stack(raw_values).mean()
                out[f"val_proxy_round{ridx}_h3_ca_rmsd"] = rv
                round_raw.append(rv)
            if aligned_values:
                av = torch.stack(aligned_values).mean()
                out[f"val_proxy_round{ridx}_h3_ca_aligned_rmsd"] = av
                round_aligned.append(av)

        for ridx, (logits, mask) in enumerate(r_pred_S_logits):
            if bool(mask.any()):
                pred_round = torch.argmax(logits[mask], dim=-1)
                out[f"val_proxy_round{ridx}_aar"] = (
                    pred_round == true_S[mask]
                ).float().mean()

        if round_raw:
            out["val_proxy_refinement_raw_rmsd_delta"] = (
                round_raw[-1] - round_raw[0]
            )
        if round_aligned:
            out["val_proxy_refinement_aligned_rmsd_delta"] = (
                round_aligned[-1] - round_aligned[0]
            )

        is_ag = self.batch_constants.get("is_ag")
        if is_ag is None:
            return out
        pred_final_ca = r_interface_X[-1][:, ca_idx].float()
        contact_f1, contact_precision, contact_recall, caar_values = [], [], [], []
        for g in range(n_graph):
            pm = interface_batch_id == g
            agm = (batch_id == g) & is_ag & (true_S != self.aa_feature.boa_idx)
            if not bool(pm.any()) or not bool(agm.any()):
                continue
            ag_ca = true_X[agm, ca_idx].float()
            native_contact = torch.cdist(true_ca[pm], ag_ca) < 8.0
            pred_contact = torch.cdist(pred_final_ca[pm], ag_ca) < 8.0
            tp = (native_contact & pred_contact).float().sum()
            fp = ((~native_contact) & pred_contact).float().sum()
            fn = (native_contact & (~pred_contact)).float().sum()
            precision = tp / (tp + fp + self.scorefm_eps)
            recall = tp / (tp + fn + self.scorefm_eps)
            f1 = 2.0 * precision * recall / (precision + recall + self.scorefm_eps)
            contact_precision.append(precision)
            contact_recall.append(recall)
            contact_f1.append(f1)
            native_res = native_contact.any(dim=-1)
            if bool(native_res.any()):
                global_par_idx = paratope_mask.nonzero(as_tuple=False).reshape(-1)[pm]
                caar_values.append((
                    pred_S[global_par_idx[native_res]]
                    == true_S[global_par_idx[native_res]]
                ).float().mean())
        if contact_f1:
            out["val_proxy_native_contact_f1"] = torch.stack(contact_f1).mean()
            out["val_proxy_native_contact_precision"] = torch.stack(contact_precision).mean()
            out["val_proxy_native_contact_recall"] = torch.stack(contact_recall).mean()
        if caar_values:
            out["val_proxy_caar"] = torch.stack(caar_values).mean()
        return out

    def compute_gradient_conflict_diagnostics(self):
        """AMP/DDP-safe observational objective-gradient probe.

        v50 differentiated each objective with respect to a DDP parameter while
        still inside bf16 autocast. PyTorch 2.0.1 can then fail in
        ``at::autocast::prioritize``. This version probes a shared activation,
        disables autocast for autograd.grad, uses fp32 diagnostics, and fails
        soft because an observational diagnostic must never stop training.
        """
        if not self.grad_conflict_diagnostics:
            self.last_gradient_diagnostics = {}
            return {}

        terms = self._diagnostic_objective_tensors
        probe = self._diagnostic_probe_tensor
        if not terms or probe is None or not torch.is_tensor(probe):
            self.last_gradient_diagnostics = {}
            return {}
        if not probe.requires_grad:
            self.last_gradient_diagnostics = {}
            return {}

        grads = {}
        self._last_gradient_diagnostic_error = ""
        amp_ctx = torch.cuda.amp.autocast(enabled=False) if probe.is_cuda else nullcontext()
        try:
            with amp_ctx:
                for name, value in terms.items():
                    if not torch.is_tensor(value) or not value.requires_grad:
                        continue
                    scalar = value.float()
                    if scalar.numel() != 1:
                        scalar = scalar.mean()
                    g = torch.autograd.grad(
                        scalar, probe, retain_graph=True, allow_unused=True
                    )[0]
                    if g is not None:
                        grads[name] = g.detach().float().reshape(-1)
        except RuntimeError as exc:
            self._last_gradient_diagnostic_error = str(exc)
            self.last_gradient_diagnostics = {
                "grad_probe_failed": probe.detach().new_tensor(1.0, dtype=torch.float32),
                "grad_probe_amp_safe": probe.detach().new_tensor(0.0, dtype=torch.float32),
            }
            return self.last_gradient_diagnostics

        out = {
            "grad_probe_failed": probe.detach().new_tensor(0.0, dtype=torch.float32),
            "grad_probe_amp_safe": probe.detach().new_tensor(1.0, dtype=torch.float32),
        }
        for name, g in grads.items():
            out[f"grad_probe_norm_{name}"] = torch.linalg.norm(g)
        pairs = [
            ("endpoint", "satc"),
            ("seq", "satc"),
            ("endpoint", "seq"),
            ("structure", "satc"),
        ]
        for a, b in pairs:
            if a in grads and b in grads:
                ga, gb = grads[a], grads[b]
                out[f"grad_probe_cos_{a}_{b}"] = (
                    torch.dot(ga, gb)
                    / (torch.linalg.norm(ga) * torch.linalg.norm(gb) + self.scorefm_eps)
                )
        self.last_gradient_diagnostics = {k: v.detach() for k, v in out.items()}
        return self.last_gradient_diagnostics

    def _trajectory_consistency_objective(
            self, *, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
            surface, residue_pos, template, lengths,
            Xt, pred_clean_X, interface_atom_mask, interface_batch_id,
            t_graph, sequence_state_for_model):
        """Local trajectory consistency for endpoint-parameterized Flow Matching.

        This objective is designed for PCS_RC_LC_R1 after the endpoint baseline
        is already strong.  It does not compare an induced score to a target
        score, and it does not add an independent prediction head.

        Given the current generated state Xt at time t and the model-predicted
        endpoint X1_hat(t), we form a short model-induced Euler step:

            v_t^theta = (X1_hat(t) - Xt) / (1 - t)
            X_{t+dt}^theta = Xt + dt * stopgrad(v_t^theta)

        We then call the same network again at (X_{t+dt}^theta, t+dt) and ask
        its predicted endpoint to stay consistent with stopgrad(X1_hat(t)).
        This directly constrains the local self-consistency of the learned
        trajectory.  The FM variant additionally asks the induced velocity at
        the neighboring state to match the previous velocity.
        """
        zero = pred_clean_X.new_tensor(0.0)
        if self.scorefm_loss_mode not in {
            "traj_consistency", "traj_consistency_fm"
        }:
            return zero, {
                "scorefm_traj_consistency": zero.detach(),
                "scorefm_traj_velocity": zero.detach(),
                "scorefm_traj_rate": zero.detach(),
            }

        if Xt is None or pred_clean_X is None or t_graph is None:
            return zero, {
                "scorefm_traj_consistency": zero.detach(),
                "scorefm_traj_velocity": zero.detach(),
                "scorefm_traj_rate": zero.detach(),
            }

        if interface_batch_id.numel() == 0:
            return zero, {
                "scorefm_traj_consistency": zero.detach(),
                "scorefm_traj_velocity": zero.detach(),
                "scorefm_traj_rate": zero.detach(),
            }

        device = pred_clean_X.device
        dtype = pred_clean_X.dtype
        n_graph = int(interface_batch_id.max().item()) + 1
        t_graph = torch.as_tensor(t_graph, device=device, dtype=dtype)
        if t_graph.dim() == 0 or t_graph.numel() == 1:
            t_graph = t_graph.reshape(1).expand(n_graph)
        else:
            t_graph = t_graph.reshape(-1)
            if t_graph.numel() != n_graph:
                raise ValueError(
                    "trajectory consistency expects graph-level t_graph with "
                    f"{n_graph} values, got {t_graph.numel()}."
                )

        # Only apply the consistency term on a safe interval.  This prevents
        # near-source states from being dominated by an unreliable early
        # prediction and prevents near-target states from suffering the
        # 1/(1-t) singularity of endpoint-parameterized velocity.
        max_dt = (1.0 - t_graph - self.scorefm_min_sigma).clamp_min(0.0)
        dt_graph = torch.minimum(
            torch.full_like(t_graph, float(self.traj_delta_t)),
            max_dt,
        )
        active_graph = (
            (t_graph >= float(self.traj_t_min))
            & (t_graph <= float(self.traj_t_max))
            & (dt_graph > self.scorefm_eps)
        )

        if not bool(active_graph.any()):
            return zero, {
                "scorefm_traj_consistency": zero.detach(),
                "scorefm_traj_velocity": zero.detach(),
                "scorefm_traj_rate": zero.detach(),
            }

        t_int = self._time_for_interface(t_graph, interface_batch_id, pred_clean_X)
        dt_int = self._time_for_interface(dt_graph, interface_batch_id, pred_clean_X)
        t_next_graph = (t_graph + dt_graph).clamp(max=1.0 - self.scorefm_min_sigma)
        t_next_int = self._time_for_interface(t_next_graph, interface_batch_id, pred_clean_X)

        sigma_int = (1.0 - t_int).clamp_min(self.scorefm_min_sigma)
        sigma_next_int = (1.0 - t_next_int).clamp_min(self.scorefm_min_sigma)

        # The step is intentionally detached.  The first prediction is already
        # trained by the endpoint loss; the trajectory term trains the same
        # network to be consistent when it is queried at the next state.  This
        # avoids high-memory second-order coupling and reduces collapse risk.
        with torch.no_grad():
            velocity_t = (pred_clean_X - Xt) / sigma_int
            x_next = Xt + dt_int * velocity_t
            endpoint_target = pred_clean_X.detach()
            velocity_target = velocity_t.detach()

        _, _, _, _, r_interface_X_next, _, _ = self._forward(
            X, S, cmask, smask, paratope_mask, X_pep, S_pep,
            surface, residue_pos, template, lengths,
            interface_init=x_next,
            sequence_init=sequence_state_for_model,
            flow_t=t_next_graph,
        )
        pred_next = r_interface_X_next[-1]

        endpoint_per_graph, endpoint_valid = (
            self._masked_residue_smooth_l1_per_graph(
                pred_next, endpoint_target, interface_atom_mask, interface_batch_id
            )
        )

        active = active_graph & endpoint_valid
        if active.any():
            endpoint_consistency = endpoint_per_graph[active].mean()
        else:
            endpoint_consistency = zero

        velocity_consistency = zero
        if self.scorefm_loss_mode == "traj_consistency_fm":
            velocity_next = (pred_next - x_next) / sigma_next_int
            velocity_per_graph, velocity_valid = (
                self._masked_residue_smooth_l1_per_graph(
                    velocity_next, velocity_target,
                    interface_atom_mask, interface_batch_id
                )
            )
            active_v = active_graph & velocity_valid
            if active_v.any():
                velocity_consistency = velocity_per_graph[active_v].mean()

        total = (
            float(self.traj_consistency_weight) * endpoint_consistency
            + float(self.traj_velocity_weight) * velocity_consistency
        )
        traj_rate = active_graph.float().mean()
        return total, {
            "scorefm_traj_consistency": endpoint_consistency.detach(),
            "scorefm_traj_velocity": velocity_consistency.detach(),
            "scorefm_traj_rate": traj_rate.detach(),
        }

    def forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, xloss_mask, context_ratio=0):
        '''
        :param X: [N, n_channel, 3], Cartesian coordinates
        :param context_ratio: float, rate of context provided in masked sequence, should be [0, 1) and anneal to 0 in training, probability of keeping ground-truth sequence context among originally masked positions.
        '''
        # import ipdb; ipdb.set_trace()
        # Do not retain a shared activation from a previous batch.
        self._diagnostic_probe_tensor = None
        self._last_gradient_diagnostic_error = ""
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
            if X_pep is not None:
                X_pep = X_pep[:, :4]
            xloss_mask = xloss_mask[:, :4]
        # clone ground truth coordinates, sequence
        true_X, true_S = X.clone(), S.clone()

        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)
        batch_id = self.batch_constants['batch_id']

        # Sequence design mask and supervision mask are deliberately separated.
        #
        # legacy:
        #   Reproduces the original curriculum: native context is injected by
        #   removing most designed residues from both the categorical path and CE.
        # loss_only:
        #   The complete H3 categorical state follows q_t for every designed
        #   residue, while CE may be subsampled for curriculum purposes.
        # off:
        #   Every designed residue follows the same train-time path used at
        #   inference and every residue is supervised.  This is the formal v51
        #   setting because hard/native context leakage is amplified by DUAL_SEQ.
        design_smask = smask.clone()
        sequence_loss_mask = design_smask.clone()
        if self.sequence_context_mode == "legacy":
            if context_ratio > 0:
                not_ctx_mask = (
                    torch.rand_like(smask, dtype=torch.float) >= context_ratio
                )
                smask = torch.logical_and(design_smask, not_ctx_mask)
                sequence_loss_mask = smask
        elif self.sequence_context_mode == "loss_only":
            smask = design_smask
            if context_ratio > 0:
                not_ctx_mask = (
                    torch.rand_like(design_smask, dtype=torch.float)
                    >= context_ratio
                )
                sequence_loss_mask = torch.logical_and(
                    design_smask, not_ctx_mask
                )
        elif self.sequence_context_mode == "off":
            smask = design_smask
            sequence_loss_mask = design_smask
        sequence_path_mask = (
            smask if self.sequence_context_mode == "legacy"
            else design_smask
        )

        gt_interface_X = true_X[paratope_mask]
        batch_size = int(self.batch_constants['batch_size'].item()) if torch.is_tensor(self.batch_constants['batch_size']) else int(self.batch_constants['batch_size'])
        interface_batch_id = self.batch_constants['interface_batch_id']
        state_path = bool(self.scorefm_state_path)

        if state_path:
            # Sample X_0/S_0 from the configured source distribution.
            #
            # reference:
            #     antigen-centered random source.
            # PCS/PCS-RC:
            #     proposal-conditioned source using X_pep/S_pep when valid.
            interface_X, interface_S = self.init_interface(
                X, S, paratope_mask, batch_id
            )
            interface_X, interface_S = self._condition_initial_interface(
                interface_X, interface_S, X_pep, S_pep
            )
            source_ca_mean = self._reference_ca_mean(
                X, S, paratope_mask, batch_id
            )

            # Continuous flow time. Avoid sigma_t = 1 - t being too small because
            # the analytic score contains 1 / sigma_t^2.
            t_graph = self._sample_flow_times(batch_size, device=X.device, dtype=X.dtype)

            # Real path weight: must use the true endpoint geometry.
            # X_t = (1 - t) X_0 + t X_1 reaches exactly X_1 at t=1.
            base_weight_graph = 1.0 - t_graph

            # Score denominator: numerically protected only for score/velocity loss.
            sigma_score_graph = base_weight_graph.clamp_min(self.scorefm_min_sigma)

            t_int = self._time_for_interface(t_graph, interface_batch_id, interface_X)
            base_weight_int = self._time_for_interface(base_weight_graph, interface_batch_id, interface_X)
            sigma_score_int = self._time_for_interface(sigma_score_graph, interface_batch_id, interface_X)

            mu_t = self.flow_matcher.interpolate(
                interface_X, gt_interface_X, t_int
            )

            si_gamma_int = None
            si_gamma_prime_int = None
            sat_eps_int = None
            sat_gamma_int = None
            sat_active_int = None
            satc_runtime = None
            satc_transport_rms_graph = None
            satc_gamma_graph = None
            gt_satc_runtime = None
            gt_satc_Xt = None
            gt_satc_delta_graph = None
            gt_satc_active_graph = None
            gt_satc_transport_graph = None
            structured_endpoint_target = None
            structured_path_details = None
            if self.scorefm_loss_mode in {
                "structured_global_endpoint", "structured_global_cfm"
            }:
                Xt, structured_endpoint_target, structured_path_details = (
                    self._structured_global_primary_path(
                        mu_t=mu_t, source_X0=interface_X, target_X1=gt_interface_X,
                        t_graph=t_graph, interface_batch_id=interface_batch_id,
                    )
                )
            elif self.scorefm_loss_mode == "structured_multiscale_cfm":
                Xt, structured_endpoint_target, structured_path_details = (
                    self._structured_multiscale_primary_path(
                        mu_t=mu_t, source_X0=interface_X, target_X1=gt_interface_X,
                        t_graph=t_graph, interface_batch_id=interface_batch_id,
                    )
                )
            elif self.scorefm_loss_mode in {"si_score", "si_score_fm"}:
                # Training-only stochastic interpolant around the PCS source-to-
                # native bridge.  This creates an analytic score target without
                # adding an independent score head:
                #   Z_t = mu_t + gamma(t) * eps,
                #   gamma(t) = gamma_scale * t * (1 - t).
                gamma_graph = (
                    float(self.si_gamma_scale)
                    * t_graph
                    * (1.0 - t_graph)
                ).clamp_min(self.scorefm_min_sigma)
                gamma_prime_graph = float(self.si_gamma_scale) * (1.0 - 2.0 * t_graph)
                si_gamma_int = self._time_for_interface(
                    gamma_graph, interface_batch_id, interface_X
                )
                si_gamma_prime_int = self._time_for_interface(
                    gamma_prime_graph, interface_batch_id, interface_X
                )
                Xt = mu_t + si_gamma_int * torch.randn_like(mu_t)
            elif self.scorefm_loss_mode in {
                "score_aware_traj_lite", "score_aware_traj_fm_lite",
                "score_aware_traj_if_lite", "score_aware_traj_if_fm_lite",
                "score_aware_traj_nt_lite", "score_aware_traj_nt_fm_lite",
                "score_aware_traj_if_nt_lite", "score_aware_traj_if_nt_fm_lite"
            }:
                # One-forward score-aware off-path training.  We perturb only
                # the model input state, keep X1 as the endpoint target, and
                # use the known perturbation direction to regularize the
                # endpoint-induced correction velocity.  This avoids a second
                # _forward call while still exposing score-defined off-path
                # states to the R1 flow.
                satc_runtime = self._satc_effective_runtime(increment_step=True)
                active_graph = (
                    (torch.rand_like(t_graph) < float(satc_runtime["apply_prob"]))
                    & (t_graph >= float(self.satc_t_min))
                    & (t_graph <= float(self.satc_t_max))
                )

                if self.satc_tube_mode == "transport_calibrated":
                    tube_atom_pos = self.aa_feature._construct_atom_pos(
                        true_S[paratope_mask]
                    )
                    tube_atom_mask = (
                        tube_atom_pos != self.aa_feature.atom_pos_pad_idx
                    )
                    (
                        sat_gamma_int,
                        satc_transport_rms_graph,
                        satc_gamma_graph,
                    ) = self._satc_transport_calibrated_gamma(
                        source_X0=interface_X,
                        target_X1=gt_interface_X,
                        atom_mask=tube_atom_mask,
                        t_graph=t_graph,
                        interface_batch_id=interface_batch_id,
                        gamma_scale=float(satc_runtime["gamma_scale"]),
                    )
                else:
                    gamma_graph = (
                        float(satc_runtime["gamma_scale"])
                        * t_graph
                        * (1.0 - t_graph)
                    ).clamp_min(self.scorefm_eps)
                    sat_gamma_int = self._time_for_interface(
                        gamma_graph, interface_batch_id, interface_X
                    )
                    satc_gamma_graph = gamma_graph

                sat_active_int = active_graph[interface_batch_id].reshape(-1, 1, 1)
                # Keep iid Gaussian noise in AbFlow's actual full-atom Cartesian
                # state.  This retains the analytic isotropic score used by the
                # existing SATC derivation and avoids importing an SO(3) or
                # residue-frame process from a different model family.
                sat_eps_int = torch.randn_like(mu_t)
                Xt = (
                    mu_t
                    + sat_active_int.to(mu_t.dtype)
                    * sat_gamma_int
                    * sat_eps_int
                )
            elif self.scorefm_loss_mode == "score_aware_graph_translation_consistency":
                # Primary endpoint training remains on the clean PCS bridge.
                # At a deterministic interval, a second query sees the same H3
                # state translated as one rigid Cartesian block.  This preserves
                # every internal atom/residue distance and targets the observed
                # global-placement failure without introducing SO(3) dynamics.
                Xt = mu_t
                gt_satc_runtime = self._satc_gt_runtime(increment_step=True)
                if bool(gt_satc_runtime["active_batch"]):
                    (
                        gt_satc_Xt,
                        gt_satc_delta_graph,
                        satc_gamma_graph,
                        gt_satc_active_graph,
                        gt_satc_transport_graph,
                    ) = self._satc_graph_translation_state(
                        clean_Xt=mu_t,
                        source_X0=interface_X,
                        target_X1=gt_interface_X,
                        t_graph=t_graph,
                        interface_batch_id=interface_batch_id,
                    )
            else:
                Xt = mu_t

            if not self.struct_only:
                St = self._sample_categorical_path(
                    true_S[paratope_mask], interface_S, t_graph, interface_batch_id,
                    corrupt_mask=sequence_path_mask[paratope_mask],
                )
                sequence_state_for_model = St
            else:
                St = interface_S
                sequence_state_for_model = None
        else:
            # Non-state evaluator: no explicit X_t/S_t/t is injected.
            interface_X = None
            interface_S = None
            Xt = None
            St = None
            t_int = None
            sigma_score_int = None
            si_gamma_int = None
            si_gamma_prime_int = None
            structured_endpoint_target = None
            structured_path_details = None
            sat_eps_int = None
            sat_gamma_int = None
            sat_active_int = None
            satc_transport_rms_graph = None
            satc_gamma_graph = None
            gt_satc_runtime = None
            gt_satc_Xt = None
            gt_satc_delta_graph = None
            gt_satc_active_graph = None
            gt_satc_transport_graph = None
            t_graph = X.new_zeros(1)
            sequence_state_for_model = None
            source_ca_mean = None

        # get results
        H, pred_S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(
            X, S, cmask, smask, paratope_mask, X_pep, S_pep,
            surface, residue_pos, template, lengths,
            interface_init=Xt if state_path else None,
            sequence_init=sequence_state_for_model if state_path else None,
            flow_t=t_graph if state_path else None
        )

        # v52 score-aware graph-translation teacher query.  It is skipped
        # during validation and on non-scheduled training steps.  Diagnostic
        # capture is temporarily disabled so the primary clean-bridge activation
        # remains the gradient-conflict probe.
        gt_satc_pred_X1 = None
        if (
            state_path
            and self.scorefm_loss_mode
            == "score_aware_graph_translation_consistency"
            and gt_satc_Xt is not None
            and gt_satc_active_graph is not None
            and bool(gt_satc_active_graph.any())
        ):
            capture_saved = bool(getattr(self, "_diagnostic_capture", False))
            probe_saved = self._diagnostic_probe_tensor
            cond_diag_saved = dict(self._latest_condition_diagnostics)
            self._diagnostic_capture = False
            try:
                (
                    _H_gt, _pred_S_gt, _logits_gt, _pred_X_gt,
                    r_interface_X_gt, _edge_gt, _prmsd_gt,
                ) = self._forward(
                    X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                    surface, residue_pos, template, lengths,
                    interface_init=gt_satc_Xt,
                    sequence_init=(
                        sequence_state_for_model if state_path else None
                    ),
                    flow_t=t_graph,
                )
                gt_satc_pred_X1 = r_interface_X_gt[-1]
            finally:
                self._diagnostic_capture = capture_saved
                self._diagnostic_probe_tensor = probe_saved
                self._latest_condition_diagnostics = cond_diag_saved

        # sequence negative log likelihood
        snll = X.new_tensor(0.0)
        total = X.new_tensor(0.0)
        if not self.struct_only:
            for logits, _round_mask in r_pred_S_logits:
                mask = sequence_loss_mask
                if mask.any():
                    snll = snll + F.cross_entropy(
                        logits[mask], true_S[mask], reduction='sum'
                    )
                    total = total + mask.sum()
            snll = snll / total.clamp_min(1.0)

        # structure loss
        struct_loss, struct_loss_details, bb_rmsd, ops = self.protein_feature.structure_loss(pred_X, true_X, true_S, cmask, batch_id, xloss_mask, self.aa_feature)

        # docking loss

        # 1. Unique coordinate objective for the shadow paratope.
        # The previous implementation added a global interface loss and a second
        # x1 auxiliary loss for the same endpoint error. Here the endpoint is
        # supervised exactly once, with per-complex normalization.
        interface_atom_pos = self.aa_feature._construct_atom_pos(
            true_S[paratope_mask]
        )
        interface_atom_mask = (
            interface_atom_pos != self.aa_feature.atom_pos_pad_idx
        )

        satc_residue_weight = None
        if state_path and self.scorefm_loss_mode in {
            "score_aware_traj_if_lite", "score_aware_traj_if_fm_lite",
            "score_aware_traj_if_nt_lite", "score_aware_traj_if_nt_fm_lite"
        }:
            satc_residue_weight = self._satc_interface_residue_weights(
                true_X, paratope_mask
            )

        if state_path:
            interface_loss, scorefm_details = (
                self._coordinate_training_objective(
                    Xt=Xt,
                    X1=gt_interface_X,
                    pred_clean_X=r_interface_X[-1],
                    atom_mask=interface_atom_mask,
                    interface_batch_id=interface_batch_id,
                    t=t_int,
                    sigma_t=sigma_score_int,
                    source_ca_mean=source_ca_mean,
                    source_X0=interface_X,
                    si_gamma_t=si_gamma_int,
                    si_gamma_prime_t=si_gamma_prime_int,
                    sat_eps_t=sat_eps_int,
                    sat_gamma_t=sat_gamma_int,
                    sat_active_t=sat_active_int,
                    satc_residue_weight=satc_residue_weight,
                    satc_score_weight_eff=(
                        None if satc_runtime is None else satc_runtime["score_weight"]
                    ),
                    satc_velocity_weight_eff=(
                        None if satc_runtime is None else satc_runtime["velocity_weight"]
                    ),
                    satc_schedule_info=satc_runtime,
                    satc_transport_rms_graph=satc_transport_rms_graph,
                    satc_gamma_graph=satc_gamma_graph,
                    structured_endpoint_target=structured_endpoint_target,
                    structured_path_details=structured_path_details,
                )
            )
        else:
            endpoint_per_graph, endpoint_valid = (
                self._masked_residue_smooth_l1_per_graph(
                    r_interface_X[-1],
                    gt_interface_X,
                    interface_atom_mask,
                    interface_batch_id,
                )
            )
            if endpoint_valid.any():
                interface_loss = endpoint_per_graph[
                    endpoint_valid
                ].mean()
            else:
                interface_loss = pred_X.new_tensor(0.0)

            zero = interface_loss.detach() * 0.0
            scorefm_details = {
                "scorefm_total": interface_loss.detach(),
                "scorefm_endpoint": interface_loss.detach(),
                "scorefm_dsm": zero,
                "scorefm_dsm_rate": zero,
                "scorefm_velocity": zero,
                "scorefm_velocity_rate": zero,
            }

        if (
            state_path
            and self.scorefm_loss_mode
            == "score_aware_graph_translation_consistency"
        ):
            zero_gt = interface_loss * 0.0
            if gt_satc_pred_X1 is not None:
                gt_aux, gt_details = self._graph_translation_satc_objective(
                    clean_Xt=Xt,
                    perturbed_Xt=gt_satc_Xt,
                    clean_pred_X1=r_interface_X[-1],
                    perturbed_pred_X1=gt_satc_pred_X1,
                    delta_graph=gt_satc_delta_graph,
                    active_graph=gt_satc_active_graph,
                    t_graph=t_graph,
                    interface_batch_id=interface_batch_id,
                    endpoint_loss=interface_loss,
                )
            else:
                gt_aux = zero_gt
                gt_details = {
                    "scorefm_gt_satc_consistency": zero_gt.detach(),
                    "scorefm_gt_satc_rate": zero_gt.detach(),
                    "scorefm_gt_satc_perturb_rms": zero_gt.detach(),
                    "scorefm_gt_satc_endpoint_shift_rms": zero_gt.detach(),
                    "scorefm_gt_satc_velocity_cos": zero_gt.detach(),
                    "scorefm_gt_satc_response_ratio": zero_gt.detach(),
                    "scorefm_gt_satc_aux_to_endpoint": zero_gt.detach(),
                }
            self._last_satc_objective_tensor = gt_aux
            interface_loss = interface_loss + gt_aux
            scorefm_details.update(gt_details)
            scorefm_details["scorefm_gt_satc_gamma_mean"] = (
                zero_gt.detach()
                if satc_gamma_graph is None
                else satc_gamma_graph.detach().mean()
            )
            scorefm_details["scorefm_gt_satc_transport_mean"] = (
                zero_gt.detach()
                if gt_satc_transport_graph is None
                else gt_satc_transport_graph.detach().mean()
            )
            scorefm_details["scorefm_gt_satc_interval"] = (
                zero_gt.detach().new_tensor(float(self.satc_gt_interval))
            )
            scorefm_details["scorefm_gt_satc_start_epoch"] = (
                zero_gt.detach().new_tensor(float(self.satc_gt_start_epoch))
            )
            scorefm_details["scorefm_total"] = interface_loss.detach()

        if state_path and self.scorefm_loss_mode in {
            "traj_consistency", "traj_consistency_fm"
        }:
            traj_loss, traj_details = self._trajectory_consistency_objective(
                X=X,
                S=S,
                cmask=cmask,
                smask=smask,
                paratope_mask=paratope_mask,
                X_pep=X_pep,
                S_pep=S_pep,
                surface=surface,
                residue_pos=residue_pos,
                template=template,
                lengths=lengths,
                Xt=Xt,
                pred_clean_X=r_interface_X[-1],
                interface_atom_mask=interface_atom_mask,
                interface_batch_id=interface_batch_id,
                t_graph=t_graph,
                sequence_state_for_model=sequence_state_for_model,
            )
            interface_loss = interface_loss + traj_loss
            scorefm_details.update(traj_details)
            scorefm_details["scorefm_total"] = interface_loss.detach()

        self.last_scorefm_losses = scorefm_details


        # 2. edge dist loss
        if self.pred_edge_dist:
            gt_edge_dist = self._get_inter_edge_dist(self.normalizer.normalize(true_X), true_S)
            ed_loss, r_ed_losses = 0, []
            for edge_dist in r_edge_dist:
                r_ed_loss = F.smooth_l1_loss(edge_dist, gt_edge_dist)
                ed_loss = ed_loss + r_ed_loss
                r_ed_losses.append(r_ed_loss)
        else:
            r_ed_losses = [0 for _ in range(self.round)]
            ed_loss = 0
        dock_loss = interface_loss + ed_loss

        if self.struct_only:
            # predicted rmsd
            prmsd_loss = F.smooth_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss
        else:
            pdev_loss, prmsd_loss = None, None

        # comprehensive loss
        loss = (
            self.seq_ce_weight * snll
            + struct_loss
            + dock_loss
            + (0 if pdev_loss is None else pdev_loss)
        )
        self._diagnostic_objective_tensors = {
            "seq": self.seq_ce_weight * snll,
            "structure": struct_loss,
            "endpoint": getattr(
                self, "_last_endpoint_objective_tensor", interface_loss
            ),
            "satc": getattr(
                self, "_last_satc_objective_tensor", interface_loss * 0.0
            ),
            "edge": ed_loss if torch.is_tensor(ed_loss) else loss * 0.0,
        }

        # AAR and conditioning diagnostics.
        with torch.no_grad():
            if sequence_loss_mask.any():
                aa_hit = (
                    pred_S[sequence_loss_mask]
                    == true_S[sequence_loss_mask]
                )
                aar = aa_hit.float().mean()
            else:
                aar = X.new_tensor(0.0)

            diag = {
                "seq_ce_weight": torch.as_tensor(self.seq_ce_weight, device=X.device),
                "scorefm_loss_mode_endpoint": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "endpoint" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_velocity_core": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "velocity_core" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_analytic_core": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "analytic_core" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_si_score": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "si_score" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_si_score_fm": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "si_score_fm" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_traj_consistency": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "traj_consistency" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_traj_consistency_fm": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "traj_consistency_fm" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_score_aware_traj_lite": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "score_aware_traj_lite" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_score_aware_traj_fm_lite": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "score_aware_traj_fm_lite" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_score_aware_traj_if_lite": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "score_aware_traj_if_lite" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_score_aware_traj_if_fm_lite": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "score_aware_traj_if_fm_lite" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_score_aware_traj_if_nt_lite": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "score_aware_traj_if_nt_lite" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_score_aware_traj_if_nt_fm_lite": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "score_aware_traj_if_nt_fm_lite" else 0.0,
                    device=X.device,
                ),
                "scorefm_loss_mode_graph_translation_satc": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode
                    == "score_aware_graph_translation_consistency" else 0.0,
                    device=X.device,
                ),
                "si_gamma_scale": torch.as_tensor(
                    float(getattr(self, "si_gamma_scale", 0.0)), device=X.device
                ),
                "si_score_weight": torch.as_tensor(
                    float(getattr(self, "si_score_weight", 0.0)), device=X.device
                ),
                "si_velocity_weight": torch.as_tensor(
                    float(getattr(self, "si_velocity_weight", 0.0)), device=X.device
                ),
                "traj_consistency_weight": torch.as_tensor(
                    float(getattr(self, "traj_consistency_weight", 0.0)), device=X.device
                ),
                "traj_velocity_weight": torch.as_tensor(
                    float(getattr(self, "traj_velocity_weight", 0.0)), device=X.device
                ),
                "traj_delta_t": torch.as_tensor(
                    float(getattr(self, "traj_delta_t", 0.0)), device=X.device
                ),
                "traj_t_min": torch.as_tensor(
                    float(getattr(self, "traj_t_min", 0.0)), device=X.device
                ),
                "traj_t_max": torch.as_tensor(
                    float(getattr(self, "traj_t_max", 0.0)), device=X.device
                ),
                "satc_apply_prob": torch.as_tensor(
                    float(getattr(self, "satc_apply_prob", 0.0)), device=X.device
                ),
                "satc_gamma_scale": torch.as_tensor(
                    float(getattr(self, "satc_gamma_scale", 0.0)), device=X.device
                ),
                "satc_score_weight": torch.as_tensor(
                    float(getattr(self, "satc_score_weight", 0.0)), device=X.device
                ),
                "satc_velocity_weight": torch.as_tensor(
                    float(getattr(self, "satc_velocity_weight", 0.0)), device=X.device
                ),
                "satc_t_min": torch.as_tensor(
                    float(getattr(self, "satc_t_min", 0.0)), device=X.device
                ),
                "satc_t_max": torch.as_tensor(
                    float(getattr(self, "satc_t_max", 0.0)), device=X.device
                ),
                "satc_interface_weight_alpha": torch.as_tensor(
                    float(getattr(self, "satc_interface_weight_alpha", 0.0)), device=X.device
                ),
                "satc_interface_cutoff": torch.as_tensor(
                    float(getattr(self, "satc_interface_cutoff", 0.0)), device=X.device
                ),
                "satc_interface_temperature": torch.as_tensor(
                    float(getattr(self, "satc_interface_temperature", 0.0)), device=X.device
                ),
                "satc_tube_mode_transport_calibrated": torch.as_tensor(
                    1.0 if self.satc_tube_mode == "transport_calibrated" else 0.0,
                    device=X.device,
                ),
                "satc_tube_mode_graph_translation_calibrated": torch.as_tensor(
                    1.0 if self.satc_tube_mode
                    == "graph_translation_calibrated" else 0.0,
                    device=X.device,
                ),
                "satc_gt_interval": torch.as_tensor(
                    float(self.satc_gt_interval), device=X.device
                ),
                "satc_gt_start_epoch": torch.as_tensor(
                    float(self.satc_gt_start_epoch), device=X.device
                ),
                "satc_gamma_abs_max": torch.as_tensor(
                    float(self.satc_gamma_abs_max), device=X.device
                ),
                "satc_projection_bound_hard_clip": torch.as_tensor(
                    1.0 if self.satc_projection_bound_mode == "hard_clip" else 0.0,
                    device=X.device,
                ),
                "satc_magnitude_loss_unbiased": torch.as_tensor(
                    1.0 if self.satc_magnitude_loss_mode == "unbiased_ratio_huber" else 0.0,
                    device=X.device,
                ),
                "scorefm_state_path": torch.as_tensor(
                    1.0 if state_path else 0.0, device=X.device
                ),
                "source_mode_reference": torch.as_tensor(
                    1.0 if getattr(self, "abflow_source_mode", "reference") == "reference" else 0.0,
                    device=X.device
                ),
                "source_mode_pcs": torch.as_tensor(
                    1.0 if getattr(self, "abflow_source_mode", "reference") == "pcs" else 0.0,
                    device=X.device
                ),
                "source_mode_pcs_rc": torch.as_tensor(
                    1.0 if getattr(self, "abflow_source_mode", "reference") == "pcs_rc" else 0.0,
                    device=X.device
                ),
                "recurrent_proposal_context": torch.as_tensor(
                    1.0 if getattr(self, "abflow_recurrent_proposal_context", False) else 0.0,
                    device=X.device
                ),
                "coord_pep_source_weight": torch.as_tensor(
                    float(getattr(self, "coord_pep_source_weight", 0.0)), device=X.device
                ),
                "seq_pep_source_weight": torch.as_tensor(
                    float(getattr(self, "seq_pep_source_weight", 0.0)), device=X.device
                ),
                "coord_pep_as_condition": torch.as_tensor(
                    1.0 if getattr(self, "coord_pep_as_condition", False) else 0.0,
                    device=X.device
                ),
                "proposal_adapter_start_round": torch.as_tensor(
                    float(getattr(self, "proposal_adapter_start_round", 0)),
                    device=X.device
                ),
                "seq_input_mode_state": torch.as_tensor(
                    1.0 if self.seq_input_mode == "state" else 0.0, device=X.device
                ),
                "seq_input_mode_pep_condition": torch.as_tensor(
                    1.0 if self.seq_input_mode == "pep_condition" else 0.0, device=X.device
                ),
                "shadow_seq_state_enabled": torch.as_tensor(
                    1.0 if getattr(self, "dual_sequence_state", False) else 0.0,
                    device=X.device,
                ),
                "dual_sequence_state_enabled": torch.as_tensor(
                    1.0 if getattr(self, "dual_sequence_state", False) else 0.0,
                    device=X.device,
                ),
                "sequence_context_mode_legacy": torch.as_tensor(
                    1.0 if self.sequence_context_mode == "legacy" else 0.0,
                    device=X.device,
                ),
                "sequence_context_mode_loss_only": torch.as_tensor(
                    1.0 if self.sequence_context_mode == "loss_only" else 0.0,
                    device=X.device,
                ),
                "sequence_context_mode_off": torch.as_tensor(
                    1.0 if self.sequence_context_mode == "off" else 0.0,
                    device=X.device,
                ),
                "final_readout_integrated_endpoint": torch.as_tensor(
                    1.0 if self.final_readout_mode == "integrated_endpoint"
                    else 0.0,
                    device=X.device,
                ),
                "deterministic_validation": torch.as_tensor(
                    1.0 if self.deterministic_validation else 0.0,
                    device=X.device,
                ),
                "sequence_path_mask_rate": sequence_path_mask.float().mean(),
                "sequence_loss_mask_rate": sequence_loss_mask.float().mean(),
                "t_mean": t_graph.detach().float().mean(),
                "t_min": t_graph.detach().float().min(),
                "t_max": t_graph.detach().float().max(),
            }
            for key, value in self._latest_condition_diagnostics.items():
                diag[key] = value.detach()

            valid_pep = (
                S_pep is not None
                and S_pep.numel() == int(paratope_mask.sum().item())
            )
            if valid_pep and smask[paratope_mask].any():
                pep_full = torch.empty_like(S)
                pep_full.copy_(S)
                pep_full[paratope_mask] = S_pep.to(device=S.device, dtype=torch.long)
                pep_mask = smask
                pred_pep_hit = pred_S[pep_mask] == pep_full[pep_mask]
                pep_native_hit = pep_full[pep_mask] == true_S[pep_mask]
                diag["seq_pred_vs_pep_aar"] = pred_pep_hit.float().mean()
                diag["seq_pep_vs_native_aar"] = pep_native_hit.float().mean()

            # Measure the proposal's own coordinate quality.  Without this
            # diagnostic, an improvement or degradation from coordinate
            # conditioning cannot be attributed to the condition mechanism
            # versus the quality of X_pep itself.
            valid_coord_pep = (
                X_pep is not None
                and X_pep.shape == gt_interface_X.shape
            )
            if valid_coord_pep:
                pep_raw = X_pep.to(
                    device=gt_interface_X.device,
                    dtype=gt_interface_X.dtype,
                )
                proposal_backbone = pep_raw[:, :3]
                proposal_valid = (
                    torch.isfinite(proposal_backbone)
                    .all(dim=-1)
                    .all(dim=-1)
                    & (
                        proposal_backbone.abs()
                        .sum(dim=-1)
                        .sum(dim=-1)
                        > self.scorefm_eps
                    )
                )
                if proposal_valid.any():
                    ca_idx = 1 if X_pep.shape[1] > 1 else 0
                    pep_ca = pep_raw[:, ca_idx]
                    native_ca = gt_interface_X[:, ca_idx]
                    pep_ca_sq = ((pep_ca - native_ca) ** 2).sum(dim=-1)

                    valid_graph_id = interface_batch_id[proposal_valid]
                    n_graph = int(interface_batch_id.max().item()) + 1
                    pep_ca_sum = torch.zeros(
                        n_graph,
                        device=pep_ca_sq.device,
                        dtype=pep_ca_sq.dtype,
                    )
                    pep_ca_count = torch.zeros(
                        n_graph,
                        device=pep_ca_sq.device,
                        dtype=pep_ca_sq.dtype,
                    )
                    pep_ca_sum.scatter_add_(
                        0,
                        valid_graph_id,
                        pep_ca_sq[proposal_valid],
                    )
                    pep_ca_count.scatter_add_(
                        0,
                        valid_graph_id,
                        torch.ones_like(pep_ca_sq[proposal_valid]),
                    )
                    valid_graph = pep_ca_count > 0
                    pep_ca_mse_graph = (
                        pep_ca_sum
                        / pep_ca_count.clamp_min(1.0)
                    )
                    diag["coord_pep_to_native_ca_rmsd"] = torch.sqrt(
                        pep_ca_mse_graph[valid_graph].clamp_min(0.0)
                    ).mean()
                    diag["coord_pep_valid_rate"] = (
                        proposal_valid.float().mean()
                    )

            if (
                state_path
                and St is not None
                and S_pep is not None
                and S_pep.numel() == St.numel()
            ):
                pep_state = S_pep.to(device=St.device, dtype=torch.long).reshape(-1)
                valid_pair = (
                    (pep_state >= 0) & (pep_state < self.num_classes)
                    & (St >= 0) & (St < self.num_classes)
                )
                if bool(valid_pair.any()):
                    diag["seq_state_vs_pep_disagreement_rate"] = (
                        St[valid_pair] != pep_state[valid_pair]
                    ).float().mean()

            if bool(getattr(self, "_diagnostic_validation_mode", False)):
                diag.update(self._validation_proxy_diagnostics(
                    true_X=true_X, true_S=true_S, pred_S=pred_S,
                    r_pred_S_logits=r_pred_S_logits,
                    r_interface_X=r_interface_X,
                    paratope_mask=paratope_mask, smask=smask,
                    batch_id=batch_id, interface_batch_id=interface_batch_id,
                ))
            self.last_abflow_diagnostics = {
                k: v.detach() if torch.is_tensor(v) else v for k, v in diag.items()
            }

        self._clean_batch_constants()
        return loss, (snll, aar), (struct_loss, *struct_loss_details), (dock_loss, interface_loss, ed_loss, r_ed_losses), (pdev_loss, prmsd_loss)


    def _sampling_time_grid(self, n_steps, device, dtype):
        """Return true interval boundaries [0, ..., 1].

        Velocity is evaluated at left endpoints t_i<1. The final model readout
        is queried at t=1 without evaluating an analytic score denominator.
        """
        n_steps = max(1, int(n_steps))
        return torch.linspace(
            0.0, 1.0, steps=n_steps + 1,
            device=device, dtype=dtype
        )

    @torch.no_grad()
    def sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
               surface, residue_pos, template, lengths, n_steps=10,
               init_noise=None, return_hidden=False, show_progress=False,
               progress_desc=None):
        n_steps = _env_int("ABFLOW_SAMPLE_N_STEPS", n_steps)
        if n_steps < 1:
            raise ValueError("ABFLOW_SAMPLE_N_STEPS must be >= 1.")

        if not bool(getattr(self, "scorefm_state_path", True)):
            return self.struct_sample(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths,
                init_noise=init_noise, return_hidden=return_hidden
            )

        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]
            if X_pep is not None:
                X_pep = X_pep[:, :4]

        gen_X, gen_S = X.clone(), S.clone()
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size_raw = self.batch_constants['batch_size']
        batch_size = (
            int(batch_size_raw.item())
            if torch.is_tensor(batch_size_raw)
            else int(batch_size_raw)
        )
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]

        best_metric = torch.full(
            (batch_size,), 1e10, dtype=torch.float, device=X.device
        )
        interface_cmask = paratope_mask[cmask]

        interface_X, interface_S = self.init_interface(
            X, S, paratope_mask, batch_id, init_noise=init_noise
        )
        interface_X, interface_S = self._condition_initial_interface(
            interface_X, interface_S, X_pep, S_pep
        )
        time_grid = self._sampling_time_grid(
            n_steps, device=X.device, dtype=X.dtype
        )
        Xt = interface_X.clone()
        St = interface_S.clone()

        step_iter = range(n_steps)
        if show_progress:
            step_iter = tqdm(
                step_iter, total=n_steps,
                desc=progress_desc or 'Sampling ODE',
                leave=False, dynamic_ncols=True
            )

        for i in step_iter:
            t = time_grid[i]
            t_next = time_grid[i + 1]
            dt = t_next - t
            flow_t_graph = t.reshape(1).expand(batch_size)
            if show_progress and hasattr(step_iter, 'set_postfix'):
                step_iter.set_postfix(t=f'{float(t):.2f}')

            sequence_state_for_model = St if not self.struct_only else None
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd = self._forward(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths,
                interface_init=Xt,
                sequence_init=sequence_state_for_model,
                flow_t=flow_t_graph
            )
            pred_clean_X = r_interface_X[-1]

            raw_residual = pred_clean_X - Xt
            if self.scorefm_sampler_mode == "residual":
                dX = raw_residual
                Xt = Xt + dX * dt
            elif self.scorefm_sampler_mode == "bridge":
                Xt = self.flow_matcher.bridge_step(
                    Xt, pred_clean_X, t, dt
                )
            else:
                raise ValueError(
                    f"Unknown sampler mode: {self.scorefm_sampler_mode}"
                )

            if not self.struct_only:
                cur_logits = r_pred_S_logits[-1][0][paratope_mask]
                cur_logits = cur_logits - cur_logits.max(
                    dim=-1, keepdim=True
                )[0]
                cur_probs = F.softmax(cur_logits, dim=-1)
                refresh_prob = self.flow_matcher.categorical_refresh_probability(
                    t, dt
                )
                proposed_S = torch.multinomial(
                    cur_probs.clamp_min(1e-8), num_samples=1
                ).squeeze(-1)
                refresh = (
                    torch.rand(St.shape, device=St.device) < refresh_prob
                )
                refresh = refresh & smask[paratope_mask]
                St = torch.where(refresh, proposed_S, St)

        # Terminal readout.
        #
        # For the bridge sampler, the last interval has
        # dt = 1 - t, hence Xt <- Xt + (X1_hat-Xt)/(1-t)*dt = X1_hat.
        # The categorical linear path has the same integrated jump probability
        # dt/(1-t)=1 on the final interval.  Therefore the loop already produces
        # a terminal state.  Querying the network again at exactly t=1 is both
        # redundant and out of the continuous training support.
        if self.final_readout_mode == "legacy_t1_query":
            X_state = X.clone()
            S_state = S.clone()
            X_state[paratope_mask] = Xt
            S_state[paratope_mask] = St

            final_t = time_grid[-1].detach()
            final_flow_t_graph = final_t.reshape(1).expand(batch_size)
            sequence_state_for_model = St if not self.struct_only else None
            (
                H_final, pred_S_final, r_pred_S_logits_final,
                pred_X_final, r_interface_X_final, _, prmsd_final
            ) = self._forward(
                X_state, S_state, cmask, smask, paratope_mask,
                X_pep, S_pep, surface, residue_pos, template, lengths,
                interface_init=Xt,
                sequence_init=sequence_state_for_model,
                flow_t=final_flow_t_graph,
            )
            interface_X_final = r_interface_X_final[-1]
            final_logits_full = (
                None if self.struct_only
                else r_pred_S_logits_final[-1][0]
            )
        else:
            # Reuse the final left-endpoint prediction and the integrated state.
            H_final = H
            pred_X_final = pred_X
            prmsd_final = prmsd
            interface_X_final = Xt
            final_logits_full = (
                None if self.struct_only
                else r_pred_S_logits[-1][0]
            )
            pred_S_final = pred_S.clone()
            if not self.struct_only and bool(smask.any()):
                if self.sequence_decode_mode == "argmax":
                    pred_S_final[smask] = torch.argmax(
                        final_logits_full[smask], dim=-1
                    )
                else:
                    # Keep the terminal CTMC sample for diverse generation.
                    pred_S_final[paratope_mask] = St

        if not self.struct_only:
            S_logits = final_logits_full[smask]
            if S_logits.shape[0] > 0:
                S_probs = torch.softmax(
                    S_logits, dim=-1
                ).max(dim=-1)[0]
                nlls = -torch.log(S_probs.clamp_min(1e-8))
                metric = scatter_mean(
                    nlls, s_batch_id, dim=0, dim_size=batch_size
                )
            else:
                metric = best_metric.new_zeros(batch_size)
        else:
            metric = scatter_mean(
                prmsd_final[interface_cmask], interface_batch_id,
                dim=0, dim_size=batch_size
            )

        update = metric < best_metric
        cupdate = cmask & update[batch_id]
        supdate = smask & update[batch_id]
        best_metric[update] = metric[update]
        gen_X[cupdate] = pred_X_final[cupdate]
        if not self.struct_only:
            gen_S[supdate] = pred_S_final[supdate]

        # Preserve the original AbFlow global-antibody alignment convention, but
        # align to the integrated terminal interface rather than a second t=1
        # network query.
        for b in range(batch_size):
            if not update[b]:
                continue
            is_cur_graph = batch_id == b
            current_paratope = is_cur_graph & paratope_mask
            ori_cdr = gen_X[current_paratope][:, :4]
            pred_cdr = interface_X_final[
                interface_batch_id == b
            ][:, :4]
            _, R, trans = kabsch_torch(
                ori_cdr.reshape(-1, 3), pred_cdr.reshape(-1, 3)
            )
            is_cur_ab = is_cur_graph & is_ab
            gen_X[is_cur_ab] = torch.matmul(
                gen_X[is_cur_ab], R.T
            ) + trans

        self._clean_batch_constants()
        if return_hidden:
            return gen_X, gen_S, best_metric, H_final
        return gen_X, gen_S, best_metric

    def struct_sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, init_noise=None, return_hidden=False):
        
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
            if X_pep is not None:
                X_pep = X_pep[:, :4]
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        batch_size = int(batch_size.item()) if torch.is_tensor(batch_size) else int(batch_size)
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]

        best_metric = torch.ones(batch_size, dtype=torch.float, device=X.device) * 1e10
        interface_cmask = paratope_mask[cmask]

        n_tries = 10 if self.struct_only else 1
        for i in range(n_tries):
        
            # generate
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd = self._forward(X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, init_noise)

            # PPL or PRMSD
            if not self.struct_only:
                S_logits = r_pred_S_logits[-1][0][smask]
                S_probs = torch.max(torch.softmax(S_logits, dim=-1), dim=-1)[0]
                nlls = -torch.log(S_probs)
                metric = scatter_mean(nlls, s_batch_id, dim=0, dim_size=batch_size)  # [batch_size]
            else:
                metric = scatter_mean(prmsd[interface_cmask], interface_batch_id, dim=0, dim_size=batch_size)  # [batch_size]

            update = metric < best_metric
            cupdate = cmask & update[batch_id]
            supdate = smask & update[batch_id]
            # update metric history
            best_metric[update] = metric[update]

            # 1. set generated part
            gen_X[cupdate] = pred_X[cupdate]
            if not self.struct_only:
                gen_S[supdate] = pred_S[supdate]
        
            interface_X = r_interface_X[-1]
            # 2. align by cdr
            for i in range(batch_size):
                if not update[i]:
                    continue
                # 1. align CDRH3
                is_cur_graph = batch_id == i
                cdrh3_cur_graph = torch.logical_and(is_cur_graph, paratope_mask)
                ori_cdr = gen_X[cdrh3_cur_graph][:, :4]  # backbone
                pred_cdr = interface_X[interface_batch_id == i][:, :4]
                _, R, t = kabsch_torch(ori_cdr.reshape(-1, 3), pred_cdr.reshape(-1, 3))

                # 2. tranform antibody
                is_cur_ab = is_cur_graph & is_ab
                ab_X = torch.matmul(gen_X[is_cur_ab], R.T) + t
                gen_X[is_cur_ab] = ab_X

        self._clean_batch_constants()

        if return_hidden:
            return gen_X, gen_S, best_metric, H
        return gen_X, gen_S, best_metric

    def sample_many(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths,
                    n_samples=5, n_steps=20, return_hidden=False, show_progress=False):
        """
        Generate multiple samples in a single call
        
        Args:
            X, S, cmask, smask, paratope_mask, residue_pos, template, lengths: 
                Same parameters as in sample() method
            n_samples: Number of samples to generate
            n_steps: Number of flow steps for each sample
            return_hidden: Whether to return hidden states
            
        Returns:
            list_gen_X: List of n_samples generated coordinates
            list_gen_S: List of n_samples generated sequences
            list_metrics: List of n_samples metrics
            list_H: (Optional) List of n_samples hidden states if return_hidden=True
        """
        list_gen_X = []
        list_gen_S = []
        list_metrics = []
        list_H = [] if return_hidden else None
        
        # Generate multiple samples with different random noise
        for i in range(n_samples):
            # Generate different noise for each sample
            if self.backbone_only:
                init_noise = torch.randn(paratope_mask.sum(), 4, 3, device=X.device)
            else:
                init_noise = torch.randn(paratope_mask.sum(), X.shape[1], 3, device=X.device)
                
            # Generate a sample
            if return_hidden:
                gen_X, gen_S, metric, H = self.sample(
                    X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos,
                    template, lengths, n_steps=n_steps, init_noise=init_noise, return_hidden=True,
                    show_progress=show_progress, progress_desc=f'Sample {i + 1}/{n_samples} ODE'
                )
                list_H.append(H)
            else:
                gen_X, gen_S, metric = self.sample(
                    X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos,
                    template, lengths, n_steps=n_steps, init_noise=init_noise,
                    show_progress=show_progress, progress_desc=f'Sample {i + 1}/{n_samples} ODE'
                )
            
            # Store results
            list_gen_X.append(gen_X)
            list_gen_S.append(gen_S)
            list_metrics.append(metric)
            
        if return_hidden:
            return list_gen_X, list_gen_S, list_metrics, list_H
        else:
            return list_gen_X, list_gen_S, list_metrics

isMEANModel = AbFlowModel
dyMEANModel = AbFlowModel