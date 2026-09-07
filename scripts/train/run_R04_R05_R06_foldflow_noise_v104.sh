#!/usr/bin/env bash
# R05MF_ORTHOGONAL_R05_ABLATION_V172: three orthogonal one-factor runs, each directly parented by R05.
# R05MF_CAUSAL_MODULES_V173: R20/R21/R22 are independent direct-R05 modules.
set -euo pipefail

MODE=${1:-}
EXP_ID=${2:-}
GPU_ID=${3:-0}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${ABFLOW_PROJECT_ROOT:-$ROOT}
SELF_LAUNCHER=$(python - "${BASH_SOURCE[0]}" <<'PYSELF'
import os, sys
print(os.path.realpath(sys.argv[1]))
PYSELF
)

# ============================================================
# AbFlow v104: standardize the stochastic R3 corruption on the R01 base
# ============================================================
# Scientific parent: R01 = AbX common frame + PCS-RC + U02 canonical
# Score--Flow, whose structural Pareto is currently the strongest.
#
# R04: R01, but replace endpoint-dependent adaptive g with FoldFlow's fixed
#      g_scaled=0.1 (g_raw=1.0 A at coordinate_scaling=0.1). Global H3 R3
#      support, U02 hybrid carrier and matched sampler remain unchanged.
# R05: R04, but change only stochastic support global-H3 -> independent
#      per-residue R3 translations (broadcast within each residue), matching
#      the Euclidean translation granularity used by FoldFlow.
# R06: R04, but change only U02's hard endpoint/canonical seam to the already
#      implemented U03 boundary-regular carrier and its matched sampler.
# R07: R05 + R06 factor combination: residue-level R3 support + U03 carrier.
#      Relative to R05, only the carrier changes; no model code is changed.
# R08: R05 + v164 closed no-MSA single/pair representation core.
#      AbX OPM supplies s->z; MF pair-first ordering supplies refined z->s;
#      final z directly conditions the sole R05 EGNN node+coordinate mechanics.
# R09: R08 + MF/AF3-inspired pair-conditioned atom representation refinement only.
# R10: R09 + design-region factored smooth-lDDT only (weight 0.1); historical evidence run.
# R05MF_DIAGNOSTIC_V167: diagnostics-only overlay; formal Train->Val->Test unchanged.
# R05MF_PARENT_AUTHORITY_V168: implements the parent-anchored non-dominant MF->R05
#      residual operator and Test-trajectory authority diagnostics.
# R05MF_AUTHORITY_LADDER_V169: new three-run causal ladder after stopping R08-R10:
# R05MF_SEQUENCE_PATH_AUTHORITY_V171: retires parent-RMS clipping as the root-cause
# hypothesis and introduces an exact categorical sequence bridge with path-noisy CE.
# The formal v171 branch preserves Train->Validation->Test exactly.
# R05MF_LIVEPAIR_DISTOGRAM_V170: retires the invalid v169 R13 whose distogram
#      consumed detached pair_carry. The official replacement R13 separates
#      pair_carry=stopgrad(z_r) from pair_live=z_R, keeps no cross-round BPTT,
#      RNG-isolates the auxiliary head, and requires a first-batch shared-gradient PASS.
# R11: R08 closed core + parent-authority closure only (pair-atom OFF).
# R12: R11 + pair-conditioned atom representation only.
# R13: R12 + MFDesign persistent-pair distogram only (weight 0.03), now on live final z_R.
#      R11-R13 are retained as historical authority experiments. R14-R16 re-branch
#      from R08 and replace the legacy external native-sequence curriculum with the
#      exact categorical bridge while preserving R05 physics/sampler/optimizer.
#      smooth-lDDT is deliberately OFF in the new ladder because its DF/DA terms
#      were empirically saturated and its measured gradient authority was weak.
#
# Physical path min_sigma is intentionally 0 for all three formal runs.  This
# keeps PCS-RC/native as exact endpoints and preserves the exact canonical
# probability-flow algebra.  ABFLOW_R3_SCORE_MIN_SIGMA=0.01 is numerical only.
# ============================================================

# ---------- common AbFlow / task settings ----------
export ABFLOW_R05_ENDPOINT_RELATION="off"
export ABFLOW_R05_GEOM_PAIR="off"
export ABFLOW_SEQUENCE_PATH_CONTRACT="legacy_context"
export ABFLOW_SEQUENCE_LOSS_SCOPE="context_mask"
export ABFLOW_SOURCE_MODE="pcs_rc"
export ABFLOW_RECURRENT_PROPOSAL_CONTEXT="on"
export ABFLOW_COORD_PEP_SOURCE_WEIGHT="1.0"
export ABFLOW_SEQ_PEP_SOURCE_WEIGHT="1.0"
export ABFLOW_COORD_PEP_AS_CONDITION="on"

export ABFLOW_SCOREFM_STATE_PATH="on"
export ABFLOW_SCOREFM_PER_SAMPLE_T="on"
export ABFLOW_SCOREFM_TIME_EMBED="on"
export ABFLOW_SCOREFM_T_SAMPLING="uniform"
export ABFLOW_SCOREFM_MIN_SIGMA="0.01"
export ABFLOW_SCOREFM_DSM_T_MIN="0.2"
export ABFLOW_SCOREFM_DSM_T_MAX="0.8"

# AbX common coordinate frame; whole physical antibody CA center.
export ABFLOW_ABX_COMMON_CENTER="on"

# FoldFlow numerical convention.  The F01 path is constructed in raw A, so
# model code converts g_scaled=0.1 -> g_raw=1.0 A exactly once.
export ABFLOW_R3_FLOW_COORDINATE_SCALING="0.1"
export ABFLOW_R3_G_MODE="foldflow_fixed_scaled"
export ABFLOW_R3_FIXED_G_SCALED="0.1"
export ABFLOW_R3_PATH_MIN_SIGMA="0.0"
export ABFLOW_R3_SCORE_MIN_SIGMA="0.01"

# Historical adaptive-g knobs remain defined only for compatibility/diagnostics;
# they are NOT consumed when ABFLOW_R3_G_MODE=foldflow_fixed_scaled.
export ABFLOW_R3_TRANSPORT_FRACTION="0.05"
export ABFLOW_R3_TRANSPORT_MAX="20.0"
export ABFLOW_F01_HYBRID_T_MIN="0.20"

# Historical runs default to the proven R05 sequence process.  The formal v171
# R14-R16 cases below override only the training-state/objective contract: the
# existing R05 categorical sampler remains unchanged, while the extra native-context
# curriculum is removed and q_t itself becomes the sole sequence-state authority.
export ABFLOW_SEQ_INPUT_MODE="pep_condition"
export ABFLOW_SHADOW_SEQ_STATE="off"
export ABFLOW_DUAL_SEQUENCE_STATE="off"
export ABFLOW_DUAL_SEQUENCE_ATOM_MODE="hidden_only"
export ABFLOW_SEQUENCE_CONTEXT_MODE="legacy"
export ABFLOW_SEQUENCE_RECYCLE_MODE="off"
export ABFLOW_SEQUENCE_SOURCE_MODE="proposal"
export ABFLOW_SEQUENCE_GENERATIVE_MODE="legacy"
export ABFLOW_RECURRENT_PROPOSAL_SEQUENCE_CONTEXT="on"
export ABFLOW_FINAL_READOUT_MODE="integrated_endpoint"
export ABFLOW_SEQUENCE_DECODE_MODE="argmax"
export ABFLOW_DETERMINISTIC_VALIDATION="on"
export ABFLOW_SEQ_CE_WEIGHT="1.0"
export ABFLOW_PROPOSAL_ADAPTER_START_ROUND="1"

# Freeze rejected / orthogonal branches.
export ABFLOW_PAIR_TIME_SCOPE="off"
export ABFLOW_SCOREFLOW_PAIR_MODE="off"
export ABFLOW_SCOREFLOW_PAIR_STOP_GRAD="on"
export ABFLOW_R3_SCORE_DSM_WEIGHT="0.0"
export ABFLOW_R3_PATHFLOW_WEIGHT="0.0"
export ABFLOW_SF2M_SCORE_WEIGHT="0.0"
export ABFLOW_SATC_APPLY_PROB="0.0"
export ABFLOW_SATC_SCORE_WEIGHT="0.0"
export ABFLOW_SATC_VELOCITY_WEIGHT="0.0"
export ABFLOW_SATC_INTERFACE_WEIGHT_ALPHA="0.0"
export ABFLOW_SUPPORT_FACTORIZED_COORD="off"
export ABFLOW_TRANSLATION_ROUND_CREDIT="off"
export ABFLOW_ROUND_CONSISTENT_COORD_SUPERVISION="off"
export ABFLOW_COORDINATE_AUTHORITY="legacy_dual"
export ABFLOW_STRUCTURE_SEQ_READOUT="off"

# R05 x MFDesign representation defaults.  Formal R08-R16 JSON configs
# override only this whitelisted representation/loss namespace; all R05 physics
# above remains launcher-owned and cannot be changed by the experiment metadata.
export ABFLOW_MF_REPR_CORE="off"
export ABFLOW_MF_DISTOGRAM="off"
export ABFLOW_MF_SMOOTH_LDDT="off"
export ABFLOW_MF_SMOOTH_LDDT_CUTOFF="15.0"
export ABFLOW_MF_SMOOTH_LDDT_INTRA_WEIGHT="1.0"
export ABFLOW_MF_SMOOTH_LDDT_SCAFFOLD_WEIGHT="1.0"
export ABFLOW_MF_SMOOTH_LDDT_ANTIGEN_WEIGHT="1.0"
export ABFLOW_MF_SINGLE_DIM="128"
export ABFLOW_MF_PAIR_DIM="64"
export ABFLOW_MF_REPR_BLOCKS="1"
export ABFLOW_MF_REPR_HEADS="4"
export ABFLOW_MF_ATOM_S="128"
export ABFLOW_MF_ATOM_DEPTH="3"
export ABFLOW_MF_ATOM_HEADS="4"
export ABFLOW_MF_PAIR_ANTIGEN_K="18"
export ABFLOW_MF_LOCAL_ANTIGEN_K="18"
export ABFLOW_MF_COORD_SCALE="0.1"
export ABFLOW_MF_RELPOS_DIM="32"
export ABFLOW_MF_OPM_DIM="64"
export ABFLOW_MF_ALLATOM_PAIR="on"
export ABFLOW_MF_ALLATOM_CHUNK="512"
export ABFLOW_MF_DETACH_STATE_CARRY="on"
# Terminal auxiliary state is OFF by default. Only replacement R13 enables the
# live final z_R path; recurrent s/z carry remains detached in every formal run.
export ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX="off"
# Private auxiliary-head initialization stream. fork_rng in the model guarantees
# that enabling distogram does not advance the generator/global training RNG.
export ABFLOW_MF_DISTOGRAM_HEAD_SEED="13003"
export ABFLOW_MF_TRIANGLE_HEADS="4"
export ABFLOW_MF_TRIANGLE_HIDDEN="128"
export ABFLOW_MF_TRIANGLE_CHECKPOINT="off"
export ABFLOW_MF_PAIR_ATOM_REFINER="off"
export ABFLOW_MF_PAIR_ATOM_DEPTH="1"
export ABFLOW_MF_PAIR_ATOM_HEADS="4"
export ABFLOW_MF_PAIR_ATOM_QUERY_CHUNK="64"
export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="off"
export ABFLOW_DDP_STATIC_GRAPH="off"
export ABFLOW_DDP_COST_BALANCED="off"
export ABFLOW_MF_DISTOGRAM_BINS="64"
export ABFLOW_MF_DISTOGRAM_MIN="2.0"
export ABFLOW_MF_DISTOGRAM_MAX="22.0"
export ABFLOW_LOSS_SEQUENCE_WEIGHT="1.0"
export ABFLOW_LOSS_STRUCTURE_WEIGHT="1.0"
export ABFLOW_LOSS_INTERFACE_WEIGHT="1.0"
export ABFLOW_LOSS_EDGE_WEIGHT="1.0"
export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.0"
export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.0"
# Observational-only forensic for the repeated epoch-0 1e5-1e6 structure summary.
export ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD="10000"

# Unused R03 direct-Cartesian controls kept at benign defaults because the v103
# model parses them at init; none of R04/R05/R06 selects the R03 loss mode.
export ABFLOW_R03_G_SCALED="0.1"
export ABFLOW_R03_PATH_MIN_SIGMA_SCALED="0.0"
export ABFLOW_R03_CENTER_RESIDUAL="on"

case "$EXP_ID" in
  R04_PCS_RC_LC_R1_ABX_FF_FIXEDG_GLOBAL_U02_SCOREFLOW)
    export ABFLOW_ABLATION_PARENT="R01_PCS_RC_LC_R1_ABX_U02_GLOBAL_R3_SCOREFLOW"
    export ABFLOW_EXPERIMENT_FACTOR="adaptive_g_to_foldflow_fixed_g_only"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R04_FF_FIXEDG_GLOBAL_U02"
    export ABFLOW_MODULE_PARENT="R01_ABX_U02"

    export ABFLOW_R3_NOISE_SCOPE="global"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"
    ;;

  R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW)
    export ABFLOW_ABLATION_PARENT="R04_PCS_RC_LC_R1_ABX_FF_FIXEDG_GLOBAL_U02_SCOREFLOW"
    export ABFLOW_EXPERIMENT_FACTOR="global_to_residue_r3_noise_support_only"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R05_FF_FIXEDG_RESIDUE_U02"
    export ABFLOW_MODULE_PARENT="R04_FF_FIXEDG_GLOBAL_U02"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"
    ;;

  R06_PCS_RC_LC_R1_ABX_FF_FIXEDG_GLOBAL_U03_SCOREFLOW)
    export ABFLOW_ABLATION_PARENT="R04_PCS_RC_LC_R1_ABX_FF_FIXEDG_GLOBAL_U02_SCOREFLOW"
    export ABFLOW_EXPERIMENT_FACTOR="u02_hard_seam_to_u03_boundary_regular_carrier_only"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R06_FF_FIXEDG_GLOBAL_U03"
    export ABFLOW_MODULE_PARENT="R04_FF_FIXEDG_GLOBAL_U02"

    export ABFLOW_R3_NOISE_SCOPE="global"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_boundary_regular_carrier"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_boundary_regular_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"
    ;;


  R07_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U03_SCOREFLOW)
    # R07 = R05 + the single carrier change already validated in R06.
    # Relative to R05 this remains a one-factor ablation:
    #   residue support stays fixed; U02 -> U03 only.
    export ABFLOW_ABLATION_PARENT="R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW"
    export ABFLOW_EXPERIMENT_FACTOR="u02_hard_seam_to_u03_boundary_regular_carrier_only_on_residue_support"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R07_FF_FIXEDG_RESIDUE_U03"
    export ABFLOW_MODULE_PARENT="R05_FF_FIXEDG_RESIDUE_U02"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_boundary_regular_carrier"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_boundary_regular_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"
    ;;

  R08_R05_MF_CLOSED_CORE_U02)
    export ABFLOW_ABLATION_PARENT="R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW"
    export ABFLOW_EXPERIMENT_FACTOR="closed_no_msa_single_pair_core_only"
    export ABFLOW_SINGLE_FACTOR_ABLATION="false"
    export ABFLOW_MODULE_ID="R08_R05_MF_CLOSED_CORE"
    export ABFLOW_MODULE_PARENT="R05_FF_FIXEDG_RESIDUE_U02"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_DDP_COST_BALANCED="on"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    export ABFLOW_MF_TRIANGLE_CHECKPOINT="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="off"
    export ABFLOW_MF_SMOOTH_LDDT="off"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.0"
    export ABFLOW_MF_DISTOGRAM="off"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.0"
    ;;

  R09_R05_MF_CLOSED_CORE_PAIRATOM_U02)
    export ABFLOW_ABLATION_PARENT="R08_R05_MF_CLOSED_CORE_U02"
    export ABFLOW_EXPERIMENT_FACTOR="add_pair_conditioned_atom_representation_only"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R09_R05_MF_CLOSED_CORE_PAIRATOM"
    export ABFLOW_MODULE_PARENT="R08_R05_MF_CLOSED_CORE"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_DDP_COST_BALANCED="on"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    export ABFLOW_MF_TRIANGLE_CHECKPOINT="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="on"
    ;;

  R10_R05_MF_CLOSED_CORE_PAIRATOM_DESIGNLDDT_U02)
    export ABFLOW_ABLATION_PARENT="R09_R05_MF_CLOSED_CORE_PAIRATOM_U02"
    export ABFLOW_EXPERIMENT_FACTOR="add_design_region_factored_smooth_lddt_on_top_of_pair_atom"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R10_R05_MF_CLOSED_CORE_PAIRATOM_DESIGNLDDT"
    export ABFLOW_MODULE_PARENT="R09_R05_MF_CLOSED_CORE_PAIRATOM"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_DDP_COST_BALANCED="on"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    export ABFLOW_MF_TRIANGLE_CHECKPOINT="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="on"
    export ABFLOW_MF_DISTOGRAM="off"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.0"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.1"
    export ABFLOW_MF_SMOOTH_LDDT="on"
    ;;

  R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED_U02)
    # R11 = R08 + one factor: close the shared MF->R05 residual authority boundary.
    # Pair-atom and all auxiliary geometry losses stay OFF to isolate the P0 mechanism.
    export ABFLOW_ABLATION_PARENT="R08_R05_MF_CLOSED_CORE_U02"
    export ABFLOW_EXPERIMENT_FACTOR="parent_anchored_nondominant_residual_authority_on_closed_core_only"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED"
    export ABFLOW_MODULE_PARENT="R08_R05_MF_CLOSED_CORE"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_DDP_COST_BALANCED="on"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    export ABFLOW_MF_TRIANGLE_CHECKPOINT="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="off"
    export ABFLOW_MF_DISTOGRAM="off"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.0"
    export ABFLOW_MF_SMOOTH_LDDT="off"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.0"
    export ABFLOW_MF_PARENT_AUTHORITY="on"
    export ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS="on"
    ;;

  R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_U02)
    # R12 = R11 + one donor-validated semantic: MF-style z->atom-pair representation.
    export ABFLOW_ABLATION_PARENT="R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED_U02"
    export ABFLOW_EXPERIMENT_FACTOR="add_pair_conditioned_atom_representation_under_fixed_parent_authority"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM"
    export ABFLOW_MODULE_PARENT="R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_DDP_COST_BALANCED="on"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    export ABFLOW_MF_TRIANGLE_CHECKPOINT="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="on"
    export ABFLOW_MF_DISTOGRAM="off"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.0"
    export ABFLOW_MF_SMOOTH_LDDT="off"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.0"
    export ABFLOW_MF_PARENT_AUTHORITY="on"
    export ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS="on"
    ;;

  R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_DISTOGRAM_U02)
    echo "ERROR: this v169 R13 ID is retired: its distogram consumed detached pair_carry and did not supervise the generator." >&2
    echo "Use: R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02" >&2
    exit 2
    ;;

  R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02)
    # R13 = R12 + one MFDesign-native objective: persistent-pair distogram CE.
    # 0.03 is the uploaded MFDesign Stage-2/3/4 prior; actual AbFlow authority is audited.
    export ABFLOW_ABLATION_PARENT="R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_U02"
    export ABFLOW_EXPERIMENT_FACTOR="add_live_final_pair_mfdesign_distogram_003_only"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM"
    export ABFLOW_MODULE_PARENT="R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_DDP_COST_BALANCED="on"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    export ABFLOW_MF_TRIANGLE_CHECKPOINT="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="on"
    export ABFLOW_MF_DISTOGRAM="on"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.03"
    export ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX="on"
    export ABFLOW_MF_DISTOGRAM_HEAD_SEED="13003"
    export ABFLOW_MF_SMOOTH_LDDT="off"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.0"
    export ABFLOW_MF_PARENT_AUTHORITY="on"
    export ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS="on"
    ;;

  R14_R05_MF_CLOSED_CORE_EXACTSEQ_U02)
    # R14 re-branches from R08 and changes one coherent scientific contract:
    # sequence state + sequence objective are both governed by the same exact
    # categorical bridge.  The external native-context curriculum is removed,
    # and CE authority is restricted to residues that actually took the
    # source/noisy branch. Parent-RMS authority is intentionally OFF.
    export ABFLOW_ABLATION_PARENT="R08_R05_MF_CLOSED_CORE_U02"
    export ABFLOW_EXPERIMENT_FACTOR="replace_external_native_context_curriculum_with_exact_bridge_path_noisy_denoising_contract"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R14_R05_MF_CLOSED_CORE_EXACTSEQ"
    export ABFLOW_MODULE_PARENT="R08_R05_MF_CLOSED_CORE"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_DDP_COST_BALANCED="on"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    export ABFLOW_MF_TRIANGLE_CHECKPOINT="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="off"
    export ABFLOW_MF_DISTOGRAM="off"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.0"
    export ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX="off"
    export ABFLOW_MF_SMOOTH_LDDT="off"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.0"
    export ABFLOW_MF_PARENT_AUTHORITY="off"
    export ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS="on"
    export ABFLOW_SEQUENCE_CONTEXT_MODE="off"
    export ABFLOW_SEQUENCE_PATH_CONTRACT="exact_categorical_bridge"
    export ABFLOW_SEQUENCE_LOSS_SCOPE="path_noisy"
    ;;

  R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_U02)
    # R15 = R14 + the already-supported pair-conditioned atom representation.
    export ABFLOW_ABLATION_PARENT="R14_R05_MF_CLOSED_CORE_EXACTSEQ_U02"
    export ABFLOW_EXPERIMENT_FACTOR="add_pair_conditioned_atom_representation_under_exact_sequence_contract"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM"
    export ABFLOW_MODULE_PARENT="R14_R05_MF_CLOSED_CORE_EXACTSEQ"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_DDP_COST_BALANCED="on"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    export ABFLOW_MF_TRIANGLE_CHECKPOINT="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="on"
    export ABFLOW_MF_DISTOGRAM="off"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.0"
    export ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX="off"
    export ABFLOW_MF_SMOOTH_LDDT="off"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.0"
    export ABFLOW_MF_PARENT_AUTHORITY="off"
    export ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS="on"
    export ABFLOW_SEQUENCE_CONTEXT_MODE="off"
    export ABFLOW_SEQUENCE_PATH_CONTRACT="exact_categorical_bridge"
    export ABFLOW_SEQUENCE_LOSS_SCOPE="path_noisy"
    ;;

  R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM_U02)
    # R16 = R15 + live terminal-pair distogram CE, weight 0.03 donor prior.
    export ABFLOW_ABLATION_PARENT="R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_U02"
    export ABFLOW_EXPERIMENT_FACTOR="add_live_final_pair_distogram_003_under_exact_sequence_contract"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM"
    export ABFLOW_MODULE_PARENT="R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_DDP_COST_BALANCED="on"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    export ABFLOW_MF_TRIANGLE_CHECKPOINT="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="on"
    export ABFLOW_MF_DISTOGRAM="on"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.03"
    export ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX="on"
    export ABFLOW_MF_DISTOGRAM_HEAD_SEED="13003"
    export ABFLOW_MF_SMOOTH_LDDT="off"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.0"
    export ABFLOW_MF_PARENT_AUTHORITY="off"
    export ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS="on"
    export ABFLOW_SEQUENCE_CONTEXT_MODE="off"
    export ABFLOW_SEQUENCE_PATH_CONTRACT="exact_categorical_bridge"
    export ABFLOW_SEQUENCE_LOSS_SCOPE="path_noisy"
    ;;


  R17_R05_EXACTSEQ_U02)
    # V172-A: direct R05 parent + one coherent sequence-path authority module.
    # No MF representation, PairAtom, distogram, smooth-lDDT or parent clipping.
    export ABFLOW_ABLATION_PARENT="R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW"
    export ABFLOW_EXPERIMENT_FACTOR="exact_categorical_sequence_path_and_path_noisy_ce_only"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R17_R05_EXACTSEQ"
    export ABFLOW_MODULE_PARENT="R05_FF_FIXEDG_RESIDUE_U02"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_MF_REPR_CORE="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="off"
    export ABFLOW_MF_DISTOGRAM="off"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.0"
    export ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX="off"
    export ABFLOW_MF_SMOOTH_LDDT="off"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.0"
    export ABFLOW_MF_PARENT_AUTHORITY="off"
    export ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS="off"

    export ABFLOW_SEQUENCE_CONTEXT_MODE="off"
    export ABFLOW_SEQUENCE_PATH_CONTRACT="exact_categorical_bridge"
    export ABFLOW_SEQUENCE_LOSS_SCOPE="path_noisy"

    export ABFLOW_DDP_COST_BALANCED="off"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="off"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    ;;

  R18_R05_MF_CLOSED_CORE_U02)
    # V172-B: direct R05 parent + closed no-MSA relational representation only.
    # Sequence path remains the historical R05 contract by design so this run
    # answers only whether the representation module itself helps/hurts R05.
    export ABFLOW_ABLATION_PARENT="R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW"
    export ABFLOW_EXPERIMENT_FACTOR="closed_no_msa_single_pair_representation_only_direct_r05"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R18_R05_MF_CLOSED_CORE"
    export ABFLOW_MODULE_PARENT="R05_FF_FIXEDG_RESIDUE_U02"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_MF_REPR_CORE="on"
    export ABFLOW_MF_PAIR_ATOM_REFINER="off"
    export ABFLOW_MF_DISTOGRAM="off"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.0"
    export ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX="off"
    export ABFLOW_MF_SMOOTH_LDDT="off"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.0"
    export ABFLOW_MF_PARENT_AUTHORITY="off"
    export ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS="on"

    export ABFLOW_SEQUENCE_CONTEXT_MODE="legacy"
    export ABFLOW_SEQUENCE_PATH_CONTRACT="legacy_context"
    export ABFLOW_SEQUENCE_LOSS_SCOPE="context_mask"

    export ABFLOW_DDP_COST_BALANCED="on"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    export ABFLOW_MF_TRIANGLE_CHECKPOINT="off"
    ;;

  R19_R05_DESIGNLDDT_U02)
    # V172-C: direct R05 parent + design-region factored smooth-lDDT only.
    # V172 decouples this coordinate objective from MF_REPR_CORE so the loss can
    # finally be tested as a genuine one-factor R05 ablation.
    export ABFLOW_ABLATION_PARENT="R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW"
    export ABFLOW_EXPERIMENT_FACTOR="design_region_factored_smooth_lddt_01_only_direct_r05"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="R19_R05_DESIGNLDDT"
    export ABFLOW_MODULE_PARENT="R05_FF_FIXEDG_RESIDUE_U02"

    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"

    export ABFLOW_MF_REPR_CORE="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="off"
    export ABFLOW_MF_DISTOGRAM="off"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.0"
    export ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX="off"
    export ABFLOW_MF_SMOOTH_LDDT="on"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.1"
    export ABFLOW_MF_PARENT_AUTHORITY="off"
    export ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS="off"

    export ABFLOW_SEQUENCE_CONTEXT_MODE="legacy"
    export ABFLOW_SEQUENCE_PATH_CONTRACT="legacy_context"
    export ABFLOW_SEQUENCE_LOSS_SCOPE="context_mask"

    export ABFLOW_DDP_COST_BALANCED="off"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="off"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    ;;

  R23_R05_ABX_PAIR_TIME_U02|R24_R05_ENDPOINT_RELATION_U02|R25_R05_GEOMETRY_PAIR_U02)
    export ABFLOW_ABLATION_PARENT="R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW"
    export ABFLOW_EXPERIMENT_FACTOR="$EXP_ID"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="$EXP_ID"
    export ABFLOW_MODULE_PARENT="R05_FF_FIXEDG_RESIDUE_U02"
    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"
    export ABFLOW_SEQUENCE_CONTEXT_MODE="legacy"
    export ABFLOW_SEQUENCE_PATH_CONTRACT="legacy_context"
    export ABFLOW_SEQUENCE_LOSS_SCOPE="context_mask"
    export ABFLOW_PAIR_TIME_SCOPE="off"
    export ABFLOW_R05_ENDPOINT_RELATION="off"
    export ABFLOW_R05_GEOM_PAIR="off"
    export ABFLOW_MF_REPR_CORE="off"
    export ABFLOW_MF_PAIR_ATOM_REFINER="off"
    export ABFLOW_MF_DISTOGRAM="off"
    export ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX="off"
    export ABFLOW_MF_SMOOTH_LDDT="off"
    export ABFLOW_LOSS_DISTOGRAM_WEIGHT="0.0"
    export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.0"
    export ABFLOW_MF_PARENT_AUTHORITY="off"
    export ABFLOW_GRAD_CONFLICT_DIAGNOSTICS="on"
    export ABFLOW_DDP_COST_BALANCED="off"
    export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="off"
    export ABFLOW_DDP_STATIC_GRAPH="off"
    case "$EXP_ID" in
      R23_R05_ABX_PAIR_TIME_U02)
        export ABFLOW_PAIR_TIME_SCOPE="residue" ;;
      R24_R05_ENDPOINT_RELATION_U02)
        export ABFLOW_R05_ENDPOINT_RELATION="on"
        export ABFLOW_MF_SMOOTH_LDDT="on"
        export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.1" ;;
      R25_R05_GEOMETRY_PAIR_U02)
        export ABFLOW_R05_GEOM_PAIR="on"
        export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on" ;;
    esac
    ;;

  R20_R05_FULLDESIGN_SEQ_U02|R21_R05_ENDPOINT_RELATION_U02|R22_R05_GEOMETRY_PAIR_U02)
    export ABFLOW_ABLATION_PARENT="R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW"
    export ABFLOW_EXPERIMENT_FACTOR="$EXP_ID"
    export ABFLOW_SINGLE_FACTOR_ABLATION="true"
    export ABFLOW_MODULE_ID="$EXP_ID"
    export ABFLOW_MODULE_PARENT="R05_FF_FIXEDG_RESIDUE_U02"
    export ABFLOW_R3_NOISE_SCOPE="residue"
    export ABFLOW_SCOREFM_LOSS_MODE="f01_r3_endpoint_canonical_hybrid"
    export ABFLOW_SCOREFM_SAMPLER_MODE="f01_canonical_carrier"
    export ABFLOW_FLOW_T_MIN="0.0"
    export ABFLOW_FLOW_T_MAX="1.0"
    export ABFLOW_GRAD_CONFLICT_DIAGNOSTICS="on"
    export ABFLOW_DDP_COST_BALANCED="off"
    # Equal training sample order to the R05 parent, not R18 cost balancing.
    case "$EXP_ID" in
      R20_R05_FULLDESIGN_SEQ_U02)
        export ABFLOW_SEQUENCE_CONTEXT_MODE="off"
        export ABFLOW_SEQUENCE_LOSS_SCOPE="context_mask" ;;
      R21_R05_ENDPOINT_RELATION_U02)
        export ABFLOW_R05_ENDPOINT_RELATION="on"
        export ABFLOW_MF_SMOOTH_LDDT="on"
        export ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT="0.1" ;;
      R22_R05_GEOMETRY_PAIR_U02)
        export ABFLOW_R05_GEOM_PAIR="on"
        export ABFLOW_DDP_FIND_UNUSED_PARAMETERS="on" ;;
    esac
    ;;

  *)
    echo "Unknown EXP_ID: $EXP_ID"
    echo "Supported v104:"
    echo "  R04_PCS_RC_LC_R1_ABX_FF_FIXEDG_GLOBAL_U02_SCOREFLOW"
    echo "  R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW"
    echo "  R06_PCS_RC_LC_R1_ABX_FF_FIXEDG_GLOBAL_U03_SCOREFLOW"
    echo "  R07_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U03_SCOREFLOW"
    echo "  R08_R05_MF_CLOSED_CORE_U02"
    echo "  R09_R05_MF_CLOSED_CORE_PAIRATOM_U02"
    echo "  R10_R05_MF_CLOSED_CORE_PAIRATOM_DESIGNLDDT_U02"
    echo "  R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED_U02"
    echo "  R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_U02"
    echo "  R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02"
    echo "  R14_R05_MF_CLOSED_CORE_EXACTSEQ_U02"
    echo "  R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_U02"
    echo "  R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM_U02"
    echo "  R17_R05_EXACTSEQ_U02"
    echo "  R18_R05_MF_CLOSED_CORE_U02"
    echo "  R19_R05_DESIGNLDDT_U02"
    echo "  R20_R05_FULLDESIGN_SEQ_U02 / R21_R05_ENDPOINT_RELATION_U02 / R22_R05_GEOMETRY_PAIR_U02"
    echo "  R23_R05_ABX_PAIR_TIME_U02 / R24_R05_ENDPOINT_RELATION_U02 / R25_R05_GEOMETRY_PAIR_U02"
    exit 2
    ;;
esac

# Diagnostics are observational only; they do not change gradients.  Formal
# R08-R16 diagnostics are printed to stdout and therefore captured by the single
# canonical version_0/run_time.log.  We intentionally do not maintain a second
# fragmented metrics/latest/alerts file tree.
export ABFLOW_CONDITION_DIAGNOSTICS="${ABFLOW_CONDITION_DIAGNOSTICS:-on}"
export ABFLOW_GRAD_DIAGNOSTIC_INTERVAL="${ABFLOW_GRAD_DIAGNOSTIC_INTERVAL:-0}"
case "$EXP_ID" in
  R08_R05_MF_CLOSED_CORE_U02|R09_R05_MF_CLOSED_CORE_PAIRATOM_U02|R10_R05_MF_CLOSED_CORE_PAIRATOM_DESIGNLDDT_U02|R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED_U02|R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_U02|R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02|R14_R05_MF_CLOSED_CORE_EXACTSEQ_U02|R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_U02|R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM_U02|R17_R05_EXACTSEQ_U02|R18_R05_MF_CLOSED_CORE_U02|R19_R05_DESIGNLDDT_U02|R20_R05_FULLDESIGN_SEQ_U02|R21_R05_ENDPOINT_RELATION_U02|R22_R05_GEOMETRY_PAIR_U02|R23_R05_ABX_PAIR_TIME_U02|R24_R05_ENDPOINT_RELATION_U02|R25_R05_GEOMETRY_PAIR_U02)
    # One gradient-authority probe per epoch is worth the tiny overhead because
    # these runs are explicitly testing a new representation state.
    export ABFLOW_GRAD_CONFLICT_DIAGNOSTICS="${ABFLOW_GRAD_CONFLICT_DIAGNOSTICS:-on}"
    ;;
  *)
    export ABFLOW_GRAD_CONFLICT_DIAGNOSTICS="${ABFLOW_GRAD_CONFLICT_DIAGNOSTICS:-off}"
    ;;
esac

# Runtime parity.
export ABFLOW_AMP="${ABFLOW_AMP:-on}"
export ABFLOW_AMP_DTYPE="${ABFLOW_AMP_DTYPE:-bf16}"
export ABFLOW_ALLOW_TF32="${ABFLOW_ALLOW_TF32:-on}"
export ABFLOW_NUM_WORKERS="${ABFLOW_NUM_WORKERS:-8}"
export ABFLOW_PREFETCH_FACTOR="${ABFLOW_PREFETCH_FACTOR:-4}"
export ABFLOW_VALID_NUM_WORKERS="${ABFLOW_VALID_NUM_WORKERS:-2}"
export ABFLOW_VALID_PREFETCH_FACTOR="${ABFLOW_VALID_PREFETCH_FACTOR:-2}"
export ABFLOW_VALID_PERSISTENT_WORKERS="${ABFLOW_VALID_PERSISTENT_WORKERS:-off}"
# v3: validation uses the SAME DDP world as training.  The Trainer shards the
# historical logical validation batches exactly once across ranks and reduces
# them back to the original global mean, with no padding or duplicate samples.
export ABFLOW_DDP_VALIDATION="${ABFLOW_DDP_VALIDATION:-on}"
export ABFLOW_LOG_INTERVAL="${ABFLOW_LOG_INTERVAL:-20}"
export ABFLOW_TQDM_MININTERVAL="${ABFLOW_TQDM_MININTERVAL:-5.0}"
export ABFLOW_SAVE_INTERVAL="${ABFLOW_SAVE_INTERVAL:-1}"

# ------------------------------------------------------------------
# Formal epoch-wise Test phase (evaluation infrastructure only)
# ------------------------------------------------------------------
# Scientific R04/R05/R06 variables above are untouched.  Test reuses the same
# DDP world as training, applies EMA inside Trainer, runs model.sample + the
# original cal_metrics.py, and restores RNG afterwards.
export ABFLOW_PROJECT_ROOT="${ABFLOW_PROJECT_ROOT:-$PROJECT_ROOT}"
export ABFLOW_EPOCH_TEST="${ABFLOW_EPOCH_TEST:-on}"
export ABFLOW_EPOCH_TEST_INTERVAL="${ABFLOW_EPOCH_TEST_INTERVAL:-1}"
export ABFLOW_EPOCH_TEST_JSON="${ABFLOW_EPOCH_TEST_JSON:-${PROJECT_ROOT}/datasets/RAbD/test.json}"
export ABFLOW_EPOCH_TEST_BATCH_SIZE="${ABFLOW_EPOCH_TEST_BATCH_SIZE:-20}"
export ABFLOW_EPOCH_TEST_N_STEPS="${ABFLOW_EPOCH_TEST_N_STEPS:-10}"
export ABFLOW_EPOCH_TEST_BASE_SEED="${ABFLOW_EPOCH_TEST_BASE_SEED:-2023}"
export ABFLOW_EPOCH_TEST_METRIC_WORKERS="${ABFLOW_EPOCH_TEST_METRIC_WORKERS:-8}"
export ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS="${ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS:-off}"
export ABFLOW_EPOCH_TEST_KEEP_STRUCTURES="${ABFLOW_EPOCH_TEST_KEEP_STRUCTURES:-off}"
export ABFLOW_EPOCH_TEST_FAIL_FAST="${ABFLOW_EPOCH_TEST_FAIL_FAST:-off}"

# The formal train path no longer uses the historical TopK watcher.  TopK
# checkpoint *retention* inside validation remains unchanged.
export ABFLOW_AUTO_TOPK_EVAL="off"

FORCE_SCRATCH=${ABFLOW_FORCE_SCRATCH:-off}
MAX_EPOCH=${ABFLOW_MAX_EPOCH:-}

_is_on() {
  local v="${1:-off}"
  case "${v,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

# ------------------------------------------------------------------
# R08-R10 v164 formal hardware/batch contract
# ------------------------------------------------------------------
# The JSON keeps the historical R05 GLOBAL batch_size=56.  train.py divides
# that global batch by the DDP world size, so a formal two-GPU run is exactly
# 28 complexes/rank.  We enforce two selected physical GPUs here to prevent an
# accidental 4-GPU launch from silently changing the per-rank batch semantics.
case "$EXP_ID" in
  R08_R05_MF_CLOSED_CORE_U02) _ABFLOW_EXPECTED_GPUS="2,3" ;;
  R09_R05_MF_CLOSED_CORE_PAIRATOM_U02) _ABFLOW_EXPECTED_GPUS="4,5" ;;
  R10_R05_MF_CLOSED_CORE_PAIRATOM_DESIGNLDDT_U02) _ABFLOW_EXPECTED_GPUS="6,7" ;;
  R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED_U02) _ABFLOW_EXPECTED_GPUS="2,3" ;;
  R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_U02) _ABFLOW_EXPECTED_GPUS="4,5" ;;
  R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02) _ABFLOW_EXPECTED_GPUS="6,7" ;;
  R14_R05_MF_CLOSED_CORE_EXACTSEQ_U02) _ABFLOW_EXPECTED_GPUS="2,3" ;;
  R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_U02) _ABFLOW_EXPECTED_GPUS="4,5" ;;
  R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM_U02) _ABFLOW_EXPECTED_GPUS="6,7" ;;
  R17_R05_EXACTSEQ_U02) _ABFLOW_EXPECTED_GPUS="2,3" ;;
  R18_R05_MF_CLOSED_CORE_U02) _ABFLOW_EXPECTED_GPUS="4,5" ;;
  R19_R05_DESIGNLDDT_U02) _ABFLOW_EXPECTED_GPUS="6,7" ;;
  R20_R05_FULLDESIGN_SEQ_U02|R21_R05_ENDPOINT_RELATION_U02|R22_R05_GEOMETRY_PAIR_U02) _ABFLOW_EXPECTED_GPUS="$GPU_ID" ;;
  R23_R05_ABX_PAIR_TIME_U02) _ABFLOW_EXPECTED_GPUS="2,3" ;;
  R24_R05_ENDPOINT_RELATION_U02) _ABFLOW_EXPECTED_GPUS="4,5" ;;
  R25_R05_GEOMETRY_PAIR_U02) _ABFLOW_EXPECTED_GPUS="6,7" ;;
  *) _ABFLOW_EXPECTED_GPUS="" ;;
esac

if [[ -n "$_ABFLOW_EXPECTED_GPUS" ]]; then
  IFS=',' read -r -a _ABFLOW_GPU_LIST <<< "$GPU_ID"
  if [[ "${#_ABFLOW_GPU_LIST[@]}" -ne 2 ]]; then
    echo "ERROR: $EXP_ID formal protocol requires exactly two GPUs; got GPU_ID=$GPU_ID" >&2
    exit 2
  fi
  if [[ "${_ABFLOW_GPU_LIST[0]}" == "${_ABFLOW_GPU_LIST[1]}" ]]; then
    echo "ERROR: the two GPU ids must be distinct: GPU_ID=$GPU_ID" >&2
    exit 2
  fi
  if [[ "$GPU_ID" != "$_ABFLOW_EXPECTED_GPUS" ]]; then
    echo "ERROR: $EXP_ID formal GPU mapping is $_ABFLOW_EXPECTED_GPUS; got GPU_ID=$GPU_ID" >&2
    exit 2
  fi
  export ABFLOW_SELECTED_GPUS="$GPU_ID"
  export ABFLOW_SELECTED_GPU_COUNT="2"
  export ABFLOW_EFFECTIVE_GLOBAL_BATCH="56"
  export ABFLOW_LOCAL_BATCH_PER_GPU="28"
  echo "[R05MFHardware] EXP=$EXP_ID GPUs=$ABFLOW_SELECTED_GPUS world_size=2 global_batch=56 local_batch=28"
else
  export ABFLOW_SELECTED_GPUS="$GPU_ID"
  export ABFLOW_SELECTED_GPU_COUNT=""
  export ABFLOW_EFFECTIVE_GLOBAL_BATCH=""
  export ABFLOW_LOCAL_BATCH_PER_GPU=""
fi

# ------------------------------------------------------------------
# Integrated AutoTopK infrastructure
# ------------------------------------------------------------------
AUTO_TOPK_EVAL=${ABFLOW_AUTO_TOPK_EVAL:-off}
AUTO_TOPK_POLL_INTERVAL=${ABFLOW_TOPK_POLL_INTERVAL:-300}
AUTO_TOPK_MAX_NEW=${ABFLOW_TOPK_MAX_NEW:-1}
AUTO_TOPK_LATEST_ONLY=${ABFLOW_TOPK_LATEST_ONLY:-off}
AUTO_TOPK_MAX_EVAL_GPUS=${ABFLOW_AUTO_TOPK_MAX_EVAL_GPUS:-1}
AUTO_TOPK_TEST_JSON=${ABFLOW_TOPK_TEST_JSON:-${PROJECT_ROOT}/datasets/RAbD/test.json}
AUTO_TOPK_EVAL_SCRIPT=${ABFLOW_TOPK_EVAL_SCRIPT:-${PROJECT_ROOT}/scripts/test/evaluate_topk_map.py}
AUTO_TOPK_FORCE=${ABFLOW_TOPK_FORCE:-off}

_csv_contains() {
  local csv=",$1,"
  local item="$2"
  [[ "$csv" == *",${item},"* ]]
}

_infer_spare_eval_gpus() {
  # v103 same-GPU policy:
  # AutoTopK uses the SAME physical GPU list as training by default.
  # Explicit ABFLOW_EVAL_GPUS / ABFLOW_EVAL_GPU still overrides this.
  #
  # Examples:
  #   train GPU_ID=2,3 -> eval pool=2,3
  #   train GPU_ID=4,5 -> eval pool=4,5
  #   train GPU_ID=6,7 -> eval pool=6,7
  if [[ -n "${ABFLOW_EVAL_GPUS:-}" ]]; then
    echo "$ABFLOW_EVAL_GPUS"
    return 0
  fi
  if [[ -n "${ABFLOW_EVAL_GPU:-}" ]]; then
    echo "$ABFLOW_EVAL_GPU"
    return 0
  fi

  echo "$GPU_ID"
}

_start_auto_topk_watcher() {
  local eval_gpus="$1"
  local run_dir="$2"
  local log_file="$run_dir/auto_topk_eval.log"
  local pid_file="$run_dir/auto_topk_eval.pid"

  if ! _is_on "$AUTO_TOPK_EVAL"; then
    echo "[AutoTopK] disabled by ABFLOW_AUTO_TOPK_EVAL=$AUTO_TOPK_EVAL"
    return 0
  fi
  if [[ -z "$eval_gpus" ]]; then
    echo "[AutoTopK] no evaluation GPU available; watcher not started."
    return 0
  fi
  if [[ ! -f "$AUTO_TOPK_TEST_JSON" ]]; then
    echo "[AutoTopK] test json not found: $AUTO_TOPK_TEST_JSON"
    return 0
  fi
  if [[ ! -f "$AUTO_TOPK_EVAL_SCRIPT" ]]; then
    echo "[AutoTopK] evaluator not found: $AUTO_TOPK_EVAL_SCRIPT"
    return 0
  fi

  mkdir -p "$run_dir"

  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" >/dev/null 2>&1; then
      echo "[AutoTopK] watcher already running pid=$old_pid"
      return 0
    fi
    rm -f "$pid_file"
  fi

  echo "[AutoTopK] starting watcher: exp=$EXP_ID eval_gpus=$eval_gpus run_dir=$run_dir" | tee -a "$log_file"
  (
    python "$AUTO_TOPK_EVAL_SCRIPT" \
      --exp-id "$EXP_ID" \
      --run-dir "$run_dir" \
      --test-json "$AUTO_TOPK_TEST_JSON" \
      --gpu-ids "$eval_gpus" \
      --project-root "$PROJECT_ROOT" \
      --launcher "$SELF_LAUNCHER" \
      --watch \
      --poll-interval "$AUTO_TOPK_POLL_INTERVAL" \
      --max-new "$AUTO_TOPK_MAX_NEW" \
      $( _is_on "$AUTO_TOPK_LATEST_ONLY" && echo --latest-only ) \
      $( _is_on "$AUTO_TOPK_FORCE" && echo --force )
  ) >> "$log_file" 2>&1 &

  echo $! > "$pid_file"
  echo "[AutoTopK] watcher pid=$(cat "$pid_file") log=$log_file"
}

_stop_auto_topk_watcher() {
  local run_dir="$1"
  local pid_file="$run_dir/auto_topk_eval.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "[AutoTopK] stopping watcher pid=$pid"
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

_run_auto_topk_once() {
  local eval_gpus="$1"
  local run_dir="$2"
  local log_file="$run_dir/auto_topk_eval_final.log"

  if ! _is_on "$AUTO_TOPK_EVAL"; then
    return 0
  fi
  [[ -z "$eval_gpus" ]] && eval_gpus="$GPU_ID"
  if [[ ! -f "$AUTO_TOPK_TEST_JSON" || ! -f "$AUTO_TOPK_EVAL_SCRIPT" ]]; then
    return 0
  fi

  echo "[AutoTopK] final catch-up evaluation: exp=$EXP_ID eval_gpus=$eval_gpus" | tee -a "$log_file"
  python "$AUTO_TOPK_EVAL_SCRIPT" \
    --exp-id "$EXP_ID" \
    --run-dir "$run_dir" \
    --test-json "$AUTO_TOPK_TEST_JSON" \
    --gpu-ids "$eval_gpus" \
    --project-root "$PROJECT_ROOT" \
    --launcher "$SELF_LAUNCHER" \
    $( _is_on "$AUTO_TOPK_LATEST_ONLY" && echo --latest-only ) \
    $( _is_on "$AUTO_TOPK_FORCE" && echo --force ) \
    >> "$log_file" 2>&1 || true
}

print_settings() {
  echo "Experiment: $EXP_ID"
  echo "ABLATION_PARENT=$ABFLOW_ABLATION_PARENT"
  echo "EXPERIMENT_FACTOR=$ABFLOW_EXPERIMENT_FACTOR"
  echo "LOSS_MODE=$ABFLOW_SCOREFM_LOSS_MODE"
  echo "SAMPLER_MODE=$ABFLOW_SCOREFM_SAMPLER_MODE"
  echo "ABX_COMMON_CENTER=$ABFLOW_ABX_COMMON_CENTER"
  echo "FLOW_T_RANGE=$ABFLOW_FLOW_T_MIN,$ABFLOW_FLOW_T_MAX"
  echo "FLOW_COORDINATE_SCALING=$ABFLOW_R3_FLOW_COORDINATE_SCALING"
  echo "R3_G_MODE=$ABFLOW_R3_G_MODE"
  echo "R3_FIXED_G_SCALED=$ABFLOW_R3_FIXED_G_SCALED"
  echo "R3_PATH_MIN_SIGMA=$ABFLOW_R3_PATH_MIN_SIGMA"
  echo "R3_NOISE_SCOPE=$ABFLOW_R3_NOISE_SCOPE"
  echo "R03_G_SCALED=$ABFLOW_R03_G_SCALED"
  echo "R03_CENTER_RESIDUAL=$ABFLOW_R03_CENTER_RESIDUAL"
  echo "DDP_VALIDATION=$ABFLOW_DDP_VALIDATION"
  echo "EPOCH_TEST=$ABFLOW_EPOCH_TEST"
  echo "EPOCH_TEST_INTERVAL=$ABFLOW_EPOCH_TEST_INTERVAL"
  echo "EPOCH_TEST_JSON=$ABFLOW_EPOCH_TEST_JSON"
  echo "EPOCH_TEST_BATCH_SIZE=$ABFLOW_EPOCH_TEST_BATCH_SIZE"
  echo "EPOCH_TEST_N_STEPS=$ABFLOW_EPOCH_TEST_N_STEPS"
  echo "EPOCH_TEST_BASE_SEED=$ABFLOW_EPOCH_TEST_BASE_SEED"
  echo "FORMAL_EPOCH_PHASES=Train->Validation->Test"
  echo "DIAGNOSTIC_REVISION=R05MF_DIAGNOSTIC_V167+R05MF_PARENT_AUTHORITY_V168+R05MF_AUTHORITY_LADDER_V169+R05MF_LIVEPAIR_DISTOGRAM_V170+R05_V175_MODULE_AUDIT"
  echo "COORDINATE_AUTHORITY=$ABFLOW_COORDINATE_AUTHORITY"
  echo "STRUCTURE_SEQ_READOUT=$ABFLOW_STRUCTURE_SEQ_READOUT"
  echo "DUAL_SEQUENCE_STATE=$ABFLOW_DUAL_SEQUENCE_STATE"
  echo "DUAL_SEQUENCE_ATOM_MODE=$ABFLOW_DUAL_SEQUENCE_ATOM_MODE"
  echo "SEQUENCE_CONTEXT_MODE=$ABFLOW_SEQUENCE_CONTEXT_MODE"
  echo "PAIR_TIME_SCOPE=$ABFLOW_PAIR_TIME_SCOPE"
  echo "R05_ENDPOINT_RELATION=$ABFLOW_R05_ENDPOINT_RELATION"
  echo "R05_GEOM_PAIR=$ABFLOW_R05_GEOM_PAIR"
  echo "SEQUENCE_PATH_CONTRACT=${ABFLOW_SEQUENCE_PATH_CONTRACT:-legacy_context}"
  echo "SEQUENCE_LOSS_SCOPE=${ABFLOW_SEQUENCE_LOSS_SCOPE:-context_mask}"
  echo "SEQUENCE_RECYCLE_MODE=$ABFLOW_SEQUENCE_RECYCLE_MODE"
  echo "SEQUENCE_SOURCE_MODE=${ABFLOW_SEQUENCE_SOURCE_MODE:-proposal}"
  echo "SEQUENCE_GENERATIVE_MODE=${ABFLOW_SEQUENCE_GENERATIVE_MODE:-legacy}"
  echo "RECURRENT_PROPOSAL_SEQUENCE_CONTEXT=${ABFLOW_RECURRENT_PROPOSAL_SEQUENCE_CONTEXT:-on}"
  echo "ROUND_CONSISTENT_COORD_SUPERVISION=${ABFLOW_ROUND_CONSISTENT_COORD_SUPERVISION:-off}"
  echo "MF_REPR_CORE=${ABFLOW_MF_REPR_CORE:-off}"
  echo "MF_SINGLE_DIM=${ABFLOW_MF_SINGLE_DIM:-}"
  echo "MF_PAIR_DIM=${ABFLOW_MF_PAIR_DIM:-}"
  echo "MF_REPR_BLOCKS=${ABFLOW_MF_REPR_BLOCKS:-}"
  echo "MF_ATOM_S=${ABFLOW_MF_ATOM_S:-}"
  echo "MF_ATOM_DEPTH=${ABFLOW_MF_ATOM_DEPTH:-}"
  echo "MF_ATOM_HEADS=${ABFLOW_MF_ATOM_HEADS:-}"
  echo "MF_LOCAL_ANTIGEN_K=${ABFLOW_MF_LOCAL_ANTIGEN_K:-}"
  echo "MF_TRIANGLE_HEADS=${ABFLOW_MF_TRIANGLE_HEADS:-}"
  echo "MF_TRIANGLE_HIDDEN=${ABFLOW_MF_TRIANGLE_HIDDEN:-}"
  echo "MF_TRIANGLE_CHECKPOINT=${ABFLOW_MF_TRIANGLE_CHECKPOINT:-off}"
  echo "MF_PAIR_ATOM_REFINER=${ABFLOW_MF_PAIR_ATOM_REFINER:-off}"
  echo "MF_PAIR_ATOM_DEPTH=${ABFLOW_MF_PAIR_ATOM_DEPTH:-}"
  echo "MF_PAIR_ATOM_HEADS=${ABFLOW_MF_PAIR_ATOM_HEADS:-}"
  echo "MF_PAIR_ATOM_QUERY_CHUNK=${ABFLOW_MF_PAIR_ATOM_QUERY_CHUNK:-}"
  echo "DDP_FIND_UNUSED_PARAMETERS=${ABFLOW_DDP_FIND_UNUSED_PARAMETERS:-off}"
  echo "DDP_STATIC_GRAPH=${ABFLOW_DDP_STATIC_GRAPH:-off}"
  echo "DDP_COST_BALANCED=${ABFLOW_DDP_COST_BALANCED:-off}"
  echo "MF_COORD_SCALE=${ABFLOW_MF_COORD_SCALE:-}"
  echo "MF_RELPOS_DIM=${ABFLOW_MF_RELPOS_DIM:-}"
  echo "MF_OPM_DIM=${ABFLOW_MF_OPM_DIM:-}"
  echo "MF_ALLATOM_PAIR=${ABFLOW_MF_ALLATOM_PAIR:-off}"
  echo "MF_ALLATOM_CHUNK=${ABFLOW_MF_ALLATOM_CHUNK:-}"
  echo "MF_DETACH_STATE_CARRY=${ABFLOW_MF_DETACH_STATE_CARRY:-off}"
  echo "MF_TERMINAL_LIVE_PAIR_AUX=${ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX:-off}"
  echo "MF_DISTOGRAM_HEAD_SEED=${ABFLOW_MF_DISTOGRAM_HEAD_SEED:-NA}"
  echo "MF_PARENT_AUTHORITY=${ABFLOW_MF_PARENT_AUTHORITY:-off}"
  echo "MF_SAMPLE_AUTHORITY_DIAGNOSTICS=${ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS:-off}"
  echo "MF_DISTOGRAM=${ABFLOW_MF_DISTOGRAM:-off}"
  echo "MF_SMOOTH_LDDT=${ABFLOW_MF_SMOOTH_LDDT:-off}"
  echo "MF_SMOOTH_LDDT_CUTOFF=${ABFLOW_MF_SMOOTH_LDDT_CUTOFF:-}"
  echo "MF_SMOOTH_LDDT_REL_WEIGHTS=${ABFLOW_MF_SMOOTH_LDDT_INTRA_WEIGHT:-}/${ABFLOW_MF_SMOOTH_LDDT_SCAFFOLD_WEIGHT:-}/${ABFLOW_MF_SMOOTH_LDDT_ANTIGEN_WEIGHT:-}"
  echo "LOSS_SEQUENCE_WEIGHT=${ABFLOW_LOSS_SEQUENCE_WEIGHT:-}"
  echo "LOSS_STRUCTURE_WEIGHT=${ABFLOW_LOSS_STRUCTURE_WEIGHT:-}"
  echo "LOSS_INTERFACE_WEIGHT=${ABFLOW_LOSS_INTERFACE_WEIGHT:-}"
  echo "LOSS_EDGE_WEIGHT=${ABFLOW_LOSS_EDGE_WEIGHT:-}"
  echo "LOSS_DISTOGRAM_WEIGHT=${ABFLOW_LOSS_DISTOGRAM_WEIGHT:-}"
  echo "LOSS_SMOOTH_LDDT_WEIGHT=${ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT:-}"
  echo "GRAD_CONFLICT_DIAGNOSTICS=$ABFLOW_GRAD_CONFLICT_DIAGNOSTICS"
  echo "GRAD_DIAGNOSTIC_INTERVAL=$ABFLOW_GRAD_DIAGNOSTIC_INTERVAL"
  echo "GPU=$GPU_ID"
  echo "GPU_COUNT=${ABFLOW_SELECTED_GPU_COUNT:-}"
  echo "EFFECTIVE_GLOBAL_BATCH=${ABFLOW_EFFECTIVE_GLOBAL_BATCH:-}"
  echo "LOCAL_BATCH_PER_GPU=${ABFLOW_LOCAL_BATCH_PER_GPU:-}"
}

# ------------------------------------------------------------------
# test bridge used by evaluate_topk_map.py
# ------------------------------------------------------------------
if [[ "$MODE" == "test" ]]; then
  CKPT=${4:-}
  RESULT_DIR=${5:-}
  TEST_JSON=${6:-${PROJECT_ROOT}/datasets/RAbD/test.json}

  if [[ -z "$CKPT" || -z "$RESULT_DIR" ]]; then
    echo "Usage: bash $0 test <EXP_ID> <GPU_ID> <CKPT> <RESULT_DIR> [TEST_JSON]"
    exit 2
  fi
  [[ -f "$CKPT" ]] || { echo "Checkpoint not found: $CKPT"; exit 2; }
  [[ -f "$TEST_JSON" ]] || { echo "Test JSON not found: $TEST_JSON"; exit 2; }

  print_settings
  echo "Checkpoint: $CKPT"
  echo "Result dir: $RESULT_DIR"
  echo "Test JSON: $TEST_JSON"

  # Serialize actual test jobs that target the same GPU id/list.
  # The watcher itself is lightweight; only the generation/evaluation subprocess
  # is protected by this lock.
  if command -v flock >/dev/null 2>&1; then
    EVAL_LOCK="/tmp/abflow_v104_eval_gpu_${GPU_ID//,/__}.lock"
    (
      flock -x 9
      GPU="$GPU_ID" bash "$PROJECT_ROOT/scripts/test/test_epoch_ddp.sh" \
        "$CKPT" "$TEST_JSON" "$RESULT_DIR" rabd
    ) 9>"$EVAL_LOCK"
  else
    echo "[AutoTopK] WARNING: flock unavailable; shared eval GPUs are not serialized." >&2
    GPU="$GPU_ID" bash "$PROJECT_ROOT/scripts/test/test_epoch_ddp.sh" \
      "$CKPT" "$TEST_JSON" "$RESULT_DIR" rabd
  fi
  exit 0
fi

# ------------------------------------------------------------------
# attach watcher to an already-running v103 experiment
# ------------------------------------------------------------------
if [[ "$MODE" == "attach_eval" ]]; then
  
RUN_ROOT=${ABFLOW_RUN_ROOT:-${PROJECT_ROOT}/results_module}
  RUN_DIR="${RUN_ROOT}/${EXP_ID}"
  EVAL_GPUS_ARG=${4:-auto}
  if [[ -z "$EVAL_GPUS_ARG" || "$EVAL_GPUS_ARG" == "auto" ]]; then
    EVAL_GPUS_ARG=$(_infer_spare_eval_gpus)
    [[ -z "$EVAL_GPUS_ARG" ]] && EVAL_GPUS_ARG="$GPU_ID"
  fi
  echo "[AutoTopK] attach mode: exp=$EXP_ID run_dir=$RUN_DIR eval_gpus=$EVAL_GPUS_ARG"
  python "$AUTO_TOPK_EVAL_SCRIPT" \
    --exp-id "$EXP_ID" \
    --run-dir "$RUN_DIR" \
    --test-json "$AUTO_TOPK_TEST_JSON" \
    --gpu-ids "$EVAL_GPUS_ARG" \
    --project-root "$PROJECT_ROOT" \
    --launcher "$SELF_LAUNCHER" \
    --watch \
    --poll-interval "$AUTO_TOPK_POLL_INTERVAL" \
    --max-new "$AUTO_TOPK_MAX_NEW" \
    $( _is_on "$AUTO_TOPK_LATEST_ONLY" && echo --latest-only ) \
    $( _is_on "$AUTO_TOPK_FORCE" && echo --force )
  exit 0
fi

if [[ "$MODE" != "train" ]]; then
  echo "Train:       bash $0 train <R04...R22 EXP_ID> 2,3 <config.json>"
  echo "Test bridge: bash $0 test  <R04...R22 EXP_ID> 2,3 <ckpt> <result_dir> [test.json]"
  echo "Attach eval: bash $0 attach_eval <R04...R22 EXP_ID> 2,3 [eval_gpus|auto]"
  exit 2
fi

BASE_CONFIG=${4:-}
[[ -n "$BASE_CONFIG" ]] || { echo "Missing BASE_CONFIG"; exit 2; }
[[ -f "$BASE_CONFIG" ]] || { echo "Base config not found: $BASE_CONFIG"; exit 2; }

# The scientific JSON is the authority for representation/objective settings and
# the two explicit torch1.11 DDP graph-contract switches used by R08-R10.  U02/R05
# path/source/frame variables remain locked by the launcher cases above.
_ABFLOW_JSON_EXPORTS=$(python - "$BASE_CONFIG" "$EXP_ID" <<'PYMFENV'
import json, shlex, sys
path, exp_id = sys.argv[1:3]
with open(path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
meta = cfg.get('_experiment', {})
expected = str(meta.get('exp_id', '') or '').strip()
if expected and expected != exp_id:
    raise SystemExit(
        f'Config/EXP_ID mismatch: config expects {expected}, launcher got {exp_id}'
    )
env = meta.get('runtime_env', {})
if not isinstance(env, dict):
    raise SystemExit('_experiment.runtime_env must be a JSON object')
allowed_ddp = {
    'ABFLOW_DDP_FIND_UNUSED_PARAMETERS',
    'ABFLOW_DDP_STATIC_GRAPH',
}
allowed_sequence = {
    'ABFLOW_SEQUENCE_CONTEXT_MODE',
    'ABFLOW_SEQUENCE_PATH_CONTRACT',
    'ABFLOW_SEQUENCE_LOSS_SCOPE',
}
for key, value in env.items():
    if not (
        key.startswith('ABFLOW_MF_')
        or key.startswith('ABFLOW_LOSS_')
        or key in allowed_ddp
        or key in allowed_sequence
        or key in {'ABFLOW_R05_ENDPOINT_RELATION', 'ABFLOW_R05_GEOM_PAIR', 'ABFLOW_PAIR_TIME_SCOPE'}
    ):
        raise SystemExit(
            f'Forbidden runtime_env key {key}: only ABFLOW_MF_*, ABFLOW_LOSS_*, '
            'the formal ABFLOW_SEQUENCE_* contract keys, Pair-Time/R05 module keys, and the two DDP graph '
            'switches are allowed'
        )
    print('export ' + key + '=' + shlex.quote(str(value)))
PYMFENV
)
eval "$_ABFLOW_JSON_EXPORTS"
unset _ABFLOW_JSON_EXPORTS

# ------------------------------------------------------------------
# V172 orthogonal direct-R05 ablation contract.
# Each experiment has the SAME scientific parent (R05) and exactly one coherent
# module.  This is deliberately separate from the historical sequential R08-R16
# ladders so a failure of one branch cannot invalidate the other two.
# ------------------------------------------------------------------
case "$EXP_ID" in
  R17_R05_EXACTSEQ_U02|R18_R05_MF_CLOSED_CORE_U02|R19_R05_DESIGNLDDT_U02)
    MODEL_FILE="$PROJECT_ROOT/models/AbFlow/AbFlow_model.py"
    TRAINER_FILE="$PROJECT_ROOT/trainer/AbFlow_trainer.py"
    [[ -f "$MODEL_FILE" ]] || { echo "ERROR: missing $MODEL_FILE" >&2; exit 2; }
    [[ -f "$TRAINER_FILE" ]] || { echo "ERROR: missing $TRAINER_FILE" >&2; exit 2; }
    grep -q "R05MF_ORTHOGONAL_R05_ABLATION_V172" "$MODEL_FILE" || {
      echo "ERROR: R17-R19 require the V172 direct-R05 model overlay." >&2; exit 2;
    }
    grep -q "R05MF_SEQUENCE_PATH_AUTHORITY_V171" "$MODEL_FILE" || {
      echo "ERROR: V172 requires the V171 exact-sequence state/objective implementation." >&2; exit 2;
    }
    grep -q "R05MF_SEQUENCE_PATH_AUTHORITY_V171" "$TRAINER_FILE" || {
      echo "ERROR: V172 requires the V171 sequence-path audit trainer." >&2; exit 2;
    }
    _is_on "$ABFLOW_EPOCH_TEST" || { echo "ERROR: V172 formal protocol requires Epoch Test ON." >&2; exit 2; }
    [[ "$ABFLOW_EPOCH_TEST_INTERVAL" == "1" ]] || { echo "ERROR: V172 formal protocol requires Epoch Test every epoch." >&2; exit 2; }
    [[ "$ABFLOW_SOURCE_MODE" == "pcs_rc" ]] || { echo "ERROR: V172 requires PCS-RC." >&2; exit 2; }
    [[ "$ABFLOW_R3_NOISE_SCOPE" == "residue" ]] || { echo "ERROR: V172 requires R05 residue-level R3 support." >&2; exit 2; }
    [[ "$ABFLOW_SCOREFM_LOSS_MODE" == "f01_r3_endpoint_canonical_hybrid" ]] || { echo "ERROR: V172 requires the R05/U02 target." >&2; exit 2; }
    [[ "$ABFLOW_SCOREFM_SAMPLER_MODE" == "f01_canonical_carrier" ]] || { echo "ERROR: V172 requires the R05/U02 sampler." >&2; exit 2; }
    [[ "$ABFLOW_PAIR_TIME_SCOPE" == "off" ]] || { echo "ERROR: V172 keeps Pair-Time off." >&2; exit 2; }
    [[ "${ABFLOW_MF_PARENT_AUTHORITY:-off}" == "off" ]] || { echo "ERROR: V172 retires parent-RMS authority." >&2; exit 2; }

    python - "$BASE_CONFIG" "$EXP_ID" <<'PYV172'
import json, math, sys
path, exp_id = sys.argv[1:3]
with open(path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
required = {
    'batch_size': 56, 'embed_dim': 64, 'hidden_size': 128,
    'n_layers': 3, 'iter_round': 3, 'use_ema': True,
    'max_epoch': 200,
}
for key, expected in required.items():
    if cfg.get(key) != expected:
        raise SystemExit(f'ERROR: {exp_id} requires {key}={expected}, got {cfg.get(key)!r}')
if not math.isclose(float(cfg.get('ema_decay', -1)), 0.999, rel_tol=0, abs_tol=1e-12):
    raise SystemExit(f'ERROR: {exp_id} requires ema_decay=0.999')
meta = cfg.get('_experiment') or {}
if meta.get('parent') != 'R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW':
    raise SystemExit(f'ERROR: {exp_id} must be directly parented by R05')
if meta.get('implementation_revision') != 'v172_r05_orthogonal_single_factor_ablation':
    raise SystemExit(f'ERROR: {exp_id} implementation_revision mismatch')
env = meta.get('runtime_env') or {}
expected = {
    'R17_R05_EXACTSEQ_U02': {
        'ABFLOW_MF_REPR_CORE':'off','ABFLOW_MF_PAIR_ATOM_REFINER':'off',
        'ABFLOW_MF_DISTOGRAM':'off','ABFLOW_LOSS_DISTOGRAM_WEIGHT':'0.0',
        'ABFLOW_MF_SMOOTH_LDDT':'off','ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT':'0.0',
        'ABFLOW_SEQUENCE_CONTEXT_MODE':'off',
        'ABFLOW_SEQUENCE_PATH_CONTRACT':'exact_categorical_bridge',
        'ABFLOW_SEQUENCE_LOSS_SCOPE':'path_noisy',
    },
    'R18_R05_MF_CLOSED_CORE_U02': {
        'ABFLOW_MF_REPR_CORE':'on','ABFLOW_MF_PAIR_ATOM_REFINER':'off',
        'ABFLOW_MF_DISTOGRAM':'off','ABFLOW_LOSS_DISTOGRAM_WEIGHT':'0.0',
        'ABFLOW_MF_SMOOTH_LDDT':'off','ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT':'0.0',
        'ABFLOW_SEQUENCE_CONTEXT_MODE':'legacy',
        'ABFLOW_SEQUENCE_PATH_CONTRACT':'legacy_context',
        'ABFLOW_SEQUENCE_LOSS_SCOPE':'context_mask',
    },
    'R19_R05_DESIGNLDDT_U02': {
        'ABFLOW_MF_REPR_CORE':'off','ABFLOW_MF_PAIR_ATOM_REFINER':'off',
        'ABFLOW_MF_DISTOGRAM':'off','ABFLOW_LOSS_DISTOGRAM_WEIGHT':'0.0',
        'ABFLOW_MF_SMOOTH_LDDT':'on','ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT':'0.1',
        'ABFLOW_SEQUENCE_CONTEXT_MODE':'legacy',
        'ABFLOW_SEQUENCE_PATH_CONTRACT':'legacy_context',
        'ABFLOW_SEQUENCE_LOSS_SCOPE':'context_mask',
    },
}[exp_id]
for key, val in expected.items():
    if str(env.get(key, '')) != val:
        raise SystemExit(f'ERROR: {exp_id} requires {key}={val}, got {env.get(key)!r}')
for key in ('ABFLOW_MF_PARENT_AUTHORITY','ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX'):
    if str(env.get(key, 'off')) != 'off':
        raise SystemExit(f'ERROR: {exp_id} requires {key}=off')
print(f'[V172DirectR05Preflight] config PASS: {exp_id}')
PYV172

    case "$EXP_ID" in
      R17_R05_EXACTSEQ_U02)
        [[ "$ABFLOW_SEQUENCE_CONTEXT_MODE" == "off" ]] || { echo "ERROR: R17 exact sequence requires context off." >&2; exit 2; }
        [[ "$ABFLOW_SEQUENCE_PATH_CONTRACT" == "exact_categorical_bridge" ]] || { echo "ERROR: R17 path contract mismatch." >&2; exit 2; }
        [[ "$ABFLOW_SEQUENCE_LOSS_SCOPE" == "path_noisy" ]] || { echo "ERROR: R17 loss scope mismatch." >&2; exit 2; }
        [[ "$ABFLOW_MF_REPR_CORE" == "off" ]] || { echo "ERROR: R17 must not enable MF core." >&2; exit 2; }
        ;;
      R18_R05_MF_CLOSED_CORE_U02)
        [[ "$ABFLOW_MF_REPR_CORE" == "on" ]] || { echo "ERROR: R18 requires MF core on." >&2; exit 2; }
        [[ "$ABFLOW_MF_PAIR_ATOM_REFINER" == "off" ]] || { echo "ERROR: R18 is core-only; PairAtom must be off." >&2; exit 2; }
        [[ "$ABFLOW_MF_DISTOGRAM" == "off" ]] || { echo "ERROR: R18 is core-only; distogram must be off." >&2; exit 2; }
        [[ "$ABFLOW_MF_SMOOTH_LDDT" == "off" ]] || { echo "ERROR: R18 is core-only; smooth-lDDT must be off." >&2; exit 2; }
        [[ "$ABFLOW_SEQUENCE_CONTEXT_MODE" == "legacy" ]] || { echo "ERROR: R18 keeps matched R05 sequence context." >&2; exit 2; }
        AMENC_FILE="$PROJECT_ROOT/models/modules/am_enc.py"
        [[ -f "$AMENC_FILE" ]] || { echo "ERROR: R18 missing $AMENC_FILE" >&2; exit 2; }
        grep -q "enriched_edge_feat" "$AMENC_FILE" || { echo "ERROR: R18 requires final-pair -> R05 EGNN coupling." >&2; exit 2; }
        grep -q "enable_pair_representation" "$AMENC_FILE" || { echo "ERROR: R18 am_enc lacks pair representation integration." >&2; exit 2; }
        grep -q "CostBalancedDistributedSampler" "$PROJECT_ROOT/train.py" || { echo "ERROR: R18 requires cost-balanced DDP train.py." >&2; exit 2; }
        [[ "${ABFLOW_DDP_COST_BALANCED:-off}" == "on" ]] || { echo "ERROR: R18 requires cost-balanced DDP." >&2; exit 2; }
        [[ "${ABFLOW_DDP_FIND_UNUSED_PARAMETERS:-off}" == "on" ]] || { echo "ERROR: R18 requires find_unused_parameters=on." >&2; exit 2; }
        ;;
      R19_R05_DESIGNLDDT_U02)
        [[ "$ABFLOW_MF_REPR_CORE" == "off" ]] || { echo "ERROR: R19 must remain R05 representation-only parent." >&2; exit 2; }
        [[ "$ABFLOW_MF_SMOOTH_LDDT" == "on" ]] || { echo "ERROR: R19 requires design-region smooth-lDDT on." >&2; exit 2; }
        [[ "$ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT" == "0.1" ]] || { echo "ERROR: R19 fixed smooth-lDDT weight must be 0.1." >&2; exit 2; }
        ;;
    esac
    echo "[V172DirectR05Preflight] source/code/physics PASS"
    ;;
esac


# R08-R13 fail-fast scientific contract (v170 replacement R13 included). This is validation infrastructure, not
# an extra method component: expensive training must not start if the selected
# JSON, source files, or R05 invariants disagree.
case "$EXP_ID" in
  R08_R05_MF_CLOSED_CORE_U02|R09_R05_MF_CLOSED_CORE_PAIRATOM_U02|R10_R05_MF_CLOSED_CORE_PAIRATOM_DESIGNLDDT_U02|R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED_U02|R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_U02|R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02|R14_R05_MF_CLOSED_CORE_EXACTSEQ_U02|R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_U02|R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM_U02)
    MODEL_FILE="$PROJECT_ROOT/models/AbFlow/AbFlow_model.py"
    AMENC_FILE="$PROJECT_ROOT/models/modules/am_enc.py"
    TRAINER_FILE="$PROJECT_ROOT/trainer/AbFlow_trainer.py"
    [[ -f "$MODEL_FILE" ]] || { echo "ERROR: missing $MODEL_FILE" >&2; exit 2; }
    [[ -f "$AMENC_FILE" ]] || { echo "ERROR: missing $AMENC_FILE" >&2; exit 2; }
    [[ -f "$TRAINER_FILE" ]] || { echo "ERROR: missing $TRAINER_FILE" >&2; exit 2; }
    grep -q "class MFRepresentationEnrichment" "$MODEL_FILE" || {
      echo "ERROR: AbFlow_model.py is not the R05 x MF representation version." >&2; exit 2;
    }
    grep -q "R05MF_CLOSED_CORE_V164" "$MODEL_FILE" || {
      echo "ERROR: R08-R10 require v164 closed-core AbFlow_model.py." >&2; exit 2;
    }
    grep -q "enriched_edge_feat" "$AMENC_FILE" || {
      echo "ERROR: R08-R10 require final-pair -> same-step R05 EGNN coordinate coupling." >&2; exit 2;
    }
    grep -q "enable_pair_representation" "$AMENC_FILE" || {
      echo "ERROR: am_enc.py lacks the persistent-pair R05 integration." >&2; exit 2;
    }
    grep -q "_print_validation_audits" "$TRAINER_FILE" || {
      echo "ERROR: AbFlow_trainer.py lacks the R05-MF observability contract." >&2; exit 2;
    }
    grep -q "R05MF_DIAGNOSTIC_V167" "$MODEL_FILE" || {
      echo "ERROR: R08-R10 resume requires the v167 diagnostics-only model overlay." >&2; exit 2;
    }
    grep -q "R05MF_DIAGNOSTIC_V167" "$TRAINER_FILE" || {
      echo "ERROR: R08-R10 resume requires the v167 diagnostics-only trainer overlay." >&2; exit 2;
    }
    case "$EXP_ID" in
      R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED_U02|R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_U02)
        grep -q "R05MF_PARENT_AUTHORITY_V168" "$MODEL_FILE" || { echo "ERROR: R11-R12 require the v168 parent-authority operator." >&2; exit 2; }
        grep -q "R05MF_PARENT_AUTHORITY_V168" "$TRAINER_FILE" || { echo "ERROR: R11-R12 require the v168 Test-trajectory diagnostics trainer." >&2; exit 2; }
        grep -q "R05MF_PARENT_AUTHORITY_V168" "$AMENC_FILE" || { echo "ERROR: R11-R12 require the v168 pair-edge authority operator." >&2; exit 2; }
        grep -q "R05MF_AUTHORITY_LADDER_V169" "$MODEL_FILE" || { echo "ERROR: R11-R12 require the v169 authority-ladder model overlay." >&2; exit 2; }
        grep -q "R05MF_AUTHORITY_LADDER_V169" "$TRAINER_FILE" || { echo "ERROR: R11-R12 require the v169 authority-ladder trainer overlay." >&2; exit 2; }
        grep -q "R05MF_AUTHORITY_LADDER_V169" "$AMENC_FILE" || { echo "ERROR: R11-R12 require the v169 authority-ladder am_enc overlay." >&2; exit 2; }
        ;;
      R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02)
        grep -q "R05MF_PARENT_AUTHORITY_V168" "$MODEL_FILE" || { echo "ERROR: replacement R13 requires v168 parent authority." >&2; exit 2; }
        grep -q "R05MF_AUTHORITY_LADDER_V169" "$AMENC_FILE" || { echo "ERROR: replacement R13 requires the v169 pair-edge authority am_enc." >&2; exit 2; }
        grep -q "R05MF_LIVEPAIR_DISTOGRAM_V170" "$MODEL_FILE" || { echo "ERROR: replacement R13 requires the v170 live-pair model overlay." >&2; exit 2; }
        grep -q "R05MF_LIVEPAIR_DISTOGRAM_V170" "$TRAINER_FILE" || { echo "ERROR: replacement R13 requires the v170 live-pair gradient-contract trainer." >&2; exit 2; }
        ;;
    esac
    case "$EXP_ID" in
      R14_R05_MF_CLOSED_CORE_EXACTSEQ_U02|R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_U02|R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM_U02)
        grep -q "R05MF_SEQUENCE_PATH_AUTHORITY_V171" "$MODEL_FILE" || { echo "ERROR: R14-R16 require the v171 exact-sequence model contract." >&2; exit 2; }
        grep -q "R05MF_SEQUENCE_PATH_AUTHORITY_V171" "$TRAINER_FILE" || { echo "ERROR: R14-R16 require the v171 sequence-path audit trainer." >&2; exit 2; }
        [[ "${ABFLOW_MF_PARENT_AUTHORITY:-off}" == "off" ]] || { echo "ERROR: R14-R16 retire parent-RMS authority; it must be off." >&2; exit 2; }
        [[ "${ABFLOW_SEQUENCE_CONTEXT_MODE:-}" == "off" ]] || { echo "ERROR: R14-R16 require SEQUENCE_CONTEXT_MODE=off." >&2; exit 2; }
        [[ "${ABFLOW_SEQUENCE_PATH_CONTRACT:-}" == "exact_categorical_bridge" ]] || { echo "ERROR: R14-R16 require exact_categorical_bridge." >&2; exit 2; }
        [[ "${ABFLOW_SEQUENCE_LOSS_SCOPE:-}" == "path_noisy" ]] || { echo "ERROR: R14-R16 require path_noisy sequence CE authority." >&2; exit 2; }
        ;;
    esac
    _is_on "$ABFLOW_EPOCH_TEST" || {
      echo "ERROR: formal R08-R16 protocol requires Epoch Test ON." >&2; exit 2;
    }
    [[ "$ABFLOW_EPOCH_TEST_INTERVAL" == "1" ]] || {
      echo "ERROR: formal R08-R16 protocol requires Epoch Test every epoch (interval=1)." >&2; exit 2;
    }

    python - "$BASE_CONFIG" "$EXP_ID" <<'PYR05MF'
import json, math, sys
path, exp_id = sys.argv[1:3]
with open(path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
required = {
    'batch_size': 56, 'embed_dim': 64, 'hidden_size': 128,
    'n_layers': 3, 'iter_round': 3, 'use_ema': True,
}
for key, expected in required.items():
    if cfg.get(key) != expected:
        raise SystemExit(f'ERROR: {exp_id} requires {key}={expected}, got {cfg.get(key)!r}')
if not math.isclose(float(cfg.get('ema_decay', -1)), 0.999, rel_tol=0, abs_tol=1e-12):
    raise SystemExit('ERROR: R08-R10 require ema_decay=0.999')
meta = cfg.get('_experiment') or {}
authority_ladder_ids = {
    'R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED_U02',
    'R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_U02',
    'R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02',
}
exact_sequence_ids = {
    'R14_R05_MF_CLOSED_CORE_EXACTSEQ_U02',
    'R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_U02',
    'R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM_U02',
}
expected_revision = (
    'v171_r05_mf_exact_sequence_path_authority'
    if exp_id in exact_sequence_ids
    else ('v170_r05_mf_livepair_distogram'
          if exp_id == 'R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02'
          else ('v169_r05_mf_authority_ladder'
                if exp_id in authority_ladder_ids
                else 'v164_r05_mf_pair_atom_pair_universe_fix'))
)
if meta.get('implementation_revision') != expected_revision:
    raise SystemExit(
        f"ERROR: {exp_id} requires implementation_revision={expected_revision}, "
        f"got {meta.get('implementation_revision')!r}"
    )
env = meta.get('runtime_env') or {}
common = {
    'ABFLOW_MF_REPR_CORE': 'on',
    'ABFLOW_MF_SINGLE_DIM': '256',
    'ABFLOW_MF_PAIR_DIM': '128',
    'ABFLOW_MF_REPR_BLOCKS': '1',
    'ABFLOW_MF_REPR_HEADS': '16',
    'ABFLOW_MF_ATOM_S': '128',
    'ABFLOW_MF_ATOM_DEPTH': '3',
    'ABFLOW_MF_ATOM_HEADS': '4',
    'ABFLOW_MF_LOCAL_ANTIGEN_K': '18',
    'ABFLOW_MF_TRIANGLE_HEADS': '4',
    'ABFLOW_MF_TRIANGLE_HIDDEN': '128',
    'ABFLOW_MF_TRIANGLE_CHECKPOINT': 'off',
    'ABFLOW_MF_PAIR_ATOM_DEPTH': '1',
    'ABFLOW_MF_PAIR_ATOM_HEADS': '4',
    'ABFLOW_MF_PAIR_ATOM_QUERY_CHUNK': '64',
    'ABFLOW_DDP_FIND_UNUSED_PARAMETERS': 'on',
    'ABFLOW_DDP_STATIC_GRAPH': 'off',
    'ABFLOW_MF_COORD_SCALE': '0.1',
    'ABFLOW_MF_RELPOS_DIM': '32',
    'ABFLOW_MF_OPM_DIM': '64',
    'ABFLOW_MF_ALLATOM_PAIR': 'on',
    'ABFLOW_MF_ALLATOM_CHUNK': '512',
    'ABFLOW_MF_DETACH_STATE_CARRY': 'on',
    'ABFLOW_MF_DISTOGRAM_HEAD_SEED': '13003',
    'ABFLOW_MF_SMOOTH_LDDT_CUTOFF': '15.0',
    'ABFLOW_MF_SMOOTH_LDDT_INTRA_WEIGHT': '1.0',
    'ABFLOW_MF_SMOOTH_LDDT_SCAFFOLD_WEIGHT': '1.0',
    'ABFLOW_MF_SMOOTH_LDDT_ANTIGEN_WEIGHT': '1.0',
    'ABFLOW_LOSS_SEQUENCE_WEIGHT': '1.0',
    'ABFLOW_LOSS_STRUCTURE_WEIGHT': '1.0',
    'ABFLOW_LOSS_INTERFACE_WEIGHT': '1.0',
    'ABFLOW_LOSS_EDGE_WEIGHT': '1.0',
}
expected_by_exp = {
    'R08_R05_MF_CLOSED_CORE_U02': ('off', 'off', 0.0, 'off', 0.0, 'off'),
    'R09_R05_MF_CLOSED_CORE_PAIRATOM_U02': ('on', 'off', 0.0, 'off', 0.0, 'off'),
    'R10_R05_MF_CLOSED_CORE_PAIRATOM_DESIGNLDDT_U02': ('on', 'off', 0.0, 'on', 0.1, 'off'),
    'R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED_U02': ('off', 'off', 0.0, 'off', 0.0, 'off'),
    'R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_U02': ('on', 'off', 0.0, 'off', 0.0, 'off'),
    'R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02': ('on', 'on', 0.03, 'off', 0.0, 'on'),
    'R14_R05_MF_CLOSED_CORE_EXACTSEQ_U02': ('off', 'off', 0.0, 'off', 0.0, 'off'),
    'R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_U02': ('on', 'off', 0.0, 'off', 0.0, 'off'),
    'R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM_U02': ('on', 'on', 0.03, 'off', 0.0, 'on'),
}
for key, expected in common.items():
    if str(env.get(key, '')) != expected:
        raise SystemExit(f'ERROR: {exp_id} requires {key}={expected}, got {env.get(key)!r}')
pair_atom, disto, disto_w, slddt, slddt_w, live_pair_aux = expected_by_exp[exp_id]
if str(env.get('ABFLOW_MF_PAIR_ATOM_REFINER', '')) != pair_atom:
    raise SystemExit(f'ERROR: {exp_id} pair-atom flag mismatch')
if str(env.get('ABFLOW_MF_DISTOGRAM', '')) != disto:
    raise SystemExit(f'ERROR: {exp_id} distogram flag mismatch')
if str(env.get('ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX', 'off')) != live_pair_aux:
    raise SystemExit(f'ERROR: {exp_id} terminal live-pair auxiliary flag mismatch')
if str(env.get('ABFLOW_MF_SMOOTH_LDDT', '')) != slddt:
    raise SystemExit(f'ERROR: {exp_id} smooth-lDDT flag mismatch')
if str(env.get('ABFLOW_MF_SMOOTH_LDDT_CUTOFF', '')) != '15.0':
    raise SystemExit(f'ERROR: {exp_id} smooth-lDDT cutoff must be 15.0 A')
if not math.isclose(float(env.get('ABFLOW_LOSS_DISTOGRAM_WEIGHT', -1)), disto_w, rel_tol=0, abs_tol=1e-12):
    raise SystemExit(f'ERROR: {exp_id} distogram weight mismatch')
if not math.isclose(float(env.get('ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT', -1)), slddt_w, rel_tol=0, abs_tol=1e-12):
    raise SystemExit(f'ERROR: {exp_id} smooth-lDDT weight mismatch')
if exp_id in authority_ladder_ids:
    if str(env.get('ABFLOW_MF_PARENT_AUTHORITY', '')) != 'on':
        raise SystemExit('ERROR: R11-R13 require ABFLOW_MF_PARENT_AUTHORITY=on')
    if str(env.get('ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS', '')) != 'on':
        raise SystemExit('ERROR: R11-R13 require Test sample authority diagnostics on')
    if exp_id == 'R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02':
        if str(env.get('ABFLOW_MF_TERMINAL_LIVE_PAIR_AUX', '')) != 'on':
            raise SystemExit('ERROR: replacement R13 requires live final pair auxiliary state')
        if str(env.get('ABFLOW_MF_DISTOGRAM_HEAD_SEED', '')) != '13003':
            raise SystemExit('ERROR: replacement R13 requires RNG-isolated distogram head seed=13003')
else:
    if str(env.get('ABFLOW_MF_PARENT_AUTHORITY', 'off')) not in {'', 'off'}:
        raise SystemExit(f'ERROR: {exp_id} must keep parent authority closure off')
if exp_id in exact_sequence_ids:
    required_seq = {
        'ABFLOW_SEQUENCE_CONTEXT_MODE': 'off',
        'ABFLOW_SEQUENCE_PATH_CONTRACT': 'exact_categorical_bridge',
        'ABFLOW_SEQUENCE_LOSS_SCOPE': 'path_noisy',
    }
    for key, expected in required_seq.items():
        if str(env.get(key, '')) != expected:
            raise SystemExit(f'ERROR: {exp_id} requires {key}={expected}')
    if str(env.get('ABFLOW_MF_SAMPLE_AUTHORITY_DIAGNOSTICS', '')) != 'on':
        raise SystemExit('ERROR: R14-R16 keep trajectory authority diagnostics on')
    if exp_id == 'R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM_U02':
        if str(env.get('ABFLOW_MF_DISTOGRAM_HEAD_SEED', '')) != '13003':
            raise SystemExit('ERROR: R16 requires RNG-isolated distogram head seed=13003')
if 'ABFLOW_MF_CLEAN_SC' in env:
    raise SystemExit(f'ERROR: {exp_id} must not define retired ABFLOW_MF_CLEAN_SC')
print(f'[R05MFPreflight] config PASS: {exp_id}')
PYR05MF

    [[ "$ABFLOW_SOURCE_MODE" == "pcs_rc" ]] || { echo "ERROR: R05MF requires PCS-RC." >&2; exit 2; }
    [[ "$ABFLOW_R3_NOISE_SCOPE" == "residue" ]] || { echo "ERROR: R05MF requires residue R3 support." >&2; exit 2; }
    [[ "$ABFLOW_SCOREFM_LOSS_MODE" == "f01_r3_endpoint_canonical_hybrid" ]] || { echo "ERROR: R05MF requires U02 hybrid target." >&2; exit 2; }
    [[ "$ABFLOW_SCOREFM_SAMPLER_MODE" == "f01_canonical_carrier" ]] || { echo "ERROR: R05MF requires the matched U02 sampler." >&2; exit 2; }
    [[ "$ABFLOW_PAIR_TIME_SCOPE" == "off" ]] || { echo "ERROR: R05MF requires Pair-Time off." >&2; exit 2; }
    grep -q "class _TriangleMultiplication" "$PROJECT_ROOT/models/AbFlow/AbFlow_model.py" || { echo "ERROR: triangle-closed representation model not installed." >&2; exit 2; }
    grep -q "CostBalancedDistributedSampler" "$PROJECT_ROOT/train.py" || { echo "ERROR: cost-balanced DDP train.py not installed." >&2; exit 2; }
    grep -q "find_unused_parameters=find_unused" "$PROJECT_ROOT/trainer/abs_trainer.py" || { echo "ERROR: dynamic-graph DDP abs_trainer.py not installed." >&2; exit 2; }
    [[ "${ABFLOW_DDP_COST_BALANCED:-off}" == "on" ]] || { echo "ERROR: R08-R16 require cost-balanced DDP sharding." >&2; exit 2; }
    [[ "${ABFLOW_MF_TRIANGLE_CHECKPOINT:-off}" == "off" ]] || { echo "ERROR: torch1.11 R08-R16 require triangle checkpoint off." >&2; exit 2; }
    [[ "${ABFLOW_DDP_FIND_UNUSED_PARAMETERS:-off}" == "on" ]] || { echo "ERROR: R08-R16 require DDP find_unused_parameters on." >&2; exit 2; }
    [[ "${ABFLOW_DDP_STATIC_GRAPH:-off}" == "off" ]] || { echo "ERROR: R08-R16 require DDP static_graph off." >&2; exit 2; }
    echo "[R05MFPreflight] source/code/physics PASS"
    ;;
esac

# V175 metric-directed direct-R05 module contract. No ValGen.
case "$EXP_ID" in
  R23_R05_ABX_PAIR_TIME_U02|R24_R05_ENDPOINT_RELATION_U02|R25_R05_GEOMETRY_PAIR_U02)
    [[ "$ABFLOW_SOURCE_MODE" == "pcs_rc" ]] || { echo "ERROR: V175 requires PCS-RC." >&2; exit 2; }
    [[ "$ABFLOW_R3_NOISE_SCOPE" == "residue" ]] || { echo "ERROR: V175 requires residue-R3 support." >&2; exit 2; }
    [[ "$ABFLOW_SCOREFM_LOSS_MODE" == "f01_r3_endpoint_canonical_hybrid" ]] || { echo "ERROR: V175 requires U02 target." >&2; exit 2; }
    [[ "$ABFLOW_SCOREFM_SAMPLER_MODE" == "f01_canonical_carrier" ]] || { echo "ERROR: V175 requires matched U02 sampler." >&2; exit 2; }
    [[ "$ABFLOW_SEQUENCE_CONTEXT_MODE" == "legacy" ]] || { echo "ERROR: V175 freezes original AbFlow sequence context." >&2; exit 2; }
    _is_on "$ABFLOW_EPOCH_TEST" || { echo "ERROR: V175 preserves Epoch Test ON." >&2; exit 2; }
    case "$EXP_ID" in
      R23_R05_ABX_PAIR_TIME_U02)
        [[ "$ABFLOW_PAIR_TIME_SCOPE" == "residue" ]] || exit 2
        [[ "$ABFLOW_R05_ENDPOINT_RELATION" == "off" && "$ABFLOW_R05_GEOM_PAIR" == "off" ]] || exit 2 ;;
      R24_R05_ENDPOINT_RELATION_U02)
        [[ "$ABFLOW_PAIR_TIME_SCOPE" == "off" && "$ABFLOW_R05_ENDPOINT_RELATION" == "on" && "$ABFLOW_R05_GEOM_PAIR" == "off" ]] || exit 2 ;;
      R25_R05_GEOMETRY_PAIR_U02)
        [[ "$ABFLOW_PAIR_TIME_SCOPE" == "off" && "$ABFLOW_R05_ENDPOINT_RELATION" == "off" && "$ABFLOW_R05_GEOM_PAIR" == "on" ]] || exit 2 ;;
    esac
    echo "[V175Preflight] R05 physics + single-module contract PASS: $EXP_ID"
    ;;
esac


case "$EXP_ID" in
  R23_R05_ABX_PAIR_TIME_U02|R24_R05_ENDPOINT_RELATION_U02|R25_R05_GEOMETRY_PAIR_U02)
    python "$PROJECT_ROOT/scripts/train/validate_R05_v175.py" --root "$PROJECT_ROOT" --config "$BASE_CONFIG" --runtime
    python "$PROJECT_ROOT/scripts/train/validate_R05_v175.py" --root "$PROJECT_ROOT" --tensor
    ;;
esac

case "$EXP_ID" in
  R20_R05_FULLDESIGN_SEQ_U02|R21_R05_ENDPOINT_RELATION_U02|R22_R05_GEOMETRY_PAIR_U02)
    python "$PROJECT_ROOT/scripts/train/validate_R05_v173.py" --root "$PROJECT_ROOT" --config "$BASE_CONFIG" --runtime
    # This is an independent CPU math/gradient check, not a GPU training smoke test.
    python "$PROJECT_ROOT/scripts/train/validate_R05_v173.py" --root "$PROJECT_ROOT" --tensor
    ;;
esac

RUN_ROOT=${ABFLOW_RUN_ROOT:-${PROJECT_ROOT}/results_module}
RUN_DIR="${RUN_ROOT}/${EXP_ID}"
CONFIG_DIR="${RUN_ROOT}/generated_configs"
RUN_CONFIG="${CONFIG_DIR}/${EXP_ID}.json"
RUNTIME_META="${RUN_DIR}/abflow_runtime.json"
mkdir -p "$RUN_DIR" "$CONFIG_DIR"

# ------------------------------------------------------------------
# Strict resume/scratch decision.
# Crucial: never overwrite a non-empty resume_checkpoint.
# ------------------------------------------------------------------
BASE_RESUME_CHECKPOINT=$(python - "$BASE_CONFIG" <<'PYRESUME'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(str(cfg.get("resume_checkpoint", "") or "").strip())
PYRESUME
)
ENV_RESUME_CHECKPOINT=$(echo "${ABFLOW_RESUME_CHECKPOINT:-}" | xargs)

# Resume priority:
#   1) explicit ABFLOW_RESUME_CHECKPOINT from the launch command;
#   2) resume_checkpoint stored in the scientific JSON;
#   3) scratch only when neither exists.
# The environment override changes only runtime state restoration; the generated
# config records the resolved absolute checkpoint path for reproducibility.
if [[ -n "$ENV_RESUME_CHECKPOINT" ]]; then
  RUN_MODE="resume"
  EFFECTIVE_RESUME_CHECKPOINT="$ENV_RESUME_CHECKPOINT"
  if _is_on "$FORCE_SCRATCH"; then
    echo "[Resume] WARNING: ABFLOW_FORCE_SCRATCH=$FORCE_SCRATCH ignored because ABFLOW_RESUME_CHECKPOINT is set." >&2
  fi
elif [[ -n "$BASE_RESUME_CHECKPOINT" ]]; then
  RUN_MODE="resume"
  EFFECTIVE_RESUME_CHECKPOINT="$BASE_RESUME_CHECKPOINT"
  if _is_on "$FORCE_SCRATCH"; then
    echo "[Resume] WARNING: ABFLOW_FORCE_SCRATCH=$FORCE_SCRATCH ignored because base config contains resume_checkpoint." >&2
  fi
elif _is_on "$FORCE_SCRATCH"; then
  RUN_MODE="scratch"
  EFFECTIVE_RESUME_CHECKPOINT=""
else
  RUN_MODE="scratch"
  EFFECTIVE_RESUME_CHECKPOINT=""
fi

if [[ "$RUN_MODE" == "resume" ]]; then
  if [[ ! -f "$EFFECTIVE_RESUME_CHECKPOINT" ]]; then
    echo "ERROR: resume checkpoint not found: $EFFECTIVE_RESUME_CHECKPOINT" >&2
    exit 2
  fi

  ckpt_real=$(realpath "$EFFECTIVE_RESUME_CHECKPOINT")
  run_real=$(realpath -m "$RUN_DIR")
  case "$ckpt_real" in
    "$run_real"/version_*/checkpoint/last_step*.pt)
      ;;
    *)
      echo "ERROR: resume checkpoint must be last_step*.pt under this experiment run:" >&2
      echo "  expected: $run_real/version_N/checkpoint/last_step*.pt" >&2
      echo "  got:      $ckpt_real" >&2
      exit 2
      ;;
  esac
else
  case "$EXP_ID" in
    R08_R05_MF_CLOSED_CORE_U02|R09_R05_MF_CLOSED_CORE_PAIRATOM_U02|R10_R05_MF_CLOSED_CORE_PAIRATOM_DESIGNLDDT_U02|R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED_U02|R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_U02|R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02|R14_R05_MF_CLOSED_CORE_EXACTSEQ_U02|R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_U02|R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM_U02|R17_R05_EXACTSEQ_U02|R18_R05_MF_CLOSED_CORE_U02|R19_R05_DESIGNLDDT_U02|R20_R05_FULLDESIGN_SEQ_U02|R21_R05_ENDPOINT_RELATION_U02|R22_R05_GEOMETRY_PAIR_U02|R23_R05_ABX_PAIR_TIME_U02|R24_R05_ENDPOINT_RELATION_U02|R25_R05_GEOMETRY_PAIR_U02)
      # Formal R08-R13 use exactly one run directory: version_0.  This removes the
      # DDP race that previously produced rank0/version_0 and rank1/version_1.
      # Existing scientific state is never deleted automatically.
      if compgen -G "$RUN_DIR/version_*" > /dev/null; then
        echo "ERROR: formal scratch run already contains version_* under: $RUN_DIR" >&2
        echo "Remove the failed/old experiment directory explicitly before a new scratch run." >&2
        exit 2
      fi
      export ABFLOW_FIXED_VERSION="0"
      ;;
    *)
      # Historical behavior for non-R08-R13 experiments.
      if [[ -d "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        if ! _is_on "${ABFLOW_ALLOW_NONEMPTY_RUN_DIR:-off}"; then
          echo "ERROR: fresh-run directory is not empty: $RUN_DIR" >&2
          echo "Set resume_checkpoint in the base config, or explicitly use a new run root." >&2
          exit 2
        fi
      fi
      ;;
  esac
fi

# One canonical human-readable log per formal experiment.  It captures the same
# stdout/stderr that is visible in tmux, including model diagnostics and errors.
# epoch_summary.csv is the only compact machine-readable epoch table.
case "$EXP_ID" in
  R08_R05_MF_CLOSED_CORE_U02|R09_R05_MF_CLOSED_CORE_PAIRATOM_U02|R10_R05_MF_CLOSED_CORE_PAIRATOM_DESIGNLDDT_U02|R11_R05_MF_CLOSED_CORE_PARENT_ANCHORED_U02|R12_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_U02|R13_R05_MF_CLOSED_CORE_PARENT_ANCHORED_PAIRATOM_LIVEPAIR_DISTOGRAM_U02|R14_R05_MF_CLOSED_CORE_EXACTSEQ_U02|R15_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_U02|R16_R05_MF_CLOSED_CORE_EXACTSEQ_PAIRATOM_LIVEPAIR_DISTOGRAM_U02|R17_R05_EXACTSEQ_U02|R18_R05_MF_CLOSED_CORE_U02|R19_R05_DESIGNLDDT_U02|R20_R05_FULLDESIGN_SEQ_U02|R21_R05_ENDPOINT_RELATION_U02|R22_R05_GEOMETRY_PAIR_U02|R23_R05_ABX_PAIR_TIME_U02|R24_R05_ENDPOINT_RELATION_U02|R25_R05_GEOMETRY_PAIR_U02)
    if [[ "$RUN_MODE" == "resume" ]]; then
      unset ABFLOW_FIXED_VERSION || true
      RUN_VERSION_DIR=$(dirname "$(dirname "$(realpath "$EFFECTIVE_RESUME_CHECKPOINT")")")
    else
      RUN_VERSION_DIR="$RUN_DIR/version_0"
    fi
    if [[ "${ABFLOW_DRY_RUN:-0}" != "1" ]]; then
      mkdir -p "$RUN_VERSION_DIR"
      RUN_TIME_LOG="$RUN_VERSION_DIR/run_time.log"
      export ABFLOW_RUN_TIME_LOG="$RUN_TIME_LOG"
      if [[ "$RUN_MODE" == "scratch" ]]; then
        : > "$RUN_TIME_LOG"
      fi
      export PYTHONUNBUFFERED=1
      exec > >(stdbuf -oL -eL tee -a "$RUN_TIME_LOG") 2>&1
      echo "[RunLog] $RUN_TIME_LOG"
      echo "[R05MFRunContract] EXP=$EXP_ID GPUs=$GPU_ID world_size=2 global_batch=56 local_batch=28 version=$(basename "$RUN_VERSION_DIR")"
    fi
    ;;
esac

echo "Run mode: $RUN_MODE"
echo "Effective resume checkpoint: ${EFFECTIVE_RESUME_CHECKPOINT:-<none>}"

# ------------------------------------------------------------------
# Generate train config while preserving resume state.
# ------------------------------------------------------------------
python - "$BASE_CONFIG" "$RUN_CONFIG" "$RUN_DIR" "$EXP_ID" "$RUNTIME_META" "$EFFECTIVE_RESUME_CHECKPOINT" <<'PYCFG'
import json, os, sys, datetime
src, dst, save_dir, exp_id, runtime_meta, resume_ckpt = sys.argv[1:7]

with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

meta = cfg.pop("_experiment", None)
if isinstance(meta, dict):
    expected = str(meta.get("exp_id", "") or "").strip()
    if expected and expected != exp_id:
        raise ValueError(
            f"Config/EXP_ID mismatch: config expects {expected}, launcher got {exp_id}"
        )

# Remove private/launcher-only uppercase keys if any.
for key in list(cfg.keys()):
    if key.isupper():
        cfg.pop(key, None)

cfg["save_dir"] = save_dir

def normalize_optional_path(value):
    text = str(value or "").strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return "" if text.lower() in {"", "none", "null"} else text

resolved_resume = normalize_optional_path(resume_ckpt)
if resolved_resume:
    cfg["resume_checkpoint"] = resolved_resume
    if not os.path.isfile(resolved_resume):
        raise FileNotFoundError(
            f"resume_checkpoint does not exist: {resolved_resume}"
        )
else:
    # Scratch means no resume field at all. This avoids historical JSON->CLI
    # wrappers turning an empty JSON string into the literal token ''.
    cfg.pop("resume_checkpoint", None)

def env_on(name, default="off"):
    return os.environ.get(name, default).strip().lower() in {"1","true","yes","y","on"}

def env_int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else int(default)

def env_float(name, default):
    value = os.environ.get(name, "").strip()
    return float(value) if value else float(default)

cfg["num_workers"] = env_int("ABFLOW_NUM_WORKERS", cfg.get("num_workers", 8))
cfg["prefetch_factor"] = env_int("ABFLOW_PREFETCH_FACTOR", cfg.get("prefetch_factor", 4))
cfg["valid_num_workers"] = env_int("ABFLOW_VALID_NUM_WORKERS", cfg.get("valid_num_workers", 2))
cfg["valid_prefetch_factor"] = env_int("ABFLOW_VALID_PREFETCH_FACTOR", cfg.get("valid_prefetch_factor", 2))
cfg["valid_persistent_workers"] = env_on(
    "ABFLOW_VALID_PERSISTENT_WORKERS",
    "on" if cfg.get("valid_persistent_workers", False) else "off",
)
cfg["log_interval"] = env_int("ABFLOW_LOG_INTERVAL", cfg.get("log_interval", 1))
cfg["tqdm_mininterval"] = env_float("ABFLOW_TQDM_MININTERVAL", cfg.get("tqdm_mininterval", 5.0))
cfg["save_interval"] = env_int("ABFLOW_SAVE_INTERVAL", cfg.get("save_interval", 1))

max_epoch_env = os.environ.get("ABFLOW_MAX_EPOCH", "").strip()
if max_epoch_env:
    cfg["max_epoch"] = int(max_epoch_env)

amp_dtype = os.environ.get("ABFLOW_AMP_DTYPE", str(cfg.get("amp_dtype", "bf16"))).strip().lower()
if amp_dtype not in {"bf16", "fp16"}:
    raise ValueError("ABFLOW_AMP_DTYPE must be bf16 or fp16")
cfg["amp_dtype"] = amp_dtype
cfg["amp"] = env_on("ABFLOW_AMP", "on" if cfg.get("amp", True) else "off")
cfg["allow_tf32"] = env_on("ABFLOW_ALLOW_TF32", "on" if cfg.get("allow_tf32", True) else "off")

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
    f.write("\n")

runtime = {
    "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "experiment_id": exp_id,
    "run_mode": "resume" if cfg.get("resume_checkpoint", "") else "scratch",
    "resume_checkpoint": cfg.get("resume_checkpoint", ""),
    "max_epoch": cfg.get("max_epoch"),
    "batch_size": cfg.get("batch_size"),
    "selected_gpus": os.environ.get("ABFLOW_SELECTED_GPUS", ""),
    "selected_gpu_count": os.environ.get("ABFLOW_SELECTED_GPU_COUNT", ""),
    "effective_global_batch": os.environ.get("ABFLOW_EFFECTIVE_GLOBAL_BATCH", ""),
    "local_batch_per_gpu": os.environ.get("ABFLOW_LOCAL_BATCH_PER_GPU", ""),
    "run_time_log": os.environ.get("ABFLOW_RUN_TIME_LOG", ""),
    "fixed_version": os.environ.get("ABFLOW_FIXED_VERSION", ""),
    "grad_conflict_diagnostics": os.environ.get("ABFLOW_GRAD_CONFLICT_DIAGNOSTICS", "off"),
    "grad_diagnostic_interval": os.environ.get("ABFLOW_GRAD_DIAGNOSTIC_INTERVAL", "0"),
    "loss_mode": os.environ.get("ABFLOW_SCOREFM_LOSS_MODE", ""),
    "sampler_mode": os.environ.get("ABFLOW_SCOREFM_SAMPLER_MODE", ""),
    "abx_common_center": os.environ.get("ABFLOW_ABX_COMMON_CENTER", ""),
    "flow_t_min": os.environ.get("ABFLOW_FLOW_T_MIN", ""),
    "flow_t_max": os.environ.get("ABFLOW_FLOW_T_MAX", ""),
    "flow_coordinate_scaling": os.environ.get("ABFLOW_R3_FLOW_COORDINATE_SCALING", ""),
    "r3_g_mode": os.environ.get("ABFLOW_R3_G_MODE", ""),
    "r3_fixed_g_scaled": os.environ.get("ABFLOW_R3_FIXED_G_SCALED", ""),
    "r3_path_min_sigma": os.environ.get("ABFLOW_R3_PATH_MIN_SIGMA", ""),
    "r3_score_min_sigma": os.environ.get("ABFLOW_R3_SCORE_MIN_SIGMA", ""),
    "r3_noise_scope": os.environ.get("ABFLOW_R3_NOISE_SCOPE", ""),
    "r03_g_scaled": os.environ.get("ABFLOW_R03_G_SCALED", ""),
    "r03_path_min_sigma_scaled": os.environ.get("ABFLOW_R03_PATH_MIN_SIGMA_SCALED", ""),
    "r03_center_residual": os.environ.get("ABFLOW_R03_CENTER_RESIDUAL", ""),
    "coordinate_authority": os.environ.get("ABFLOW_COORDINATE_AUTHORITY", ""),
    "structure_seq_readout": os.environ.get("ABFLOW_STRUCTURE_SEQ_READOUT", ""),
    "dual_sequence_state": os.environ.get("ABFLOW_DUAL_SEQUENCE_STATE", ""),
    "dual_sequence_atom_mode": os.environ.get("ABFLOW_DUAL_SEQUENCE_ATOM_MODE", ""),
    "source_sha256": {
        rel: __import__('hashlib').sha256(open(os.path.join(os.environ.get('ABFLOW_PROJECT_ROOT','.'), rel),'rb').read()).hexdigest()
        for rel in ['models/AbFlow/AbFlow_model.py','models/AbFlow/abflow_r3_matcher.py','models/modules/am_enc.py','trainer/AbFlow_trainer.py','trainer/abs_trainer.py','scripts/train/run_R04_R05_R06_foldflow_noise_v104.sh']
    },
    "sequence_context_mode": os.environ.get("ABFLOW_SEQUENCE_CONTEXT_MODE", ""),
    "sequence_path_contract": os.environ.get("ABFLOW_SEQUENCE_PATH_CONTRACT", "legacy_context"),
    "sequence_loss_scope": os.environ.get("ABFLOW_SEQUENCE_LOSS_SCOPE", "context_mask"),
    "sequence_recycle_mode": os.environ.get("ABFLOW_SEQUENCE_RECYCLE_MODE", ""),
    "auto_topk_eval": os.environ.get("ABFLOW_AUTO_TOPK_EVAL", "off"),
    "epoch_test": os.environ.get("ABFLOW_EPOCH_TEST", "off"),
    "epoch_test_interval": os.environ.get("ABFLOW_EPOCH_TEST_INTERVAL", "1"),
    "epoch_test_json": os.environ.get("ABFLOW_EPOCH_TEST_JSON", ""),
    "epoch_test_batch_size": os.environ.get("ABFLOW_EPOCH_TEST_BATCH_SIZE", "20"),
    "epoch_test_n_steps": os.environ.get("ABFLOW_EPOCH_TEST_N_STEPS", "10"),
    "ddp_validation": os.environ.get("ABFLOW_DDP_VALIDATION", "on"),
    "epoch_test_base_seed": os.environ.get("ABFLOW_EPOCH_TEST_BASE_SEED", "2023"),
    "epoch_test_protocol": "logical_batch_seeded_v1",
    "eval_gpus": os.environ.get("ABFLOW_EVAL_GPUS", ""),
}
os.makedirs(os.path.dirname(runtime_meta), exist_ok=True)
with open(runtime_meta, "w", encoding="utf-8") as f:
    json.dump(runtime, f, indent=2, ensure_ascii=False)
    f.write("\n")
PYCFG

print_settings
echo "Config: $RUN_CONFIG"
echo "Save dir: $RUN_DIR"
echo "Runtime metadata: $RUNTIME_META"

python - "$RUN_CONFIG" <<'PYCHECK'
import json, os, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
print("Generated resume/runtime config:")
for key in [
    "max_epoch", "batch_size", "save_topk", "use_ema", "ema_decay",
    "num_workers", "valid_num_workers", "save_interval", "resume_checkpoint",
]:
    print(f"  {key}={cfg.get(key, '')}")
resume = str(cfg.get("resume_checkpoint", "") or "").strip()
if resume:
    if not os.path.isfile(resume):
        raise SystemExit(f"ERROR: generated resume checkpoint missing: {resume}")
    if not os.path.basename(resume).startswith("last_step") or not resume.endswith(".pt"):
        raise SystemExit(
            "ERROR: strict resume requires a last_step*.pt train-state checkpoint"
        )
PYCHECK

if [[ "${ABFLOW_DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY RUN] no training launched."
  exit 0
fi

# Formal v104 path does not use the historical TopK watcher.  A watcher from
# a previous invocation may still be alive when resuming the same run, so stop
# that stale process before the DDP Trainer starts.  This changes only
# evaluation infrastructure; validation TopK checkpoint retention is untouched.
_stop_auto_topk_watcher "$RUN_DIR"

# Formal v104 path: Test is the Trainer's third epoch phase.  No watcher is
# spawned and no second model process competes with the training DDP job.
set +e
GPU="$GPU_ID" bash "$PROJECT_ROOT/scripts/train/train.sh" "$RUN_CONFIG"
TRAIN_STATUS=$?
set -e

exit "$TRAIN_STATUS"
