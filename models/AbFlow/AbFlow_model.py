#!/usr/bin/python
# -*- coding:utf-8 -*-
import hashlib
import math, time, os
import functools as fn
import numpy as np
from contextlib import nullcontext
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from torch_scatter import scatter_mean

from data.pdb_utils import VOCAB
from utils.nn_utils import SeparatedAminoAcidFeature, ProteinFeature
from utils.nn_utils import GMEdgeConstructor, SeperatedCoordNormalizer
from utils.nn_utils import _knn_edges
from evaluation.rmsd import kabsch_torch

from ..modules.am_enc import AMEncoder
from ..modules.am_egnn import AMEGNN
from .abflow_conditional_matcher import AbFlowConditionalMatcher
from .abflow_r3_matcher import AbFlowR3Matcher


# V211_EVIDENCE_DRIVEN_R28_PARENT: all three formal experiments use the attached
# source-faithful AbX single/pair/Seqformer -> native R05 EGNN integration.
# R29 adds Distogram to R28; R30 independently adds donor smooth-lDDT to R28.
# The donor operators are unchanged.  The integration boundary is an exact
# zero-start reparameterization of the parent R05 input/edge maps; optional
# modules are RNG-isolated so a child objective cannot silently change shared
# initialization.  Task localization happens only in authoritative masks.
# V211_GOLD_STANDARD_TASK_CONTRACT: preserve the configured design regions as
# an exact JSON-list -> CLI-list -> Dataset -> Model contract.  The model does
# not guess H3, repair a missing CDR, or silently clip the task. ``smask`` is
# checked against the configured paratope union; ``cmask`` remains the original
# template-coordinate initialization mask and is not repurposed as a sequence
# design mask.  Generated residues outside the task mask are immutable.
#
# V193_RUNTIME_CLOSURE: V190 + AbX checkpointing, framework-anchor antigen patch, DDP-safe empty-surface semantics, finite guards.
# Source-faithful AbX ResidueEmbedding+PairEmbedding+Seqformer is localized to
# original AM_E_GCL/MS_E_GCL native edge_attr; R05/U02/PCS-RC/physical recurrence
# remain the sole coordinate/flow authority. AbX recycling is deliberately absent.


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


def _normalise_task_regions(value):
    """Return a stable tuple of canonical task-region names.

    ``argparse(nargs='+')`` supplies lists, while the legacy JSON launcher
    expects scalar strings.  Keeping this normaliser at the model boundary
    makes the runtime contract independent of that representation detail.
    Malformed strings such as ``"['H3']"`` are deliberately *not* accepted.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = [value]

    regions = []
    for item in raw:
        name = str(item).strip().upper()
        if name and name not in regions:
            regions.append(name)
    return tuple(regions)


def _abx_memory_diag_enabled():
    return _env_flag("ABFLOW_ABX_MEMORY_DIAGNOSTICS", False)


def _abx_memory_diag_limit():
    return max(0, _env_int("ABFLOW_ABX_MEMORY_LOG_CALLS", 12))


def _abx_dist_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return 0


def _abx_diag_active(context):
    if not _abx_memory_diag_enabled() or context is None:
        return False
    return int(context.get("call", -1)) < _abx_memory_diag_limit()


def _abx_cuda_mem_string(device):
    if not torch.cuda.is_available():
        return "cuda=off"
    dev = torch.device(device)
    if dev.type != "cuda":
        return "cuda=off"
    gib = float(1024 ** 3)
    return (
        f"alloc={torch.cuda.memory_allocated(dev)/gib:.3f}GiB "
        f"reserved={torch.cuda.memory_reserved(dev)/gib:.3f}GiB "
        f"peak_alloc={torch.cuda.max_memory_allocated(dev)/gib:.3f}GiB "
        f"peak_reserved={torch.cuda.max_memory_reserved(dev)/gib:.3f}GiB"
    )


def _abx_mem_log(tag, context, tensor=None, extra=""):
    if not _abx_diag_active(context):
        return
    # Suppress only torch.checkpoint backward recomputation duplicates.
    if bool(context.get("checkpoint_block", False)) and torch.is_grad_enabled():
        return
    if tensor is not None and torch.is_tensor(tensor):
        device = tensor.device
    else:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    call = int(context.get("call", -1))
    round_idx = int(context.get("round", -1))
    suffix = f" {extra}" if extra else ""
    print(
        f"[AbXMemory] rank={_abx_dist_rank()} call={call} round={round_idx} "
        f"stage={tag} {_abx_cuda_mem_string(device)}{suffix}",
        flush=True,
    )



def _abx_finite_guard_enabled():
    return _env_flag("ABFLOW_NONFINITE_FAILFAST", True)


def _assert_finite_tensor(name, tensor, context=None):
    """Fail-fast diagnostic only; valid tensors are never modified.

    Finite checking has its own call budget so it is not accidentally disabled
    when verbose memory logging is shortened.  The formal V193 configs cover
    the first eight train steps (3 R05 rounds/step = 24 AbX calls).
    """
    if not _abx_finite_guard_enabled() or tensor is None:
        return
    if not torch.is_tensor(tensor) or not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
        return
    if context is not None:
        guard_calls = max(0, _env_int("ABFLOW_NONFINITE_GUARD_CALLS", 24))
        if int(context.get("call", -1)) >= guard_calls:
            return
    finite = torch.isfinite(tensor)
    if bool(finite.all().detach().cpu().item()):
        return
    with torch.no_grad():
        bad = int((~finite).sum().detach().cpu().item())
        total = int(tensor.numel())
        vals = tensor.detach().float()[finite]
        if vals.numel():
            vmin = float(vals.min().cpu().item())
            vmax = float(vals.max().cpu().item())
            vabs = float(vals.abs().max().cpu().item())
        else:
            vmin = vmax = vabs = float("nan")
    call = -1 if context is None else int(context.get("call", -1))
    rnd = -1 if context is None else int(context.get("round", -1))
    print(
        f"[NonFiniteTensor] rank={_abx_dist_rank()} call={call} round={rnd} "
        f"name={name} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"bad={bad}/{total} finite_min={vmin:.6g} finite_max={vmax:.6g} "
        f"finite_absmax={vabs:.6g}", flush=True
    )
    raise FloatingPointError(f"non-finite tensor detected at {name}")


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

class _ResidueConstants:
    restype_num = 20
    num_ab_regions = 14
    # AbFlow's fixed 14-slot convention starts N, CA, C, O, ... .  These are
    # exactly the backbone indices consumed by AbX pseudo_beta_fn_v2.
    atom_order = {'N': 0, 'CA': 1, 'C': 2}
residue_constants = _ResidueConstants()

# ---- exact AbX common_modules.py initialization/dropout/geometry semantics ----
class Linear_common(nn.Linear):
    def __init__(self, input_dim, output_dim, init, bias=True):
        super().__init__(input_dim, output_dim, bias=bias)
        assert init in ['gate', 'final', 'attn', 'relu', 'linear']
        if init in ['gate', 'final']:
            nn.init.constant_(self.weight, 0.)
        elif init == 'attn':
            torch.nn.init.xavier_uniform_(self.weight)
        elif init in ['relu', 'linear']:
            distribution_stddev = 0.87962566103423978
            scale = 2. if init == 'relu' else 1.
            stddev = np.sqrt(scale / input_dim) / distribution_stddev
            nn.init.trunc_normal_(self.weight, mean=0., std=stddev)
        else:
            raise NotImplementedError(f'{init} not Implemented')
        if bias:
            if init == 'gate':
                nn.init.constant_(self.bias, 1.)
            else:
                nn.init.constant_(self.bias, 0.)

def Linear(input_dim, output_dim, init, bias=True, config=None):
    assert init in ['gate', 'final', 'attn', 'relu', 'linear']
    return Linear_common(input_dim, output_dim, init, bias)

def apply_dropout(tensor, rate, is_training, broadcast_dim=None):
    if is_training and rate > 0.0:
        if broadcast_dim is not None:
            shape = list(tensor.shape)
            shape[broadcast_dim] = 1
            with torch.no_grad():
                scale = 1. / (1. - rate)
                keep_rate = torch.full(shape, 1. - rate, dtype=tensor.dtype, device=tensor.device)
                keep = torch.bernoulli(keep_rate)
            return scale * keep * tensor
        else:
            return F.dropout(tensor, rate)
    else:
        return tensor

def pseudo_beta_fn_v2(aatype, all_atom_positions, all_atom_masks=None):
    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']
    N = all_atom_positions[..., n_idx, :]
    CA = all_atom_positions[..., ca_idx, :]
    C = all_atom_positions[..., c_idx, :]
    b = CA - N
    c = C - CA
    a = torch.cross(b, c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    if all_atom_masks is not None:
        CB_mask = torch.all(torch.stack([
            all_atom_masks[...,n_idx], all_atom_masks[...,ca_idx], all_atom_masks[...,c_idx]
        ], dim=-1), dim=-1)
        return CB, CB_mask
    return CB

def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    breaks = torch.linspace(min_bin, max_bin, steps=num_bins-1, device=positions.device)
    sq_breaks = torch.square(breaks)
    dist2 = torch.sum(torch.square(
        rearrange(positions, 'b l c -> b l () c') -
        rearrange(positions, 'b l c -> b () l c')), dim=-1, keepdims=True)
    true_bins = torch.sum(dist2 > sq_breaks, axis=-1).long()
    return true_bins


RESTYPES = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
RESTYPE_1TO3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
ATOM14_NAMES = {
'ALA':['N','CA','C','O','CB','','','','','','','','',''],
'ARG':['N','CA','C','O','CB','CG','CD','NE','CZ','NH1','NH2','','',''],
'ASN':['N','CA','C','O','CB','CG','OD1','ND2','','','','','',''],
'ASP':['N','CA','C','O','CB','CG','OD1','OD2','','','','','',''],
'CYS':['N','CA','C','O','CB','SG','','','','','','','',''],
'GLN':['N','CA','C','O','CB','CG','CD','OE1','NE2','','','','',''],
'GLU':['N','CA','C','O','CB','CG','CD','OE1','OE2','','','','',''],
'GLY':['N','CA','C','O','','','','','','','','','',''],
'HIS':['N','CA','C','O','CB','CG','ND1','CD2','CE1','NE2','','','',''],
'ILE':['N','CA','C','O','CB','CG1','CG2','CD1','','','','','',''],
'LEU':['N','CA','C','O','CB','CG','CD1','CD2','','','','','',''],
'LYS':['N','CA','C','O','CB','CG','CD','CE','NZ','','','','',''],
'MET':['N','CA','C','O','CB','CG','SD','CE','','','','','',''],
'PHE':['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','','',''],
'PRO':['N','CA','C','O','CB','CG','CD','','','','','','',''],
'SER':['N','CA','C','O','CB','OG','','','','','','','',''],
'THR':['N','CA','C','O','CB','OG1','CG2','','','','','','',''],
'TRP':['N','CA','C','O','CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],
'TYR':['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','OH','',''],
'VAL':['N','CA','C','O','CB','CG1','CG2','','','','','','',''],
'UNK':['','','','','','','','','','','','','','']}
CHI_ATOMS = {
'ALA':[], 'ARG':[['N','CA','CB','CG'],['CA','CB','CG','CD'],['CB','CG','CD','NE'],['CG','CD','NE','CZ']],
'ASN':[['N','CA','CB','CG'],['CA','CB','CG','OD1']], 'ASP':[['N','CA','CB','CG'],['CA','CB','CG','OD1']],
'CYS':[['N','CA','CB','SG']], 'GLN':[['N','CA','CB','CG'],['CA','CB','CG','CD'],['CB','CG','CD','OE1']],
'GLU':[['N','CA','CB','CG'],['CA','CB','CG','CD'],['CB','CG','CD','OE1']], 'GLY':[],
'HIS':[['N','CA','CB','CG'],['CA','CB','CG','ND1']], 'ILE':[['N','CA','CB','CG1'],['CA','CB','CG1','CD1']],
'LEU':[['N','CA','CB','CG'],['CA','CB','CG','CD1']], 'LYS':[['N','CA','CB','CG'],['CA','CB','CG','CD'],['CB','CG','CD','CE'],['CG','CD','CE','NZ']],
'MET':[['N','CA','CB','CG'],['CA','CB','CG','SD'],['CB','CG','SD','CE']], 'PHE':[['N','CA','CB','CG'],['CA','CB','CG','CD1']],
'PRO':[['N','CA','CB','CG'],['CA','CB','CG','CD']], 'SER':[['N','CA','CB','OG']], 'THR':[['N','CA','CB','OG1']],
'TRP':[['N','CA','CB','CG'],['CA','CB','CG','CD1']], 'TYR':[['N','CA','CB','CG'],['CA','CB','CG','CD1']], 'VAL':[['N','CA','CB','CG1']]}
ATOM14_INDEX = {r:{a:i for i,a in enumerate(names) if a} for r,names in ATOM14_NAMES.items()}
ATOM14_MASK_TABLE = torch.tensor([[1.0 if a else 0.0 for a in ATOM14_NAMES[RESTYPE_1TO3[r]]] for r in RESTYPES] + [[0.0]*14])

# V202_JSON_BATCH_MEMORY_CLOSURE
# V201_LOCALIZED_WIDTH_CONTRACT_CLOSURE
# - localized/donor width closure retained from V201
# - triangle chunk is runtime-configured (JSON runtime_env), not hard-coded
# - batch size remains a launcher/config concern; model code never owns batch size
# Fixes the V200 auxiliary-head width leak: every AbX consumer now derives its
# input width from the active trunk config (localized or donor), rather than
# silently retaining donor defaults. Scientific state / losses are unchanged.
# V200_PERSISTENT_PAIR_LOCALIZED_RUNTIME
# V199_ABX_OBSERVED_ATOM_MASK_FORMAL_CLOSURE
# AbX's own l2_normalize uses sqrt(sum(square(v)) + epsilon).  Our online
# atom14->torsion adapter is differentiable w.r.t. the recurrent R05 state, so
# the epsilon must be inside sqrt; sqrt(r2).clamp_min(eps) still executes the
# singular SqrtBackward0 at r2==0 before clamp can protect the gradient.
_TORSION_NORM_AUDIT_COUNT = 0

def _dihedral_sin_cos(p0,p1,p2,p3,epsilon=None,diag_tag=None):
    global _TORSION_NORM_AUDIT_COUNT
    if epsilon is None:
        epsilon = _env_float("ABFLOW_TORSION_NORM_EPS", 1e-12)
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError(f"ABFLOW_TORSION_NORM_EPS must be > 0, got {epsilon}")

    # Keep the geometric construction in FP32 even under BF16 autocast.  This
    # mirrors the fact that donor AbX torsions are precomputed numeric features
    # while preserving the gradient from current-state geometry in our adapter.
    with torch.cuda.amp.autocast(enabled=False):
        p0f,p1f,p2f,p3f = p0.float(),p1.float(),p2.float(),p3.float()
        b0 = p0f-p1f; b1=p2f-p1f; b2=p3f-p2f
        b1_sq = (b1*b1).sum(-1,keepdim=True)
        b1n = b1 / torch.sqrt(b1_sq + epsilon)
        v = b0 - (b0*b1n).sum(-1,keepdim=True)*b1n
        w = b2 - (b2*b1n).sum(-1,keepdim=True)*b1n
        x = (v*w).sum(-1)
        y = (torch.cross(b1n,v,dim=-1)*w).sum(-1)
        xy_sq = x*x+y*y
        n = torch.sqrt(xy_sq + epsilon)
        out = torch.stack([y/n,x/n],dim=-1)

    if _env_flag("ABFLOW_TORSION_DIAGNOSTICS", False):
        limit = max(0, _env_int("ABFLOW_TORSION_LOG_CALLS", 16))
        axis_bad = bool((b1_sq.detach() <= epsilon).any().cpu().item())
        plane_bad = bool((xy_sq.detach() <= epsilon).any().cpu().item())
        if (axis_bad or plane_bad) and _TORSION_NORM_AUDIT_COUNT < limit:
            call = _TORSION_NORM_AUDIT_COUNT
            _TORSION_NORM_AUDIT_COUNT += 1
            axis_min = float(b1_sq.detach().min().cpu().item())
            plane_min = float(xy_sq.detach().min().cpu().item())
            print(
                f"[TorsionNormAudit] rank={_abx_dist_rank()} call={call} "
                f"tag={diag_tag or 'unknown'} axis_sq_min={axis_min:.3e} "
                f"plane_sq_min={plane_min:.3e} eps={epsilon:.1e} "
                f"axis_degenerate={int(axis_bad)} plane_degenerate={int(plane_bad)}"
            )
    return out.to(dtype=p0.dtype) if p0.dtype in (torch.float16, torch.bfloat16) else out

def _atom14_chemical_mask(seq, coords):
    safe = seq.long().clamp(min=0, max=20)
    table = ATOM14_MASK_TABLE.to(device=coords.device, dtype=coords.dtype)
    chem = table[safe]
    finite = torch.isfinite(coords).all(dim=-1).to(coords.dtype)
    return chem * finite

def _abflow_ca_fill_observed_mask(seq, coords):
    """Infer AbFlow's resolved-atom mask from its CA-fill sentinel.

    AbFlow preprocessing initializes every atom slot to the residue CA and only
    overwrites slots that are actually resolved.  This fallback is used at
    inference by legacy generate.py paths that historically removed xloss_mask.
    Training/validation use the explicit xloss_mask and audit this fallback.
    """
    chem = _atom14_chemical_mask(seq, coords).bool()
    if coords.shape[-2] < 2:
        return chem.to(dtype=coords.dtype)
    ca = coords[..., 1:2, :]
    d2 = ((coords - ca) ** 2).sum(dim=-1)
    # A real non-CA atom cannot physically coincide with CA.  Use a tiny FP32
    # tolerance only to survive centering/scaling roundoff. CA itself is valid
    # whenever chemically present and finite.
    tol2 = float(_env_float("ABFLOW_ABX_CA_FILL_TOL2", 1e-12))
    observed = d2 > tol2
    observed[..., 1] = True
    return (chem & observed).to(dtype=coords.dtype)

def _atom14_exists_from_seq(seq, coords, observed_mask=None):
    chem = _atom14_chemical_mask(seq, coords)
    if observed_mask is None:
        observed = _abflow_ca_fill_observed_mask(seq, coords)
    else:
        if tuple(observed_mask.shape) != tuple(coords.shape[:-1]):
            raise ValueError(
                f"atom observed-mask shape mismatch: {tuple(observed_mask.shape)} "
                f"vs coords {tuple(coords.shape)}"
            )
        observed = observed_mask.to(device=coords.device, dtype=coords.dtype)
    return chem * observed

# V200_VECTOR_TORSION_TABLES: exact atom-index lookup, no Python/GPU sync loop.
_CHI_INDEX_ROWS = []
_CHI_VALID_ROWS = []
for _aa1 in RESTYPES:
    _res3 = RESTYPE_1TO3[_aa1]
    _idxmap = ATOM14_INDEX[_res3]
    _idx_row, _valid_row = [], []
    for _chi_i in range(4):
        if _chi_i < len(CHI_ATOMS[_res3]):
            _names = CHI_ATOMS[_res3][_chi_i]
            _ok = all(_n in _idxmap for _n in _names)
            _idx_row.append([_idxmap[_n] if _ok else 0 for _n in _names])
            _valid_row.append(_ok)
        else:
            _idx_row.append([0, 0, 0, 0])
            _valid_row.append(False)
    _CHI_INDEX_ROWS.append(_idx_row)
    _CHI_VALID_ROWS.append(_valid_row)
# unknown/MASK row
_CHI_INDEX_ROWS.append([[0, 0, 0, 0] for _ in range(4)])
_CHI_VALID_ROWS.append([False] * 4)
CHI_ATOM_INDEX_TABLE = torch.tensor(_CHI_INDEX_ROWS, dtype=torch.long)
CHI_VALID_TABLE = torch.tensor(_CHI_VALID_ROWS, dtype=torch.bool)


def _torsions_from_atom14(seq, coords, chain_id, mask, atom_exists=None):
    """Vectorized exact AbX 7-torsion construction for padded atom14 tensors.

    This is algebraically the same pre-omega/phi/psi/chi1..4 construction as
    V199.  It removes per-residue ``.item()/bool`` CUDA synchronizations only;
    invalid torsions retain the donor default [sin,cos]=[0,1].
    """
    B, L = seq.shape
    out = coords.new_zeros((B, L, 7, 2))
    out[..., 1] = 1.0
    exists = (
        _atom14_exists_from_seq(seq, coords).bool()
        if atom_exists is None
        else atom_exists.to(device=coords.device).bool()
    )
    valid_res = mask.bool()

    # Neighbor validity without wrap-around leakage.
    prev_mask = torch.roll(valid_res, shifts=1, dims=1)
    next_mask = torch.roll(valid_res, shifts=-1, dims=1)
    prev_chain = torch.roll(chain_id, shifts=1, dims=1)
    next_chain = torch.roll(chain_id, shifts=-1, dims=1)
    same_prev = valid_res & prev_mask & (chain_id == prev_chain)
    same_next = valid_res & next_mask & (chain_id == next_chain)
    if L:
        same_prev[:, 0] = False
        same_next[:, -1] = False

    prev_coords = torch.roll(coords, shifts=1, dims=1)
    next_coords = torch.roll(coords, shifts=-1, dims=1)
    prev_exists = torch.roll(exists, shifts=1, dims=1)
    next_exists = torch.roll(exists, shifts=-1, dims=1)

    pre_ok = same_prev & prev_exists[..., 1] & prev_exists[..., 2] & exists[..., 0] & exists[..., 1]
    phi_ok = same_prev & prev_exists[..., 2] & exists[..., 0] & exists[..., 1] & exists[..., 2]
    psi_ok = same_next & exists[..., 0] & exists[..., 1] & exists[..., 2] & next_exists[..., 0]

    pre = _dihedral_sin_cos(prev_coords[..., 1, :], prev_coords[..., 2, :], coords[..., 0, :], coords[..., 1, :], diag_tag="vector:pre_omega")
    phi = _dihedral_sin_cos(prev_coords[..., 2, :], coords[..., 0, :], coords[..., 1, :], coords[..., 2, :], diag_tag="vector:phi")
    psi = _dihedral_sin_cos(coords[..., 0, :], coords[..., 1, :], coords[..., 2, :], next_coords[..., 0, :], diag_tag="vector:psi")
    out[..., 0, :] = torch.where(pre_ok[..., None], pre, out[..., 0, :])
    out[..., 1, :] = torch.where(phi_ok[..., None], phi, out[..., 1, :])
    out[..., 2, :] = torch.where(psi_ok[..., None], psi, out[..., 2, :])

    # Four chi torsions from a residue-type lookup table.
    safe_seq = seq.long().clamp(min=0, max=20)
    chi_idx_table = CHI_ATOM_INDEX_TABLE.to(device=coords.device)
    chi_valid_table = CHI_VALID_TABLE.to(device=coords.device)
    chi_idx = chi_idx_table[safe_seq]                # [B,L,4,4]
    chi_type_ok = chi_valid_table[safe_seq] & valid_res[..., None]

    coords4 = coords[:, :, None, :, :].expand(B, L, 4, 14, 3)
    gather_idx = chi_idx[..., None].expand(B, L, 4, 4, 3)
    chi_pts = torch.gather(coords4, dim=3, index=gather_idx)  # [B,L,4,4,3]

    exists4 = exists[:, :, None, :].expand(B, L, 4, 14)
    chi_exists = torch.gather(exists4, dim=3, index=chi_idx)
    chi_ok = chi_type_ok & chi_exists.all(dim=-1)
    chi = _dihedral_sin_cos(
        chi_pts[..., 0, :], chi_pts[..., 1, :],
        chi_pts[..., 2, :], chi_pts[..., 3, :],
        diag_tag="vector:chi",
    )
    out[..., 3:7, :] = torch.where(
        chi_ok[..., None], chi, out[..., 3:7, :]
    )
    return out

class _Cfg(dict):
    """OmegaConf-like minimal attribute/dict compatibility used only by the local donor port."""
    def __getattr__(self, key):
        try: return self[key]
        except KeyError as exc: raise AttributeError(key) from exc
    def __setattr__(self, key, value): self[key] = value
    @classmethod
    def from_dict(cls,d):
        return cls({k:(cls.from_dict(v) if isinstance(v,dict) else v) for k,v in d.items()})

def abx_v4_l3_seqformer_config(recycle_features=False,recycle_pos=False):
    """AbX operators with an AbFlow-localized width profile.

    V200 keeps the donor operator graph (ResidueEmbedding, PairEmbedding,
    SeqAttentionWithPairBias, OPM, triangle multiplication/attention and pair
    transition) but does not treat donor numeric widths as scientific state.
    The original AbX configuration was designed together with ESM/recycling;
    this project deliberately has neither.  The localized profile therefore
    matches the representation scale to the 128-d R05 backbone while retaining
    every operator and the same pair semantics.
    """
    profile = _env_str("ABFLOW_ABX_WIDTH_PROFILE", "localized").lower()
    if profile == "donor":
        seq_channel, pair_channel, index_embed = 512, 128, 32
        seq_heads, outer_channel, tri_hidden, tri_heads = 32, 64, 128, 4
    elif profile == "localized":
        seq_channel, pair_channel, index_embed = 256, 64, 16
        seq_heads, outer_channel, tri_hidden, tri_heads = 8, 32, 64, 4
    else:
        raise ValueError(
            "ABFLOW_ABX_WIDTH_PROFILE must be 'localized' or 'donor', "
            f"got {profile!r}"
        )
    if (seq_channel + index_embed) % seq_heads != 0:
        raise ValueError("localized AbX single width must be divisible by seq heads")
    if (pair_channel + 2 * index_embed) % tri_heads != 0:
        raise ValueError("localized AbX pair width must be divisible by triangle heads")
    return _Cfg.from_dict({
      'width_profile':profile,
      'seqformer_num_block':1,'seq_channel':seq_channel,'pair_channel':pair_channel,
      'max_relative_feature':32,'index_embed_size':index_embed,
      'recycle_features':bool(recycle_features),'recycle_pos':bool(recycle_pos),
      'prev_pos':{'min_bin':3.375,'num_bins':15,'max_bin':21.375},
      'seqformer':{
        'seq_attention_with_pair_bias':{'orientation':'per_row','num_head':seq_heads,'inp_kernels':[],'dropout_rate':0.1,'shared_dropout':True},
        'seq_transition':{'orientation':'per_row','num_intermediate_factor':4,'dropout_rate':0.0,'shared_dropout':True},
        'outer_product_mean':{'orientation':'per_row','num_outer_channel':outer_channel,'dropout_rate':0.0,'shared_dropout':True},
        'triangle_multiplication_outgoing':{'orientation':'per_row','num_intermediate_channel':tri_hidden,'gating':True,'num_head':tri_heads,'inp_kernels':[],'dropout_rate':0.1,'shared_dropout':False},
        'triangle_multiplication_incoming':{'orientation':'per_column','num_intermediate_channel':tri_hidden,'gating':True,'num_head':tri_heads,'inp_kernels':[],'dropout_rate':0.1,'shared_dropout':False},
        'triangle_attention_starting_node':{'orientation':'per_row','num_head':tri_heads,'gating':True,'inp_kernels':[],'dropout_rate':0.1,'shared_dropout':False},
        'triangle_attention_ending_node':{'orientation':'per_column','num_head':tri_heads,'gating':True,'inp_kernels':[],'dropout_rate':0.1,'shared_dropout':False},
        'pair_transition':{'orientation':'per_row','num_intermediate_factor':4,'dropout_rate':0.0,'shared_dropout':True},
      }
    })


# ===== AbX donor ResidueEmbedding / PairEmbedding: source-faithful local port =====
class ResidueEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = config.seq_channel
        self.max_aa_types = residue_constants.restype_num
        self.aatype_embed = nn.Embedding(self.max_aa_types+3, feat_dim)
        self.cdr_embed = nn.Embedding(residue_constants.num_ab_regions+1, feat_dim)
        
        self.coordinate_embed = nn.Sequential(
            Linear(14*3 + 7*2, feat_dim, init='linear', bias=True), 
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),             
        )

        infeat_dim = feat_dim * 3 + 2
        self.mlp = nn.Sequential(
            Linear(infeat_dim, feat_dim*2, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim*2, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
        )

    def forward(self, batch, seq, atom14_positions, angles_sin_cos):
        """
        Args:
            aa:         (N, L).
            residx:     (N, L).
            chain_nb:   (N, L).
            pos_atoms:  (N, L, A, 3).
            mask_atoms: (N, L, A).
            fragment_type:  (N, L).
        """
        mask, fixed_mask = batch['mask'], batch['fixed_mask']
        mask = torch.logical_and(mask, fixed_mask)
        N, L = mask.shape

        # Amino acid, Chain id, residue number and cdr definition
        aa, chain_ids, residx, cdr_def, coords, torsion_angle = seq, batch['chain_id'], batch['residx'], batch['cdr_def'], atom14_positions, angles_sin_cos

        aa_feat = self.aatype_embed(aa.long()) # (N, L, feat)
        aa_feat = aa_feat * mask[:, :, None]
        cdr_feat = self.cdr_embed(cdr_def)
        # Coordinates and torsion angles
        coord_feat = self.coordinate_embed(torch.cat((coords.reshape(N, L, -1), torsion_angle.reshape(N, L, -1)), dim=-1))
        out_feat = self.mlp(torch.cat([aa_feat, chain_ids[..., None], residx[..., None], cdr_feat, coord_feat], dim=-1)) # (N, L, F)

        out_feat = out_feat * mask[:, :, None]
        return out_feat



class PairEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = config.pair_channel
        self.dgram_config = config.prev_pos
        self.num_bins = self.dgram_config.num_bins
        self.max_num_atoms = 14
        self.max_aa_types = residue_constants.restype_num + 3
        self.max_relpos = 32
        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2*self.max_relpos+1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, self.max_num_atoms*self.max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            Linear(self.max_num_atoms*self.max_num_atoms, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
            nn.ReLU(),
        )

        self.dgram_embed = nn.Embedding(self.num_bins, feat_dim)

        infeat_dim = feat_dim * 4
        self.out_mlp = nn.Sequential(
            Linear(infeat_dim, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
        )

    def forward(self, batch, seq, atom14_positions, atom14_gt_exists):
        """
        Args:
            aa: (N, L).
            residx: (N, L).
            chain_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            structure_mask: (N, L)
            sequence_mask:  (N, L), mask out unknown amino acids to generate.

        Returns:
            (N, L, L, feat_dim)
        """

        mask, fixed_mask = batch['mask'], batch['fixed_mask']
        mask = torch.logical_and(mask, fixed_mask)
        mask_pair = mask[:, :, None] * mask[:, None, :]
        N, L = mask.shape

        # Amino acid, Chain id, residue number 
        aa, chain_ids, residx, coords, coords_mask = seq, batch['chain_id'], batch['residx'], atom14_positions, atom14_gt_exists
        mask_atoms = coords_mask[..., residue_constants.atom_order['CA']]

        aa_pair = aa[:,:,None]*self.max_aa_types + aa[:,None,:]    # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair.long())
    
        # Relative sequential positions
        same_chain = (chain_ids[:, :, None] == chain_ids[:, None, :])
        relpos = torch.clamp(
            residx[:,:,None] - residx[:,None,:], 
            min=-self.max_relpos, max=self.max_relpos,
        )   # (N, L, L)
        feat_relpos = self.relpos_embed(relpos + self.max_relpos) * same_chain[:,:,:,None]

        # V200_EXACT_PAIR_DISTANCE_KERNEL
        # Exact same 14x14 Euclidean distances as the donor broadcast equation,
        # but without materializing [B,L,L,14,14,3].  Flattening residue atoms
        # and using cdist preserves the mathematical pair feature and autograd;
        # only the execution kernel / temporary-memory footprint changes.
        flat_coords = coords.reshape(N, L * self.max_num_atoms, 3).float()
        distance = torch.cdist(
            flat_coords, flat_coords, p=2,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        distance = distance.reshape(
            N, L, self.max_num_atoms, L, self.max_num_atoms
        ).permute(0, 1, 3, 2, 4).contiguous()
        distance = (distance / 10.0).reshape(N, L, L, -1).to(coords.dtype)
        distance_coef = F.softplus(self.aapair_to_distcoef(aa_pair.long()))    # (N, L, L, A*A)
        d_gauss = torch.exp(-1 * distance_coef * distance**2)

        mask_atom_pair = mask_atoms[:,:,None,None]*mask_atoms[:,None,:,None]
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)

        # Dgram
        peusdo_beta = pseudo_beta_fn_v2(aa, coords)
        disto_bins = dgram_from_positions(peusdo_beta, **self.dgram_config)

        feat_dgram = self.dgram_embed(disto_bins)

        
        # All
        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dgram], dim=-1)
        feat_all = self.out_mlp(feat_all)   # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all


# ===== AbX donor timestep embedding =====
def pair_concat(pair_1, pair_2):
    assert pair_1.shape[0] == pair_2.shape[0] and pair_1.shape[-1] == pair_2.shape[-1]
    assert pair_1.device == pair_2.device
    device = pair_1.device
    batch_size = pair_1.shape[0]
    channel = pair_1.shape[-1]

    length_1 = pair_1.shape[1]
    length_2 = pair_2.shape[1]
    concat_dim1 = torch.cat(
        (
        pair_1, 
        torch.zeros((batch_size, length_2, length_1, channel), device=device)
        ), dim=1)
    
    concat_dim2 = torch.cat(
        (
        torch.zeros((batch_size, length_1, length_2, channel), device=device), 
        pair_2
        ), dim=1)
    pair_all = torch.cat([concat_dim1, concat_dim2], dim=2)
    return pair_all



def abx_get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    """
    From Fairseq.Build sinusoidal embeddings.This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: # Zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Embedder(nn.Module):
    """
    A module for encoding diffusion timesteps and embedding them into sequence and pair representations.
    
    Returns:
        node_embed: [B, N,  seq_channel + index_embed_size]
        edge_embed: [B, N, N, pair_channel + 2*index_embed_size]
    """
    def __init__(self, model_conf):
        super(Embedder,self).__init__()
        self._embed_conf = model_conf
        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        
        self.timestep_embedder = fn.partial(
            abx_get_timestep_embedding,
            embedding_dim = index_embed_size
        )


    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])


    def forward(self, seq_act, pair_act, batch):
        num_batch, num_res = batch['seq'].shape
        # node_feats = []
        t = batch['t'] 
        fixed_mask = batch['fixed_mask']

        # Embedding times t
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1,num_res,1)
        ) # [B, N, index_embed_size]
        #import ipdb; ipdb.set_trace()
        # Evoformer embedding
        pair_act = pair_act.reshape([num_batch, num_res**2, -1])

        seq_feats = [seq_act]
        pair_feats = [pair_act]
        seq_feats.append(prot_t_embed)
        pair_feats.append(self._cross_concat(prot_t_embed, num_batch, num_res))

        
        pair_feats = torch.cat(pair_feats, dim=-1).float()
        seq_feats = torch.cat(seq_feats, dim=-1).float()
        pair_feats = pair_feats.reshape([num_batch, num_res, num_res, -1])

        return seq_feats, pair_feats
    



# ===== AbX donor Seqformer operators: source-faithful local port =====
class Attention(nn.Module):
    def __init__(self, input_dim, key_dim, value_dim, output_dim, num_head,split_first=True, gating=True,inp_kernels=None, config=None):
        super().__init__()
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0

        self.key_dim, self.value_dim = key_dim, value_dim

        self.num_head = num_head
        
        self.split_first = split_first

        if self.split_first:
            self.proj_q = Linear(input_dim, key_dim, init='attn', bias=False, config=config)
            self.proj_k = Linear(input_dim, key_dim, init='attn', bias=False, config=config)
            self.proj_v = Linear(input_dim, value_dim, init='attn', bias=False, config=config)
        else:
            assert (key_dim == value_dim)
            self.proj_in = Linear(input_dim, key_dim * 3, init='attn', bias=False, config=config)
        
        self.gating = gating
        if gating:
            self.gate= Linear(input_dim, value_dim, init='gate', config=config)

        self.proj_out = Linear(value_dim, output_dim, init='final', config=config)
         
        self.inp_kernels = inp_kernels
        self._diag_name = None
        self._diag_context = None
        if inp_kernels:
            self.inp_q = SpatialDepthWiseInception(key_dim // num_head, inp_kernels)
            self.inp_k = SpatialDepthWiseInception(key_dim // num_head, inp_kernels)
            self.inp_v = SpatialDepthWiseInception(value_dim // num_head, inp_kernels)

    def forward(self, q_data, k_data=None, bias=None, k_mask=None):
        """
        Arguments:
            q_data: (batch_size, N_seqs, N_queries, q_channel)
            k_data: (batch_size, N_seqs, N_keys, k_channel)
            k_mask: (batch_size, N_seqs, N_keys)
            bias  : (batch_size, N_queries, N_keys). shared by all seqs
        Returns:
            (b s l c)
        """
        key_dim, value_dim = self.key_dim // self.num_head, self.value_dim // self.num_head
        
        if self.split_first:
            assert (k_data is not None)
            q = self.proj_q(q_data) 
            k = self.proj_k(k_data)
            v = self.proj_v(k_data)
            q, k, v = map(lambda t: rearrange(t, 'b s l (h d) -> b s h l d', h = self.num_head), (q, k, v))
        else:
            assert (k_data is None)
            t = rearrange(self.proj_in(q_data), "... l (h d) -> ... h l d", h=self.num_head)
            q, k, v = torch.chunk(t, 3, dim=-1)
        
        if self.inp_kernels:
            q, k, v = map(lambda t: rearrange(t, 'b s h l d-> b (s h) l d'), (q, k, v))
            q = self.inp_q(q)
            k = self.inp_k(k)
            v = self.inp_v(v)
            q, k, v = map(lambda t: rearrange(t, 'b (s h) l d-> b s h l d', h = self.num_head), (q, k, v))
        
        q = q* key_dim**(-0.5)

        if self._diag_name is not None and _abx_diag_active(self._diag_context):
            logits_shape = tuple(q.shape[:-2]) + (q.shape[-2], k.shape[-2])
            logits_numel = 1
            for dim in logits_shape:
                logits_numel *= int(dim)
            logits_gib = (
                logits_numel * int(q.element_size()) / float(1024 ** 3)
            )
            _abx_mem_log(
                f"{self._diag_name}:pre_logits",
                self._diag_context,
                q,
                extra=(
                    f"q={tuple(q.shape)}/{q.dtype} "
                    f"k={tuple(k.shape)}/{k.dtype} "
                    f"logits={logits_shape} est={logits_gib:.3f}GiB"
                ),
            )

        logits = torch.einsum('... h q d, ... h k d -> ... h q k', q, k)

        if self._diag_name is not None and _abx_diag_active(self._diag_context):
            _abx_mem_log(
                f"{self._diag_name}:post_logits",
                self._diag_context,
                logits,
                extra=(
                    f"logits={tuple(logits.shape)}/{logits.dtype} "
                    f"tensor={logits.numel()*logits.element_size()/float(1024**3):.3f}GiB"
                ),
            )

        if bias is not None:
            logits = logits + rearrange(bias,  'b h q k -> b () h q k')

        if k_mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            k_mask = rearrange(k_mask, 'b s k -> b s () () k')
            logits = logits.masked_fill(~k_mask.bool(), mask_value)

        weights = F.softmax(logits, dim = -1)
        if self._diag_name is not None and _abx_diag_active(self._diag_context):
            _abx_mem_log(
                f"{self._diag_name}:post_softmax",
                self._diag_context,
                weights,
                extra=f"weights={tuple(weights.shape)}/{weights.dtype}",
            )
        weighted_avg = torch.einsum('b s h q k, b s h k d -> b s h q d', weights, v)
        weighted_avg = rearrange(weighted_avg, 'b s h q d -> b s q (h d)')
        
        if self.gating:
            gate_values = torch.sigmoid(self.gate(q_data))
            weighted_avg = weighted_avg * gate_values

        output = self.proj_out(weighted_avg)

        return output

class SeqAttentionWithPairBias(nn.Module):
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel):
        super().__init__()
        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.seq_norm = LayerNorm(num_in_seq_channel)
        self.pair_norm = LayerNorm(num_in_pair_channel)
        self.proj_pair = Linear(num_in_pair_channel, c.num_head, init='linear', bias = False, config=LoRA_conf)

        self.attn = Attention(
                input_dim=num_in_seq_channel,
                key_dim=num_in_seq_channel,
                value_dim=num_in_seq_channel,
                output_dim=num_in_seq_channel,
                num_head=c.num_head,
                split_first=False,
                inp_kernels=c.inp_kernels,
                config=LoRA_conf)

        self.config = config

    def forward(self, seq_act, pair_act, mask):
        """
        Arguments:
            seq_act: (b l c)
            pair_act: (b l l c)
            mask: (b l), padding mask
        Returns:
            (b l c)
        """
        mask = rearrange(mask, 'b l -> b () l')
        seq_act = self.seq_norm(seq_act)
        
        pair_act = self.pair_norm(pair_act)
        bias = rearrange(self.proj_pair(pair_act), 'b i j h -> b h i j')
        
        seq_act = rearrange(seq_act, 'b l c -> b () l c')
        seq_act = self.attn(q_data=seq_act, bias=bias, k_mask=mask)
        seq_act = rearrange(seq_act, 'b s l c -> (b s) l c')
        return seq_act

class Transition(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()

        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        intermediate_channel = num_in_channel * c.num_intermediate_factor
        self.transition = nn.Sequential(
                LayerNorm(num_in_channel),
                Linear(num_in_channel, intermediate_channel, init='linear', config=LoRA_conf),
                nn.ReLU(),
                Linear(intermediate_channel, num_in_channel, init='final', config=LoRA_conf),
                )

    def forward(self, act, mask):
        return self.transition(act)

# AF2 and ESM-FOLD have different implementations
# Here we just follow ESMFOLD
class OuterProductMean(nn.Module):
    def __init__(self, config, num_in_channel, num_out_channel):
        super().__init__()

        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.norm = LayerNorm(num_in_channel)
        self.left_proj = Linear(num_in_channel, c.num_outer_channel, init='linear', config=LoRA_conf)
        self.right_proj = Linear(num_in_channel, c.num_outer_channel, init='linear', config=LoRA_conf)

        self.out_proj = Linear(2 * c.num_outer_channel, num_out_channel, init='final', config=LoRA_conf)

    def forward(self, act, mask):
        """
        act: (b l c)
        mask: (b l)
        """
        mask = rearrange(mask, 'b l -> b l ()')
        act = self.norm(act)
        left_act = mask * self.left_proj(act)
        right_act = mask * self.right_proj(act)
        
        prod = left_act[:, None, :, :] * right_act[:, :, None, :]
        diff = left_act[:, None, :, :] - right_act[:, :, None, :]

        act = torch.cat([prod, diff], dim=-1)
        act = self.out_proj(act)

        return act

class TriangleMultiplication(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config
        assert c.orientation in ['per_row', 'per_column']
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.norm = LayerNorm(num_in_channel)

        self.left_proj = Linear(num_in_channel, c.num_intermediate_channel, init='linear', config=LoRA_conf)
        self.right_proj = Linear(num_in_channel, c.num_intermediate_channel, init='linear', config=LoRA_conf)

        self.final_norm = LayerNorm(c.num_intermediate_channel)
        
        if c.gating:
            self.left_gate = Linear(num_in_channel, c.num_intermediate_channel, init='gate', config=LoRA_conf)
            self.right_gate = Linear(num_in_channel, c.num_intermediate_channel, init='gate', config=LoRA_conf)
            self.final_gate = Linear(num_in_channel, num_in_channel, init='gate', config=LoRA_conf)
        
        self.proj_out = Linear(c.num_intermediate_channel, num_in_channel, init='final', config=LoRA_conf)

        
        if c.inp_kernels:
            self.inp_left = SpatialDepthWiseInception(c.num_intermediate_channel // c.num_head, c.inp_kernels)
            self.inp_right = SpatialDepthWiseInception(c.num_intermediate_channel // c.num_head, c.inp_kernels)

        self.config = c

    def forward(self, act, mask):
        """
        act: (b l l c)
        mask: (b l)
        """
        c = self.config

        #pair_mask = rearrange(mask, 'b l -> b l () ()') * rearrange(mask, 'b l -> b () l ()')
        pair_mask = mask[:,:,None,None] * mask[:,None,:,None]
        
        act = self.norm(act)

        input_act = act

        left_proj_act = self.left_proj(act)
        right_proj_act = self.right_proj(act)
        
        if c.inp_kernels:
            if c.orientation == 'per_row':
                equation = 'b i j (h d) -> b (i h) j d'
            else:
                equation = 'b i j (h d) -> b (j h) i d'

            left_proj_act, right_proj_act = map(
                    lambda t: rearrange(t, equation, h = c.num_head), (left_proj_act, right_proj_act))

            left_proj_act = self.inp_left(left_proj_act)
            right_proj_act = self.inp_right(right_proj_act)
            
            if c.orientation == 'per_row':
                equation = 'b (i h) j d -> b i j (h d)'
            else:
                equation = 'b (j h) i d -> b i j (h d)'
            
            left_proj_act, right_proj_act = map(
                    lambda t: rearrange(t, equation, h = c.num_head), (left_proj_act, right_proj_act))
        
        left_proj_act = pair_mask * left_proj_act
        right_proj_act = pair_mask * right_proj_act
        
        if c.gating:
            left_gate_values = torch.sigmoid(self.left_gate(act))
            right_gate_values = torch.sigmoid(self.right_gate(act))

            left_proj_act = left_proj_act * left_gate_values
            right_proj_act = right_proj_act * right_gate_values

        if c.orientation == 'per_row':
            act = torch.einsum('b i k c, b j k c -> b i j c', left_proj_act, right_proj_act)
        elif c.orientation == 'per_column':
            act = torch.einsum('b k i c, b k j c -> b i j c', left_proj_act, right_proj_act)
        else:
            raise NotImplementedError(f'{self.orientation} not Implemented')

        act = self.final_norm(act)
        act = self.proj_out(act)
        
        if c.gating:
            gate_values = torch.sigmoid(self.final_gate(input_act))
            act = act * gate_values

        return act

class TriangleAttention(nn.Module):
    def __init__(self, config, num_in_pair_channel):
        super().__init__()
        c = config

        assert c.orientation in ['per_row', 'per_column']
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None

        self.norm = LayerNorm(num_in_pair_channel)
        self.proj_pair = Linear(num_in_pair_channel, c.num_head, init='linear', bias = False, config=LoRA_conf)
        self.attn = Attention(
                input_dim=num_in_pair_channel,
                key_dim=num_in_pair_channel,
                value_dim=num_in_pair_channel,
                output_dim=num_in_pair_channel,
                num_head=c.num_head,
                gating=c.gating,
                inp_kernels=c.inp_kernels,
                config=LoRA_conf)
        self.attn._diag_name = f"triangle_attention_{c.orientation}"
        self._diag_context = None

        self.config = config

    def forward(self, pair_act, seq_mask):
        '''
        pair_act: (b l l c)
        seq_mask: (b l)
        '''
        c = self.config
        if c.orientation == 'per_column':
            pair_act = rearrange(pair_act, 'b i j c -> b j i c')

        pair_act = self.norm(pair_act)
        seq_mask = rearrange(seq_mask, 'b l -> b () l')

        # V200_EXACT_TRIANGLE_CHUNKING
        # Triangle attention is independent along the outer ``s`` axis.  Chunk
        # only that axis while keeping the full q/k bias matrix. Concatenating
        # the chunks is algebraically identical to the dense donor call, but the
        # cubic logits/softmax temporary becomes [B,chunk,H,L,L] instead of
        # [B,L,H,L,L].  No approximation, sparsification or attention deletion.
        bias = rearrange(self.proj_pair(pair_act), 'b i j h -> b h i j')
        self.attn._diag_context = self._diag_context
        # V202_JSON_RUNTIME_TRIANGLE_CHUNK
        # The chunk size is an execution parameter supplied by the selected JSON
        # (via _experiment.runtime_env -> launcher export).  It is deliberately
        # not tied to batch_size and does not alter the attention equation.
        # A non-positive value would silently fall back to the full cubic
        # temporary and can OOM on 48GB cards, so fail fast instead.
        chunk = _env_int("ABFLOW_ABX_TRIANGLE_CHUNK_SIZE", 32)
        if chunk <= 0:
            raise ValueError(
                "ABFLOW_ABX_TRIANGLE_CHUNK_SIZE must be a positive integer; "
                f"got {chunk}. Set it in the selected JSON runtime_env."
            )
        if int(pair_act.shape[1]) > chunk:
            outs = []
            for start in range(0, int(pair_act.shape[1]), chunk):
                stop = min(start + chunk, int(pair_act.shape[1]))
                pc = pair_act[:, start:stop]
                outs.append(self.attn(
                    q_data=pc, k_data=pc, bias=bias, k_mask=seq_mask
                ))
            pair_act = torch.cat(outs, dim=1)
        else:
            pair_act = self.attn(
                q_data=pair_act, k_data=pair_act, bias=bias, k_mask=seq_mask
            )

        if c.orientation == 'per_column':
            pair_act = rearrange(pair_act, 'b i j c -> b j i c')

        return pair_act

class SeqformerIteration(nn.Module):
    def __init__(self, config, seq_channel, pair_channel):
        super().__init__()
        c = config

        self.seq_attn = SeqAttentionWithPairBias(c.seq_attention_with_pair_bias, seq_channel, pair_channel)
        self.seq_transition = Transition(c.seq_transition, seq_channel)
        self.outer_product_mean = OuterProductMean(c.outer_product_mean, seq_channel, pair_channel)
        
        self.triangle_multiplication_outgoing = TriangleMultiplication(c.triangle_multiplication_outgoing, pair_channel)
        self.triangle_multiplication_incoming = TriangleMultiplication(c.triangle_multiplication_incoming, pair_channel)
        self.triangle_attention_starting_node = TriangleAttention(c.triangle_attention_starting_node, pair_channel)
        self.triangle_attention_ending_node = TriangleAttention(c.triangle_attention_ending_node, pair_channel)
        self.pair_transition = Transition(c.pair_transition, pair_channel)

        self.config = config

    def forward(self, seq_act, pair_act, seq_mask, diag_context=None):
        """
        seq_act: (b l c)
        pair_act: (b l l c)
        seq_mask: (b l)
        """
        c = self.config

        def dropout_fn(input_act, act, config):
            if self.training and config.dropout_rate > 0.:
                if config.shared_dropout:
                    if config.orientation == 'per_row':
                        broadcast_dim = 1
                    else:
                        broadcast_dim = 2
                else:
                    broadcast_dim = None
                act = apply_dropout(act, config.dropout_rate,
                        is_training=True, broadcast_dim=broadcast_dim)
            return input_act + act
        
        seq_act = dropout_fn(
                seq_act, self.seq_attn(seq_act, pair_act, seq_mask), c.seq_attention_with_pair_bias)
        seq_act = seq_act + self.seq_transition(seq_act, seq_mask)
        
        pair_act = pair_act + self.outer_product_mean(seq_act, seq_mask)
        _abx_mem_log(
            "seqformer:after_opm", diag_context, pair_act,
            extra=f"seq={tuple(seq_act.shape)}/{seq_act.dtype} pair={tuple(pair_act.shape)}/{pair_act.dtype}",
        )
        
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_outgoing(pair_act, seq_mask), c.triangle_multiplication_outgoing)
        _abx_mem_log("seqformer:after_tri_mul_out", diag_context, pair_act)
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_incoming(pair_act, seq_mask), c.triangle_multiplication_incoming)
        _abx_mem_log("seqformer:after_tri_mul_in", diag_context, pair_act)

        self.triangle_attention_starting_node._diag_context = diag_context
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_starting_node(pair_act, seq_mask), c.triangle_attention_starting_node)
        _abx_mem_log("seqformer:after_tri_attn_start", diag_context, pair_act)

        self.triangle_attention_ending_node._diag_context = diag_context
        pair_act = dropout_fn(
                pair_act, self.triangle_attention_ending_node(pair_act, seq_mask), c.triangle_attention_ending_node)
        _abx_mem_log("seqformer:after_tri_attn_end", diag_context, pair_act)
        pair_act = pair_act + self.pair_transition(pair_act, seq_mask)
        _abx_mem_log("seqformer:after_pair_transition", diag_context, pair_act)
        
        return seq_act, pair_act

class Seqformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        self.blocks = nn.ModuleList([SeqformerIteration(c.seqformer, c.seq_channel+c.index_embed_size, c.pair_channel+2*c.index_embed_size) for _ in range(c.seqformer_num_block)])

    def forward(self, seq_act, pair_act, mask, is_recycling=True, diag_context=None):
        checkpoint_enabled = bool(
            self.training and not is_recycling
            and _env_flag("ABFLOW_ABX_ACTIVATION_CHECKPOINT", True)
        )
        if checkpoint_enabled and _abx_diag_active(diag_context):
            print(
                f"[AbXCheckpoint] rank={_abx_dist_rank()} "
                f"call={int(diag_context.get('call', -1))} "
                f"round={int(diag_context.get('round', -1))} "
                "blocks=all preserve_rng_state=1", flush=True
            )
        for it, block in enumerate(self.blocks):
            block_context = diag_context
            if checkpoint_enabled and diag_context is not None:
                block_context = dict(diag_context)
                block_context["checkpoint_block"] = True
            block_fn = fn.partial(block, seq_mask=mask, diag_context=block_context)
            if checkpoint_enabled:
                # AbX v4_l3 has one block, so the donor's historical it>0 gate
                # never checkpointed anything.  Since V200 the persistent AbX
                # trunk runs once per outer forward (outside the three R05 physical
                # rounds).  Checkpointing this one block preserves equations/RNG/
                # gradients while dropping its retained intra-block activations.
                seq_act, pair_act = checkpoint(
                    block_fn, seq_act, pair_act, preserve_rng_state=True
                )
            else:
                seq_act, pair_act = block_fn(seq_act, pair_act)
        return seq_act, pair_act

class SpatialDepthWiseConvolution(nn.Module):
    def __init__(self, head_dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=head_dim, out_channels=head_dim,
                kernel_size=(kernel_size,),
                # padding=(kernel_size - 1,),
                padding=kernel_size//2,
                groups=head_dim)
    
    def forward(self, x: torch.Tensor):
        batch_size, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size * heads, head_dim, seq_len)
        x = self.conv(x)
        #if self.kernel_size>1:
        #    x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, head_dim, seq_len)
        x = x.permute(0, 1, 3, 2)
        return x

class SpatialDepthWiseInception(nn.Module):
    def __init__(self, head_dim, kernels):
        super().__init__()
       
        assert len(kernels) > 1 and  kernels[0] == 1

        self.convs = torch.nn.ModuleList([SpatialDepthWiseConvolution(head_dim, kernel_size=k) for k in kernels[1:]])
        self.kernels = kernels
    def forward(self, x):
        # x: (batch, num_heads, len, head_dim)
        
        assert x.shape[1] % len(self.kernels) == 0
        group_num_head = x.shape[1] // len(self.kernels)
        
        outputs = [x[:,:group_num_head]]

        for i, conv in enumerate(self.convs):
            outputs.append(conv(x[:,group_num_head*(i+1):group_num_head*(i+2)]))

        outputs = torch.cat(outputs, dim=1)

        return outputs

# ===== AbX donor boundary-local classes =====

class AbXEmbeddingAndSeqformerLocal(nn.Module):
    """Local rewrite of AbX EmbeddingAndSeqformer with the R05 boundary only."""
    def __init__(self, recycle_features=False, recycle_pos=False):
        super().__init__(); c=abx_v4_l3_seqformer_config(recycle_features,recycle_pos); self.config=c
        # V184 dual-route time contract.
        # The scalar is the SAME R05 transport time t, but it is intentionally
        # represented in two validated destinations:
        #   (1) R05 flow-time -> task-specific dynamic node state;
        #   (2) AbX donor timestep embedding -> AbX single and pair channels.
        # This preserves the source-faithful AbX time-conditioned s/z trunk.
        # It therefore intentionally enables pair-time conditioning inside the
        # AbX pair representation; this is part of the formal V184 factor, not
        # a hidden second time variable or accidental contamination.
        self.use_abx_time = _env_flag("ABFLOW_ABX_TIME_EMBED", True)
        self.num_token=23; self.num_region=15
        self.proj_aa_type=nn.Embedding(self.num_token,c.seq_channel,padding_idx=20)
        self.encode_residue_emb=ResidueEmbedding(c); self.encode_pair_emb=PairEmbedding(c)
        self.aa_proj=nn.Sequential(LayerNorm(c.seq_channel),Linear(c.seq_channel,c.seq_channel,init='linear',bias=True),nn.ReLU(),Linear(c.seq_channel,c.seq_channel,init='linear',bias=True))
        # Project-level frozen boundary: no ESM branch. All residue/pair/Seqformer operators remain donor-faithful.
        self.proj_rel_pos=nn.Embedding(c.max_relative_feature*2+2,c.pair_channel)
        if c.recycle_features:
            self.prev_seq_norm=LayerNorm(c.seq_channel+c.index_embed_size)
            self.prev_pair_norm=LayerNorm(c.pair_channel+2*c.index_embed_size)
        if c.recycle_pos:
            self.proj_prev_pos=nn.Embedding(c.prev_pos.num_bins,c.pair_channel+2*c.index_embed_size)
        self.seqformer=Seqformer(c); self.t_embeder=Embedder(c)

    def forward(self,batch):
        c=self.config; seq,mask,seq_pos=batch['seq_t'],batch['mask'],batch['residx']; antibody_len=batch['antibody_len']
        diag_context = batch.get('_abx_diag_context', None)
        _abx_mem_log(
            "embedder:start", diag_context, seq,
            extra=f"seq={tuple(seq.shape)} valid={mask.sum(-1).tolist()} antibody_len={antibody_len.tolist()}",
        )
        # Padded batches can have graph-specific antibody lengths. Build donor base embeddings graph-wise.
        B,L=seq.shape
        raw_seq_act=self.proj_aa_type(seq.long())
        pos_index=torch.arange(L,device=seq.device)[None,:]
        ab_mask=(pos_index<antibody_len[:,None]) & mask.bool()
        ag_mask=(~ab_mask) & mask.bool()
        # Exact donor asymmetry: antibody keeps the raw aa embedding while antigen
        # passes through aa_proj.  torch.where avoids an in-place autograd hazard.
        # AMP-safe donor asymmetry.
        # Under torch.cuda.amp.autocast(bf16), aa_proj(raw_seq_act) can be
        # bfloat16 while the embedding output raw_seq_act remains float32.
        # torch.where in torch 1.11 requires both value branches to share dtype.
        # Cast only the projected antigen branch back to the donor/base embedding
        # dtype; this changes precision plumbing only, not AbX representation semantics.
        ag_seq_act = self.aa_proj(raw_seq_act).to(dtype=raw_seq_act.dtype)
        seq_act = torch.where(ag_mask[..., None], ag_seq_act, raw_seq_act)
        # Exact pair_concat semantics without in-place slicing: antibody-antibody
        # and antigen-antigen get relative-position embeddings; cross blocks are 0.
        off=seq_pos[:,None,:]-seq_pos[:,:,None]  # j - i, same orientation as donor
        rel=torch.clip(off+c.max_relative_feature,min=0,max=2*c.max_relative_feature)+1
        same_group=(ab_mask[:,:,None]&ab_mask[:,None,:]) | (ag_mask[:,:,None]&ag_mask[:,None,:])
        pair_act=self.proj_rel_pos(rel.long())*same_group[...,None].to(raw_seq_act.dtype)
        enc_s=self.encode_residue_emb(batch,batch['seq_t'],batch['atom14_gt_positions'],batch['torsion_angles_sin_cos'])
        _abx_mem_log(
            "embedder:after_residue_embedding", diag_context, enc_s,
            extra=f"enc_s={tuple(enc_s.shape)}/{enc_s.dtype}",
        )
        enc_z=self.encode_pair_emb(batch,batch['seq_t'],batch['atom14_gt_positions'],batch['atom14_gt_exists'])
        _abx_mem_log(
            "embedder:after_pair_embedding", diag_context, enc_z,
            extra=f"enc_z={tuple(enc_z.shape)}/{enc_z.dtype}",
        )
        seq_act=seq_act+enc_s; pair_act=pair_act+enc_z
        ie_seq_act=seq_act.detach().contiguous(); ie_pair_act=pair_act.detach().contiguous()
        if self.use_abx_time:
            seq_act,pair_act=self.t_embeder(seq_act,pair_act,batch)
        else:
            # Compatibility fallback only. Formal V184 configs keep AbX time
            # ON. If an external ablation disables it, preserve donor tensor
            # widths from the active config with zero channels so that turning
            # time off does not simultaneously change Seqformer width/capacity.
            iz = int(c.index_embed_size)
            seq_act = torch.cat([
                seq_act, seq_act.new_zeros((B, L, iz))
            ], dim=-1)
            pair_act = torch.cat([
                pair_act, pair_act.new_zeros((B, L, L, 2 * iz))
            ], dim=-1)
        if c.recycle_features:
            if batch.get('prev_seq') is not None: seq_act=seq_act+self.prev_seq_norm(batch['prev_seq'])
            if batch.get('prev_pair') is not None: pair_act=pair_act+self.prev_pair_norm(batch['prev_pair'])
        if c.recycle_pos and batch.get('prev_pos') is not None:
            pair_act=pair_act+self.proj_prev_pos(batch['prev_pos'])
        _abx_mem_log(
            "embedder:before_seqformer", diag_context, pair_act,
            extra=f"seq={tuple(seq_act.shape)}/{seq_act.dtype} pair={tuple(pair_act.shape)}/{pair_act.dtype}",
        )
        seq_act,pair_act=self.seqformer(
            seq_act,pair_act,mask=mask,
            is_recycling=batch.get('is_recycling',False),
            diag_context=diag_context,
        )
        _abx_mem_log(
            "embedder:after_seqformer", diag_context, pair_act,
            extra=f"seq={tuple(seq_act.shape)}/{seq_act.dtype} pair={tuple(pair_act.shape)}/{pair_act.dtype}",
        )
        return seq_act,pair_act,ie_seq_act,ie_pair_act

class AbXDistogramHeadLocal(nn.Module):
    """Exact donor DistogramHead equation with an explicit runtime pair width.

    V200 localized AbX uses pair_channel=64 and index_embed_size=16, hence the
    Seqformer output ``z`` is 64 + 2*16 = 96 channels.  The old constructor
    retained donor defaults (128 + 2*32 = 192), which made R29/R30 fail before
    the first training step.  The head equation itself is unchanged; only its
    boundary dimension is now supplied by the active trunk config.
    """
    def __init__(self, pair_dim, num_bins=64, first_break=2.3125, last_break=21.6875):
        super().__init__()
        self.input_dim = int(pair_dim)
        self.num_bins = int(num_bins)
        if self.input_dim <= 0:
            raise ValueError(f"distogram pair_dim must be positive, got {self.input_dim}")
        self.register_buffer(
            'breaks',
            torch.linspace(first_break, last_break, steps=self.num_bins - 1),
            persistent=False,
        )
        self.proj = Linear(self.input_dim, self.num_bins, init='final')

    def forward(self, pair):
        if int(pair.shape[-1]) != self.input_dim:
            raise RuntimeError(
                "AbX distogram width contract violated: "
                f"z.shape[-1]={int(pair.shape[-1])}, head.input_dim={self.input_dim}. "
                "The auxiliary head must use the same active AbX width profile as the trunk."
            )
        x = self.proj(pair)
        return (x + rearrange(x, 'b i j c -> b j i c')) * 0.5

def abx_distogram_loss_local(logits,pseudo_beta,pseudo_beta_mask,min_bin=2.3125,max_bin=21.6875,no_bins=64,eps=1e-6):
    boundaries=torch.linspace(min_bin,max_bin,no_bins-1,device=logits.device)**2
    dists=torch.sum((pseudo_beta[...,None,:]-pseudo_beta[...,None,:,:])**2,dim=-1,keepdims=True)
    true_bins=torch.sum(dists>boundaries,dim=-1)
    errors=F.cross_entropy(logits.reshape(-1,no_bins),true_bins.reshape(-1),reduction='none').reshape_as(true_bins)
    square_mask=pseudo_beta_mask[...,None]*pseudo_beta_mask[...,None,:]
    denom=eps+torch.sum(square_mask,dim=(-1,-2))
    mean=torch.sum(errors*square_mask,dim=(-1,-2))/denom
    return torch.mean(mean)



# ============================================================================
# V182: AbX source-faithful single/pair trunk -> original R05 EGNN native edge_attr
# ============================================================================
class R05AbXNativeTrunk(nn.Module):
    """AbX representation operators with an R05-only boundary adapter.

    The *operators* (ResidueEmbedding, PairEmbedding, time embedding and the full
    one-block Seqformer) are source-faithful to the supplied AbX implementation.
    Only data routing is adapted:

    * AbFlow is flat/ragged, AbX is padded [B,L,...]; this class packs/unpacks.
    * ESM is intentionally absent because the project has a frozen no-ESM
      boundary. No Seqformer operator is removed.
    * AbX recycling is deliberately disabled. R05's three physical refinement
      rounds remain the sole recurrent mechanism in this experiment family.
    * ``fixed_mask`` retains AbX's information barrier: design-region structural
      features are masked in ResidueEmbedding/PairEmbedding. This is important
      here because R05 already exposes the current flow state through its native
      full-atom radial geometry; copying clean/native design geometry into the
      donor trunk would leak X1, while redundantly re-encoding the shadow Xt would
      create a dual-coordinate-state ambiguity between ctx and local branches.
    """
    # Widths are instance-level because V200 supports both donor and localized
    # profiles.  ``single_dim`` / ``pair_dim`` are derived from trunk.config in
    # __init__ and are the single source of truth for all downstream consumers.

    _CDR_CODE = {"H1": 1, "H2": 2, "H3": 3, "L1": 4, "L2": 5, "L3": 6}
    _CHOTHIA = {
        "H1": (26, 32), "H2": (52, 56), "H3": (95, 102),
        "L1": (24, 34), "L2": (50, 56), "L3": (89, 97),
    }

    def __init__(self, enable_distogram=False,
                 distogram_pair_scope="all_resolved",
                 forward_seed=271828):
        super().__init__()
        # User requirement: no AbX prev_s/prev_z/prev_pos recycling in V182.
        self.trunk = AbXEmbeddingAndSeqformerLocal(False, False)
        self.single_dim = int(
            self.trunk.config.seq_channel + self.trunk.config.index_embed_size
        )
        self.pair_dim = int(
            self.trunk.config.pair_channel + 2 * self.trunk.config.index_embed_size
        )
        self.enable_distogram = bool(enable_distogram)
        self.forward_seed = int(forward_seed)
        if self.forward_seed < 0:
            raise ValueError("AbX forward_seed must be non-negative")
        self._training_forward_call = 0
        self.distogram_pair_scope = str(distogram_pair_scope).strip().lower()
        if self.distogram_pair_scope not in {
            "all_resolved", "design_anchored"
        }:
            raise ValueError(
                "distogram_pair_scope must be all_resolved or "
                f"design_anchored, got {self.distogram_pair_scope!r}"
            )
        self.distogram_head = (
            AbXDistogramHeadLocal(pair_dim=self.pair_dim)
            if self.enable_distogram else None
        )
        if self.distogram_head is not None and self.distogram_head.input_dim != self.pair_dim:
            raise RuntimeError(
                "AbX width initialization contract violated: "
                f"pair_dim={self.pair_dim}, distogram_input={self.distogram_head.input_dim}"
            )
        self.last_diagnostics = {}
        self._memory_diag_call = 0

    @staticmethod
    def _ordered_nodes(valid_mask, batch_id, is_antigen, X, design_mask,
                       segment_ids, antigen_context_mask=None):
        """Full antibody + AbX-style H3-framework-anchor antigen patch.

        AbX first restricts antigen context around the residues immediately
        flanking each CDR.  For our H3-only design setting we mirror that
        semantics without native-H3 leakage: the two framework residues just
        outside the current H3 design mask are the anchors; antigen CA residues
        are ranked by current-state distance to those anchors and capped at the
        donor reference maximum (32 by default).  Parent R05 sparse geometry and
        surface/interaction edges are NOT cropped by this representation-only
        selection.
        """
        nodes, antibody_lens, source_ag_counts, selected_ag_counts = [], [], [], []
        B = int(batch_id.max().item()) + 1 if batch_id.numel() else 0
        cap = max(1, _env_int(
            "ABFLOW_ABX_MAX_ANTIGEN",
            _env_int("ABFLOW_ABX_REFERENCE_MAX_ANTIGEN", 32),
        ))
        mode = _env_str("ABFLOW_ABX_ANTIGEN_CONTEXT_MODE", "framework_anchor_ca").lower()
        if mode not in {"framework_anchor_ca", "all"}:
            raise ValueError(
                "ABFLOW_ABX_ANTIGEN_CONTEXT_MODE must be framework_anchor_ca or all"
            )
        for gid in range(B):
            g = valid_mask & (batch_id == gid)
            ab = torch.nonzero(g & (~is_antigen), as_tuple=False).flatten()
            ag_all = torch.nonzero(g & is_antigen, as_tuple=False).flatten()
            source_ag_counts.append(int(ag_all.numel()))

            # R05 already defines a local epitope/context universe.  Prefer that
            # antigen subset first; only fall back to all biological antigen if
            # the parent local mask is empty for this graph.
            ag = ag_all
            if antigen_context_mask is not None and ag.numel():
                local_ag = ag[antigen_context_mask[ag].bool()]
                if local_ag.numel():
                    ag = local_ag

            if mode == "framework_anchor_ca" and ag.numel() > cap:
                design = torch.nonzero(
                    g & (~is_antigen) & design_mask.bool(), as_tuple=False
                ).flatten()
                anchors = []
                if design.numel():
                    design_seg = segment_ids[design[0]]
                    chain_ab = ab[segment_ids[ab] == design_seg]
                    left = chain_ab[chain_ab < design.min()]
                    right = chain_ab[chain_ab > design.max()]
                    if left.numel():
                        anchors.append(left[-1])
                    if right.numel():
                        anchors.append(right[0])

                if anchors:
                    anchor_idx = torch.stack(anchors)
                    # Context selection is discrete.  Detach explicitly so the
                    # ranking operation never retains a coordinate autograd graph.
                    anchor_ca = X[anchor_idx, 1].detach().float()
                    ag_ca = X[ag, 1].detach().float()
                    if not bool(
                        torch.isfinite(anchor_ca).all().item()
                        and torch.isfinite(ag_ca).all().item()
                    ):
                        raise FloatingPointError(
                            "non-finite CA before AbX framework-anchor antigen patch"
                        )
                    score = torch.cdist(anchor_ca, ag_ca).amin(dim=0)
                    keep = torch.topk(score, k=cap, largest=False).indices
                    ag = torch.sort(ag[keep]).values
                else:
                    # Deterministic fallback for malformed/edge-case masks.
                    ag = ag[:cap]

            selected_ag_counts.append(int(ag.numel()))
            nodes.append(torch.cat([ab, ag], dim=0))
            antibody_lens.append(int(ab.numel()))
        return nodes, antibody_lens, source_ag_counts, selected_ag_counts

    @staticmethod
    def _remap_chain_ids(seg, mask):
        out = seg.new_zeros(seg.shape)
        for b in range(seg.shape[0]):
            seen = []
            for j in range(int(mask[b].sum().item())):
                value = int(seg[b, j].item())
                if value not in seen:
                    seen.append(value)
                out[b, j] = seen.index(value)
        return out

    @classmethod
    def _cdr_definition(cls, residue_index, chain_id, mask, antibody_len, cdr_type, design_mask):
        """Reconstruct AbX CDR codes from R05 metadata without H3 hard-coding.

        R05 does not carry AbX's explicit ``cdr_def`` tensor. For RAbD/Chothia
        numbering we reconstruct H1/H2/H3/L1/L2/L3 from residue indices and the
        antibody-chain order (first antibody chain=heavy, second=light), then
        override the task-design residues with ``cdr_type`` when it is one of the
        six named CDRs. For a multi-CDR design, the Chothia-derived labels remain.
        """
        out = residue_index.new_zeros(residue_index.shape)
        B, L = residue_index.shape
        for b in range(B):
            n = int(mask[b].sum().item())
            ab_n = int(antibody_len[b].item())
            # Unique antibody chain ids in first-occurrence order.
            chains = []
            for j in range(ab_n):
                cid = int(chain_id[b, j].item())
                if cid not in chains:
                    chains.append(cid)
            for j in range(min(n, ab_n)):
                cid = int(chain_id[b, j].item())
                if cid not in chains:
                    continue
                chain_kind = "H" if chains.index(cid) == 0 else "L"
                r = int(residue_index[b, j].item())
                for name, (lo, hi) in cls._CHOTHIA.items():
                    if name.startswith(chain_kind) and lo <= r <= hi:
                        out[b, j] = cls._CDR_CODE[name]
                        break
        names = _normalise_task_regions(cdr_type)
        if len(names) == 1 and names[0] in cls._CDR_CODE:
            name = names[0]
            out = torch.where(
                design_mask,
                torch.full_like(out, cls._CDR_CODE[name]),
                out,
            )
        return out

    def _pack(self, X, S, segment_ids, residue_pos, batch_id, valid_mask,
              is_antigen, design_mask, flow_t, cdr_type, antigen_context_mask=None,
              atom_observed_mask=None):
        nodes, ab_lens, source_ag_counts, selected_ag_counts = self._ordered_nodes(
            valid_mask, batch_id, is_antigen, X, design_mask,
            segment_ids, antigen_context_mask,
        )
        B = len(nodes)
        L = max([int(idx.numel()) for idx in nodes] or [0])
        if L == 0:
            raise RuntimeError("AbX trunk received no biological residues")
        if X.shape[1] != 14:
            raise ValueError(
                f"V184 AbX port requires AbFlow's formal 14-slot full-atom state; got {tuple(X.shape)}"
            )

        Sp = S.new_full((B, L), 20)
        Xp = X.new_zeros((B, L, 14, 3))
        Obsp = X.new_zeros((B, L, 14)) if atom_observed_mask is not None else None
        Seg = segment_ids.new_zeros((B, L))
        Rp = torch.zeros((B, L), device=X.device, dtype=torch.long)
        M = torch.zeros((B, L), device=X.device, dtype=torch.bool)
        Design = torch.zeros_like(M)
        IsAg = torch.zeros_like(M)
        GI = S.new_full((B, L), -1)
        if atom_observed_mask is not None:
            if tuple(atom_observed_mask.shape) != tuple(X.shape[:-1]):
                raise ValueError(
                    f"AbFlow xloss_mask shape mismatch: {tuple(atom_observed_mask.shape)} "
                    f"vs X {tuple(X.shape)}"
                )
        for b, idx in enumerate(nodes):
            n = int(idx.numel())
            if n == 0:
                continue
            Sp[b, :n] = S[idx].long().clamp(0, 22)
            Xp[b, :n] = X[idx]
            if Obsp is not None:
                Obsp[b, :n] = atom_observed_mask[idx].to(dtype=Xp.dtype)
            Seg[b, :n] = segment_ids[idx]
            rp = residue_pos[idx]
            rp = rp[..., 0] if rp.dim() > 1 else rp
            # torch<=1.11 CUDA does not implement round() for Long tensors.
            # R05 residue_pos is normally already integer-valued; round only a
            # floating representation before the final semantic cast.
            if torch.is_floating_point(rp):
                rp = torch.round(rp)
            Rp[b, :n] = rp.to(dtype=torch.long)
            M[b, :n] = True
            Design[b, :n] = design_mask[idx].bool()
            IsAg[b, :n] = is_antigen[idx].bool()
            GI[b, :n] = idx

        Chain = self._remap_chain_ids(Seg, M)
        antibody_len = torch.tensor(ab_lens, device=X.device, dtype=torch.long)
        Cdr = self._cdr_definition(Rp, Chain, M, antibody_len, cdr_type, Design)
        # V199 formal atom-observation contract. AbFlow stores unresolved atom
        # slots as a copy of CA and records the true observation state in
        # xloss_mask; donor AbX stores unresolved atom14 coordinates as zero.
        # Translate only this private donor input representation. Parent R05 X,
        # EGNN geometry, losses and coordinate authority are untouched.
        Exists = _atom14_exists_from_seq(Sp, Xp, Obsp) * M[..., None].to(Xp.dtype)
        Xp_donor = torch.where(Exists.bool()[..., None], Xp, torch.zeros_like(Xp))

        # AbX semantics: design region is not a clean/native structural condition.
        # Fixed scaffold/antigen structural embeddings are visible; design
        # ResidueEmbedding/PairEmbedding geometry is blocked. The current Xt still
        # enters R05 AM_E_GCL through its original multi-channel radial path.
        Fixed = M & (~Design)
        Tors = _torsions_from_atom14(
            Sp, Xp_donor, Chain, Fixed, atom_exists=Exists.bool()
        )

        # One compact boundary audit is retained in formal runs.  It verifies
        # that the inference fallback (CA-fill sentinel) agrees with the explicit
        # AbFlow xloss_mask on fixed biological residues. No per-layer logging.
        audit_limit = max(0, _env_int("ABFLOW_ABX_ATOM_MASK_AUDIT_CALLS", 1))
        audit_count = int(getattr(self, "_atom_mask_audit_count", 0))
        if audit_count < audit_limit:
            inferred = _atom14_exists_from_seq(Sp, Xp, None).bool() & M[..., None]
            explicit = Exists.bool() & M[..., None]
            fixed_atom = Fixed[..., None] & _atom14_chemical_mask(Sp, Xp).bool()
            mismatch = int(((inferred ^ explicit) & fixed_atom).sum().detach().cpu().item())
            unresolved = int((fixed_atom & (~explicit)).sum().detach().cpu().item())
            resolved = int((fixed_atom & explicit).sum().detach().cpu().item())
            print(
                f"[AbXAtomMaskContract] rank={_abx_dist_rank()} call={audit_count} "
                f"source={'xloss_mask' if Obsp is not None else 'ca_fill_fallback'} "
                f"resolved={resolved} unresolved_zeroed={unresolved} "
                f"fallback_mismatch={mismatch}"
            )
            self._atom_mask_audit_count = audit_count + 1

        t = flow_t
        if t is None:
            t = Xp.new_zeros(B)
        elif t.dim() == 0:
            t = t.expand(B)
        elif t.numel() != B:
            t = torch.stack([t[batch_id == gid].reshape(-1)[0] for gid in range(B)])

        batch = {
            'seq': Sp, 'seq_t': Sp, 'mask': M, 'fixed_mask': Fixed,
            'chain_id': Chain, 'residx': Rp, 'cdr_def': Cdr,
            'atom14_gt_positions': Xp_donor, 'atom14_gt_exists': Exists,
            'torsion_angles_sin_cos': Tors, 't': t.to(Xp.dtype),
            'antibody_len': antibody_len, 'is_recycling': False,
            '_design_mask': Design, '_is_antigen': IsAg,
        }
        return batch, GI, nodes, source_ag_counts, selected_ag_counts

    @staticmethod
    def _node_lookup(GI, mask, n_global):
        graph = GI.new_full((n_global,), -1)
        local = GI.new_full((n_global,), -1)
        for b in range(mask.shape[0]):
            n = int(mask[b].sum().item())
            if n == 0:
                continue
            idx = GI[b, :n].long()
            graph[idx] = b
            local[idx] = torch.arange(n, device=GI.device, dtype=GI.dtype)
        return graph, local

    @staticmethod
    def gather_pair(z, query_edges_global, node_graph, node_local):
        """Map dense AbX z_ij to the exact sparse EGNN edge order."""
        if query_edges_global.numel() == 0:
            return z.new_zeros((0, z.shape[-1]))
        row, col = query_edges_global.long()
        gr, gc = node_graph[row], node_graph[col]
        lr, lc = node_local[row], node_local[col]
        valid = (gr >= 0) & (gr == gc) & (lr >= 0) & (lc >= 0)
        out = z.new_zeros((row.numel(), z.shape[-1]))
        if bool(valid.any()):
            out[valid] = z[gr[valid], lr[valid], lc[valid]]
        return out

    def forward(self, X, S, segment_ids, residue_pos, batch_id, valid_mask,
                is_antigen, design_mask, flow_t, cdr_type, round_idx=-1,
                antigen_context_mask=None, atom_observed_mask=None):
        call_idx = int(self._memory_diag_call)
        self._memory_diag_call += 1
        diag_context = {"call": call_idx, "round": int(round_idx)}
        _assert_finite_tensor("r05_abx.input_X", X, diag_context)
        batch, GI, nodes, source_ag_counts, selected_ag_counts = self._pack(
            X, S, segment_ids, residue_pos, batch_id, valid_mask,
            is_antigen, design_mask, flow_t, cdr_type, antigen_context_mask,
            atom_observed_mask=atom_observed_mask,
        )
        batch['_abx_diag_context'] = diag_context

        if _abx_diag_active(diag_context):
            ab_counts, ag_counts, design_counts, total_counts = [], [], [], []
            for idx in nodes:
                total_counts.append(int(idx.numel()))
                if idx.numel() == 0:
                    ab_counts.append(0)
                    ag_counts.append(0)
                    design_counts.append(0)
                    continue
                ab_counts.append(int((~is_antigen[idx]).sum().item()))
                ag_counts.append(int(is_antigen[idx].sum().item()))
                design_counts.append(int(design_mask[idx].sum().item()))

            B, L = batch['mask'].shape
            tri_heads = int(
                self.trunk.config.seqformer.triangle_attention_starting_node.num_head
            )
            tri_elems = int(B) * int(L) * tri_heads * int(L) * int(L)
            tri_bf16_gib = tri_elems * 2 / float(1024 ** 3)
            tri_fp32_gib = tri_elems * 4 / float(1024 ** 3)
            atom14_diff_fp32_gib = (
                int(B) * int(L) * int(L) * 14 * 14 * 3 * 4
                / float(1024 ** 3)
            )
            ref_cap = _env_int("ABFLOW_ABX_REFERENCE_MAX_ANTIGEN", 32)
            active_cap = max(1, _env_int("ABFLOW_ABX_MAX_ANTIGEN", ref_cap))
            patch_mode = _env_str("ABFLOW_ABX_ANTIGEN_CONTEXT_MODE", "framework_anchor_ca")
            print(
                f"[AbXInputAudit] rank={_abx_dist_rank()} call={call_idx} "
                f"round={int(round_idx)} B={B} Lpad={L} total={total_counts} "
                f"antibody={ab_counts} antigen_source={source_ag_counts} "
                f"antigen_selected={selected_ag_counts} design={design_counts} "
                f"AbX_ref_antigen_cap={ref_cap} source_over_cap="
                f"{sum(int(n > ref_cap) for n in source_ag_counts)} "
                f"triangle_logits_one_op=bf16:{tri_bf16_gib:.3f}GiB/"
                f"fp32:{tri_fp32_gib:.3f}GiB "
                f"pair_atom14_diff_fp32={atom14_diff_fp32_gib:.3f}GiB",
                flush=True,
            )
            if source_ag_counts != selected_ag_counts:
                print(
                    f"[AbXAntigenPatch] rank={_abx_dist_rank()} call={call_idx} "
                    f"round={int(round_idx)} source={source_ag_counts} "
                    f"selected={selected_ag_counts} cap={active_cap} "
                    f"mode={patch_mode} R05_geometry_edges=unchanged",
                    flush=True,
                )
            _abx_mem_log("r05_abx:before_trunk", diag_context, X)

        # AbX donor dropout remains fully active, but it must not advance the
        # RNG stream later consumed by the parent R05 EGNN dropout. Otherwise a
        # mathematically zero bridge would still alter the parent's stochastic
        # training function. A call-indexed, rank-local donor RNG gives fresh
        # masks while fork_rng restores parent CPU/current-CUDA states on exit.
        if self.training:
            call_index = int(self._training_forward_call)
            self._training_forward_call += 1
            donor_seed = (
                self.forward_seed
                + 1_000_003 * int(_abx_dist_rank())
                + call_index
            )
            rng_devices = (
                [int(X.device.index)]
                if X.is_cuda and X.device.index is not None else []
            )
            with torch.random.fork_rng(devices=rng_devices):
                torch.default_generator.manual_seed(donor_seed)
                if X.is_cuda:
                    with torch.cuda.device(X.device):
                        torch.cuda.manual_seed(donor_seed)
                s, z, _, _ = self.trunk(batch)
        else:
            s, z, _, _ = self.trunk(batch)
        _assert_finite_tensor("r05_abx.single", s, diag_context)
        _assert_finite_tensor("r05_abx.pair", z, diag_context)
        _abx_mem_log(
            "r05_abx:after_trunk", diag_context, z,
            extra=f"single={tuple(s.shape)}/{s.dtype} pair={tuple(z.shape)}/{z.dtype}",
        )
        mask = batch['mask']
        N = int(S.shape[0])
        global_s = s.new_zeros((N, s.shape[-1]))
        for b, idx in enumerate(nodes):
            n = int(idx.numel())
            if n:
                global_s[idx] = s[b, :n]
        node_graph, node_local = self._node_lookup(GI, mask, N)
        logits = self.distogram_head(z) if self.distogram_head is not None else None
        with torch.no_grad():
            self.last_diagnostics = {
                'abx_single_rms': torch.sqrt(global_s.float().pow(2).mean() + 1e-8).to(X.dtype),
                'abx_pair_rms': torch.sqrt(z.float().pow(2).mean() + 1e-8).to(X.dtype),
                'abx_token_count': mask.sum().to(X.dtype),
                'abx_pair_count': (mask.sum(-1).float().pow(2).sum()).to(X.dtype),
                'abx_design_embedding_mask_rate': (batch['fixed_mask'] == 0).logical_and(mask).float().sum().div(mask.float().sum().clamp_min(1)).to(X.dtype),
            }
        return {
            'single_global': global_s,
            'pair_dense': z,
            'diag_context': diag_context,
            'node_graph': node_graph,
            'node_local': node_local,
            'global_index': GI,
            'mask': mask,
            'seq_padded': batch['seq'],
            'atom_exists_padded': batch['atom14_gt_exists'].bool(),
            'design_padded': batch['_design_mask'].bool(),
            'is_antigen_padded': batch['_is_antigen'].bool(),
            'distogram_logits': logits,
            'biological_mask': valid_mask.bool(),
            'diag': self.last_diagnostics,
        }

    def distogram_loss_from_native(self, state, true_X, true_S):
        """Exact AbX donor distogram objective plus observational decomposition.

        The trainable objective preserves the donor equation: symmetric logits,
        64 bins over 2.3125--21.6875 Angstrom, pseudo-beta targets, resolved
        pseudo-beta validity, per-complex cardinality normalization and then a
        batch mean.  ``design_anchored`` changes only the task support mask to
        pairs with at least one JSON-selected design endpoint. Relation-specific
        terms and the donor all-pair loss are detached audits; they never
        rebalance the optimized objective.
        """
        zero = true_X.sum() * 0.0
        if not self.enable_distogram or state.get('distogram_logits') is None:
            return zero, {}
        GI, token_mask = state['global_index'], state['mask']
        logits = state['distogram_logits']
        B, L = token_mask.shape
        Xp = true_X.new_zeros((B, L, 14, 3))
        Sp = true_S.new_full((B, L), 20)
        Obs = torch.zeros(
            (B, L, 14), device=true_X.device, dtype=torch.bool
        )
        packed_obs = state['atom_exists_padded'].bool()
        for b in range(B):
            n = int(token_mask[b].sum().item())
            if n:
                idx = GI[b, :n].long()
                Xp[b, :n] = true_X[idx]
                Sp[b, :n] = true_S[idx].long().clamp(0, 22)
                Obs[b, :n] = packed_obs[b, :n]

        pseudo_beta, pseudo_beta_mask = pseudo_beta_fn_v2(Sp, Xp, Obs)
        pseudo_beta_mask = pseudo_beta_mask.bool() & token_mask
        boundaries = torch.linspace(
            2.3125, 21.6875, 63,
            device=logits.device, dtype=pseudo_beta.dtype,
        )
        d2 = torch.sum(
            (pseudo_beta[:, :, None, :] - pseudo_beta[:, None, :, :]).square(),
            dim=-1, keepdim=True,
        )
        target = torch.sum(d2 > boundaries.square(), dim=-1).long()
        ce = F.cross_entropy(
            logits.reshape(-1, 64), target.reshape(-1), reduction='none'
        ).reshape(B, L, L)
        donor_pair_mask = (
            pseudo_beta_mask[:, :, None] & pseudo_beta_mask[:, None, :]
        )
        design = state['design_padded'].bool() & pseudo_beta_mask
        if self.distogram_pair_scope == "design_anchored":
            # Exact donor CE/bins/symmetry/reduction are retained.  The sole
            # AbFlow localization is the task-authoritative pair support:
            # at least one endpoint belongs to the JSON-selected design mask.
            pair_mask = donor_pair_mask & (
                design[:, :, None] | design[:, None, :]
            )
        else:
            pair_mask = donor_pair_mask

        denom = pair_mask.sum(dim=(-1, -2)).to(ce.dtype) + 1e-6
        per_graph = (
            ce * pair_mask.to(ce.dtype)
        ).sum(dim=(-1, -2)) / denom
        loss = per_graph.mean()

        donor_denom = donor_pair_mask.sum(dim=(-1, -2)).to(ce.dtype) + 1e-6
        donor_per_graph = (
            ce * donor_pair_mask.to(ce.dtype)
        ).sum(dim=(-1, -2)) / donor_denom
        donor_all_pair_loss = donor_per_graph.mean()

        with torch.no_grad():
            antigen = state['is_antigen_padded'].bool() & pseudo_beta_mask
            scaffold = pseudo_beta_mask & (~design) & (~antigen)
            context = pseudo_beta_mask & (~design)

            masks = {
                'design_design': donor_pair_mask & design[:, :, None] & design[:, None, :],
                'design_framework': donor_pair_mask & (
                    (design[:, :, None] & scaffold[:, None, :])
                    | (scaffold[:, :, None] & design[:, None, :])
                ),
                'design_antigen': donor_pair_mask & (
                    (design[:, :, None] & antigen[:, None, :])
                    | (antigen[:, :, None] & design[:, None, :])
                ),
                'context_context': donor_pair_mask & context[:, :, None] & context[:, None, :],
            }

            def masked_mean(value, mask):
                n = mask.sum()
                return (
                    (value * mask.to(value.dtype)).sum()
                    / n.clamp_min(1).to(value.dtype)
                )

            probs = torch.softmax(logits.float(), dim=-1)
            last_contact_bin = int((boundaries < 8.0).sum().item())
            contact_bins = (
                torch.arange(64, device=logits.device) <= last_contact_bin
            )
            contact_prob = probs[..., contact_bins].sum(dim=-1)
            native_contact = d2.squeeze(-1) < (8.0 ** 2)
            da_direct = (
                donor_pair_mask
                & design[:, :, None]
                & antigen[:, None, :]
            )
            soft_tp = (contact_prob * native_contact.float() * da_direct.float()).sum()
            soft_pred = (contact_prob * da_direct.float()).sum()
            chemical = _atom14_chemical_mask(Sp, Xp).bool() & token_mask[..., None]
            audit = {
                'disto_raw_loss': loss.detach(),
                'disto_all_pair_raw_loss': donor_all_pair_loss.detach(),
                'disto_resolved_pseudo_beta_rate': (
                    pseudo_beta_mask.float().sum() / token_mask.float().sum().clamp_min(1)
                ),
                'disto_valid_pairs_total': pair_mask.sum().to(loss.dtype),
                'disto_donor_valid_pairs_total': donor_pair_mask.sum().to(loss.dtype),
                'disto_task_pair_fraction': (
                    pair_mask.sum().to(loss.dtype)
                    / donor_pair_mask.sum().clamp_min(1).to(loss.dtype)
                ),
                'disto_scope_design_anchored': loss.new_tensor(
                    1.0 if self.distogram_pair_scope == "design_anchored" else 0.0
                ),
                'disto_contact_precision_8A_design_antigen': (
                    soft_tp / soft_pred.clamp_min(1e-8)
                ).to(loss.dtype),
                'disto_resolved_atom_rate': (
                    (Obs & chemical).float().sum()
                    / chemical.float().sum().clamp_min(1)
                ).to(loss.dtype),
            }
            for name, rel_mask in masks.items():
                audit[f'disto_{name}_pairs'] = rel_mask.sum().to(loss.dtype)
                audit[f'disto_ce_{name}'] = masked_mean(ce, rel_mask).to(loss.dtype)
        return loss, audit


def design_region_smooth_lddt_loss(
    pred_X, true_X, valid_atom_mask, design_residue_mask, batch_id,
    is_antigen_mask=None, cutoff=15.0,
):
    """MF/Boltz smooth-lDDT with only the task pair-mask localized.

    Donor invariants are unchanged: all resolved protein atoms, one pair mask,
    one cardinality denominator per complex, the 15 Angstrom native-distance
    cutoff, and the 0.5/1/2/4 Angstrom sigmoid score.  AbFlow localization adds
    exactly one condition: at least one atom in a scored pair must belong to the
    JSON-selected design region.  Relation breakdowns are observational only.
    """
    if is_antigen_mask is None:
        is_antigen_mask = torch.zeros_like(design_residue_mask, dtype=torch.bool)
    cutoff = float(cutoff)
    graph_losses = []
    rel_values = {'intra': [], 'scaffold': [], 'antigen': []}
    pair_counts = {'intra': 0, 'scaffold': 0, 'antigen': 0}

    for gid_t in torch.unique(batch_id):
        graph = batch_id == gid_t
        pred, true = pred_X[graph], true_X[graph]
        valid = valid_atom_mask[graph].bool()
        design = design_residue_mask[graph].bool()
        antigen = is_antigen_mask[graph].bool()
        if not bool(design.any()):
            continue
        nr, na = valid.shape
        resolved = valid.reshape(-1)
        if not bool(resolved.any()):
            continue
        pred_atom = pred.reshape(-1, 3).float()
        true_atom = true.reshape(-1, 3).float()
        design_atom = design[:, None].expand(nr, na).reshape(-1)
        antigen_atom = antigen[:, None].expand(nr, na).reshape(-1)
        scaffold_atom = (~design[:, None].expand(nr, na).reshape(-1)) & (~antigen_atom)

        d_true = torch.cdist(true_atom, true_atom)
        d_pred = torch.cdist(pred_atom, pred_atom)
        delta = (d_pred - d_true).abs()
        score = 0.25 * (
            torch.sigmoid(0.5 - delta)
            + torch.sigmoid(1.0 - delta)
            + torch.sigmoid(2.0 - delta)
            + torch.sigmoid(4.0 - delta)
        )
        pair = resolved[:, None] & resolved[None, :]
        pair = pair & (~torch.eye(pair.shape[0], device=pair.device, dtype=torch.bool))
        pair = pair & (d_true < cutoff)
        design_pair = pair & (design_atom[:, None] | design_atom[None, :])
        denominator = design_pair.sum().to(score.dtype)
        if bool(design_pair.any()):
            graph_losses.append(
                1.0 - (score * design_pair.to(score.dtype)).sum()
                / denominator.clamp_min(1.0)
            )

        with torch.no_grad():
            relation_masks = {
                'intra': pair & design_atom[:, None] & design_atom[None, :],
                'scaffold': pair & (
                    (design_atom[:, None] & scaffold_atom[None, :])
                    | (scaffold_atom[:, None] & design_atom[None, :])
                ),
                'antigen': pair & (
                    (design_atom[:, None] & antigen_atom[None, :])
                    | (antigen_atom[:, None] & design_atom[None, :])
                ),
            }
            for name, relation_mask in relation_masks.items():
                n = int(relation_mask.sum().item())
                if n:
                    rel_values[name].append(
                        1.0 - score[relation_mask].mean().detach()
                    )
                    pair_counts[name] += n
    zero = pred_X.sum() * 0.0
    total = torch.stack(graph_losses).mean().to(pred_X.dtype) if graph_losses else zero
    def dm(k): return torch.stack(rel_values[k]).mean().to(pred_X.dtype) if rel_values[k] else zero.detach()
    return total, {
        'intra': dm('intra'), 'scaffold': dm('scaffold'), 'antigen': dm('antigen'),
        'intra_pairs': pred_X.detach().new_tensor(float(pair_counts['intra'])),
        'scaffold_pairs': pred_X.detach().new_tensor(float(pair_counts['scaffold'])),
        'antigen_pairs': pred_X.detach().new_tensor(float(pair_counts['antigen'])),
    }


class AbFlowModel(nn.Module):
    def _enforce_task_sequence_mask_contract(
            self, cmask, smask, paratope_mask, template, stage):
        """Validate, but never guess or repair, the JSON-defined design task.

        V211 keeps ``cdr`` and ``paratope`` as lists from JSON through the CLI,
        dataset and model.  ``paratope_mask`` is therefore the authoritative
        task mask.  The model is allowed to verify the contract, but it must not
        silently turn a missing task into H3 or clip a broader task to H3.

        ``cmask`` is intentionally distinct: the template owns one row for each
        true coordinate-mask entry, so framework coordinate rows may legitimately
        exist outside the designed sequence region.  We validate and preserve it.
        """
        if not self.task_mask_contract:
            return cmask, smask
        for name, value in (
            ("cmask", cmask),
            ("smask", smask),
            ("paratope_mask", paratope_mask),
        ):
            if not torch.is_tensor(value):
                raise TypeError(
                    f"[V211TaskMaskFAIL] stage={stage} {name} must be a tensor; "
                    f"got {type(value).__name__}."
                )
        if cmask.shape != paratope_mask.shape or smask.shape != paratope_mask.shape:
            raise ValueError(
                "[V211TaskMaskFAIL] mask shape mismatch: "
                f"stage={stage} cmask={tuple(cmask.shape)} "
                f"smask={tuple(smask.shape)} "
                f"paratope={tuple(paratope_mask.shape)}."
            )

        target = paratope_mask.bool()
        coord = cmask.bool()
        seq = smask.bool()
        if not bool(target.any().item()):
            raise RuntimeError(
                f"[V211TaskMaskFAIL] stage={stage} paratope mask is empty."
            )

        coord_missing = target & ~coord
        seq_missing = target & ~seq
        if bool(coord_missing.any().item()):
            raise RuntimeError(
                "[V211TaskMaskFAIL] coordinate/template mask omits design residues: "
                f"stage={stage} missing={int(coord_missing.sum().item())}."
            )
        if (not self.struct_only) and self.pep_seq and bool(seq_missing.any().item()):
            raise RuntimeError(
                "[V211TaskMaskFAIL] sequence mask omits design residues: "
                f"stage={stage} missing={int(seq_missing.sum().item())}."
            )

        coord_rows = int(coord.sum().item())
        if not torch.is_tensor(template):
            raise TypeError(
                f"[V211CoordinateTemplateFAIL] stage={stage} template must be "
                f"a tensor; got {type(template).__name__}."
            )
        if template.dim() < 1 or int(template.shape[0]) != coord_rows:
            raise RuntimeError(
                "[V211CoordinateTemplateFAIL] cmask/template row mismatch: "
                f"stage={stage} cmask_rows={coord_rows} "
                f"template_rows={int(template.shape[0]) if template.dim() else 0}."
            )

        seq_outside = seq & ~target
        if bool(seq_outside.any().item()):
            raise RuntimeError(
                "[V211TaskMaskFAIL] sequence design mask contains residues outside "
                f"the JSON-defined paratope: stage={stage} "
                f"outside={int(seq_outside.sum().item())}."
            )
        if not self._task_mask_contract_logged:
            if _abx_dist_rank() == 0:
                print(
                    "[V211TaskMaskPASS] "
                    f"stage={stage} cdr={list(self.cdr_regions)} "
                    f"paratope={list(self.paratope_regions)} "
                    f"coord_rows_preserved={coord_rows} "
                    f"coord_outside_design={int((coord & ~target).sum().item())} "
                    f"template_rows={int(template.shape[0])} "
                    f"seq_outside=0 design_residues={int(target.sum().item())}",
                    flush=True,
                )
            self._task_mask_contract_logged = True
        return coord, seq

    def _assert_framework_sequence_immutable(self, generated_S, input_S,
                                             paratope_mask, stage):
        if self.task_mask_contract and not self.struct_only:
            changed = (generated_S != input_S) & ~paratope_mask.bool()
            if bool(changed.any().item()):
                raise RuntimeError(
                    "[V211FrameworkSequenceFAIL] generated sequence changed "
                    f"outside the task paratope: stage={stage} "
                    f"changed={int(changed.sum().item())}."
                )

    @torch.no_grad()
    def shared_initialization_fingerprint(self):
        """SHA256 over parameters shared by R28/R29/R30.

        The optional Distogram projection is deliberately excluded.  Equal
        hashes across the three scratch runs prove that a child head did not
        consume RNG and silently change the parent/trunk/GNN initialization.
        This is an audit only; it never enters optimization or checkpoints.
        """
        digest = hashlib.sha256()
        count = 0
        elements = 0
        excluded_prefix = "abx_repr.distogram_head."
        for name, parameter in sorted(self.named_parameters()):
            if name.startswith(excluded_prefix):
                continue
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            # PyTorch 1.11 rejects dtype-view on a zero-dimensional tensor
            # (for example a scalar Long parameter).  Flattening first keeps
            # the exact same raw-byte fingerprint while satisfying the
            # dtype-view requirement that self.dim() must be greater than 0.
            byte_view = value.reshape(-1).view(torch.uint8)
            digest.update(byte_view.numpy().tobytes())
            count += 1
            elements += int(value.numel())
        return {
            "sha256": digest.hexdigest(),
            "parameter_tensors": count,
            "parameter_elements": elements,
            "excluded_prefix": excluded_prefix,
        }

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
        self.task_mask_contract = _env_flag("ABFLOW_TASK_MASK_CONTRACT", True)
        cdr_regions = _normalise_task_regions(cdr_type)
        paratope_regions = _normalise_task_regions(paratope)
        self._task_mask_contract_logged = False
        if self.task_mask_contract:
            if not cdr_regions or not paratope_regions:
                raise ValueError(
                    "[V211TaskContractFAIL] cdr/paratope must arrive from JSON; "
                    f"got cdr={cdr_type!r}, paratope={paratope!r}."
                )
            if cdr_regions != paratope_regions:
                raise ValueError(
                    "[V211TaskContractFAIL] this co-design protocol requires the "
                    f"same CDR and paratope regions; got {cdr_regions} vs "
                    f"{paratope_regions}."
                )
        self.cdr_regions = cdr_regions
        self.paratope_regions = paratope_regions
        self.cdr_type = list(cdr_regions) if cdr_regions else cdr_type
        self.paratope = list(paratope_regions) if paratope_regions else paratope

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
        # =========================================================
        # V184 clean AbX representation -> original AM_E_GCL edge_attr
        # =========================================================
        # R28, R29 and R30 all use the source-faithful AbX representation trunk.
        # Their only formal differences are the JSON-controlled auxiliary losses:
        # R28 none, R29 Distogram, R30 donor smooth-lDDT.
        self.abx_native_repr = _env_flag("ABFLOW_ABX_NATIVE_REPR", True)
        self.abx_distogram = _env_flag("ABFLOW_ABX_DISTOGRAM", False)
        self.abx_init_seed = _env_int("ABFLOW_ABX_INIT_SEED", 314159)
        if self.abx_init_seed < 0:
            raise ValueError("ABFLOW_ABX_INIT_SEED must be non-negative")
        self.abx_forward_seed = _env_int("ABFLOW_ABX_FORWARD_SEED", 271828)
        if self.abx_forward_seed < 0:
            raise ValueError("ABFLOW_ABX_FORWARD_SEED must be non-negative")
        self.abx_bridge_mode = _env_str(
            "ABFLOW_ABX_BRIDGE_MODE", "zero_reparameterized"
        ).strip().lower()
        if self.abx_bridge_mode != "zero_reparameterized":
            raise ValueError(
                "V211 formal runs require ABFLOW_ABX_BRIDGE_MODE="
                f"zero_reparameterized, got {self.abx_bridge_mode!r}"
            )
        self.distogram_pair_scope = _env_str(
            "ABFLOW_DISTOGRAM_PAIR_SCOPE", "all_resolved"
        ).strip().lower()
        if self.distogram_pair_scope not in {
            "all_resolved", "design_anchored"
        }:
            raise ValueError(
                "ABFLOW_DISTOGRAM_PAIR_SCOPE must be all_resolved or "
                f"design_anchored, got {self.distogram_pair_scope!r}"
            )
        self.mf_smooth_lddt = _env_flag(
            "ABFLOW_MF_SMOOTH_LDDT",
            _env_flag("ABFLOW_ABX_SMOOTH_LDDT", False),
        )
        self.smooth_lddt_context_mode = _env_str(
            "ABFLOW_SMOOTH_LDDT_CONTEXT_MODE", "fixed_observed"
        ).strip().lower()
        if self.smooth_lddt_context_mode != "fixed_observed":
            raise ValueError(
                "V211 formal runs require ABFLOW_SMOOTH_LDDT_CONTEXT_MODE="
                f"fixed_observed, got {self.smooth_lddt_context_mode!r}"
            )
        # Every top-level coefficient is explicit and JSON-controlled.  Setting
        # the four parent weights to 1.0 is algebraically identical to R05.
        self.loss_sequence_weight = _env_float("ABFLOW_LOSS_SEQUENCE_WEIGHT", 1.0)
        self.loss_structure_weight = _env_float("ABFLOW_LOSS_STRUCTURE_WEIGHT", 1.0)
        self.loss_interface_weight = _env_float("ABFLOW_LOSS_INTERFACE_WEIGHT", 1.0)
        self.loss_edge_weight = _env_float("ABFLOW_LOSS_EDGE_WEIGHT", 1.0)
        self.loss_distogram_weight = _env_float("ABFLOW_LOSS_DISTOGRAM_WEIGHT", 0.0)
        self.loss_smooth_lddt_weight = _env_float("ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT", 0.0)
        self.mf_smooth_lddt_cutoff = _env_float("ABFLOW_MF_SMOOTH_LDDT_CUTOFF", 15.0)
        for name, value in (
            ("sequence", self.loss_sequence_weight),
            ("structure", self.loss_structure_weight),
            ("interface", self.loss_interface_weight),
            ("edge", self.loss_edge_weight),
            ("distogram", self.loss_distogram_weight),
            ("smooth_lddt", self.loss_smooth_lddt_weight),
        ):
            if value < 0.0:
                raise ValueError(f"loss weight {name} must be non-negative, got {value}")
        if self.loss_distogram_weight != 0.0 and not self.abx_distogram:
            raise ValueError("non-zero distogram weight requires ABFLOW_ABX_DISTOGRAM=on")
        if self.abx_distogram and not self.abx_native_repr:
            raise ValueError("AbX distogram requires ABFLOW_ABX_NATIVE_REPR=on")
        if self.loss_smooth_lddt_weight != 0.0 and not self.mf_smooth_lddt:
            raise ValueError("non-zero smooth-lDDT weight requires ABFLOW_MF_SMOOTH_LDDT=on")

        # Optional child modules must not consume the RNG stream used to
        # initialize shared R05 parameters.  Without this guard, merely adding
        # the zero-initialized Distogram head changed all subsequently created
        # GNN weights and invalidated R28-vs-R29 attribution.
        if self.abx_native_repr:
            with torch.random.fork_rng(devices=[]):
                # Dedicated CPU generator stream: the donor trunk is identical
                # across experiments yet statistically independent from the
                # restored R05 initialization stream.
                torch.default_generator.manual_seed(self.abx_init_seed)
                self.abx_repr = R05AbXNativeTrunk(
                    enable_distogram=self.abx_distogram,
                    distogram_pair_scope=self.distogram_pair_scope,
                    forward_seed=self.abx_forward_seed,
                )
        else:
            self.abx_repr = None
        gnn_input_dim = embed_size
        gnn_single_dim = self.abx_repr.single_dim if self.abx_repr is not None else 0
        gnn_pair_dim = self.abx_repr.pair_dim if self.abx_repr is not None else 0
        self.gnn = AMEncoder(
            gnn_input_dim,
            hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=gnn_pair_dim,
            in_single_nf=gnn_single_dim,
            num_verts=num_verts, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False,
        )
        bridge_parameters = [
            (name, parameter)
            for name, parameter in self.gnn.named_parameters()
            if name == "single_linear.weight"
            or name.endswith("edge_attr_linear.weight")
        ]
        if self.abx_native_repr and not bridge_parameters:
            raise RuntimeError(
                "V211 bridge initialization contract found no Single/Pair adapters"
            )
        bridge_max_abs = max(
            (float(parameter.detach().abs().max().item())
             for _, parameter in bridge_parameters),
            default=0.0,
        )
        if bridge_max_abs != 0.0:
            raise RuntimeError(
                "V211 parent-preserving bridge must initialize exactly at zero; "
                f"observed max_abs={bridge_max_abs}"
            )
        self._last_abx_state = {}
        self._diagnostic_pair_probe_tensor = None
        self.last_distogram_audit = {}
        self.normalizer = SeperatedCoordNormalizer()
        if _abx_dist_rank() == 0:
            print(
                "[GeometryUnitContract] flow_coordinate_scaling="
                f"{_env_float('ABFLOW_R3_FLOW_COORDINATE_SCALING', 0.1):g} "
                "aux_coord_unit=angstrom distogram_bins_A=[2.3125,21.6875] "
                "smooth_lddt_thresholds_A=[0.5,1,2,4] "
                f"smooth_lddt_cutoff_A={self.mf_smooth_lddt_cutoff:g} "
                "double_scaling=off",
                flush=True,
            )
            print(
                "[LossWeightContract] source=json "
                f"sequence={self.loss_sequence_weight:g} "
                f"structure={self.loss_structure_weight:g} "
                f"interface={self.loss_interface_weight:g} "
                f"edge={self.loss_edge_weight:g} "
                f"distogram={self.loss_distogram_weight:g} "
                f"smooth_lddt={self.loss_smooth_lddt_weight:g}",
                flush=True,
            )
            print(
                "[V211BridgeContract] "
                f"mode={self.abx_bridge_mode} base=exact_R05 "
                "single=W_base*h+b+W_s*s pair=W_base*edge+b+W_z*z "
                "single_init=zero pair_init=zero optional_init_rng=isolate "
                f"abx_init_seed={self.abx_init_seed} "
                f"abx_forward_seed={self.abx_forward_seed} "
                "donor_dropout_rng=isolate "
                f"distogram_pair_scope={self.distogram_pair_scope} "
                f"smooth_lddt_context={self.smooth_lddt_context_mode}",
                flush=True,
            )
            print(
                "[V211BridgeInitPASS] "
                f"adapter_tensors={len(bridge_parameters)} max_abs={bridge_max_abs:.1f} "
                "parent_output_identity=exact",
                flush=True,
            )

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
        if self.scorefm_loss_mode in {
            "foldflow_r3_global_endpoint", "r3_global_endpoint",
            "ff_r3_global_endpoint",
        }:
            self.scorefm_loss_mode = "foldflow_r3_global_endpoint"
        # R05/U02 canonical Score--Flow modes. These aliases are part of the
        # validated R05 scientific contract and must not be dropped by backbone refactors.
        if self.scorefm_loss_mode in {
            "f01_r3_canonical_carrier",
            "ff_r3_canonical_carrier",
            "f01_canonical_scoreflow_carrier",
        }:
            self.scorefm_loss_mode = "f01_r3_canonical_carrier"
        if self.scorefm_loss_mode in {
            "f01_r3_endpoint_canonical_hybrid",
            "ff_r3_endpoint_canonical_hybrid",
            "f01_endpoint_scoreflow_hybrid",
        }:
            self.scorefm_loss_mode = "f01_r3_endpoint_canonical_hybrid"
        if self.scorefm_loss_mode in {
            "foldflow_r3_residue_endpoint", "r3_residue_endpoint",
            "ff_r3_residue_endpoint",
        }:
            self.scorefm_loss_mode = "foldflow_r3_residue_endpoint"
        if self.scorefm_loss_mode in {
            "foldflow_r3_residue_cfm", "r3_residue_cfm",
            "ff_r3_residue_cfm",
        }:
            self.scorefm_loss_mode = "foldflow_r3_residue_cfm"
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
            "foldflow_r3_global_endpoint",
            "f01_r3_canonical_carrier",
            "f01_r3_endpoint_canonical_hybrid",
            "foldflow_r3_residue_endpoint",
            "foldflow_r3_residue_cfm",
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
                "structured_global_cfm, structured_multiscale_cfm, foldflow_r3_global_endpoint, "
                "f01_r3_canonical_carrier, f01_r3_endpoint_canonical_hybrid, "
                "foldflow_r3_residue_endpoint, foldflow_r3_residue_cfm."
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

        # =========================================================
        # FoldFlow-R3-inspired primary path (v59)
        # =========================================================
        # transport_fraction is a path-width parameter, NOT a loss weight.
        # It is kept at 0.05 by default solely to match S01's already-tested
        # midpoint H3 displacement scale while changing the temporal profile to
        # FoldFlow R3's sqrt(t(1-t)) variance law.
        self.r3_transport_fraction = _env_float(
            "ABFLOW_R3_TRANSPORT_FRACTION", 0.05
        )
        self.r3_path_min_sigma = _env_float(
            "ABFLOW_R3_PATH_MIN_SIGMA", 0.0
        )
        self.r3_transport_max = _env_float(
            "ABFLOW_R3_TRANSPORT_MAX", 20.0
        )
        # R05 v104 stochastic-width/support contract.
        self.r3_g_mode = _env_str(
            "ABFLOW_R3_G_MODE", "adaptive_transport"
        ).strip().lower()
        self.r3_fixed_g_scaled = _env_float(
            "ABFLOW_R3_FIXED_G_SCALED", 0.1
        )
        self.r3_noise_scope = _env_str(
            "ABFLOW_R3_NOISE_SCOPE", "global"
        ).strip().lower()
        self.flow_coordinate_scaling = _env_float(
            "ABFLOW_R3_FLOW_COORDINATE_SCALING", 0.1
        )
        if not (0.0 < self.r3_transport_fraction <= 1.0):
            raise ValueError("ABFLOW_R3_TRANSPORT_FRACTION must be in (0,1].")
        if self.r3_path_min_sigma < 0.0:
            raise ValueError("ABFLOW_R3_PATH_MIN_SIGMA must be non-negative.")
        if self.r3_transport_max <= 0.0:
            raise ValueError("ABFLOW_R3_TRANSPORT_MAX must be positive.")
        if self.r3_g_mode not in {"adaptive_transport", "foldflow_fixed_scaled"}:
            raise ValueError("ABFLOW_R3_G_MODE must be adaptive_transport or foldflow_fixed_scaled.")
        if self.r3_fixed_g_scaled <= 0.0:
            raise ValueError("ABFLOW_R3_FIXED_G_SCALED must be positive.")
        if self.r3_noise_scope not in {"global", "residue"}:
            raise ValueError("ABFLOW_R3_NOISE_SCOPE must be global or residue.")
        if not (0.0 < self.flow_coordinate_scaling <= 1.0):
            raise ValueError("ABFLOW_R3_FLOW_COORDINATE_SCALING must be in (0,1].")
        self.r3_matcher = AbFlowR3Matcher(
            transport_fraction=self.r3_transport_fraction,
            path_min_sigma=self.r3_path_min_sigma,
            eps=self.scorefm_eps,
        )

        # R05/U02 canonical carrier boundary. U02 deliberately keeps the clean
        # endpoint anchor below t=0.20 and uses the canonical carrier above it.
        self.f01_canonical_t_min = _env_float("ABFLOW_F01_CANONICAL_T_MIN", 0.05)
        self.f01_hybrid_t_min = _env_float("ABFLOW_F01_HYBRID_T_MIN", 0.20)
        if not (0.0 < self.f01_canonical_t_min < 1.0):
            raise ValueError("ABFLOW_F01_CANONICAL_T_MIN must be in (0,1).")
        if not (0.0 < self.f01_hybrid_t_min < 1.0):
            raise ValueError("ABFLOW_F01_HYBRID_T_MIN must be in (0,1).")

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

        # R05 node-level flow-time conditioning.  The same flow_t is also fed
        # through the source-faithful AbX timestep embedder, which conditions
        # both AbX single and pair states. These are two representation routes
        # for one transport-time variable, not two independent clocks.
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
        if self.scorefm_sampler_mode not in {
            "residual", "bridge", "f01_canonical_carrier"
        }:
            raise ValueError(
                "Unknown ABFLOW_SCOREFM_SAMPLER_MODE="
                f"{self.scorefm_sampler_mode}. Choose from residual, bridge, "
                "f01_canonical_carrier."
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
        self._diagnostic_pair_probe_tensor = None
        self._last_gradient_diagnostic_error = ""
        self._diagnostic_validation_mode = False
        # Set by the trainer only on recorded/probed steps so diagnostics do not
        # turn every expensive training batch into a synchronization point.
        self._diagnostic_capture = False

        # Backward-compatible attribute used by historical diagnostics.  The
        # canonical source is ABFLOW_LOSS_SEQUENCE_WEIGHT in the JSON.
        self.seq_ce_weight = self.loss_sequence_weight

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
                        sequence_state_full=None,
                        abx_persistent_state=None):
        # embeddings, hidden state, (internal edges, external edges),
        # (A : c*d, w : c*1)
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(
            X, S, batch_id, self.k_neighbors, residue_pos,
            smooth_prob=smooth_prob, smooth_mask=smooth_mask
        )
        # Audit-only snapshot of the original R05 residue feature before time
        # and condition residuals.  V211 never subtracts it: the parent feature
        # path remains intact and AbX single enters through a zero-start affine
        # residual inside AMEncoder.
        parent_static_H = H_0

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

        # ---------------------------------------------------------
        # V182 AbX s/z -> native R05 EGNN interfaces
        # ---------------------------------------------------------
        # Biological nodes are real amino acids plus the active design residues
        # (which may carry a donor MASK token). BOH/BOL/BOA global helper nodes
        # are intentionally absent from AbX; their edge attributes are zero.
        # V200_PERSISTENT_ABX_OUTSIDE_R05
        # AbX single/pair is a representation condition, not a recurrent state.
        # The full AbX trunk is executed exactly once before the R05x3 loop.
        # This round only gathers the persistent dense z_ij to the CURRENT sparse
        # R05 edges, whose topology remains geometry-dependent and round-specific.
        abx_state = abx_persistent_state
        ctx_pair_attr = inter_pair_attr = surf_pair_attr = None
        if abx_state is not None:
            biological = abx_state['biological_mask']
            self._last_abx_state = abx_state
            abx_single = abx_state['single_global'].to(
                device=H_0.device, dtype=H_0.dtype
            )

            # Parent-preserving boundary.  R05 supplies the complete input;
            # AbX single is passed separately to the zero-start W_s adapter.
            # This has the same eventual affine capacity as concatenation but
            # is exactly the parent function while W_s=0.
            parent_dynamic = H_0 - parent_static_H
            H_gnn = H_0

            if diagnostics_active and bool(biological.any()):
                self._last_condition_diagnostics.update({
                    'r05_parent_static_bio_rms': torch.sqrt(parent_static_H[biological].float().pow(2).mean() + 1e-8).to(H_0.dtype),
                    'r05_parent_dynamic_bio_rms': torch.sqrt(parent_dynamic[biological].float().pow(2).mean() + 1e-8).to(H_0.dtype),
                    'abx_single_bio_rms': torch.sqrt(abx_single[biological].float().pow(2).mean() + 1e-8).to(H_0.dtype),
                    'r05_time_embed_on': H_0.new_tensor(1.0 if getattr(self, 'scorefm_time_embed', False) else 0.0),
                    'abx_time_embed_on': H_0.new_tensor(1.0 if self.abx_repr.trunk.use_abx_time else 0.0),
                })

            # Dense donor z_ij is gathered in the exact current sparse edge
            # order of the original R05 EGNN. No second coordinate decoder.
            ctx_pair_attr = self.abx_repr.gather_pair(
                abx_state['pair_dense'], ctx_edges,
                abx_state['node_graph'], abx_state['node_local'],
            ).to(H_0.dtype)
            local_global = torch.nonzero(local_mask, as_tuple=False).flatten()
            local_edges_global = local_global[local_edges]
            surf_edges_global = local_global[aligned_local_inter_edges]
            inter_pair_attr = self.abx_repr.gather_pair(
                abx_state['pair_dense'], local_edges_global,
                abx_state['node_graph'], abx_state['node_local'],
            ).to(H_0.dtype)
            surf_pair_attr = self.abx_repr.gather_pair(
                abx_state['pair_dense'], surf_edges_global,
                abx_state['node_graph'], abx_state['node_local'],
            ).to(H_0.dtype)
        else:
            # Compatibility route for non-V211 configs. All three formal V211
            # experiments keep Pair enabled and therefore do not take this path.
            H_gnn = H_0

        if diagnostics_active:
            def _edge_rms(v):
                if v is None or v.numel() == 0:
                    return H_0.new_tensor(0.0)
                return torch.sqrt(v.float().pow(2).mean() + 1e-8).to(H_0.dtype)
            self._last_condition_diagnostics.update({
                'abx_ctx_edge_attr_rms': _edge_rms(ctx_pair_attr),
                'abx_inter_edge_attr_rms': _edge_rms(inter_pair_attr),
                'abx_surf_edge_attr_rms': _edge_rms(surf_pair_attr),
                'abx_ctx_edge_count': H_0.new_tensor(float(0 if ctx_pair_attr is None else ctx_pair_attr.shape[0])),
                'abx_inter_edge_count': H_0.new_tensor(float(0 if inter_pair_attr is None else inter_pair_attr.shape[0])),
                'abx_surf_edge_count': H_0.new_tensor(float(0 if surf_pair_attr is None else surf_pair_attr.shape[0])),
            })

        H, pred_X, pred_local_X = self.gnn(
            H_gnn, X, ctx_edges, local_mask, local_X, surf, local_edges,
            paratope_mask, local_is_ab, aligned_local_inter_edges, epi_index,
            channel_attr=atom_embeddings, channel_weights=atom_weights,
            ctx_edge_attr=ctx_pair_attr, inter_edge_attr=inter_pair_attr,
            surf_edge_attr=surf_pair_attr,
            single_attr=(abx_single if abx_state is not None else None),
            capture_bridge_diagnostics=diagnostics_active,
        )
        if diagnostics_active:
            self._last_condition_diagnostics.update({
                key: value.detach()
                for key, value in getattr(
                    self.gnn, "last_bridge_diagnostics", {}
                ).items()
            })
        _diag_ctx = abx_state.get('diag_context', None) if abx_state is not None else None
        _assert_finite_tensor("r05_gnn.H", H, _diag_ctx)
        _assert_finite_tensor("r05_gnn.pred_X", pred_X, _diag_ctx)
        _assert_finite_tensor("r05_gnn.pred_local_X", pred_local_X, _diag_ctx)

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
    def _foldflow_r3_primary_path(
            self, *, source_X0, target_X1, t_graph, t_int,
            interface_batch_id, noise_scope, cfm_target):
        """FoldFlow-R3-inspired primary stochastic state for H3.

        Common mean:
            mu_t = (1-t) X0 + t X1.

        Temporal width follows uploaded FoldFlow R3:
            sigma_t = sqrt(g^2 t(1-t) + sigma_min^2).

        Adaptation to full-atom antibodies:
          global  : one 3D translation shared by every H3 atom;
          residue : one 3D translation per H3 residue, broadcast to all atoms
                    in that residue.  This preserves intra-residue atom geometry.

        We intentionally DO NOT apply FoldFlow's whole-chain COM recentering,
        because H3/framework/antigen relative placement is itself a target signal.

        Targets:
          endpoint : clean native X1 (denoising endpoint regression).
          cfm      : FoldFlow Euclidean CFM u*=X1-X0, encoded as
                     Y*=Xt+(1-t)u* for AbFlow's endpoint parameterization.
        """
        mu_t = self.r3_matcher.linear_mean(source_X0, target_X1, t_int)
        if interface_batch_id.numel() == 0:
            zero = target_X1.new_zeros(1)
            return mu_t, target_X1, {
                "r3_transport_mean": zero,
                "r3_sigma_mean": zero,
                "r3_noise_rms": zero,
                "r3_target_shift_rms": zero,
                "r3_noise_scope": zero,
                "r3_cfm_target": zero,
            }

        n_graph = int(interface_batch_id.max().item()) + 1
        ca_idx = 1 if source_X0.shape[1] > 1 else 0
        src_centroid = scatter_mean(
            source_X0[:, ca_idx].float(), interface_batch_id,
            dim=0, dim_size=n_graph,
        )
        tgt_centroid = scatter_mean(
            target_X1[:, ca_idx].float(), interface_batch_id,
            dim=0, dim_size=n_graph,
        )
        transport = torch.linalg.norm(
            tgt_centroid - src_centroid, dim=-1
        ).clamp(min=0.0, max=float(self.r3_transport_max))

        if self.r3_g_mode == "adaptive_transport":
            g_graph = self.r3_matcher.graph_g_from_transport(transport)
            g_mode_code = 0.0
            g_raw_value = float(g_graph.mean().detach().cpu().item())
        elif self.r3_g_mode == "foldflow_fixed_scaled":
            g_raw = self.r3_matcher.foldflow_scaled_g_to_raw(
                g_scaled=float(self.r3_fixed_g_scaled),
                coordinate_scaling=float(self.flow_coordinate_scaling),
            )
            g_graph = torch.full_like(transport, float(g_raw))
            g_mode_code = 1.0
            g_raw_value = float(g_raw)
        else:
            raise RuntimeError(f"Unsupported R3 g mode: {self.r3_g_mode}")
        t_graph_f = torch.as_tensor(
            t_graph, device=target_X1.device, dtype=torch.float32
        ).reshape(-1)
        if t_graph_f.numel() == 1 and n_graph > 1:
            t_graph_f = t_graph_f.expand(n_graph)
        if t_graph_f.numel() != n_graph:
            raise ValueError(
                f"FoldFlow-R3 path expects {n_graph} graph times, "
                f"got {t_graph_f.numel()}."
            )
        sigma_graph = self.r3_matcher.sigma_t(t_graph_f, g_graph)

        if noise_scope == "global":
            if self.deterministic_validation and not self.training:
                eps_graph = self._deterministic_standard_normal(
                    (n_graph, 3), target_X1.device, torch.float32
                )
            else:
                eps_graph = torch.randn(
                    (n_graph, 3), device=target_X1.device, dtype=torch.float32
                )
            shift_graph = sigma_graph[:, None] * eps_graph
            shift_res = shift_graph[interface_batch_id]
            scope_code = 0.0
        elif noise_scope == "residue":
            n_res = int(interface_batch_id.numel())
            if self.deterministic_validation and not self.training:
                eps_res = self._deterministic_standard_normal(
                    (n_res, 3), target_X1.device, torch.float32
                )
            else:
                eps_res = torch.randn(
                    (n_res, 3), device=target_X1.device, dtype=torch.float32
                )
            # FoldFlow R3 acts on residue translations.  In our all-atom state,
            # broadcast that residue translation to every atom in the residue.
            shift_res = sigma_graph[interface_batch_id, None] * eps_res
            scope_code = 1.0
        else:
            raise ValueError(f"Unknown R3 noise_scope={noise_scope}")

        shift_res = shift_res.to(mu_t.dtype)
        Xt = mu_t + shift_res[:, None, :]

        if cfm_target:
            clean_u = self.r3_matcher.clean_conditional_velocity(
                source_X0, target_X1
            )
            target = self.r3_matcher.endpoint_target_for_velocity(
                Xt, clean_u, t_int
            )
        else:
            target = target_X1

        with torch.no_grad():
            target_shift = target - target_X1
            return Xt, target, {
                "r3_transport_mean": transport.mean().to(target_X1.dtype),
                "r3_sigma_mean": sigma_graph.mean().to(target_X1.dtype),
                "r3_noise_rms": torch.sqrt(
                    shift_res.pow(2).sum(dim=-1).mean().clamp_min(0.0)
                ).to(target_X1.dtype),
                "r3_target_shift_rms": torch.sqrt(
                    target_shift.pow(2).mean().clamp_min(0.0)
                ).to(target_X1.dtype),
                "r3_noise_scope": target_X1.new_tensor(scope_code),
                "r3_g_mode": target_X1.new_tensor(g_mode_code),
                "r3_g_raw": target_X1.new_tensor(g_raw_value),
                "r3_cfm_target": target_X1.new_tensor(1.0 if cfm_target else 0.0),
            }

    @torch.no_grad()
    def _f01_unified_scoreflow_target(
            self, *, Xt, source_X0, target_X1, t_int, t_min):
        """R05/U02 single coordinate target for endpoint-parameterized Score--Flow.

        For t < t_min, supervise the clean endpoint X1. For t >= t_min,
        supervise the g-free canonical carrier
            Y* = X1 + (Xt - ((1-t)X0 + tX1)) / (2t).
        This is target replacement, not an auxiliary score/flow loss.
        """
        canonical = self.r3_matcher.canonical_carrier_target_gfree(
            Xt, source_X0, target_X1, t_int, boundary_eps=float(t_min)
        )
        t_res = torch.as_tensor(t_int, device=Xt.device, dtype=Xt.dtype)
        while t_res.dim() < Xt.dim():
            t_res = t_res.unsqueeze(-1)
        active = t_res >= float(t_min)
        target = torch.where(active, canonical, target_X1)
        with torch.no_grad():
            target_shift = target - target_X1
            active_res = active.reshape(active.shape[0], -1).any(dim=-1)
            return target, {
                "r3_canonical_active_rate": active_res.float().mean(),
                "r3_canonical_target_shift_rms": torch.sqrt(
                    target_shift.pow(2).mean().clamp_min(0.0)
                ),
                "r3_canonical_t_min": target_X1.new_tensor(float(t_min)),
            }

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
            self.scorefm_loss_mode in {
                "structured_global_cfm", "structured_multiscale_cfm",
                "foldflow_r3_residue_cfm",
                "f01_r3_canonical_carrier",
                "f01_r3_endpoint_canonical_hybrid",
            }
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

        # FoldFlow-style t-stratified diagnostics: observational only.
        # This mirrors the useful diagnostic principle in experiments_utils.py
        # without changing any gradient or training weight.
        tbin_details = {}
        try:
            n_graph_diag = int(endpoint_per_graph.shape[0])
            t_res_diag = torch.as_tensor(
                t, device=pred_clean_X.device, dtype=torch.float32
            ).reshape(interface_batch_id.numel(), -1).mean(dim=-1)
            t_graph_diag = scatter_mean(
                t_res_diag, interface_batch_id, dim=0, dim_size=n_graph_diag
            )
            for _bi, (_lo, _hi) in enumerate(
                [(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0001)]
            ):
                _m = endpoint_valid & (t_graph_diag >= _lo) & (t_graph_diag < _hi)
                _v = (
                    endpoint_per_graph[_m].mean().detach()
                    if bool(_m.any()) else endpoint_loss.detach() * 0.0
                )
                tbin_details[f"scorefm_tbin_{_bi}_loss"] = _v
        except Exception:
            # Diagnostics must never change training behavior.
            tbin_details = {}

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
            "foldflow_r3_global_endpoint",
            "f01_r3_canonical_carrier",
            "f01_r3_endpoint_canonical_hybrid",
            "foldflow_r3_residue_endpoint",
            "foldflow_r3_residue_cfm",
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
                "scorefm_r3_transport_mean": torch.as_tensor(
                    _spd.get("r3_transport_mean", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_r3_sigma_mean": torch.as_tensor(
                    _spd.get("r3_sigma_mean", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_r3_noise_rms": torch.as_tensor(
                    _spd.get("r3_noise_rms", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_r3_target_shift_rms": torch.as_tensor(
                    _spd.get("r3_target_shift_rms", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_r3_noise_scope": torch.as_tensor(
                    _spd.get("r3_noise_scope", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_r3_g_mode": torch.as_tensor(
                    _spd.get("r3_g_mode", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_r3_g_raw": torch.as_tensor(
                    _spd.get("r3_g_raw", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_r3_canonical_active_rate": torch.as_tensor(
                    _spd.get("r3_canonical_active_rate", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_r3_canonical_target_shift_rms": torch.as_tensor(
                    _spd.get("r3_canonical_target_shift_rms", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_r3_canonical_t_min": torch.as_tensor(
                    _spd.get("r3_canonical_t_min", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_r3_unified_scoreflow": pred_clean_X.new_tensor(
                    1.0 if self.scorefm_loss_mode in {
                        "f01_r3_canonical_carrier",
                        "f01_r3_endpoint_canonical_hybrid",
                    } else 0.0
                ).detach(),
                "scorefm_r3_cfm_target": torch.as_tensor(
                    _spd.get("r3_cfm_target", 0.0),
                    device=pred_clean_X.device, dtype=pred_clean_X.dtype
                ).detach(),
                "scorefm_structured_cfm": pred_clean_X.new_tensor(
                    1.0 if self.scorefm_loss_mode in {"structured_global_cfm", "structured_multiscale_cfm", "foldflow_r3_residue_cfm"} else 0.0
                ).detach(),
                "scorefm_dsm": zero,
                "scorefm_dsm_rate": zero,
                "scorefm_velocity": zero,
                "scorefm_velocity_rate": zero,
                "scorefm_traj_consistency": zero,
                "scorefm_traj_velocity": zero,
                "scorefm_traj_rate": zero,
            }
            details.update(tbin_details)
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
                 interface_init=None, sequence_init=None, flow_t=None,
                 flow_source_init=None):
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

        # Keep the queried transport state fixed across the three refinement
        # rounds.  The previous round's Endpoint changes; X_t and X_0 do not.
        scoreflow_state_X = interface_X.clone()
        scoreflow_source_X = None
        if flow_source_init is not None:
            if flow_source_init.shape != interface_X.shape:
                raise ValueError(
                    "flow_source_init/interface shape mismatch: "
                    f"{tuple(flow_source_init.shape)} vs "
                    f"{tuple(interface_X.shape)}"
                )
            scoreflow_source_X = self._raw_interface_to_model_frame(
                flow_source_init, paratope_mask, batch_id
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

        # V200_PERSISTENT_ABX_OUTSIDE_R05
        # R05 is the sole recurrence authority. AbX s_i/z_ij is computed once
        # from the outer transport state and reused across all three physical
        # refinement rounds. Design-region donor geometry remains masked by the
        # existing fixed_mask contract; antigen/framework context and flow_t are
        # unchanged across rounds. Sparse EGNN edge sets are still rebuilt each
        # round and gather from this same dense z_ij.
        abx_persistent_state = None
        self._last_abx_state = {}
        if self.abx_repr is not None:
            abx_S_static = sequence_state_full if sequence_state_full is not None else S
            abx_biological = paratope_mask.bool() | (
                (abx_S_static >= 0) & (abx_S_static < self.num_classes)
            )
            abx_X_static = X.clone()
            abx_X_static[paratope_mask] = interface_X.to(dtype=abx_X_static.dtype)
            abx_persistent_state = self.abx_repr(
                X=abx_X_static,
                S=abx_S_static,
                segment_ids=self.batch_constants['segment_ids'],
                residue_pos=residue_pos,
                batch_id=batch_id,
                valid_mask=abx_biological,
                is_antigen=self.batch_constants['is_ag'],
                design_mask=paratope_mask,
                flow_t=flow_t,
                cdr_type=self.cdr_type,
                round_idx=-1,
                antigen_context_mask=self.batch_constants['local_mask'],
                atom_observed_mask=self.batch_constants.get('xloss_mask'),
            )
            self._last_abx_state = abx_persistent_state
            if bool(getattr(self, "_diagnostic_capture", False)):
                self._diagnostic_pair_probe_tensor = abx_persistent_state['pair_dense']
            if (
                _abx_dist_rank() == 0
                and not bool(getattr(self, "_v208_representation_audit_printed", False))
            ):
                c = self.abx_repr.trunk.config
                print(
                    "[V211RepresentationContract] mode=R05+AbX "
                    f"trunk_per_outer=1 R05_rounds={int(self.round)} "
                    "single=persistent dense_pair=persistent sparse_pair_gather=per_round "
                    f"profile={c.width_profile} single_dim={self.abx_repr.single_dim} "
                    f"pair_dim={self.abx_repr.pair_dim} "
                    f"triangle_chunk={_env_int('ABFLOW_ABX_TRIANGLE_CHUNK_SIZE', 32)}",
                    flush=True,
                )
                self._v208_representation_audit_printed = True
        elif _abx_dist_rank() == 0 and not bool(
            getattr(self, "_v208_representation_audit_printed", False)
        ):
            print(
                "[V211RepresentationContract] mode=compat-direct-R05 AbX=off "
                f"R05_rounds={int(self.round)} native_edge_attr=none",
                flush=True,
            )
            self._v208_representation_audit_printed = True

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
                abx_persistent_state=abx_persistent_state,
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
            # Sparse surface routes can legitimately be empty in only some
            # rounds.  Average diagnostics available in every round rather
            # than assuming identical optional key sets.
            keys = set(condition_diag_rounds[0])
            for round_diag in condition_diag_rounds[1:]:
                keys.intersection_update(round_diag)
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
            paratope_mask, smask, batch_id, interface_batch_id, t_graph):
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

        final_sequence_diag = None
        for ridx, (logits, mask) in enumerate(r_pred_S_logits):
            if bool(mask.any()):
                logits_masked = logits[mask].float()
                probs = torch.softmax(logits_masked, dim=-1)
                pred_round = torch.argmax(logits_masked, dim=-1)
                out[f"val_proxy_round{ridx}_aar"] = (
                    pred_round == true_S[mask]
                ).float().mean()
                entropy = -(
                    probs * probs.clamp_min(1.0e-8).log()
                ).sum(dim=-1).mean()
                sorted_prob = probs.topk(k=min(2, probs.shape[-1]), dim=-1).values
                top1_margin = (
                    (sorted_prob[:, 0] - sorted_prob[:, 1]).mean()
                    if sorted_prob.shape[-1] > 1
                    else sorted_prob[:, 0].mean()
                )
                native_prob = probs.gather(
                    1, true_S[mask].long().unsqueeze(-1)
                ).squeeze(-1).mean()
                bincount = torch.bincount(
                    pred_round, minlength=self.num_classes
                ).float()
                round_diag = {
                    "seq_entropy": entropy,
                    "seq_max_prob": probs.max(dim=-1).values.mean(),
                    "seq_native_prob": native_prob,
                    "seq_top1_margin": top1_margin,
                    "seq_dominant_map_fraction": (
                        bincount.max() / bincount.sum().clamp_min(1.0)
                    ),
                    "seq_unique_map_classes": (bincount > 0).float().sum(),
                }
                for name, value in round_diag.items():
                    out[f"val_proxy_round{ridx}_{name}"] = value
                final_sequence_diag = round_diag

        if final_sequence_diag is not None:
            for name, value in final_sequence_diag.items():
                out[f"val_proxy_{name}"] = value

        if round_raw:
            out["val_proxy_refinement_raw_rmsd_delta"] = (
                round_raw[-1] - round_raw[0]
            )
        if round_aligned:
            out["val_proxy_refinement_aligned_rmsd_delta"] = (
                round_aligned[-1] - round_aligned[0]
            )

        # Per-time-bin sums are emitted with their cardinalities so the DDP
        # trainer can aggregate exact sample-weighted means.  The prior logger
        # declared these fields but the model never produced them, yielding NaN.
        t_values = torch.as_tensor(
            t_graph, device=true_ca.device, dtype=true_ca.dtype
        ).reshape(-1)
        if t_values.numel() == 1:
            t_values = t_values.expand(n_graph)
        if t_values.numel() == n_graph:
            final_ca_for_bins = r_interface_X[-1][:, ca_idx].float()
            for g in range(n_graph):
                m = interface_batch_id == g
                if not bool(m.any()):
                    continue
                bin_idx = min(4, max(0, int(torch.floor(
                    t_values[g].clamp(0.0, 1.0) * 5.0
                ).item())))
                p, q = final_ca_for_bins[m], true_ca[m]
                raw = torch.sqrt(((p - q) ** 2).sum(-1).mean())
                count_key = f"val_proxy_timebin{bin_idx}_count"
                sum_key = f"val_proxy_timebin{bin_idx}_h3_ca_rmsd_sum"
                out[count_key] = out.get(count_key, raw.new_zeros(())) + 1.0
                out[sum_key] = out.get(sum_key, raw.new_zeros(())) + raw
                if p.shape[0] >= 3:
                    try:
                        _, rot, trans = kabsch_torch(p, q)
                        aligned_p = torch.matmul(p, rot.T) + trans
                        aligned = torch.sqrt(
                            ((aligned_p - q) ** 2).sum(-1).mean()
                        )
                        acount = f"val_proxy_timebin{bin_idx}_aligned_count"
                        asum = (
                            f"val_proxy_timebin{bin_idx}_h3_ca_aligned_rmsd_sum"
                        )
                        out[acount] = out.get(acount, aligned.new_zeros(())) + 1.0
                        out[asum] = out.get(asum, aligned.new_zeros(())) + aligned
                    except Exception:
                        pass

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
            # V185 diagnostics only: quantify whether the new pair/local-geometry
            # auxiliaries agree with the established R05 endpoint/structure path.
            ("endpoint", "distogram"),
            ("seq", "distogram"),
            ("structure", "distogram"),
            ("endpoint", "smooth_lddt"),
            ("structure", "smooth_lddt"),
        ]
        for a, b in pairs:
            if a in grads and b in grads:
                ga, gb = grads[a], grads[b]
                out[f"grad_probe_cos_{a}_{b}"] = (
                    torch.dot(ga, gb)
                    / (torch.linalg.norm(ga) * torch.linalg.norm(gb) + self.scorefm_eps)
                )

        # Distogram bypasses the final R05 H_0 activation, while smooth-lDDT
        # reaches coordinates through R05. Both auxiliaries share the earlier
        # dense AbX pair state z with the generator in formal V211 experiments,
        # so probe z directly to compare their actual representation gradients.
        pair_probe = self._diagnostic_pair_probe_tensor
        if (
            pair_probe is not None
            and torch.is_tensor(pair_probe)
            and pair_probe.requires_grad
        ):
            pair_grads = {}
            try:
                with (torch.cuda.amp.autocast(enabled=False)
                      if pair_probe.is_cuda else nullcontext()):
                    for name in (
                        "endpoint", "structure", "seq", "distogram", "smooth_lddt"
                    ):
                        value = terms.get(name)
                        if not torch.is_tensor(value) or not value.requires_grad:
                            continue
                        scalar = value.float()
                        if scalar.numel() != 1:
                            scalar = scalar.mean()
                        g = torch.autograd.grad(
                            scalar, pair_probe, retain_graph=True, allow_unused=True
                        )[0]
                        if g is not None:
                            pair_grads[name] = g.detach().float().reshape(-1)
            except RuntimeError as exc:
                self._last_gradient_diagnostic_error = (
                    self._last_gradient_diagnostic_error + " | pair-z: " + str(exc)
                ).strip(" |")
            for name, grad in pair_grads.items():
                out[f"grad_pair_norm_{name}"] = torch.linalg.norm(grad)
            for a, b in (
                ("distogram", "endpoint"),
                ("distogram", "structure"),
                ("distogram", "seq"),
                ("smooth_lddt", "endpoint"),
                ("smooth_lddt", "structure"),
                ("smooth_lddt", "seq"),
            ):
                if a in pair_grads and b in pair_grads:
                    ga, gb = pair_grads[a], pair_grads[b]
                    out[f"grad_pair_cos_{a}_{b}"] = (
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
        self._diagnostic_pair_probe_tensor = None
        self._last_gradient_diagnostic_error = ""
        cmask, smask = self._enforce_task_sequence_mask_contract(
            cmask, smask, paratope_mask, template, stage="forward"
        )
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
            if X_pep is not None:
                X_pep = X_pep[:, :4]
            xloss_mask = xloss_mask[:, :4]
        # clone ground truth coordinates, sequence
        true_X, true_S = X.clone(), S.clone()

        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)
        # xloss_mask is the authoritative AbFlow resolved-atom mask.  Keep it in
        # per-batch constants so all three R05 rounds and any internal re-query
        # share the identical donor boundary without changing trainer APIs.
        self.batch_constants['xloss_mask'] = xloss_mask.bool()
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
            if self.scorefm_loss_mode == "foldflow_r3_global_endpoint":
                Xt, structured_endpoint_target, structured_path_details = (
                    self._foldflow_r3_primary_path(
                        source_X0=interface_X, target_X1=gt_interface_X,
                        t_graph=t_graph, t_int=t_int,
                        interface_batch_id=interface_batch_id,
                        noise_scope="global", cfm_target=False,
                    )
                )
            elif self.scorefm_loss_mode in {
                "f01_r3_canonical_carrier",
                "f01_r3_endpoint_canonical_hybrid",
            }:
                # R05/U02: keep the configured R3 support/g policy and replace
                # only the coordinate target by the unified canonical carrier.
                Xt, _, structured_path_details = (
                    self._foldflow_r3_primary_path(
                        source_X0=interface_X, target_X1=gt_interface_X,
                        t_graph=t_graph, t_int=t_int,
                        interface_batch_id=interface_batch_id,
                        noise_scope=self.r3_noise_scope, cfm_target=False,
                    )
                )
                _t_min = (
                    self.f01_canonical_t_min
                    if self.scorefm_loss_mode == "f01_r3_canonical_carrier"
                    else self.f01_hybrid_t_min
                )
                structured_endpoint_target, _canon_diag = (
                    self._f01_unified_scoreflow_target(
                        Xt=Xt, source_X0=interface_X,
                        target_X1=gt_interface_X, t_int=t_int, t_min=_t_min,
                    )
                )
                structured_path_details = dict(structured_path_details or {})
                structured_path_details.update(_canon_diag)
            elif self.scorefm_loss_mode == "foldflow_r3_residue_endpoint":
                Xt, structured_endpoint_target, structured_path_details = (
                    self._foldflow_r3_primary_path(
                        source_X0=interface_X, target_X1=gt_interface_X,
                        t_graph=t_graph, t_int=t_int,
                        interface_batch_id=interface_batch_id,
                        noise_scope="residue", cfm_target=False,
                    )
                )
            elif self.scorefm_loss_mode == "foldflow_r3_residue_cfm":
                Xt, structured_endpoint_target, structured_path_details = (
                    self._foldflow_r3_primary_path(
                        source_X0=interface_X, target_X1=gt_interface_X,
                        t_graph=t_graph, t_int=t_int,
                        interface_batch_id=interface_batch_id,
                        noise_scope="residue", cfm_target=True,
                    )
                )
            elif self.scorefm_loss_mode in {
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
            flow_t=t_graph if state_path else None,
            flow_source_init=interface_X if state_path else None,
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

        # 3. Gold-standard localized donor objectives.
        # R28/R29/R30 share the same AbX representation path. R29 adds only the
        # AbX Distogram objective; R30 adds only the MF/Boltz smooth-lDDT
        # coordinate objective. Thus both are controlled children of R28.
        abx_distogram_loss = X.new_tensor(0.0)
        distogram_audit = {}
        if self.abx_distogram:
            abx_distogram_loss, distogram_audit = self.abx_repr.distogram_loss_from_native(
                self._last_abx_state, true_X, true_S
            )
        self.last_distogram_audit = distogram_audit

        smooth_lddt_loss = X.new_tensor(0.0)
        smooth_lddt_diag = {
            'intra': X.new_tensor(0.0), 'scaffold': X.new_tensor(0.0),
            'antigen': X.new_tensor(0.0), 'intra_pairs': X.new_tensor(0.0),
            'scaffold_pairs': X.new_tensor(0.0), 'antigen_pairs': X.new_tensor(0.0),
            'fixed_context_pred_drift_rms': X.new_tensor(0.0),
            'fixed_context_restored_rate': X.new_tensor(0.0),
        }
        if self.mf_smooth_lddt:
            # xloss_mask is the resolved-coordinate authority. Chemical atom
            # existence alone is insufficient because PDB atoms may be missing.
            valid_atom_mask = self.batch_constants['xloss_mask'].bool()
            # Match the actual generator semantics.  _forward predicts all
            # residues internally, but sample() writes back only ``cmask``;
            # every other coordinate is observed fixed context.  Scoring the
            # discarded internal context prediction made the old design-antigen
            # term saturate near 1.0 and optimized a geometry never generated.
            # Restoring fixed context is not target leakage: those coordinates
            # are already supplied as conditioning input at inference time.
            aux_pred_X = true_X.clone()
            aux_pred_X[cmask] = pred_X[cmask]
            fixed_atom_mask = valid_atom_mask & (~cmask[:, None])
            if bool(fixed_atom_mask.any()):
                fixed_drift = torch.sqrt(
                    (pred_X.detach()[fixed_atom_mask].float()
                     - true_X.detach()[fixed_atom_mask].float())
                    .square().mean()
                ).to(X.dtype)
            else:
                fixed_drift = X.new_tensor(0.0)
            smooth_lddt_loss, smooth_lddt_diag = design_region_smooth_lddt_loss(
                pred_X=aux_pred_X, true_X=true_X,
                valid_atom_mask=valid_atom_mask,
                design_residue_mask=paratope_mask, batch_id=batch_id,
                is_antigen_mask=self.batch_constants['is_ag'],
                cutoff=self.mf_smooth_lddt_cutoff,
            )
            smooth_lddt_diag['fixed_context_pred_drift_rms'] = fixed_drift
            smooth_lddt_diag['fixed_context_restored_rate'] = (
                fixed_atom_mask.float().sum()
                / valid_atom_mask.float().sum().clamp_min(1.0)
            ).to(X.dtype)

        scorefm_details.update({
            'abx_native_repr_enabled': X.detach().new_tensor(float(self.abx_native_repr)),
            'abx_distogram_enabled': X.detach().new_tensor(float(self.abx_distogram)),
            'mf_smooth_lddt_enabled': X.detach().new_tensor(float(self.mf_smooth_lddt)),
            'abx_distogram_loss': abx_distogram_loss.detach(),
            'mf_smooth_lddt_loss': smooth_lddt_loss.detach(),
            'mf_smooth_lddt_intra_loss': smooth_lddt_diag['intra'].detach(),
            'mf_smooth_lddt_scaffold_loss': smooth_lddt_diag['scaffold'].detach(),
            'mf_smooth_lddt_antigen_loss': smooth_lddt_diag['antigen'].detach(),
            'mf_smooth_lddt_intra_pairs': smooth_lddt_diag['intra_pairs'].detach(),
            'mf_smooth_lddt_scaffold_pairs': smooth_lddt_diag['scaffold_pairs'].detach(),
            'mf_smooth_lddt_antigen_pairs': smooth_lddt_diag['antigen_pairs'].detach(),
            'mf_smooth_lddt_fixed_context_pred_drift_rms': smooth_lddt_diag[
                'fixed_context_pred_drift_rms'
            ].detach(),
            'mf_smooth_lddt_fixed_context_restored_rate': smooth_lddt_diag[
                'fixed_context_restored_rate'
            ].detach(),
            'mf_smooth_lddt_antigen_score': (
                1.0 - smooth_lddt_diag['antigen']
            ).detach(),
            **{k: v.detach() for k, v in distogram_audit.items()},
            **({k: v.detach() for k, v in self.abx_repr.last_diagnostics.items()}
               if self.abx_repr is not None else {}),
        })
        scorefm_details['disto_weighted_loss'] = (
            self.loss_distogram_weight * abx_distogram_loss
        ).detach()
        scorefm_details['mf_smooth_lddt_weighted_loss'] = (
            self.loss_smooth_lddt_weight * smooth_lddt_loss
        ).detach()

        if self.struct_only:
            # predicted rmsd
            prmsd_loss = F.smooth_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss
        else:
            pdev_loss, prmsd_loss = None, None

        # comprehensive loss
        loss = (
            self.loss_sequence_weight * snll
            + self.loss_structure_weight * struct_loss
            + self.loss_interface_weight * interface_loss
            + self.loss_edge_weight * ed_loss
            + self.loss_distogram_weight * abx_distogram_loss
            + self.loss_smooth_lddt_weight * smooth_lddt_loss
            + (0 if pdev_loss is None else pdev_loss)
        )
        self._diagnostic_objective_tensors = {
            "seq": self.loss_sequence_weight * snll,
            "structure": self.loss_structure_weight * struct_loss,
            "endpoint": self.loss_interface_weight * getattr(
                self, "_last_endpoint_objective_tensor", interface_loss
            ),
            "satc": getattr(
                self, "_last_satc_objective_tensor", interface_loss * 0.0
            ),
            "edge": self.loss_edge_weight * ed_loss if torch.is_tensor(ed_loss) else loss * 0.0,
            "distogram": self.loss_distogram_weight * abx_distogram_loss,
            "smooth_lddt": self.loss_smooth_lddt_weight * smooth_lddt_loss,
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
                "seq_ce_weight": torch.as_tensor(self.loss_sequence_weight, device=X.device),
                "loss_structure_weight": torch.as_tensor(self.loss_structure_weight, device=X.device),
                "loss_interface_weight": torch.as_tensor(self.loss_interface_weight, device=X.device),
                "loss_edge_weight": torch.as_tensor(self.loss_edge_weight, device=X.device),
                "loss_distogram_weight": torch.as_tensor(self.loss_distogram_weight, device=X.device),
                "loss_smooth_lddt_weight": torch.as_tensor(self.loss_smooth_lddt_weight, device=X.device),
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
                "scorefm_loss_mode_f01_endpoint_canonical_hybrid": torch.as_tensor(
                    1.0 if self.scorefm_loss_mode == "f01_r3_endpoint_canonical_hybrid" else 0.0,
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
                diag["seq_change_from_proposal_rate"] = (
                    ~pred_pep_hit
                ).float().mean()
                if bool(pep_native_hit.any()):
                    correct_proposal_pred_native = (
                        pred_S[pep_mask][pep_native_hit]
                        == true_S[pep_mask][pep_native_hit]
                    )
                    preservation = correct_proposal_pred_native.float().mean()
                    diag["seq_proposal_correct_preservation_rate"] = preservation
                    diag["seq_proposal_correct_damage_rate"] = 1.0 - preservation
                proposal_wrong = ~pep_native_hit
                if bool(proposal_wrong.any()):
                    diag["seq_proposal_wrong_correction_rate"] = (
                        pred_S[pep_mask][proposal_wrong]
                        == true_S[pep_mask][proposal_wrong]
                    ).float().mean()

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
                    t_graph=t_graph,
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
               progress_desc=None, xloss_mask=None):
        n_steps = _env_int("ABFLOW_SAMPLE_N_STEPS", n_steps)
        if n_steps < 1:
            raise ValueError("ABFLOW_SAMPLE_N_STEPS must be >= 1.")

        if not bool(getattr(self, "scorefm_state_path", True)):
            return self.struct_sample(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths,
                init_noise=init_noise, return_hidden=return_hidden
            )

        cmask, smask = self._enforce_task_sequence_mask_contract(
            cmask, smask, paratope_mask, template, stage="sample"
        )

        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]
            if X_pep is not None:
                X_pep = X_pep[:, :4]

        gen_X, gen_S = X.clone(), S.clone()
        self._prepare_batch_constants(S, paratope_mask, lengths)
        if xloss_mask is not None:
            if self.backbone_only:
                xloss_mask = xloss_mask[:, :4]
            self.batch_constants['xloss_mask'] = xloss_mask.bool()
        else:
            # Legacy generate.py historically drops xloss_mask. Infer the
            # AbFlow CA-fill observation mask ONCE from the original input and
            # cache it for the whole sampling trajectory; never re-infer after
            # R05 coordinate updates, because unresolved CA-fill slots may move.
            self.batch_constants['xloss_mask'] = _abflow_ca_fill_observed_mask(
                S, X
            ).bool()

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
        flow_source_X0 = interface_X.clone()
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
                flow_t=flow_t_graph,
                flow_source_init=flow_source_X0,
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
            elif self.scorefm_sampler_mode == "f01_canonical_carrier":
                if self.scorefm_loss_mode == "f01_r3_canonical_carrier":
                    _t_min = float(self.f01_canonical_t_min)
                elif self.scorefm_loss_mode == "f01_r3_endpoint_canonical_hybrid":
                    _t_min = float(self.f01_hybrid_t_min)
                else:
                    raise RuntimeError(
                        "f01_canonical_carrier sampler requires an F01 canonical loss mode."
                    )
                Xt, _ = self.r3_matcher.exact_carrier_scoreflow_step_gfree(
                    x_t=Xt, x0=flow_source_X0, carrier=pred_clean_X,
                    t=t, t_next=t_next, canonical_t_min=_t_min,
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
                flow_source_init=flow_source_X0,
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

        self._assert_framework_sequence_immutable(
            gen_S, S, paratope_mask, stage="sample"
        )
        self._clean_batch_constants()
        if return_hidden:
            return gen_X, gen_S, best_metric, H_final
        return gen_X, gen_S, best_metric

    def struct_sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, init_noise=None, return_hidden=False):
        cmask, smask = self._enforce_task_sequence_mask_contract(
            cmask, smask, paratope_mask, template, stage="struct_sample"
        )

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

        self._assert_framework_sequence_immutable(
            gen_S, S, paratope_mask, stage="struct_sample"
        )
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
