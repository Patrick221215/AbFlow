#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Modern single/pair + atom-structure replacement for the historical AbFlow AMEncoder.

This file is intentionally a *drop-in* replacement for:
    models/modules/am_enc.py

Public API kept unchanged
-------------------------
AMEncoder.__init__(
    in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf, radial_nf,
    in_edge_nf=0, num_verts=50, act_fn=nn.SiLU(), n_layers=4,
    residual=True, dropout=0.1, dense=False
)

AMEncoder.forward(...) -> (H, pred_X, pred_local_X)

Scientific boundary
-------------------
1. PCS-RC/source construction, Score-Flow path, carrier, inversion, loss and
   sampler remain owned by AbFlow_model.py and are NOT reimplemented here.
2. AbFlow fixed full-atom Cartesian layout [N, n_channel, 3] is preserved.
3. No MSA/profile/deletion features are introduced.
4. Pair-Time is not implemented here.  The modern backbone owns the pair state.
5. The validated AbFlow surface operator (MS_E_GCL) is retained as the surface-
   only final refinement rather than inventing a new unvalidated surface feature.

What is replaced
----------------
Historical sparse ctx/inter residue GCL reasoning is replaced by:
    input single state s_i
        +
    explicit dense pair state z_ij
        -> Pairformer
        -> atom attention encoder
        -> pair-conditioned token transformer
        -> atom attention decoder
        -> Cartesian residual update

The Pairformer/attention/triangle operation order follows the supplied
MFDesign/Boltz implementation.  The atom network follows the supplied
AtomAttentionEncoder -> DiffusionTransformer -> AtomAttentionDecoder structure,
adapted only at the data boundary to AbFlow's fixed 14-slot atom layout.
"""

from __future__ import annotations

import math
import os
import json
import inspect
import random
from functools import partial
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

try:
    from scipy.stats import truncnorm
except Exception:
    truncnorm = None

# Keep the already-validated AbFlow surface geometry operator.
from .am_egnn import MS_E_GCL
from utils.nn_utils import _knn_edges
from torch_scatter import scatter_mean, scatter_sum
from data.pdb_utils import VOCAB





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

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else float(default)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else int(default)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else str(default)


def _runtime_trace_enabled() -> bool:
    return _env_flag("ABFLOW_RUNTIME_TRACE", False)


def _runtime_trace_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return 0


def _runtime_tensor_mib(value) -> float:
    if not torch.is_tensor(value):
        return 0.0
    return float(value.numel() * value.element_size()) / (1024.0 * 1024.0)


def _runtime_trace_file_line(message: str):
    """Append a diagnostic line to the current version_x rank log.

    File I/O is enabled only when ABFLOW_RUNTIME_TRACE_DIR is set by Trainer.
    It never touches CUDA tensors or changes execution order.
    """
    trace_dir = os.environ.get("ABFLOW_RUNTIME_TRACE_DIR", "").strip()
    if not trace_dir:
        return
    try:
        os.makedirs(trace_dir, exist_ok=True)
        rank = _runtime_trace_rank()
        path = os.path.join(
            trace_dir, f"runtime_memory_rank{rank}.log"
        )
        with open(path, "a", encoding="utf-8") as fout:
            fout.write(str(message).rstrip("\n") + "\n")
    except Exception:
        # Diagnostics must never alter/fail the scientific run.
        pass


def _runtime_cuda_line(tag: str, device, extra: str = ""):
    """Read-only allocator snapshot: no sync, no empty_cache, no detach."""
    if not _runtime_trace_enabled():
        return
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda":
        return
    mib = 1024.0 * 1024.0
    msg = (
        f"[BackboneMem] rank={_runtime_trace_rank()} tag={tag} "
        f"alloc={torch.cuda.memory_allocated(device)/mib:.1f}MiB "
        f"reserved={torch.cuda.memory_reserved(device)/mib:.1f}MiB "
        f"peak={torch.cuda.max_memory_allocated(device)/mib:.1f}MiB"
    )
    if extra:
        msg += " " + str(extra)
    print(msg, flush=True)
    _runtime_trace_file_line(msg)


def _torch_major_minor():
    raw = str(torch.__version__).split("+", 1)[0]
    parts = raw.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return (0, 0)


_TORCH_MAJOR_MINOR = _torch_major_minor()


# PyTorch 1.11 already exposes ``use_reentrant``.  DDP has a critical
# distinction here: reentrant checkpointing cannot safely checkpoint the same
# parameterized layer multiple times in one iteration, while non-reentrant
# checkpointing is required by the current graph-wise AbFlow execution.
_CHECKPOINT_HAS_USE_REENTRANT = (
    "use_reentrant" in inspect.signature(torch_checkpoint).parameters
)


def _capture_cuda_autocast_state():
    """Capture the exact CUDA AMP state seen by the scientific forward."""
    return {
        "enabled": bool(torch.is_autocast_enabled()),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": bool(torch.is_autocast_cache_enabled()),
    }


def _checkpoint_cuda_states(*args):
    """PyTorch-1.11-compatible CUDA RNG state capture."""
    devices = list({
        int(arg.get_device())
        for arg in args
        if torch.is_tensor(arg) and arg.is_cuda
    })
    states = []
    for device in devices:
        with torch.cuda.device(device):
            states.append(torch.cuda.get_rng_state())
    return devices, states


def _restore_checkpoint_cuda_states(devices, states):
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)


def _checkpoint_nonreentrant_111_cleanup(function, *args):
    """Torch-1.11 non-reentrant checkpoint with end-of-backward storage cleanup.

    This is the torch 1.11 `_checkpoint_without_reentrant` algorithm, with two
    project-required properties:

    1. Forward and recomputation both use the exact captured CUDA autocast
       state (including BF16 dtype), matching the already validated v117/v118
       path.
    2. The recomputation `storage` list is cleared by an autograd-engine
       end-of-GraphTask callback. Torch 1.11's experimental implementation
       appends all recomputed saved tensors to this closure but does not
       explicitly clear it. In this project that storage survives into the next
       iteration and produces a 20+ GiB live-allocation baseline.

    The callback runs only AFTER the current backward GraphTask has completed.
    No tensor required by the current gradient computation is removed early.
    No model function, parameter, input, RNG sequence, loss, or optimizer rule
    is changed.
    """
    preserve_rng_state = True
    amp_state = _capture_cuda_autocast_state()

    fwd_cpu_state = torch.get_rng_state()
    had_cuda_in_fwd = False
    fwd_gpu_devices = []
    fwd_gpu_states = []
    if torch.cuda._initialized:
        had_cuda_in_fwd = True
        fwd_gpu_devices, fwd_gpu_states = _checkpoint_cuda_states(*args)

    storage = []
    counter = 0
    cleanup_queued = False

    def run_exact(*inner_args):
        with torch.cuda.amp.autocast(
            enabled=amp_state["enabled"],
            dtype=amp_state["dtype"],
            cache_enabled=amp_state["cache_enabled"],
        ):
            return function(*inner_args)

    def clear_storage_after_graph_task():
        nonlocal cleanup_queued
        before = None
        if (
            _env_flag("ABFLOW_DEEP_CHECKPOINT_TRACE", False)
            and torch.cuda.is_available()
            and torch.cuda._initialized
        ):
            before = torch.cuda.memory_allocated()

        count = len(storage)
        if _env_flag("ABFLOW_PERF_DIAGNOSTICS", False):
            _ATTN_RUNTIME_STATS["checkpoint_cleanup_calls"] += 1
            _ATTN_RUNTIME_STATS["checkpoint_storage_tensors"] += int(count)

        storage.clear()
        cleanup_queued = False

        # Detailed per-checkpoint allocator logging was useful while diagnosing
        # the torch-1.11 leak, but it performs a file open/write for every
        # checkpoint callback and can dominate short runtime benchmarks.
        # Keep it available only behind the explicit deep-trace switch.
        if before is not None and _env_flag(
            "ABFLOW_DEEP_CHECKPOINT_TRACE", False
        ):
            after = torch.cuda.memory_allocated()
            _runtime_trace_file_line(
                "[Checkpoint111Cleanup] "
                f"rank={_runtime_trace_rank()} "
                f"storage_tensors={count} "
                f"alloc_before={before/(1024.0*1024.0):.1f}MiB "
                f"alloc_after={after/(1024.0*1024.0):.1f}MiB "
                f"released={(before-after)/(1024.0*1024.0):.1f}MiB"
            )

    def pack(x):
        nonlocal counter
        idx = counter
        counter += 1
        return idx

    def unpack(x):
        nonlocal cleanup_queued

        if len(storage) == 0:
            def inner_pack(inner):
                storage.append(inner)
                return None

            def inner_unpack(_packed):
                raise RuntimeError(
                    "Backward requested a tensor hidden inside checkpoint "
                    "recomputation. This matches the torch-1.11 invariant."
                )

            rng_devices = (
                fwd_gpu_devices
                if preserve_rng_state and had_cuda_in_fwd
                else []
            )

            with torch.random.fork_rng(
                devices=rng_devices,
                enabled=preserve_rng_state,
            ):
                if preserve_rng_state:
                    torch.set_rng_state(fwd_cpu_state)
                    if had_cuda_in_fwd:
                        _restore_checkpoint_cuda_states(
                            fwd_gpu_devices, fwd_gpu_states
                        )

                with torch.enable_grad():
                    with torch.autograd.graph.saved_tensors_hooks(
                        inner_pack, inner_unpack
                    ):
                        _unused = run_exact(*args)

            # PyTorch 1.11 exposes queue_callback internally and DDP itself
            # uses it. It runs after the whole current backward GraphTask,
            # i.e. after every unpack consumer is finished.
            if not cleanup_queued:
                engine = torch.autograd.Variable._execution_engine
                if not hasattr(engine, "queue_callback"):
                    raise RuntimeError(
                        "torch 1.11 checkpoint cleanup requires "
                        "autograd execution-engine queue_callback"
                    )
                engine.queue_callback(clear_storage_after_graph_task)
                cleanup_queued = True

        return storage[x]

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = run_exact(*args)

    if torch.cuda._initialized and not had_cuda_in_fwd:
        raise RuntimeError(
            "CUDA was initialized inside a checkpointed forward; "
            "this is unsupported by the torch-1.11 checkpoint contract."
        )
    return output


def _checkpoint_nonreentrant_amp_exact(function, *args):
    """DDP-safe exact-BF16 checkpoint with a torch-1.11 lifetime fix.

    Torch 1.11 uses the project-local compatibility implementation above.
    Modern runtimes use PyTorch's native non-reentrant implementation.
    """
    if not _CHECKPOINT_HAS_USE_REENTRANT:
        return torch_checkpoint(function, *args)

    if _TORCH_MAJOR_MINOR == (1, 11):
        return _checkpoint_nonreentrant_111_cleanup(function, *args)

    amp_state = _capture_cuda_autocast_state()

    def run_with_exact_amp(*inner_args):
        with torch.cuda.amp.autocast(
            enabled=amp_state["enabled"],
            dtype=amp_state["dtype"],
            cache_enabled=amp_state["cache_enabled"],
        ):
            return function(*inner_args)

    return torch_checkpoint(
        run_with_exact_amp,
        *args,
        use_reentrant=False,
        preserve_rng_state=True,
    )


def _checkpoint_pairformer_layer(layer, s, z, mask, pair_mask):
    return _checkpoint_nonreentrant_amp_exact(
        layer, s, z, mask, pair_mask
    )


def _checkpoint_tensor_module(module, *args):
    return _checkpoint_nonreentrant_amp_exact(module, *args)


def _choose_num_heads(dim: int, maximum: int = 8) -> int:
    for heads in (maximum, 4, 2, 1):
        if heads <= maximum and dim % heads == 0:
            return heads
    return 1


class DistanceBinner(nn.Module):
    """Stable distance -> categorical-bin helper.

    The binning convention matches the supplied Boltz/MFDesign distogram code:
    K bins are represented by K-1 monotonically increasing boundaries and
    ``bin = number_of_boundaries(distance > boundary)``.
    """

    def __init__(self, num_bins: int, min_dist: float, max_dist: float):
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if max_dist <= min_dist:
            raise ValueError("max_dist must be > min_dist")
        self.num_bins = int(num_bins)
        self.min_dist = float(min_dist)
        self.max_dist = float(max_dist)
        self.register_buffer(
            "boundaries",
            torch.linspace(min_dist, max_dist, num_bins - 1),
            persistent=True,
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return (distances.unsqueeze(-1) > self.boundaries.to(distances)).sum(-1).long()


class ErrorBinner(nn.Module):
    """Continuous error -> uniform categorical bins on [0, max_error]."""

    def __init__(self, num_bins: int, max_error: float):
        super().__init__()
        self.num_bins = int(num_bins)
        self.max_error = float(max_error)

    def forward(self, errors: torch.Tensor) -> torch.Tensor:
        scaled = torch.floor(
            errors.clamp_min(0.0) * float(self.num_bins) / self.max_error
        ).long()
        return scaled.clamp(max=self.num_bins - 1)


# ============================================================================
# Initialization helpers -- same semantics as the supplied MFDesign layers.
# ============================================================================

def _fan(weight):
    if weight.ndim != 2:
        return max(1, weight.numel())
    return max(1, int(weight.shape[1]))


def lecun_normal_init_(weight):
    """LeCun fan-in truncated normal used by the supplied MFDesign layers."""
    fan_in = _fan(weight)
    if truncnorm is None:
        # Numerically close fallback when SciPy is unavailable.
        # torch.trunc_normal_ is deterministic under the normal torch RNG.
        std = 1.0 / math.sqrt(fan_in)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        return
    a, b = -2.0, 2.0
    trunc_std = truncnorm.std(a=a, b=b, loc=0.0, scale=1.0)
    std = math.sqrt(1.0 / fan_in) / trunc_std
    values = truncnorm.rvs(a=a, b=b, loc=0.0, scale=std, size=weight.numel())
    values = np.asarray(values).reshape(tuple(weight.shape))
    with torch.no_grad():
        weight.copy_(torch.as_tensor(values, device=weight.device, dtype=weight.dtype))


def final_init_(weight):
    with torch.no_grad():
        weight.zero_()


def gating_init_(weight):
    with torch.no_grad():
        weight.zero_()


def normal_init_(weight):
    nn.init.kaiming_normal_(weight, nonlinearity="linear")


# ============================================================================
# MFDesign/Boltz-style primitive layers
# ============================================================================

class LinearNoBias(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)


class LayerNormNoBias(nn.Module):
    """LayerNorm with learnable scale gamma and no additive beta.

    MFDesign/Boltz AdaLN uses a modern PyTorch LayerNorm with bias disabled.
    PyTorch 1.11 does not expose that constructor keyword.  This module keeps
    exactly the intended map:
        y = gamma * (x - mean) / sqrt(var + eps)
    with no additive beta parameter.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            None,
            self.eps,
        )

    def extra_repr(self):
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            "elementwise_affine=True, no_additive_bias=True"
        )


class Transition(nn.Module):
    """MFDesign/Boltz gated transition block."""

    def __init__(self, dim: int, hidden: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()
        with torch.no_grad():
            self.norm.weight.fill_(1.0)
            self.norm.bias.zero_()
        lecun_normal_init_(self.fc1.weight)
        lecun_normal_init_(self.fc2.weight)
        final_init_(self.fc3.weight)

    def forward(self, x):
        x = self.norm(x)
        return self.fc3(self.silu(self.fc1(x)) * self.fc2(x))


def get_dropout_mask(dropout: float, z: torch.Tensor, training: bool,
                     columnwise: bool = False) -> torch.Tensor:
    dropout = float(dropout) * bool(training)
    if dropout <= 0:
        v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
        return torch.ones_like(v)
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    d = (torch.rand_like(v) > dropout).to(z.dtype)
    return d / (1.0 - dropout)


class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)
        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        with torch.no_grad():
            self.norm_in.weight.fill_(1.0)
            self.norm_in.bias.zero_()
            self.norm_out.weight.fill_(1.0)
            self.norm_out.bias.zero_()
        lecun_normal_init_(self.p_in.weight)
        gating_init_(self.g_in.weight)
        final_init_(self.p_out.weight)
        gating_init_(self.g_out.weight)

    def forward(self, x, mask):
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x.float(), 2, dim=-1)
        x = torch.einsum("bikd,bjkd->bijd", a, b).to(x_in.dtype)
        return self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()


class TriangleMultiplicationIncoming(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)
        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        with torch.no_grad():
            self.norm_in.weight.fill_(1.0)
            self.norm_in.bias.zero_()
            self.norm_out.weight.fill_(1.0)
            self.norm_out.bias.zero_()
        lecun_normal_init_(self.p_in.weight)
        gating_init_(self.g_in.weight)
        final_init_(self.p_out.weight)
        gating_init_(self.g_out.weight)

    def forward(self, x, mask):
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()
        x = x * mask.unsqueeze(-1)
        a, b = torch.chunk(x.float(), 2, dim=-1)
        x = torch.einsum("bkid,bkjd->bijd", a, b).to(x_in.dtype)
        return self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()


class AFLinear(nn.Linear):
    """Small AlphaFold-style Linear used by triangle attention."""

    def __init__(self, in_dim, out_dim, bias=True, init="default"):
        super().__init__(in_dim, out_dim, bias=bias)
        if bias:
            nn.init.zeros_(self.bias)
        if init == "default":
            lecun_normal_init_(self.weight)
        elif init == "glorot":
            nn.init.xavier_uniform_(self.weight, gain=1.0)
        elif init == "gating":
            gating_init_(self.weight)
            if bias:
                nn.init.ones_(self.bias)
        elif init == "normal":
            normal_init_(self.weight)
        elif init == "final":
            final_init_(self.weight)
        else:
            raise ValueError(f"unsupported init={init}")


# ---------------------------------------------------------------------------
# v123 focused runtime statistics.
#
# These are plain Python counters, enabled only for short speed-profile runs.
# They never hold Tensors and never participate in state_dict/autograd.
# ---------------------------------------------------------------------------
_ATTN_RUNTIME_STATS = {
    "full_calls": 0,
    "lma_calls": 0,
    "checkpoint_cleanup_calls": 0,
    "checkpoint_storage_tensors": 0,
    # v126 batched-runtime counters (plain Python integers only).
    "batched_pairformer_calls": 0,
    "graphwise_equiv_pairformer_calls": 0,
    "batched_atom_calls": 0,
    "batched_shadow_atom_calls": 0,
    "real_tokens": 0,
    "padded_tokens": 0,
    "sc_teacher_calls": 0,
    "sc_teacher_graphs": 0,
    "sc_formal_calls": 0,
    "confidence_calls": 0,
}


def consume_runtime_perf_stats():
    out = dict(_ATTN_RUNTIME_STATS)
    for key in _ATTN_RUNTIME_STATS:
        _ATTN_RUNTIME_STATS[key] = 0
    return out


class AFAttention(nn.Module):
    """AlphaFold/MFDesign attention with an exact low-memory backend.

    The scientific attention function is unchanged:
        softmax(QK^T / sqrt(d) + sum(biases)) V,
    followed by the same gate and output projection.

    ``full`` materializes the complete FP32 logits tensor.
    ``lma`` evaluates the same softmax by exact blockwise log-sum-exp.
    No token, pair, key, value or bias is removed.
    """

    def __init__(self, c_q, c_k, c_v, c_hidden, no_heads, gating=True):
        super().__init__()
        self.c_q = c_q
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = AFLinear(
            c_q, c_hidden * no_heads, bias=False, init="glorot"
        )
        self.linear_k = AFLinear(
            c_k, c_hidden * no_heads, bias=False, init="glorot"
        )
        self.linear_v = AFLinear(
            c_v, c_hidden * no_heads, bias=False, init="glorot"
        )
        self.linear_o = AFLinear(
            c_hidden * no_heads, c_q, bias=False, init="final"
        )
        self.linear_g = (
            AFLinear(c_q, c_hidden * no_heads, bias=False, init="gating")
            if gating else None
        )

        self.backend = _env_str(
            "ABFLOW_TRIANGLE_ATTN_BACKEND", "auto"
        ).strip().lower()
        if self.backend not in {"auto", "full", "lma"}:
            raise ValueError(
                "ABFLOW_TRIANGLE_ATTN_BACKEND must be auto/full/lma"
            )

        self.full_logits_limit_mb = max(
            1.0,
            _env_float("ABFLOW_TRIANGLE_FULL_LOGITS_LIMIT_MB", 256.0),
        )
        self.lma_q_chunk_size = max(
            1, _env_int("ABFLOW_TRIANGLE_LMA_Q_CHUNK_SIZE", 64)
        )
        self.lma_kv_chunk_size = max(
            1, _env_int("ABFLOW_TRIANGLE_LMA_KV_CHUNK_SIZE", 128)
        )
        self.perf_stats_enabled = _env_flag(
            "ABFLOW_PERF_DIAGNOSTICS", False
        )

    @staticmethod
    def _slice_bias(bias, q_start, q_end, kv_start, kv_end):
        """Slice the Q/K axes while preserving broadcast dimensions."""
        sl = [slice(None)] * bias.ndim
        if bias.shape[-2] != 1:
            sl[-2] = slice(q_start, q_end)
        if bias.shape[-1] != 1:
            sl[-1] = slice(kv_start, kv_end)
        return bias[tuple(sl)]

    @staticmethod
    def _full_logits_megabytes(q, k):
        # q: [*, H, Q, D], k: [*, H, K, D]
        # logits: [*, H, Q, K], accumulated in FP32.
        n_logits = 1
        for size in q.shape[:-1]:
            n_logits *= int(size)
        n_logits *= int(k.shape[-2])
        return float(n_logits * 4) / (1024.0 * 1024.0)

    def _full_attention(self, q, k, v, biases):
        attn = torch.matmul(q.float(), k.float().transpose(-1, -2))
        for bias in biases:
            bias_fp32 = (
                bias if bias.dtype == torch.float32 else bias.float()
            )
            attn.add_(bias_fp32)
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v.float()).to(v.dtype)

    def _low_memory_attention(self, q, k, v, biases):
        """Exact blockwise softmax with full-autograd log-sum-exp merging.

        Inspired by MFDesign/Boltz LMA, but intentionally does NOT detach the
        block maxima.  This preserves the full-attention Jacobian to FP32
        rounding error, which is a stricter requirement for this project.

        For KV blocks j:
            m_j = max a_j
            w_j = sum exp(a_j - m_j)
            u_j = sum exp(a_j - m_j) v_j

        Let M = max_j m_j. Then:
            softmax(A)V =
              sum_j exp(m_j-M) u_j / sum_j exp(m_j-M) w_j.

        This is an algebraic identity; all Q/K/V and all biases participate.
        """
        no_q = int(q.shape[-2])
        no_kv = int(k.shape[-2])
        q_chunk_size = int(self.lma_q_chunk_size)
        kv_chunk_size = int(self.lma_kv_chunk_size)

        q_outputs = []

        for q_start in range(0, no_q, q_chunk_size):
            q_end = min(no_q, q_start + q_chunk_size)
            q_chunk = q[..., q_start:q_end, :].float()

            block_maxes = []
            block_weights = []
            block_values = []

            for kv_start in range(0, no_kv, kv_chunk_size):
                kv_end = min(no_kv, kv_start + kv_chunk_size)
                k_chunk = k[..., kv_start:kv_end, :].float()
                v_chunk = v[..., kv_start:kv_end, :].float()

                logits = torch.matmul(
                    q_chunk, k_chunk.transpose(-1, -2)
                )
                for bias in biases:
                    b = self._slice_bias(
                        bias, q_start, q_end, kv_start, kv_end
                    )
                    if b.dtype != torch.float32:
                        b = b.float()
                    logits.add_(b)

                local_max = logits.amax(dim=-1, keepdim=True)
                exp_logits = torch.exp(logits - local_max)

                block_maxes.append(local_max.squeeze(-1))
                block_weights.append(exp_logits.sum(dim=-1))
                block_values.append(torch.matmul(exp_logits, v_chunk))

            # Shapes end with:
            # max/weight [..., H, n_kv_chunks, Qc]
            # value      [..., H, n_kv_chunks, Qc, D]
            block_maxes = torch.stack(block_maxes, dim=-2)
            block_weights = torch.stack(block_weights, dim=-2)
            block_values = torch.stack(block_values, dim=-3)

            global_max = block_maxes.amax(dim=-2, keepdim=True)
            correction = torch.exp(block_maxes - global_max)

            numerator = (
                block_values * correction.unsqueeze(-1)
            ).sum(dim=-3)
            denominator = (
                block_weights * correction
            ).sum(dim=-2).unsqueeze(-1)

            q_outputs.append((numerator / denominator).to(v.dtype))

        return torch.cat(q_outputs, dim=-2)

    def forward(self, q_x, kv_x, biases=None):
        biases = [] if biases is None else biases

        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(
            q.shape[:-1] + (self.no_heads, -1)
        ).transpose(-2, -3)
        k = k.view(
            k.shape[:-1] + (self.no_heads, -1)
        ).transpose(-2, -3)
        v = v.view(
            v.shape[:-1] + (self.no_heads, -1)
        ).transpose(-2, -3)

        q = q / math.sqrt(self.c_hidden)

        backend = self.backend
        if backend == "auto":
            full_mb = self._full_logits_megabytes(q, k)
            backend = (
                "full"
                if full_mb <= self.full_logits_limit_mb
                else "lma"
            )

        if self.perf_stats_enabled:
            if backend == "lma":
                _ATTN_RUNTIME_STATS["lma_calls"] += 1
            else:
                _ATTN_RUNTIME_STATS["full_calls"] += 1

        if backend == "lma":
            out = self._low_memory_attention(q, k, v, biases)
        else:
            out = self._full_attention(q, k, v, biases)

        out = out.transpose(-2, -3)

        if self.linear_g is not None:
            gate = torch.sigmoid(self.linear_g(q_x))
            gate = gate.view(
                gate.shape[:-1] + (self.no_heads, -1)
            )
            out = out * gate

        out = out.reshape(out.shape[:-2] + (-1,))
        return self.linear_o(out)


class TriangleAttention(nn.Module):
    """MFDesign triangle-attention semantics with exact anchor chunking.

    MFDesign/Boltz treats the leading ``I`` pair axis as a batch-like axis for
    triangle attention and supports chunking it.  The old AbFlow adaptation
    discarded ``chunk_size`` entirely, materializing the full
    [B, I, H, J, J] FP32 attention tensor.

    Chunking I is mathematically exact because no softmax normalization crosses
    different anchor-I slices.  Q/K/J attention inside each anchor is unchanged.
    """

    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        super().__init__()
        self.starting = starting
        self.inf = inf
        self.layer_norm = nn.LayerNorm(c_in, eps=1e-5)
        self.linear = AFLinear(c_in, no_heads, bias=False, init="normal")
        self.mha = AFAttention(c_in, c_in, c_in, c_hidden, no_heads)
        # Runtime-only scheduling policy.  Chunking is exclusively along the
        # independent anchor-I axis, never along the softmax normalization axis.
        # Therefore full and chunked execution implement the same attention map.
        self.chunk_policy = _env_str(
            "ABFLOW_TRIANGLE_ATTENTION_CHUNK_SIZE", "0"
        ).strip().lower()
        self.auto_full_max_tokens = max(
            1, _env_int("ABFLOW_TRIANGLE_AUTO_FULL_MAX_TOKENS", 320)
        )
        self.auto_medium_max_tokens = max(
            self.auto_full_max_tokens,
            _env_int("ABFLOW_TRIANGLE_AUTO_MEDIUM_MAX_TOKENS", 448),
        )
        self.auto_medium_chunk = max(
            1, _env_int("ABFLOW_TRIANGLE_AUTO_MEDIUM_CHUNK", 128)
        )
        self.auto_large_chunk = max(
            1, _env_int("ABFLOW_TRIANGLE_AUTO_LARGE_CHUNK", 64)
        )

    @staticmethod
    def _slice_dim(t, dim, start, length):
        return t.narrow(dim, int(start), int(length))

    def _chunked_mha(self, x, mask_bias, triangle_bias, chunk_size):
        # x:             [*, I, J, C]
        # mask_bias:     [*, I, 1, 1, J]   -> slice anchor I
        # triangle_bias: [*, 1, H, I, J]   -> I here is the query-J bias axis;
        #                                     do NOT slice it by anchor chunks.
        n_anchor = int(x.shape[-3])
        outputs = []
        for start in range(0, n_anchor, int(chunk_size)):
            length = min(int(chunk_size), n_anchor - start)
            x_chunk = self._slice_dim(x, -3, start, length)
            mask_chunk = self._slice_dim(mask_bias, -4, start, length)
            outputs.append(
                self.mha(
                    q_x=x_chunk,
                    kv_x=x_chunk,
                    biases=[mask_chunk, triangle_bias],
                )
            )
        return torch.cat(outputs, dim=-3)

    def forward(self, x, mask=None, chunk_size=None):
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        # input [B,I,J,C] -> [B,H,I,J] -> [B,1,H,I,J]
        triangle_bias = self.linear(x).permute(0, 3, 1, 2).unsqueeze(1)

        if chunk_size is None:
            policy = self.chunk_policy
            n_anchor = int(x.shape[-3])
            if policy in {"", "auto"}:
                if n_anchor <= self.auto_full_max_tokens:
                    chunk_size = 0
                elif n_anchor <= self.auto_medium_max_tokens:
                    chunk_size = self.auto_medium_chunk
                else:
                    chunk_size = self.auto_large_chunk
            elif policy in {"0", "off", "none", "full"}:
                chunk_size = 0
            else:
                chunk_size = int(policy)
        chunk_size = int(chunk_size or 0)

        if 0 < chunk_size < int(x.shape[-3]):
            x = self._chunked_mha(
                x, mask_bias, triangle_bias, chunk_size
            )
        else:
            x = self.mha(
                q_x=x, kv_x=x, biases=[mask_bias, triangle_bias]
            )

        if not self.starting:
            x = x.transpose(-2, -3)
        return x


class TriangleAttentionStartingNode(TriangleAttention):
    def __init__(self, c_in, c_hidden, no_heads, inf=1e9):
        super().__init__(c_in, c_hidden, no_heads, starting=True, inf=inf)


class TriangleAttentionEndingNode(TriangleAttention):
    def __init__(self, c_in, c_hidden, no_heads, inf=1e9):
        super().__init__(c_in, c_hidden, no_heads, starting=False, inf=inf)


class AttentionPairBias(nn.Module):
    """MFDesign/Boltz single attention with learned bias from z_ij."""

    def __init__(self, c_s, c_z, num_heads, inf=1e6, initial_norm=True):
        super().__init__()
        if c_s % num_heads != 0:
            raise ValueError(f"c_s={c_s} must be divisible by num_heads={num_heads}")
        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.initial_norm = initial_norm
        if initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)
        self.norm_z = nn.LayerNorm(c_z)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        final_init_(self.proj_o.weight)

    def forward(self, s, z, mask, multiplicity=1, to_keys=None, model_cache=None):
        del model_cache
        B = s.shape[0]
        if self.initial_norm:
            s = self.norm_s(s)
        if to_keys is not None:
            k_in = to_keys(s)
            mask_k = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s
            mask_k = mask

        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        z_bias = self.proj_z(self.norm_z(z))
        # [B,Q,K,H] -> [B,H,Q,K]
        z_bias = z_bias.permute(0, 3, 1, 2)
        z_bias = z_bias.repeat_interleave(multiplicity, 0)

        gate = torch.sigmoid(self.proj_g(s))
        attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        attn = attn / math.sqrt(self.head_dim) + z_bias.float()
        attn = attn + (1.0 - mask_k[:, None, None].float()) * -self.inf
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        out = out.reshape(B, -1, self.c_s)
        return self.proj_o(gate * out)


# ============================================================================
# Pairformer
# ============================================================================

class PairformerLayer(nn.Module):
    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        checkpoint_mode: str = "off",
    ):
        super().__init__()
        self.dropout = dropout
        self.checkpoint_mode = str(checkpoint_mode).strip().lower()
        if self.checkpoint_mode not in {"off", "triangle"}:
            raise ValueError(
                "PairformerLayer checkpoint_mode must be off or triangle"
            )
        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.transition_z = Transition(token_z, token_z * 4)
        self.attention = AttentionPairBias(token_s, token_z, num_heads)
        self.transition_s = Transition(token_s, token_s * 4)

    def forward(self, s, z, mask, pair_mask):
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        if (
            self.training
            and self.checkpoint_mode == "triangle"
            and bool(z.requires_grad)
        ):
            tri_start = _checkpoint_tensor_module(
                self.tri_att_start, z, pair_mask
            )
        else:
            tri_start = self.tri_att_start(z, mask=pair_mask)
        z = z + dropout * tri_start

        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        if (
            self.training
            and self.checkpoint_mode == "triangle"
            and bool(z.requires_grad)
        ):
            tri_end = _checkpoint_tensor_module(
                self.tri_att_end, z, pair_mask
            )
        else:
            tri_end = self.tri_att_end(z, mask=pair_mask)
        z = z + dropout * tri_end

        z = z + self.transition_z(z)
        s = s + self.attention(s, z, mask)
        s = s + self.transition_s(s)
        return s, z


class PairformerModule(nn.Module):
    def __init__(
        self,
        token_s,
        token_z,
        num_blocks,
        num_heads=8,
        dropout=0.1,
        checkpoint_mode=None,
    ):
        super().__init__()
        self.activation_checkpoint = _env_flag(
            "ABFLOW_PAIRFORMER_ACTIVATION_CHECKPOINT", True
        )
        mode = (
            _env_str("ABFLOW_PAIRFORMER_CHECKPOINT_MODE", "triangle")
            if checkpoint_mode is None
            else str(checkpoint_mode)
        ).strip().lower()
        if not self.activation_checkpoint:
            mode = "off"
        if mode not in {"off", "triangle", "layer"}:
            raise ValueError(
                "ABFLOW_PAIRFORMER_CHECKPOINT_MODE must be "
                "off, triangle, or layer"
            )
        self.checkpoint_mode = mode

        pair_heads = max(
            1, int(_cfg_value("architecture", "pairwise_heads", _env_int("ABFLOW_MFDESIGN_PAIRWISE_HEADS", 4)))
        )
        pair_width = max(
            1, int(_cfg_value("architecture", "pairwise_head_width", _env_int("ABFLOW_MFDESIGN_PAIRWISE_HEAD_WIDTH", 32)))
        )
        layer_mode = "triangle" if mode == "triangle" else "off"
        self.layers = nn.ModuleList([
            PairformerLayer(
                token_s=token_s,
                token_z=token_z,
                num_heads=num_heads,
                dropout=dropout,
                pairwise_head_width=pair_width,
                pairwise_num_heads=pair_heads,
                checkpoint_mode=layer_mode,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, s, z, mask, pair_mask):
        for layer in self.layers:
            can_layer_checkpoint = (
                self.training
                and self.checkpoint_mode == "layer"
                and (bool(s.requires_grad) or bool(z.requires_grad))
            )
            if can_layer_checkpoint:
                s, z = _checkpoint_pairformer_layer(
                    layer, s, z, mask, pair_mask
                )
            else:
                s, z = layer(s, z, mask, pair_mask)
        return s, z


# ============================================================================
# MFDesign-style conditioned token/atom transformer
# ============================================================================

class AdaLN(nn.Module):
    def __init__(self, dim, dim_single_cond):
        super().__init__()
        # PyTorch 1.11 compatibility while preserving MFDesign AdaLN:
        # a_norm has no affine parameters; s_norm has gamma but no beta.
        self.a_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.s_norm = LayerNormNoBias(dim_single_cond)
        self.s_scale = nn.Linear(dim_single_cond, dim)
        self.s_bias = LinearNoBias(dim_single_cond, dim)

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        return torch.sigmoid(self.s_scale(s)) * a + self.s_bias(s)


class ConditionedTransitionBlock(nn.Module):
    def __init__(self, dim_single, dim_single_cond, expansion_factor=2):
        super().__init__()
        self.adaln = AdaLN(dim_single, dim_single_cond)
        dim_inner = int(dim_single * expansion_factor)
        self.swish_gate = LinearNoBias(dim_single, dim_inner * 2)
        self.a_to_b = LinearNoBias(dim_single, dim_inner)
        self.b_to_a = LinearNoBias(dim_inner, dim_single)
        output_projection = nn.Linear(dim_single_cond, dim_single)
        nn.init.zeros_(output_projection.weight)
        nn.init.constant_(output_projection.bias, -2.0)
        self.output_projection = output_projection

    @staticmethod
    def _swiglu(x):
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x

    def forward(self, a, s):
        a = self.adaln(a, s)
        b = self._swiglu(self.swish_gate(a)) * self.a_to_b(a)
        return torch.sigmoid(self.output_projection(s)) * self.b_to_a(b)


class DiffusionTransformerLayer(nn.Module):
    def __init__(self, heads, dim, dim_single_cond, dim_pairwise):
        super().__init__()
        self.adaln = AdaLN(dim, dim_single_cond)
        self.pair_bias_attn = AttentionPairBias(
            c_s=dim, c_z=dim_pairwise, num_heads=heads, initial_norm=False
        )
        output_projection = nn.Linear(dim_single_cond, dim)
        nn.init.zeros_(output_projection.weight)
        nn.init.constant_(output_projection.bias, -2.0)
        self.output_projection = output_projection
        self.transition = ConditionedTransitionBlock(dim, dim_single_cond)

    def forward(self, a, s, z, mask, to_keys=None):
        b = self.adaln(a, s)
        b = self.pair_bias_attn(
            s=b, z=z, mask=mask, to_keys=to_keys
        )
        b = torch.sigmoid(self.output_projection(s)) * b
        a = a + b
        a = a + self.transition(a, s)
        return a


class DiffusionTransformer(nn.Module):
    def __init__(self, depth, heads, dim, dim_single_cond, dim_pairwise):
        super().__init__()
        self.layers = nn.ModuleList([
            DiffusionTransformerLayer(
                heads=heads,
                dim=dim,
                dim_single_cond=dim_single_cond,
                dim_pairwise=dim_pairwise,
            )
            for _ in range(depth)
        ])

    def forward(self, a, s, z, mask, to_keys=None):
        for layer in self.layers:
            a = layer(a, s, z, mask=mask, to_keys=to_keys)
        return a


def get_indexing_matrix(K, W, H, device):
    if W % 2 != 0 or H % (W // 2) != 0:
        raise ValueError(f"window sizes require W even and H divisible by W/2; W={W}, H={H}")
    h = H // (W // 2)
    if h % 2 != 0:
        raise ValueError(f"H/(W/2) must be even; got {h}")
    arange = torch.arange(2 * K, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(
        min=0, max=h + 1
    )
    index = index.view(K, 2, 2 * K)[:, 0, :]
    onehot = F.one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    return onehot.reshape(2 * K, h * K).float()


def single_to_keys(single, indexing_matrix, W, H):
    B, N, D = single.shape
    K = N // W
    single = single.view(B, 2 * K, W // 2, D)
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )


class AtomTransformer(nn.Module):
    def __init__(self, dim, dim_single_cond, dim_pairwise, depth, heads,
                 attn_window_queries=32, attn_window_keys=128):
        super().__init__()
        self.W = attn_window_queries
        self.H = attn_window_keys
        self.transformer = DiffusionTransformer(
            depth=depth,
            heads=heads,
            dim=dim,
            dim_single_cond=dim_single_cond,
            dim_pairwise=dim_pairwise,
        )

    def forward(self, q, c, p, mask, to_keys):
        W, H = self.W, self.H
        B, N, D = q.shape
        if N % W != 0:
            raise ValueError(f"atom count {N} must be padded to a multiple of W={W}")
        NW = N // W
        q_w = q.view(B * NW, W, D)
        c_w = c.view(B * NW, W, -1)
        mask_w = mask.view(B * NW, W)
        p_w = p.view(p.shape[0] * NW, W, H, -1)

        def to_keys_new(t):
            t = t.view(B, NW * W, -1)
            return to_keys(t).view(B * NW, H, -1)

        q_w = self.transformer(
            a=q_w,
            s=c_w,
            z=p_w,
            mask=mask_w.float(),
            to_keys=to_keys_new,
        )
        return q_w.view(B, N, D)


class FixedSlotAtomStructureNetwork(nn.Module):
    """MFDesign atom architecture adapted to AbFlow fixed atom slots.

    The *architecture* is AtomAttentionEncoder -> token transformer ->
    AtomAttentionDecoder.  Only the mapping boundary differs: AbFlow already
    has a deterministic token x atom-slot layout, so atom_to_token is implicit.
    """

    def __init__(
        self,
        token_s,
        token_z,
        atom_input_dim,
        n_channel,
        depth,
        dropout=0.0,
        atom_s=None,
        atom_z=None,
        atom_heads=4,
        token_heads=8,
        W=32,
        H=128,
    ):
        super().__init__()
        del dropout
        self.token_s = token_s
        self.token_z = token_z
        self.n_channel = n_channel
        self.W = W
        self.H = H
        # Diagnostics only.
        self._runtime_trace_call_count = 0
        # MFDesign released gold values: atom_s=128, atom_z=16.
        atom_s = (
            int(_cfg_value("architecture", "atom_s", _env_int("ABFLOW_MFDESIGN_ATOM_S", 128)))
            if atom_s is None else int(atom_s)
        )
        atom_heads = max(
            1, int(_cfg_value("architecture", "atom_encoder_heads", _env_int("ABFLOW_MFDESIGN_ATOM_HEADS", atom_heads)))
        )
        if atom_s % atom_heads != 0:
            raise ValueError(
                f"atom_s={atom_s} must be divisible by atom_heads={atom_heads}"
            )
        atom_z = (
            int(_cfg_value("architecture", "atom_z", _env_int("ABFLOW_MFDESIGN_ATOM_Z", 16)))
            if atom_z is None else int(atom_z)
        )
        # Stable public diagnostics.  These are metadata only, not new state.
        self.atom_s = int(atom_s)
        self.atom_z = int(atom_z)
        self.atom_heads = int(atom_heads)

        self.embed_atom_features = LinearNoBias(atom_input_dim, atom_s)
        self.embed_atompair_pos = LinearNoBias(3, atom_z)
        self.embed_atompair_dist = LinearNoBias(1, atom_z)
        self.embed_atompair_mask = LinearNoBias(1, atom_z)

        self.s_to_c = nn.Sequential(nn.LayerNorm(token_s), LinearNoBias(token_s, atom_s))
        final_init_(self.s_to_c[1].weight)
        self.z_to_p = nn.Sequential(nn.LayerNorm(token_z), LinearNoBias(token_z, atom_z))
        final_init_(self.z_to_p[1].weight)

        self.r_to_q = LinearNoBias(3, atom_s)
        final_init_(self.r_to_q.weight)

        self.c_to_p_q = nn.Sequential(nn.ReLU(), LinearNoBias(atom_s, atom_z))
        self.c_to_p_k = nn.Sequential(nn.ReLU(), LinearNoBias(atom_s, atom_z))
        final_init_(self.c_to_p_q[1].weight)
        final_init_(self.c_to_p_k[1].weight)

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
        )
        final_init_(self.p_mlp[-1].weight)

        atom_encoder_depth = max(
            1, int(_cfg_value("architecture", "atom_encoder_depth", _env_int("ABFLOW_MFDESIGN_ATOM_ENCODER_DEPTH", 3)))
        )
        atom_decoder_depth = max(
            1, int(_cfg_value("architecture", "atom_decoder_depth", _env_int("ABFLOW_MFDESIGN_ATOM_DECODER_DEPTH", 3)))
        )
        token_depth = max(
            1, int(_cfg_value("architecture", "token_transformer_depth", _env_int("ABFLOW_MFDESIGN_TOKEN_TRANSFORMER_DEPTH", 8)))
        )
        token_heads = max(
            1, int(_cfg_value("architecture", "token_transformer_heads", _env_int("ABFLOW_MFDESIGN_TOKEN_TRANSFORMER_HEADS", 16)))
        )
        self.token_heads = int(token_heads)

        self.atom_encoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            dim_pairwise=atom_z,
            depth=atom_encoder_depth,
            heads=atom_heads,
            attn_window_queries=W,
            attn_window_keys=H,
        )

        # MFDesign structure module uses 2*token_s token activations.
        self.structure_token_s = 2 * int(token_s)
        if self.structure_token_s % token_heads != 0:
            raise ValueError(
                "2*token_s must be divisible by the MFDesign token-transformer "
                f"head count; got {self.structure_token_s} and {token_heads}"
            )
        self.atom_to_token = nn.Sequential(
            LinearNoBias(atom_s, self.structure_token_s),
            nn.ReLU(),
        )

        # AbFlow already injects its legal Score--Flow time/state condition into
        # the trunk token state.  This adapter changes only representation width:
        # the U02 path/sampler remains owned by AbFlow_model.py.
        self.s_to_a = nn.Sequential(
            nn.LayerNorm(token_s),
            LinearNoBias(token_s, self.structure_token_s),
        )
        self.token_transformer = DiffusionTransformer(
            depth=token_depth,
            heads=token_heads,
            dim=self.structure_token_s,
            dim_single_cond=self.structure_token_s,
            dim_pairwise=token_z,
        )
        self.a_norm = nn.LayerNorm(self.structure_token_s)

        self.a_to_q = LinearNoBias(self.structure_token_s, atom_s)
        final_init_(self.a_to_q.weight)
        self.atom_decoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            dim_pairwise=atom_z,
            depth=atom_decoder_depth,
            heads=atom_heads,
            attn_window_queries=W,
            attn_window_keys=H,
        )
        self.to_xyz = nn.Sequential(nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3))
        final_init_(self.to_xyz[1].weight)

        # Public AbFlow AMEncoder API remains token_s-wide.  The sequence head
        # receives a separate MFDesign-width adapter below, so this projection
        # is only an API boundary, not a new generative authority.
        self.token_out = LinearNoBias(self.structure_token_s, token_s)

    def _pad_atoms(self, value, pad_len, fill=0.0):
        if pad_len <= 0:
            return value
        shape = list(value.shape)
        shape[1] = pad_len
        pad = value.new_full(shape, fill)
        return torch.cat([value, pad], dim=1)

    def forward(
        self, s, z, coords, atom_attr, atom_weights,
        residue_update_mask=None, token_valid_mask=None,
    ):
        # One complex at a time: [L,D], [L,L,Dz], [L,C,3].
        L, C, _ = coords.shape
        if C != self.n_channel:
            raise ValueError(f"expected n_channel={self.n_channel}, got {C}")

        atom_mask = (atom_weights != 0).to(coords.dtype)
        if token_valid_mask is not None:
            token_valid_mask = token_valid_mask.to(
                device=coords.device, dtype=coords.dtype
            ).reshape(L)
            atom_mask = atom_mask * token_valid_mask[:, None]
        atom_feat = torch.cat([atom_attr, atom_mask.unsqueeze(-1)], dim=-1)

        A = L * C
        pad_A = (self.W - (A % self.W)) % self.W
        A_pad = A + pad_A

        _trace_atom = (
            _runtime_trace_enabled()
            and self.training
            and self._runtime_trace_call_count
                < max(0, _env_int("ABFLOW_RUNTIME_TRACE_ATOM_CALLS", 24))
        )
        _atom_call = int(self._runtime_trace_call_count)
        self._runtime_trace_call_count += 1
        if _trace_atom:
            _runtime_cuda_line(
                "atom.entry",
                coords.device,
                extra=(
                    f"call={_atom_call} L={L} C={C} A={A} "
                    f"A_pad={A_pad} K={A_pad // self.W} "
                    f"W={self.W} H={self.H} token_z={self.token_z}"
                ),
            )

        coords_f = coords.reshape(1, A, 3)
        mask_f = atom_mask.reshape(1, A)
        feat_f = atom_feat.reshape(1, A, atom_feat.shape[-1])
        token_ids = torch.arange(L, device=coords.device).repeat_interleave(C).view(1, A)

        coords_f = self._pad_atoms(coords_f, pad_A, fill=0.0)
        mask_f = self._pad_atoms(mask_f.unsqueeze(-1), pad_A, fill=0.0).squeeze(-1)
        feat_f = self._pad_atoms(feat_f, pad_A, fill=0.0)
        token_ids = self._pad_atoms(
            token_ids.unsqueeze(-1).to(coords.dtype), pad_A, fill=-1.0
        ).squeeze(-1).long()

        c = self.embed_atom_features(feat_f)
        valid_tid = token_ids.clamp(min=0)
        token_cond = s[valid_tid]
        token_cond = token_cond * (token_ids >= 0).unsqueeze(-1)
        c = c + self.s_to_c(token_cond)

        q = c + self.r_to_q(coords_f)

        K = A_pad // self.W
        indexing = get_indexing_matrix(K, self.W, self.H, coords.device)
        to_keys = partial(single_to_keys, indexing_matrix=indexing, W=self.W, H=self.H)

        q_coords = coords_f.view(1, K, self.W, 1, 3)
        k_coords = to_keys(coords_f).view(1, K, 1, self.H, 3)
        d = k_coords - q_coords
        inv_dist = 1.0 / (1.0 + torch.sum(d * d, dim=-1, keepdim=True))

        q_mask = mask_f.view(1, K, self.W, 1).bool()
        k_mask = to_keys(mask_f.unsqueeze(-1)).view(1, K, 1, self.H).bool()
        valid = (q_mask & k_mask).float().unsqueeze(-1)

        p = self.embed_atompair_pos(d) * valid
        p = p + self.embed_atompair_dist(inv_dist) * valid
        p = p + self.embed_atompair_mask(valid) * valid
        if _trace_atom:
            _runtime_cuda_line(
                "atom.geom_pair",
                coords.device,
                extra=f"call={_atom_call} p={tuple(p.shape)} {_runtime_tensor_mib(p):.1f}MiB",
            )

        # Inject token pair state z_ij into atom pair state p_ab.
        q_tid = token_ids.view(1, K, self.W, 1)
        k_tid = to_keys(token_ids.unsqueeze(-1).float()).view(1, K, 1, self.H).long()
        q_valid = q_tid >= 0
        k_valid = k_tid >= 0
        q_safe = q_tid.clamp(min=0, max=max(0, L - 1))
        k_safe = k_tid.clamp(min=0, max=max(0, L - 1))
        # MFDesign gold order: project token-pair 128 -> atom-pair 16 BEFORE
        # atom-window lifting.  Pointwise LayerNorm+Linear commutes with the
        # valid-index gather, while avoiding the large [...,128] window tensor.
        z0_projected = self.z_to_p(z[0])
        z_atom_projected = z0_projected[q_safe, k_safe]
        z_atom_projected = z_atom_projected * (
            q_valid & k_valid
        ).unsqueeze(-1)
        if _trace_atom:
            _runtime_cuda_line(
                "atom.z_project_before_window",
                coords.device,
                extra=(
                    f"call={_atom_call} projected_window="
                    f"{tuple(z_atom_projected.shape)} "
                    f"{_runtime_tensor_mib(z_atom_projected):.1f}MiB"
                ),
            )
        p = p + z_atom_projected

        p = p + self.c_to_p_q(c.view(1, K, self.W, 1, -1))
        p = p + self.c_to_p_k(to_keys(c).view(1, K, 1, self.H, -1))
        p = p + self.p_mlp(p)
        if _trace_atom:
            _runtime_cuda_line(
                "atom.pair_mlp",
                coords.device,
                extra=f"call={_atom_call} p={_runtime_tensor_mib(p):.1f}MiB",
            )

        q_skip, c_skip, p_skip = q, c, p
        q = self.atom_encoder(q=q, c=c, p=p, mask=mask_f, to_keys=to_keys)
        if _trace_atom:
            _runtime_cuda_line(
                "atom.encoder_done",
                coords.device,
                extra=f"call={_atom_call} q={tuple(q.shape)}",
            )

        # Atom -> token mean.
        q_real = q[:, :A].reshape(L, C, -1)
        mask_real = atom_mask.unsqueeze(-1)
        denom = mask_real.sum(dim=1).clamp_min(1.0)
        a = (self.atom_to_token(q_real) * mask_real).sum(dim=1) / denom
        a = a.unsqueeze(0)

        s_cond = self.s_to_a(s).unsqueeze(0)
        a = a + s_cond
        token_mask = (
            torch.ones((1, L), dtype=coords.dtype, device=coords.device)
            if token_valid_mask is None else token_valid_mask.view(1, L)
        )
        a = self.token_transformer(
            a=a,
            s=s_cond,
            z=z,
            mask=token_mask,
        )
        a = self.a_norm(a)
        if _trace_atom:
            _runtime_cuda_line(
                "atom.token_transformer_done",
                coords.device,
                extra=f"call={_atom_call} a={tuple(a.shape)}",
            )

        # Token -> atom and decoder.
        a_atom = a[0][valid_tid] * (token_ids >= 0).unsqueeze(-1)
        q = q_skip + self.a_to_q(a_atom)
        q = self.atom_decoder(q=q, c=c_skip, p=p_skip, mask=mask_f, to_keys=to_keys)
        if _trace_atom:
            _runtime_cuda_line(
                "atom.decoder_done",
                coords.device,
                extra=f"call={_atom_call} q={tuple(q.shape)}",
            )
        update = self.to_xyz(q[:, :A]).reshape(L, C, 3)
        update = update * atom_mask.unsqueeze(-1)

        # Preserve the existing AbFlow coordinate-authority mask.  Context
        # residues participate in representation/attention but must not be moved
        # by the structure decoder unless the original model says they are
        # coordinate-generation residues.
        if residue_update_mask is not None:
            if residue_update_mask.ndim != 1 or residue_update_mask.shape[0] != L:
                raise ValueError(
                    "residue_update_mask must be [L]; "
                    f"got {tuple(residue_update_mask.shape)} for L={L}"
                )
            update = update * residue_update_mask.to(update).view(L, 1, 1)

        # Endpoint-like residual coordinate readout.
        pred = coords + update
        return self.token_out(a[0]), pred

    def forward_batched(
        self,
        s,
        z,
        coords,
        atom_attr,
        atom_weights,
        residue_update_mask=None,
        token_valid_mask=None,
    ):
        """MFDesign-style batch-first fixed-slot AtomStructure.

        Parameters
        ----------
        s : [B,L,Ds]
        z : [B,L,L,Dz]
        coords : [B,L,C,3]
        atom_attr : [B,L,C,Da]
        atom_weights : [B,L,C]
        residue_update_mask : [B,L] or None
        token_valid_mask : [B,L] or None

        The equations are exactly the same as ``forward``.  The only change is
        that the existing MFDesign batch dimension is used directly instead of
        invoking the module B separate times from Python.
        """
        if s.ndim != 3 or z.ndim != 4 or coords.ndim != 4:
            raise ValueError(
                "forward_batched expects s[B,L,D], z[B,L,L,Dz], "
                "coords[B,L,C,3]"
            )
        B, L, C, _ = coords.shape
        if C != self.n_channel:
            raise ValueError(
                f"expected n_channel={self.n_channel}, got C={C}"
            )
        if z.shape[:3] != (B, L, L):
            raise ValueError(
                f"z shape mismatch: got {tuple(z.shape)}, "
                f"expected [{B},{L},{L},Dz]"
            )

        if _env_flag("ABFLOW_PERF_DIAGNOSTICS", False):
            _ATTN_RUNTIME_STATS["batched_atom_calls"] += 1

        atom_mask = (atom_weights != 0).to(coords.dtype)
        if token_valid_mask is None:
            token_valid_mask = torch.ones(
                (B, L), dtype=coords.dtype, device=coords.device
            )
        else:
            token_valid_mask = token_valid_mask.to(
                device=coords.device, dtype=coords.dtype
            ).reshape(B, L)
        atom_mask = atom_mask * token_valid_mask[:, :, None]
        atom_feat = torch.cat(
            [atom_attr, atom_mask.unsqueeze(-1)], dim=-1
        )

        A = L * C
        pad_A = (self.W - (A % self.W)) % self.W
        A_pad = A + pad_A

        coords_f = coords.reshape(B, A, 3)
        mask_f = atom_mask.reshape(B, A)
        feat_f = atom_feat.reshape(B, A, atom_feat.shape[-1])
        token_ids = (
            torch.arange(L, device=coords.device)
            .repeat_interleave(C)
            .view(1, A)
            .expand(B, A)
        )

        coords_f = self._pad_atoms(coords_f, pad_A, fill=0.0)
        mask_f = self._pad_atoms(
            mask_f.unsqueeze(-1), pad_A, fill=0.0
        ).squeeze(-1)
        feat_f = self._pad_atoms(feat_f, pad_A, fill=0.0)
        token_ids = self._pad_atoms(
            token_ids.unsqueeze(-1).to(coords.dtype),
            pad_A,
            fill=-1.0,
        ).squeeze(-1).long()

        c = self.embed_atom_features(feat_f)
        valid_tid = token_ids.clamp(min=0)
        token_cond = torch.gather(
            s,
            1,
            valid_tid.unsqueeze(-1).expand(
                B, A_pad, s.shape[-1]
            ),
        )
        token_cond = token_cond * (token_ids >= 0).unsqueeze(-1)
        c = c + self.s_to_c(token_cond)

        q = c + self.r_to_q(coords_f)

        K = A_pad // self.W
        indexing = get_indexing_matrix(
            K, self.W, self.H, coords.device
        )
        to_keys = partial(
            single_to_keys,
            indexing_matrix=indexing,
            W=self.W,
            H=self.H,
        )

        q_coords = coords_f.view(B, K, self.W, 1, 3)
        k_coords = to_keys(coords_f).view(B, K, 1, self.H, 3)
        d = k_coords - q_coords
        inv_dist = 1.0 / (
            1.0 + torch.sum(d * d, dim=-1, keepdim=True)
        )

        q_mask = mask_f.view(B, K, self.W, 1).bool()
        k_mask = to_keys(
            mask_f.unsqueeze(-1)
        ).view(B, K, 1, self.H).bool()
        valid = (q_mask & k_mask).float().unsqueeze(-1)

        p = self.embed_atompair_pos(d) * valid
        p = p + self.embed_atompair_dist(inv_dist) * valid
        p = p + self.embed_atompair_mask(valid) * valid

        # Same token-pair -> atom-window lookup as the validated single path.
        q_tid = token_ids.view(B, K, self.W, 1)
        k_tid = to_keys(
            token_ids.unsqueeze(-1).float()
        ).view(B, K, 1, self.H).long()
        q_valid = q_tid >= 0
        k_valid = k_tid >= 0
        q_safe = q_tid.clamp(min=0, max=max(0, L - 1))
        k_safe = k_tid.clamp(min=0, max=max(0, L - 1))

        b_idx = torch.arange(
            B, device=coords.device
        ).view(B, 1, 1, 1)
        # MFDesign gold execution order: 128 -> atom_z projection first,
        # then lift only the narrow tensor into atom windows.
        z_projected = self.z_to_p(z)
        z_atom_projected = z_projected[b_idx, q_safe, k_safe]
        z_atom_projected = z_atom_projected * (
            q_valid & k_valid
        ).unsqueeze(-1)
        p = p + z_atom_projected

        p = p + self.c_to_p_q(
            c.view(B, K, self.W, 1, -1)
        )
        p = p + self.c_to_p_k(
            to_keys(c).view(B, K, 1, self.H, -1)
        )
        p = p + self.p_mlp(p)

        q_skip, c_skip, p_skip = q, c, p
        q = self.atom_encoder(
            q=q, c=c, p=p, mask=mask_f, to_keys=to_keys
        )

        # Atom -> token mean, now batched.
        q_real = q[:, :A].reshape(B, L, C, -1)
        mask_real = atom_mask.unsqueeze(-1)
        denom = mask_real.sum(dim=2).clamp_min(1.0)
        a = (
            self.atom_to_token(q_real) * mask_real
        ).sum(dim=2) / denom

        s_cond = self.s_to_a(s)
        a = a + s_cond
        a = self.token_transformer(
            a=a,
            s=s_cond,
            z=z,
            mask=token_valid_mask,
        )
        a = self.a_norm(a)

        # Token -> atom gather and decoder.
        a_atom = torch.gather(
            a,
            1,
            valid_tid.unsqueeze(-1).expand(
                B, A_pad, a.shape[-1]
            ),
        )
        a_atom = a_atom * (token_ids >= 0).unsqueeze(-1)
        q = q_skip + self.a_to_q(a_atom)
        q = self.atom_decoder(
            q=q,
            c=c_skip,
            p=p_skip,
            mask=mask_f,
            to_keys=to_keys,
        )

        update = self.to_xyz(q[:, :A]).reshape(B, L, C, 3)
        update = update * atom_mask.unsqueeze(-1)

        if residue_update_mask is not None:
            if residue_update_mask.shape != (B, L):
                raise ValueError(
                    "batched residue_update_mask must be [B,L]; "
                    f"got {tuple(residue_update_mask.shape)}"
                )
            update = update * residue_update_mask.to(
                update
            ).view(B, L, 1, 1)

        pred = coords + update
        return self.token_out(a), pred



# ============================================================================
# v111 — MFDesign-standard token semantics at the AbFlow fixed-slot boundary
# ============================================================================

class _ZeroPositionEmbedding(nn.Module):
    """Compatibility shim: pair RelativePositionEncoder owns positional semantics."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, residue_pos):
        return torch.zeros(
            residue_pos.shape[0], self.dim,
            device=residue_pos.device, dtype=torch.float32,
        )


class _AbFlowEmbeddingTables(nn.Module):
    """Fixed-slot residue/atom lookup tables; not a graph/message-passing module."""
    def __init__(self, residue_dim, atom_dim, num_aa, num_atom_type, num_atom_pos,
                 atom_pad_idx):
        super().__init__()
        self.residue_embedding = nn.Embedding(num_aa, residue_dim)
        self.res_pos_embedding = _ZeroPositionEmbedding(residue_dim)
        self.atom_embedding = nn.Embedding(num_atom_type, atom_dim)
        self.atom_pos_embedding = nn.Embedding(num_atom_pos, atom_dim)
        self.atom_pad_id = int(atom_pad_idx)


class _LocalEdgeBookkeeper:
    """Non-learned surface/local-neighborhood bookkeeping only.

    MFDesign Pairformer no longer receives these sparse edges as its pair
    representation.  They are retained solely because the deferred historical
    MS_E_GCL surface operator still consumes local KNN edges.
    """
    def __init__(self, atom_pos_pad_idx, ag_seg_id):
        self.atom_pos_pad_idx = int(atom_pos_pad_idx)
        self.ag_seg_id = int(ag_seg_id)

    @staticmethod
    def get_batch_edges(batch_id):
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)
        N = batch_id.shape[0]
        max_n = int(torch.max(lengths).item()) if lengths.numel() else 0
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], (1, 0), value=0)
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]
        if N == 0 or max_n == 0:
            empty = torch.empty(0, dtype=torch.long, device=batch_id.device)
            return (empty, empty), (offsets, max_n, gni2lni)
        same = batch_id[:, None] == batch_id[None, :]
        same.fill_diagonal_(False)
        row, col = torch.nonzero(same, as_tuple=True)
        return (row, col), (offsets, max_n, gni2lni)


class MFDesignAbFlowFeatureAdapter(nn.Module):
    """Replace SeparatedAminoAcidFeature as the AbFlow fixed-slot data adapter.

    This module owns only token/atom identity tables and the temporary sparse
    neighborhood bookkeeping required by the *deferred* surface refiner.  It
    does not own the modern representation: single/pair semantics are produced
    by MFDesign-style Input/RelativePosition/Pairformer code in ``AMEncoder``.
    """
    def __init__(self, embed_size, atom_embed_size, fix_atom_weights=False,
                 backbone_only=False):
        super().__init__()
        del fix_atom_weights
        self.backbone_only = bool(backbone_only)
        self.num_aa_type = len(VOCAB)
        self.num_atom_type = VOCAB.get_num_atom_type()
        self.num_atom_pos = VOCAB.get_num_atom_pos()
        self.atom_mask_idx = VOCAB.get_atom_mask_idx()
        self.atom_pad_idx = VOCAB.get_atom_pad_idx()
        self.atom_pos_mask_idx = VOCAB.get_atom_pos_mask_idx()
        self.atom_pos_pad_idx = VOCAB.get_atom_pos_pad_idx()
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)
        self.boh_idx = VOCAB.symbol_to_idx(VOCAB.BOH)
        self.bol_idx = VOCAB.symbol_to_idx(VOCAB.BOL)
        self.mask_idx = VOCAB.get_mask_idx()
        self.ag_seg_id, self.hc_seg_id, self.lc_seg_id = 1, 2, 3
        n_channel = 4 if self.backbone_only else VOCAB.MAX_ATOM_NUMBER

        residue_atom_type, residue_atom_pos = [], []
        backbone = [VOCAB.atom_to_idx(atom[0]) for atom in VOCAB.backbone_atoms]
        special_mask = VOCAB.get_special_mask()
        for i in range(len(VOCAB)):
            if i in {self.boa_idx, self.boh_idx, self.bol_idx, self.mask_idx}:
                residue_atom_type.append([self.atom_mask_idx] * n_channel)
                residue_atom_pos.append([self.atom_pos_mask_idx] * n_channel)
            elif special_mask[i] == 1:
                residue_atom_type.append([self.atom_pad_idx] * n_channel)
                residue_atom_pos.append([self.atom_pos_pad_idx] * n_channel)
            else:
                sidechain = VOCAB.get_sidechain_info(VOCAB.idx_to_symbol(i))
                atom_type = list(backbone)
                atom_pos = [VOCAB.atom_pos_to_idx(VOCAB.atom_pos_bb)] * len(backbone)
                if not self.backbone_only:
                    atom_type += [VOCAB.atom_to_idx(atom[0]) for atom in sidechain]
                    atom_pos += [VOCAB.atom_pos_to_idx(atom[1]) for atom in sidechain]
                pad = n_channel - len(atom_type)
                residue_atom_type.append(atom_type + [self.atom_pad_idx] * pad)
                residue_atom_pos.append(atom_pos + [self.atom_pos_pad_idx] * pad)

        self.register_buffer(
            'residue_atom_type', torch.tensor(residue_atom_type, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            'residue_atom_pos', torch.tensor(residue_atom_pos, dtype=torch.long),
            persistent=True,
        )

        # Fixed chemical topology required only by AbFlow's diagnostic bond/chi
        # calculations.  These are data tables, not a learned representation.
        if not self.backbone_only:
            sc_bonds, sc_bonds_mask = [], []
            sc_chis, sc_chis_mask = [], []
            max_chi, max_bond = 0, 0
            raw_chis, raw_bonds = [], []
            for i in range(len(VOCAB)):
                if special_mask[i] == 1:
                    chis_i, bonds_i = [], []
                else:
                    symbol = VOCAB.idx_to_symbol(i)
                    atom_layout = list(VOCAB.backbone_atoms) + list(
                        VOCAB.get_sidechain_info(symbol)
                    )
                    atom2channel = {atom: j for j, atom in enumerate(atom_layout)}
                    chi_atoms, bond_atoms = VOCAB.get_sidechain_geometry(symbol)
                    chis_i = [
                        [atom2channel[a] for a in atoms] for atoms in chi_atoms
                    ]
                    bonds_i = []
                    for src_atom in bond_atoms:
                        for dst_atom in bond_atoms[src_atom]:
                            bonds_i.append((atom2channel[src_atom], atom2channel[dst_atom]))
                raw_chis.append(chis_i)
                raw_bonds.append(bonds_i)
                max_chi = max(max_chi, len(chis_i))
                max_bond = max(max_bond, len(bonds_i))
            for chis_i, bonds_i in zip(raw_chis, raw_bonds):
                sc_chis_mask.append([1] * len(chis_i) + [0] * (max_chi - len(chis_i)))
                sc_bonds_mask.append([1] * len(bonds_i) + [0] * (max_bond - len(bonds_i)))
                sc_chis.append(chis_i + [[-1, -1, -1, -1]] * (max_chi - len(chis_i)))
                sc_bonds.append(bonds_i + [(-1, -1)] * (max_bond - len(bonds_i)))
            self.register_buffer('sidechain_chi_angle_atoms', torch.tensor(sc_chis, dtype=torch.long), persistent=True)
            self.register_buffer('sidechain_chi_mask', torch.tensor(sc_chis_mask, dtype=torch.bool), persistent=True)
            self.register_buffer('sidechain_bonds', torch.tensor(sc_bonds, dtype=torch.long), persistent=True)
            self.register_buffer('sidechain_bonds_mask', torch.tensor(sc_bonds_mask, dtype=torch.bool), persistent=True)

        self.aa_embedding = _AbFlowEmbeddingTables(
            embed_size, atom_embed_size, self.num_aa_type,
            self.num_atom_type, self.num_atom_pos, self.atom_pad_idx,
        )
        # MFDesign sequence semantics, shared with the D3PM head.
        self.type_embedding = nn.Embedding(4, embed_size, padding_idx=0)
        self.region_embedding = nn.Embedding(10, embed_size, padding_idx=0)
        self.edge_constructor = _LocalEdgeBookkeeper(
            self.atom_pos_pad_idx, self.ag_seg_id
        )

    def _is_global(self, S):
        return (S == self.boa_idx) | (S == self.boh_idx) | (S == self.bol_idx)

    def _construct_residue_pos(self, S):
        # Kept only as a compatibility field; modern positional semantics live
        # in MFDesign RelativePositionEncoder.
        glbl = self._is_global(S)
        glbl_idx = torch.nonzero(glbl, as_tuple=False).flatten()
        if glbl_idx.numel() == 0:
            return torch.arange(S.shape[0], device=S.device, dtype=S.dtype)
        shift = F.pad(glbl_idx[:-1] - glbl_idx[1:] + 1, (1, 0), value=1)
        pos = torch.ones_like(S)
        pos[glbl] = shift
        return torch.cumsum(pos, dim=0)

    def _construct_segment_ids(self, S):
        glbl = self._is_global(S)
        glbl_nodes = S[glbl].clone()
        if glbl_nodes.numel() == 0:
            return torch.zeros_like(S)
        glbl_nodes[glbl_nodes == self.boa_idx] = self.ag_seg_id
        glbl_nodes[glbl_nodes == self.boh_idx] = self.hc_seg_id
        glbl_nodes[glbl_nodes == self.bol_idx] = self.lc_seg_id
        segment_ids = torch.zeros_like(S)
        segment_ids[glbl] = glbl_nodes - F.pad(glbl_nodes[:-1], (1, 0), value=0)
        return torch.cumsum(segment_ids, dim=0)

    def _construct_atom_pos(self, S):
        return self.residue_atom_pos[S]

    @torch.no_grad()
    def get_sidechain_chi_angles_atoms(self, S):
        if self.backbone_only:
            empty_idx = torch.empty((S.shape[0], 0, 4), dtype=torch.long, device=S.device)
            empty_mask = torch.empty((S.shape[0], 0), dtype=torch.bool, device=S.device)
            return empty_idx, empty_mask
        return self.sidechain_chi_angle_atoms[S], self.sidechain_chi_mask[S]

    @torch.no_grad()
    def get_sidechain_bonds(self, S):
        if self.backbone_only:
            empty_idx = torch.empty((S.shape[0], 0, 2), dtype=torch.long, device=S.device)
            empty_mask = torch.empty((S.shape[0], 0), dtype=torch.bool, device=S.device)
            return empty_idx, empty_mask
        return self.sidechain_bonds[S], self.sidechain_bonds_mask[S]

    def get_atom_weights(self, residue_types):
        # MFDesign uses an atom-valid mask rather than learned per-AA atom
        # weights.  This binary authority is exact and has no hidden topology
        # parameterization.
        return (self.residue_atom_pos[residue_types] != self.atom_pos_pad_idx).float()

    def residue_hidden(self, S, residue_pos=None, token_type=None, token_region=None):
        del residue_pos  # Pair RelativePositionEncoder owns residue-position semantics.
        H = self.aa_embedding.residue_embedding(S)
        if token_type is not None:
            H = H + self.type_embedding(token_type.clamp(0, 3))
        if token_region is not None:
            H = H + self.region_embedding(token_region.clamp(0, 9))
        return H

    def construct_edges(self, X, S, batch_id, k_neighbors, atom_pos=None,
                        segment_ids=None):
        # This sparse graph is surface/bookkeeping only.  Pairformer never sees
        # it as z_ij initialization.
        if atom_pos is None:
            atom_pos = self._construct_atom_pos(S)
        if segment_ids is None:
            segment_ids = self._construct_segment_ids(S)
        (row, col), info = self.edge_constructor.get_batch_edges(batch_id)
        offsets, max_n, gni2lni = info

        # utils.nn_utils._knn_edges owns the historical four-field batch-info
        # contract:
        #
        #     (offsets, batch_id, max_n, gni2lni)
        #
        # _LocalEdgeBookkeeper intentionally mirrors only the bookkeeping part
        # of EdgeConstructor.get_batch_edges(), which returns the other three
        # tensors.  Because this adapter calls _knn_edges directly (rather than
        # through historical EdgeConstructor._construct_*_edges), it must supply
        # the already-available batch_id explicitly here.
        #
        # This is surface/local-neighborhood compatibility only; Pairformer z_ij
        # never consumes these sparse KNN edges as representation authority.
        knn_batch_info = (offsets, batch_id, max_n, gni2lni)

        if row.numel() == 0:
            empty = torch.empty((2, 0), dtype=torch.long, device=X.device)
            return empty, empty
        non_global = ~(self._is_global(S[row]) | self._is_global(S[col]))
        row, col = row[non_global], col[non_global]
        row_ag = segment_ids[row] == self.ag_seg_id
        col_ag = segment_ids[col] == self.ag_seg_id
        ctx = row_ag == col_ag
        inter = row_ag != col_ag

        def knn(mask):
            if not bool(mask.any()):
                return torch.empty((2, 0), dtype=torch.long, device=X.device)
            pairs = torch.stack([row[mask], col[mask]], dim=-1)
            return _knn_edges(
                X, atom_pos, pairs, self.atom_pos_pad_idx, k_neighbors,
                knn_batch_info,
            )

        return knn(ctx), knn(inter)

    def update_global_coordinates(self, X, S, atom_pos=None):
        X = X.clone()
        if atom_pos is None:
            atom_pos = self._construct_atom_pos(S)
        glbl = self._is_global(S)
        if not bool(glbl.any()):
            return X
        chain_id = torch.cumsum(glbl.long(), dim=0)
        chain_id[glbl] = 0
        chain_id = chain_id[:, None].expand(-1, atom_pos.shape[-1])
        not_global = ~glbl
        valid = (atom_pos != self.atom_pos_pad_idx)[not_global]
        if bool(valid.any()):
            coords = X[not_global][valid]
            ids = chain_id[not_global][valid]
            global_x = scatter_mean(
                coords, ids, dim=0, dim_size=int(glbl.sum().item()) + 1
            )
            X[glbl] = global_x[1:].unsqueeze(1)
        return X

    def forward(self, X, S, batch_id, k_neighbors, residue_pos=None,
                smooth_prob=None, smooth_mask=None, token_type=None,
                token_region=None):
        H = self.residue_hidden(
            S, residue_pos=residue_pos,
            token_type=token_type, token_region=token_region,
        )
        # Preserve historical smooth-probability diagnostics without restoring
        # SeparatedAminoAcidFeature as representation authority.
        if smooth_prob is not None and smooth_mask is not None and bool(smooth_mask.any()):
            table = self.aa_embedding.residue_embedding(
                torch.arange(smooth_prob.shape[-1], device=S.device, dtype=S.dtype)
            )
            H = H.clone()
            H[smooth_mask] = smooth_prob.to(table.dtype).mm(table).to(H.dtype)
            if token_type is not None:
                H[smooth_mask] += self.type_embedding(token_type[smooth_mask].clamp(0,3))
            if token_region is not None:
                H[smooth_mask] += self.region_embedding(token_region[smooth_mask].clamp(0,9))

        atom_type = self.residue_atom_type[S]
        atom_pos = self.residue_atom_pos[S]
        atom_embedding = (
            self.aa_embedding.atom_embedding(atom_type)
            + self.aa_embedding.atom_pos_embedding(atom_pos)
        )
        atom_weights = self.get_atom_weights(S)
        ctx_edges, inter_edges = self.construct_edges(
            X, S, batch_id, k_neighbors, atom_pos=atom_pos
        )
        return H, (ctx_edges, inter_edges), (atom_embedding, atom_weights)


class MFDesignRelativePositionEncoder(nn.Module):
    """1:1 MFDesign/Boltz RelativePositionEncoder semantics."""
    def __init__(self, token_z, r_max=32, s_max=2):
        super().__init__()
        self.r_max = int(r_max)
        self.s_max = int(s_max)
        self.linear_layer = LinearNoBias(
            4 * (self.r_max + 1) + 2 * (self.s_max + 1) + 1,
            token_z,
        )

    def forward(self, feats):
        b_same_chain = torch.eq(
            feats['asym_id'][:, :, None], feats['asym_id'][:, None, :]
        )
        b_same_residue = torch.eq(
            feats['residue_index'][:, :, None], feats['residue_index'][:, None, :]
        )
        b_same_entity = torch.eq(
            feats['entity_id'][:, :, None], feats['entity_id'][:, None, :]
        )
        d_residue = torch.clip(
            feats['residue_index'][:, :, None]
            - feats['residue_index'][:, None, :]
            + self.r_max,
            0, 2 * self.r_max,
        )
        d_residue = torch.where(
            b_same_chain,
            d_residue,
            torch.zeros_like(d_residue) + 2 * self.r_max + 1,
        )
        a_rel_pos = F.one_hot(d_residue, 2 * self.r_max + 2)

        d_token = torch.clip(
            feats['token_index'][:, :, None]
            - feats['token_index'][:, None, :]
            + self.r_max,
            0, 2 * self.r_max,
        )
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            torch.zeros_like(d_token) + 2 * self.r_max + 1,
        )
        a_rel_token = F.one_hot(d_token, 2 * self.r_max + 2)

        d_chain = torch.clip(
            feats['sym_id'][:, :, None]
            - feats['sym_id'][:, None, :]
            + self.s_max,
            0, 2 * self.s_max,
        )
        d_chain = torch.where(
            b_same_chain,
            torch.zeros_like(d_chain) + 2 * self.s_max + 1,
            d_chain,
        )
        a_rel_chain = F.one_hot(d_chain, 2 * self.s_max + 2)
        return self.linear_layer(torch.cat([
            a_rel_pos.float(), a_rel_token.float(),
            b_same_entity.unsqueeze(-1).float(), a_rel_chain.float(),
        ], dim=-1))


class MFDesignSequenceD3PMConditioner(nn.Module):
    """MFDesign SequenceD3PM head with an explicit AbFlow boundary adapter.

    MFDesign Stage-1 uses hidden_dim=768 because its structure token activation
    is 2*token_s (2*384).  AbFlow's public AMEncoder API remains token_s-wide,
    so v131 performs one explicit input projection before the otherwise
    MFDesign-identical SequenceD3PM MLP.
    """
    def __init__(self, hidden_dim, vocab_size, dropout=0.1):
        super().__init__()
        input_dim = int(hidden_dim)
        hidden_dim = int(int(_cfg_value("architecture", "sequence_hidden", _env_int("ABFLOW_MFDESIGN_SEQUENCE_HIDDEN", 256))))
        self.input_proj = (
            nn.Identity()
            if input_dim == hidden_dim
            else LinearNoBias(input_dim, hidden_dim)
        )
        self.type_embed = nn.Embedding(4, hidden_dim, padding_idx=0)
        self.region_embed = nn.Embedding(10, hidden_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, res_feat, cond):
        res_feat = self.input_proj(res_feat)
        res = self.encoder(res_feat)
        type_embed = self.type_embed(cond['type'].clamp(0, 3))
        region_embed = self.region_embed(cond['region'].clamp(0, 9))
        res = torch.cat([res, type_embed, region_embed], dim=-1)
        res = self.dropout(self.LayerNorm(self.proj(res)))
        return self.decoder(res)




class DetachedConfidenceModule(nn.Module):
    """MFDesign-style confidence refiner with strict generator detachment.

    Faithfully absorbed MFDesign semantics:
      * normalize detached s_inputs/s/z;
      * add s_inputs -> z row/column projections;
      * add factorized s_i*s_j pair product;
      * add RelativePositionEncoder and token_bonds;
      * add predicted-coordinate distogram;
      * Pairformer confidence refinement;
      * pLDDT/PDE/resolved/PAE heads and aggregate pTM/ipTM/interface scores.

    The protein-only AbFlow fixed-slot representation omits MFDesign ligand
    branches but preserves the protein/interface definitions.
    """
    def __init__(self, token_s, token_z, num_heads, num_plddt_bins=50,
                 num_pde_bins=64, num_pae_bins=64, max_dist=22.0,
                 coord_scale_angstrom=10.0):
        super().__init__()
        self.coord_scale_angstrom = float(coord_scale_angstrom)
        self.dist_binner = DistanceBinner(64, 2.0, max_dist)
        self.pred_dist_embedding = nn.Embedding(64, token_z)
        gating_init_(self.pred_dist_embedding.weight)

        self.s_inputs_norm = nn.LayerNorm(token_s)
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)
        self.s_input_to_s = LinearNoBias(token_s, token_s)
        gating_init_(self.s_input_to_s.weight)
        self.s_to_z = LinearNoBias(token_s, token_z)
        self.s_to_z_transpose = LinearNoBias(token_s, token_z)
        gating_init_(self.s_to_z.weight)
        gating_init_(self.s_to_z_transpose.weight)
        self.s_to_z_prod_in1 = LinearNoBias(token_s, token_z)
        self.s_to_z_prod_in2 = LinearNoBias(token_s, token_z)
        self.s_to_z_prod_out = LinearNoBias(token_z, token_z)
        gating_init_(self.s_to_z_prod_out.weight)
        self.rel_pos = MFDesignRelativePositionEncoder(token_z)
        self.token_bonds = nn.Linear(1, token_z, bias=False)

        self.refiner = PairformerModule(
            token_s=token_s, token_z=token_z,
            num_blocks=1, num_heads=num_heads, dropout=0.0,
        )
        self.to_plddt_logits = LinearNoBias(token_s, num_plddt_bins)
        self.to_pde_logits = LinearNoBias(token_z, num_pde_bins)
        self.to_resolved_logits = LinearNoBias(token_s, 2)
        self.to_pae_logits = LinearNoBias(token_z, num_pae_bins)

    @staticmethod
    def _aggregate(logits, end=1.0):
        K = logits.shape[-1]
        centers = (
            torch.arange(K, device=logits.device, dtype=logits.dtype) + 0.5
        ) * (float(end) / float(K))
        return (torch.softmax(logits, dim=-1) * centers).sum(-1)

    @staticmethod
    def _ptm_from_pae(pae_logits, valid, asym_id, interface=False):
        K = pae_logits.shape[-1]
        centers = (
            torch.arange(K, device=pae_logits.device, dtype=pae_logits.dtype)
            + 0.5
        ) * (32.0 / float(K))
        Nres = valid.float().sum().clamp_min(1.0)
        d0 = 1.24 * (torch.clamp(Nres, min=19.0) - 15.0).pow(1.0/3.0) - 1.8
        tm = 1.0 / (1.0 + (centers / d0.clamp_min(1e-6)) ** 2)
        expected = (torch.softmax(pae_logits, dim=-1) * tm).sum(-1)
        pair = valid[:, None] & valid[None, :]
        if interface:
            pair = pair & (asym_id[:, None] != asym_id[None, :])
        denom = pair.float().sum(-1).clamp_min(1e-5)
        per_anchor = (expected * pair.float()).sum(-1) / denom
        valid_anchor = valid & pair.any(-1)
        if bool(valid_anchor.any()):
            return per_anchor[valid_anchor].max()
        return expected.new_tensor(0.0)

    def forward(self, s_inputs, s, z, pred_coords, feats,
                pred_distogram_logits=None, valid_residue_mask=None):
        # Generator isolation is enforced inside the module.
        s_inputs = self.s_inputs_norm(s_inputs.detach()).unsqueeze(0)
        s = self.s_norm(s.detach()).unsqueeze(0)
        z = self.z_norm(z.detach())
        pred_coords = pred_coords.detach()

        s = s + self.s_input_to_s(s_inputs)
        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )
        z = z + self.s_to_z_prod_out(
            self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
            * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
        )
        z = z + self.rel_pos(feats)
        z = z + self.token_bonds(feats['token_bonds'].float())

        ca_idx = 1 if pred_coords.shape[1] > 1 else 0
        pred_ca = pred_coords[:, ca_idx].float() * self.coord_scale_angstrom
        pred_d = torch.cdist(pred_ca, pred_ca).to(pred_coords.dtype)
        d_bin = self.dist_binner(pred_d)
        z = z + self.pred_dist_embedding(d_bin).unsqueeze(0)

        L = pred_coords.shape[0]
        if valid_residue_mask is None:
            valid = torch.ones(L, dtype=torch.bool, device=pred_coords.device)
        else:
            valid = valid_residue_mask.bool()
        valid = valid & feats['token_pad_mask'][0].bool()
        mask = valid.to(pred_coords.dtype).view(1, L)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s, z = self.refiner(s, z, mask=mask, pair_mask=pair_mask)

        plddt_logits = self.to_plddt_logits(s[0])
        pde_logits = self.to_pde_logits(z[0] + z[0].transpose(0, 1))
        resolved_logits = self.to_resolved_logits(s[0])
        pae_logits = self.to_pae_logits(z[0])

        plddt = self._aggregate(plddt_logits, end=1.0)
        pde = self._aggregate(pde_logits, end=32.0)
        pae = self._aggregate(pae_logits, end=32.0)
        valid_f = valid.float()
        complex_plddt = (plddt * valid_f).sum() / valid_f.sum().clamp_min(1.0)

        asym = feats['asym_id'][0]
        interface_pair = (
            valid[:, None] & valid[None, :]
            & (asym[:, None] != asym[None, :])
            & (pred_d < 8.0)
        )
        interface_token = interface_pair.any(-1)
        complex_iplddt = (
            plddt[interface_token].mean()
            if bool(interface_token.any()) else plddt.new_tensor(0.0)
        )

        if pred_distogram_logits is not None:
            probs = torch.softmax(pred_distogram_logits.detach(), dim=-1)
            # 64 bins over 2..22 Å: first ~20 bins correspond to <~8 Å,
            # mirroring MFDesign's contact-weighted PDE aggregation.
            contact_prob = probs[..., :20].sum(-1)
        else:
            contact_prob = (pred_d < 8.0).to(pde.dtype)
        eye = torch.eye(L, dtype=torch.bool, device=pred_d.device)
        pair_valid = valid[:, None] & valid[None, :] & (~eye)
        weight = contact_prob * pair_valid.to(contact_prob.dtype)
        complex_pde = (pde * weight).sum() / weight.sum().clamp_min(1e-5)
        iweight = weight * (asym[:, None] != asym[None, :]).to(weight.dtype)
        complex_ipde = (pde * iweight).sum() / iweight.sum().clamp_min(1e-5)
        ptm = self._ptm_from_pae(pae_logits, valid, asym, interface=False)
        iptm = self._ptm_from_pae(pae_logits, valid, asym, interface=True)

        return {
            'plddt_logits': plddt_logits,
            'pde_logits': pde_logits,
            'resolved_logits': resolved_logits,
            'pae_logits': pae_logits,
            'plddt': plddt,
            'pde': pde,
            'pae': pae,
            'complex_plddt': complex_plddt,
            'complex_iplddt': complex_iplddt,
            'complex_pde': complex_pde,
            'complex_ipde': complex_ipde,
            'ptm': ptm,
            'iptm': iptm,
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def _ca_valid_from_xloss(xloss_mask: torch.Tensor, n_channel: int) -> torch.Tensor:
    ca_idx = 1 if n_channel > 1 else 0
    if xloss_mask.ndim != 2:
        raise ValueError(f"xloss_mask must be [N,C], got {tuple(xloss_mask.shape)}")
    return xloss_mask[:, ca_idx].bool()


def _frame_valid_from_xloss(xloss_mask: torch.Tensor) -> torch.Tensor:
    if xloss_mask.shape[1] < 3:
        return xloss_mask[:, 0].bool()
    return xloss_mask[:, 0].bool() & xloss_mask[:, 1].bool() & xloss_mask[:, 2].bool()


def _express_ca_in_residue_frames(coords: torch.Tensor):
    """Express every CA_j in the N-CA-C frame of residue i.

    Returns
    -------
    expressed : [L, L, 3]
    frame_valid_geometry : [L]
    """
    if coords.shape[1] < 3:
        L = coords.shape[0]
        return coords.new_zeros((L, L, 3)), torch.zeros(
            L, dtype=torch.bool, device=coords.device
        )

    n = coords[:, 0].float()
    ca = coords[:, 1].float()
    c = coords[:, 2].float()

    w1_raw = n - ca
    w2_raw = c - ca
    norm1 = torch.linalg.norm(w1_raw, dim=-1)
    norm2 = torch.linalg.norm(w2_raw, dim=-1)
    w1 = w1_raw / norm1.clamp_min(1e-6).unsqueeze(-1)
    w2 = w2_raw / norm2.clamp_min(1e-6).unsqueeze(-1)

    e1_raw = w1 + w2
    e2_raw = w2 - w1
    e1_norm = torch.linalg.norm(e1_raw, dim=-1)
    e2_norm = torch.linalg.norm(e2_raw, dim=-1)
    e1 = e1_raw / e1_norm.clamp_min(1e-6).unsqueeze(-1)
    e2 = e2_raw / e2_norm.clamp_min(1e-6).unsqueeze(-1)
    e3 = torch.linalg.cross(e1, e2, dim=-1)

    # Similar validity idea to the supplied Boltz frame code.
    cos_angle = (w1 * w2).sum(-1).abs()
    valid = (
        (norm1 > 1e-2)
        & (norm2 > 1e-2)
        & (e1_norm > 1e-3)
        & (e2_norm > 1e-3)
        & (cos_angle < 0.9063)
    )

    # d[i,j] = CA_j - CA_i
    d = ca[None, :, :] - ca[:, None, :]
    expressed = torch.stack(
        [
            torch.einsum("ijd,id->ij", d, e1),
            torch.einsum("ijd,id->ij", d, e2),
            torch.einsum("ijd,id->ij", d, e3),
        ],
        dim=-1,
    )
    return expressed, valid


def _lddt_target_per_residue(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    valid_residue_mask: torch.Tensor,
    cutoff: float = 15.0,
):
    true_d = torch.cdist(true_ca.float(), true_ca.float())
    pred_d = torch.cdist(pred_ca.float(), pred_ca.float())
    L = true_ca.shape[0]
    eye = torch.eye(L, dtype=torch.bool, device=true_ca.device)
    pair = (
        valid_residue_mask[:, None]
        & valid_residue_mask[None, :]
        & (~eye)
        & (true_d < cutoff)
    )
    diff = (pred_d - true_d).abs()
    score = 0.25 * (
        (diff < 0.5).float()
        + (diff < 1.0).float()
        + (diff < 2.0).float()
        + (diff < 4.0).float()
    )
    denom = pair.float().sum(-1)
    target = (score * pair.float()).sum(-1) / denom.clamp_min(1.0)
    has_neighbors = denom > 0
    return target, has_neighbors


# ============================================================================
# Batch/component helpers
# ============================================================================

@torch.no_grad()
def _component_labels(num_nodes, ctx_edges, inter_mask, inter_edges):
    """Infer complex membership from the already-existing AbFlow graph.

    The current AMEncoder API does not receive batch_id.  AbFlow guarantees that
    graph edges never cross complexes.  We therefore use the exact existing
    ctx/local edge topology only to recover connected components; this does not
    create a new scientific feature.

    local inter_edges are mapped back to global indices via inter_mask so the
    antigen and antibody subgraphs of the same complex are connected.
    """
    if num_nodes == 0:
        return torch.empty(0, dtype=torch.long, device=ctx_edges.device)

    local_global = torch.nonzero(inter_mask, as_tuple=False).reshape(-1)
    edge_parts = [ctx_edges]
    if inter_edges is not None and inter_edges.numel() > 0 and local_global.numel() > 0:
        edge_parts.append(local_global[inter_edges])
    edges = torch.cat(edge_parts, dim=1) if len(edge_parts) > 1 else edge_parts[0]

    # Union-find on CPU; labels are discrete bookkeeping only.
    parent = list(range(num_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if edges.numel() > 0:
        e_cpu = edges.detach().to("cpu", non_blocking=False)
        for a, b in zip(e_cpu[0].tolist(), e_cpu[1].tolist()):
            union(int(a), int(b))

    roots = [find(i) for i in range(num_nodes)]
    # Preserve the natural contiguous batch order.
    root_to_label = {}
    labels = []
    for root in roots:
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels.append(root_to_label[root])
    return torch.tensor(labels, dtype=torch.long, device=ctx_edges.device)




# ============================================================================
# Drop-in AMEncoder
# ============================================================================

class AMEncoder(nn.Module):
    """Modern AbFlow backbone with the historical AMEncoder API."""

    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        n_channel,
        channel_nf,
        radial_nf,
        in_edge_nf=0,
        num_verts=50,
        act_fn=nn.SiLU(),
        n_layers=4,
        residual=True,
        dropout=0.1,
        dense=False,
    ):
        super().__init__()
        del in_edge_nf, act_fn, residual

        if dense:
            raise ValueError(
                "Modern AMEncoder does not support historical dense=True; "
                "the explicit z_ij pair representation is already dense."
            )

        self.hidden_nf = int(hidden_nf)
        self.out_node_nf = int(out_node_nf)
        self.n_channel = int(n_channel)
        self.channel_nf = int(channel_nf)
        # v131: token pair width follows MFDesign gold independently of
        # AbFlow token_s/hidden width.  Historical v127 implicitly tied
        # token_z to hidden_nf through radial_nf, which is not MFDesign parity.
        self.pair_nf = int(int(_cfg_value("architecture", "token_z", _env_int("ABFLOW_MFDESIGN_TOKEN_Z", 128))))
        self.n_layers = int(n_layers)
        self.num_verts = int(num_verts)
        self.dropout = nn.Dropout(dropout)

        # -------------------- Module 6-8 + confidence controls
        self.clean_self_condition = bool(_cfg_value(
            "self_conditioning", "enabled",
            _env_flag("ABFLOW_CLEAN_SELF_CONDITION", True),
        ))
        self.clean_self_condition_prob = float(_cfg_value(
            "self_conditioning", "probability",
            _env_float("ABFLOW_CLEAN_SELF_CONDITION_PROB", 0.5),
        ))
        if not (0.0 <= self.clean_self_condition_prob <= 1.0):
            raise ValueError(
                "ABFLOW_CLEAN_SELF_CONDITION_PROB must be in [0,1]"
            )

        self.pair_distogram_enabled = _env_flag(
            "ABFLOW_PAIR_DISTOGRAM", True
        )
        self.pair_distogram_scope = _env_str(
            "ABFLOW_PAIR_DISTOGRAM_SCOPE", "design"
        ).lower()
        if self.pair_distogram_scope not in {"design", "local"}:
            raise ValueError(
                "ABFLOW_PAIR_DISTOGRAM_SCOPE must be design or local"
            )

        self.confidence_enabled = bool(_cfg_value(
            "confidence", "enabled", _env_flag("ABFLOW_CONFIDENCE", True)
        ))
        self.confidence_pae_enabled = bool(_cfg_value(
            "confidence", "pae_enabled", _env_flag("ABFLOW_CONFIDENCE_PAE", True)
        ))
        self.coord_scale_angstrom = _env_float(
            "ABFLOW_MODEL_COORD_SCALE_ANGSTROM", 10.0
        )
        if self.coord_scale_angstrom <= 0:
            raise ValueError(
                "ABFLOW_MODEL_COORD_SCALE_ANGSTROM must be positive"
            )

        # MFDesign/Boltz distogram convention: 64 bins spanning 2..22 Å.
        self.distogram_binner = DistanceBinner(64, 2.0, 22.0)
        self.pde_binner = ErrorBinner(64, 32.0)
        self.pae_binner = ErrorBinner(64, 32.0)

        # Per-forward caches are consumed by the existing AbFlow trainer/model
        # loss path. They are not persistent state and never cross batches.
        self._modern_aux_cache = []
        self.last_modern_auxiliary_losses = {}
        self.last_modern_confidence = {}

        # Diagnostics only; plain Python integer, not model state.
        self._runtime_trace_forward_count = 0

        # -------------------- v111 MFDesign-standard input/pair authority
        self.linear_in = nn.Linear(in_node_nf, hidden_nf)
        self.s_init = nn.Linear(hidden_nf, hidden_nf, bias=False)
        self.z_init_1 = nn.Linear(hidden_nf, self.pair_nf, bias=False)
        self.z_init_2 = nn.Linear(hidden_nf, self.pair_nf, bias=False)
        self.rel_pos = MFDesignRelativePositionEncoder(self.pair_nf)
        self.token_bonds = nn.Linear(1, self.pair_nf, bias=False)

        # Current Score--Flow state geometry is a direct pair condition.  The
        # distance-bin representation follows MFDesign's validated predicted-
        # geometry pair injection; unlike v110, Pairformer therefore sees the
        # actual X_t geometry rather than only a three-class sparse-edge label.
        self.current_state_pair_geometry = _env_flag(
            'ABFLOW_CURRENT_STATE_PAIR_GEOMETRY', True
        )
        self.current_state_dist_embedding = nn.Embedding(64, self.pair_nf)
        nn.init.normal_(self.current_state_dist_embedding.weight, 0.0, 0.02)

        # True MFDesign internal recycling becomes the refinement authority.
        self.recycling_steps = max(0, int(_cfg_value("architecture", "recycling_steps", _env_int("ABFLOW_MFDESIGN_RECYCLING_STEPS", 2))))
        self.random_recycling = bool(_cfg_value("architecture", "random_recycling", _env_flag("ABFLOW_MFDESIGN_RANDOM_RECYCLING", True)))

        # v126 MFDesign-style runtime authority.
        self.batched_runtime = _env_flag(
            "ABFLOW_MFDESIGN_BATCHED_RUNTIME", False
        )
        self.batched_parity_check = _env_flag(
            "ABFLOW_BATCHED_PARITY_CHECK", False
        )
        self.batched_parity_fail_fast = _env_flag(
            "ABFLOW_BATCHED_PARITY_FAIL_FAST", True
        )
        # v127 parity policy:
        # batched GEMM/reduction is mathematically identical to per-sample
        # execution but is not expected to be bitwise-identical in FP32.
        # Use both pointwise and aggregate criteria so we do not "pass" a
        # systematic padding/mask error by merely relaxing one max threshold.
        self.batched_parity_hidden_max_tol = _env_float(
            "ABFLOW_BATCHED_PARITY_HIDDEN_MAX_TOL", 2e-4
        )
        self.batched_parity_hidden_rms_tol = _env_float(
            "ABFLOW_BATCHED_PARITY_HIDDEN_RMS_TOL", 2e-5
        )
        self.batched_parity_hidden_rel_l2_tol = _env_float(
            "ABFLOW_BATCHED_PARITY_HIDDEN_REL_L2_TOL", 2e-5
        )
        self.batched_parity_coord_max_tol = _env_float(
            "ABFLOW_BATCHED_PARITY_COORD_MAX_TOL", 1e-5
        )
        self._batched_parity_done = False

        # -------------------- Module 3: Pairformer
        # v131 MFDesign-gold authority.  These defaults match the released
        # MFDesign Stage-1/2/3 configuration where the AbFlow data boundary
        # permits an exact architectural mapping.
        token_heads = max(1, int(_cfg_value("architecture", "pairformer_heads", _env_int("ABFLOW_MFDESIGN_PAIRFORMER_HEADS", 16))))
        if hidden_nf % token_heads != 0:
            raise ValueError(
                "MFDesign-gold Pairformer requires token_s divisible by "
                f"num_heads; token_s={hidden_nf}, heads={token_heads}. "
                "Use hidden_size=384 for the gold profile."
            )
        pairformer_blocks = max(
            1, int(_cfg_value("architecture", "pairformer_blocks", _env_int("ABFLOW_MFDESIGN_PAIRFORMER_BLOCKS", 4)))
        )
        pairformer_dropout = float(_cfg_value(
            "architecture", "pairformer_dropout",
            _env_float("ABFLOW_MFDESIGN_PAIRFORMER_DROPOUT", 0.1),
        ))
        self.pairformer = PairformerModule(
            token_s=hidden_nf,
            token_z=self.pair_nf,
            num_blocks=pairformer_blocks,
            num_heads=token_heads,
            dropout=pairformer_dropout,
        )

        # Instantiate optional trainable modules only when enabled.  This keeps
        # DDP parameter usage clean for nested ablations and avoids hidden unused
        # parameters in off configurations.
        if self.clean_self_condition:
            self.clean_sc_embedding = nn.Embedding(64, self.pair_nf)
            # Zero-start: enabling SC preserves the parent function at step 0.
            nn.init.zeros_(self.clean_sc_embedding.weight)
        else:
            self.clean_sc_embedding = None

        if self.pair_distogram_enabled:
            self.pair_distogram_head = LinearNoBias(self.pair_nf, 64)
        else:
            self.pair_distogram_head = None

        if self.confidence_enabled:
            self.confidence_module = DetachedConfidenceModule(
                token_s=hidden_nf,
                token_z=self.pair_nf,
                num_heads=token_heads,
                num_plddt_bins=50,
                num_pde_bins=64,
                num_pae_bins=64,
                max_dist=22.0,
                coord_scale_angstrom=self.coord_scale_angstrom,
            )
        else:
            self.confidence_module = None

        # Exact MFDesign recycle projections. Formal v111 uses AbFlow outer
        # iter_round=1 and internal recycling=3, so there is one refinement
        # authority rather than an accidental 3x3 nested recurrence.
        self.s_recycle_norm = nn.LayerNorm(hidden_nf)
        self.z_recycle_norm = nn.LayerNorm(self.pair_nf)
        self.s_recycle = nn.Linear(hidden_nf, hidden_nf, bias=False)
        self.z_recycle = nn.Linear(self.pair_nf, self.pair_nf, bias=False)
        gating_init_(self.s_recycle.weight)
        gating_init_(self.z_recycle.weight)

        # -------------------- Module 4: atom structure network
        self.atom_structure = FixedSlotAtomStructureNetwork(
            token_s=hidden_nf,
            token_z=self.pair_nf,
            atom_input_dim=channel_nf + 1,
            n_channel=n_channel,
            depth=max(1, self.n_layers),  # compatibility only; gold depths use envs
            atom_s=int(_cfg_value("architecture", "atom_s", _env_int("ABFLOW_MFDESIGN_ATOM_S", 128))),
            atom_z=int(_cfg_value("architecture", "atom_z", _env_int("ABFLOW_MFDESIGN_ATOM_Z", 16))),
            atom_heads=int(_cfg_value("architecture", "atom_encoder_heads", _env_int("ABFLOW_MFDESIGN_ATOM_HEADS", 4))),
            token_heads=int(_cfg_value(
                "architecture", "token_transformer_heads",
                _env_int("ABFLOW_MFDESIGN_TOKEN_TRANSFORMER_HEADS", 16),
            )),
            W=int(_cfg_value("architecture", "atom_window_queries", _env_int("ABFLOW_MFDESIGN_ATOM_WINDOW_Q", 32))),
            H=int(_cfg_value("architecture", "atom_window_keys", _env_int("ABFLOW_MFDESIGN_ATOM_WINDOW_K", 128))),
        )

        # -------------------- Keep the validated AbFlow surface geometry
        # Surface remains an AbFlow-standard condition until it is separately
        # modernized; do not invent an unvalidated surface encoder here.
        self.surface_refiner = MS_E_GCL(
            hidden_nf,
            hidden_nf,
            hidden_nf,
            n_channel,
            channel_nf,
            radial_nf,
            surf_nf=num_verts,
            edges_in_d=0,
            act_fn=nn.SiLU(),
            residual=True,
            dropout=dropout,
        )

        self.linear_out = nn.Linear(hidden_nf, out_node_nf)

        # All referenced modules now exist.  This log is observational only and
        # is intentionally placed at the end of __init__ to avoid init-order bugs.
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(
                "[V136Config][Backbone] "
                f"token_s={hidden_nf} token_z={self.pair_nf} "
                f"pairformer_blocks={len(self.pairformer.layers)} "
                f"pairformer_heads={token_heads} "
                f"pairformer_dropout={pairformer_dropout:.3g} "
                f"recycle_max={self.recycling_steps} "
                f"atom_s={self.atom_structure.atom_s} "
                f"atom_z={self.atom_structure.atom_z} "
                f"atom_enc_depth={len(self.atom_structure.atom_encoder.transformer.layers)} "
                f"token_tf_depth={len(self.atom_structure.token_transformer.layers)} "
                f"token_tf_heads={self.atom_structure.token_heads} "
                f"atom_dec_depth={len(self.atom_structure.atom_decoder.transformer.layers)} "
                f"SC={self.clean_self_condition}/p={self.clean_self_condition_prob:.2f} "
                f"confidence={self.confidence_enabled}/PAE={self.confidence_pae_enabled} "
                f"checkpoint={self.pairformer.checkpoint_mode} "
                f"batched={self.batched_runtime}",
                flush=True,
            )


    @staticmethod
    def _max_valid_diff(a, b, mask):
        if mask is None:
            return float((a.float() - b.float()).abs().max().item())
        while mask.ndim < a.ndim:
            mask = mask.unsqueeze(-1)
        selected = (a.float() - b.float()).abs() * mask.to(a)
        return float(selected.max().item())

    def consume_runtime_perf_stats(self):
        return consume_runtime_perf_stats()

    def _batched_runtime_log(self, message):
        if not _env_flag("ABFLOW_BATCHED_RUNTIME_DIAGNOSTICS", False):
            return
        rank = _runtime_trace_rank()
        line = f"[BatchedRuntime] rank={rank} {message}"
        if rank == 0:
            print(line, flush=True)
        _runtime_trace_file_line(line)

    def _run_batched_complexes(
        self,
        s_raw,
        x,
        atom_attr,
        atom_weights,
        feats,
        residue_update_mask,
        recycling_steps,
    ):
        """Batch-first MFDesign trunk + full-complex AtomStructure."""
        B, L, _ = s_raw.shape

        s_init = self.s_init(s_raw)
        z_init = (
            self.z_init_1(s_raw)[:, :, None, :]
            + self.z_init_2(s_raw)[:, None, :, :]
        )
        z_init = z_init + self.rel_pos(feats)
        z_init = z_init + self.token_bonds(
            feats["token_bonds"].float()
        )

        if self.current_state_pair_geometry:
            ca_idx = 1 if x.shape[2] > 1 else 0
            d_ang = torch.cdist(
                x[:, :, ca_idx].float(),
                x[:, :, ca_idx].float(),
            ) * float(self.coord_scale_angstrom)
            geom_bin = self.distogram_binner(d_ang)
            z_init = z_init + self.current_state_dist_embedding(
                geom_bin
            )

        mask = feats["token_pad_mask"].to(s_raw.dtype)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        steps = int(recycling_steps)

        for recycle_idx in range(steps + 1):
            final_pass = recycle_idx == steps
            grad_on = bool(
                self.training
                and final_pass
                and torch.is_grad_enabled()
            )

            # The same AMP correctness barrier as the validated graphwise path,
            # but now it is executed ONCE for the whole local batch rather than
            # once per complex.
            if (
                grad_on
                and recycle_idx > 0
                and torch.is_autocast_enabled()
            ):
                torch.clear_autocast_cache()

            with torch.set_grad_enabled(grad_on):
                s = s_init + self.s_recycle(
                    self.s_recycle_norm(s)
                )
                z = z_init + self.z_recycle(
                    self.z_recycle_norm(z)
                )
                s, z = self.pairformer(
                    s,
                    z,
                    mask=mask,
                    pair_mask=pair_mask,
                )

            if _env_flag("ABFLOW_PERF_DIAGNOSTICS", False):
                _ATTN_RUNTIME_STATS[
                    "batched_pairformer_calls"
                ] += 1
                _ATTN_RUNTIME_STATS[
                    "graphwise_equiv_pairformer_calls"
                ] += B

        s_struct, pred_x = self.atom_structure.forward_batched(
            s=s,
            z=z,
            coords=x,
            atom_attr=atom_attr,
            atom_weights=atom_weights,
            residue_update_mask=residue_update_mask,
            token_valid_mask=mask,
        )
        return s, z, s_struct, pred_x

    def _clean_self_condition_z_batched(
        self,
        s_local,
        z_local,
        current_coords,
        atom_attr,
        atom_weights,
        residue_update_mask,
        token_valid_mask,
        active_graph_ids,
    ):
        """Batch-first clean self-conditioning with lazy active-subset teacher.

        One Bernoulli gate is sampled per active graph in graph order.  Only
        gate=1 graphs enter the detached teacher AtomStructure call.  The formal
        prediction remains one padded-batched call for the whole local batch.
        """
        B = s_local.shape[0]

        if not self.clean_self_condition:
            s_out, pred = self.atom_structure.forward_batched(
                s=s_local,
                z=z_local,
                coords=current_coords,
                atom_attr=atom_attr,
                atom_weights=atom_weights,
                residue_update_mask=residue_update_mask,
                token_valid_mask=token_valid_mask,
            )
            if _env_flag("ABFLOW_PERF_DIAGNOSTICS", False):
                _ATTN_RUNTIME_STATS["batched_shadow_atom_calls"] += 1
            gates = current_coords.new_zeros((B,))
            return s_out, pred, z_local, gates

        # Preserve graphwise RNG semantics: one scalar draw per graph.
        if self.training:
            gate_list = []
            for _graph_id in active_graph_ids:
                del _graph_id
                gate_list.append(
                    (
                        torch.rand((), device=current_coords.device)
                        < float(self.clean_self_condition_prob)
                    ).to(current_coords.dtype)
                )
            gates = torch.stack(gate_list, dim=0)
        else:
            gates = current_coords.new_ones((B,))

        active = gates > 0.5
        if bool(active.any()):
            active_idx = torch.nonzero(active, as_tuple=False).reshape(-1)
            with torch.no_grad():
                _, clean0_active = self.atom_structure.forward_batched(
                    s=s_local.index_select(0, active_idx),
                    z=z_local.index_select(0, active_idx),
                    coords=current_coords.index_select(0, active_idx),
                    atom_attr=atom_attr.index_select(0, active_idx),
                    atom_weights=atom_weights.index_select(0, active_idx),
                    residue_update_mask=residue_update_mask.index_select(0, active_idx),
                    token_valid_mask=token_valid_mask.index_select(0, active_idx),
                )
                if _env_flag("ABFLOW_PERF_DIAGNOSTICS", False):
                    _ATTN_RUNTIME_STATS["batched_shadow_atom_calls"] += 1
                    _ATTN_RUNTIME_STATS["sc_teacher_calls"] += 1
                    _ATTN_RUNTIME_STATS["sc_teacher_graphs"] += int(active_idx.numel())

                ca_idx = 1 if clean0_active.shape[2] > 1 else 0
                d_ang = torch.cdist(
                    clean0_active[:, :, ca_idx].float(),
                    clean0_active[:, :, ca_idx].float(),
                ) * float(self.coord_scale_angstrom)
                sc_bins = self.distogram_binner(d_ang)

            sc_update_active = self.clean_sc_embedding(sc_bins)
            sc_update = torch.zeros_like(z_local).index_copy(
                0, active_idx, sc_update_active
            )
            z_sc = z_local + sc_update
        else:
            # DDP-safe zero dependency without the no-grad teacher call.
            z_sc = z_local + 0.0 * self.clean_sc_embedding.weight.sum()

        s_out, pred = self.atom_structure.forward_batched(
            s=s_local,
            z=z_sc,
            coords=current_coords,
            atom_attr=atom_attr,
            atom_weights=atom_weights,
            residue_update_mask=residue_update_mask,
            token_valid_mask=token_valid_mask,
        )
        if _env_flag("ABFLOW_PERF_DIAGNOSTICS", False):
            _ATTN_RUNTIME_STATS["batched_shadow_atom_calls"] += 1
            _ATTN_RUNTIME_STATS["sc_formal_calls"] += 1
        return s_out, pred, z_sc, gates

    def _maybe_check_batched_parity(
        self,
        s_raw_b,
        x_b,
        atom_attr_b,
        atom_weights_b,
        feats_b,
        update_b,
        lengths,
    ):
        """First-batch deterministic eval parity: batched vs graphwise.

        The check is intentionally:
        - no-grad;
        - FP32 (autocast disabled);
        - eval-mode (dropout disabled);
        - limited to at most two complexes.

        Therefore it validates padding/masking/batch algebra without consuming
        training RNG or changing the actual training graph.
        """
        if (
            not self.batched_parity_check
            or self._batched_parity_done
            or len(lengths) == 0
        ):
            return
        self._batched_parity_done = True

        Bv = min(2, len(lengths))
        Lv = max(int(v) for v in lengths[:Bv])

        pair_training = self.pairformer.training
        atom_training = self.atom_structure.training
        self.pairformer.eval()
        self.atom_structure.eval()

        # This diagnostic is supposed to test *algebraic* batched-vs-graphwise
        # equivalence, not mixed-precision kernel equivalence.  Disabling AMP
        # alone is insufficient on Ampere: global allow_tf32=True still lets
        # FP32 matmul/einsum use TF32 tensor-core kernels, and different batch
        # shapes can select different kernels/accumulation orders.  That can
        # create O(1e-4~1e-3) hidden-state differences even when the equations
        # are identical.  Temporarily disable TF32 only for this no-grad parity
        # probe, then restore the training flags exactly.
        _parity_cuda_tf32 = None
        _parity_cudnn_tf32 = None
        if torch.cuda.is_available():
            _parity_cuda_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
            _parity_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        pair_s_err = 0.0
        pair_z_err = 0.0
        atom_s_err = 0.0
        atom_x_err = 0.0

        pair_s_sse = 0.0
        pair_s_ref_sse = 0.0
        pair_s_count = 0
        pair_z_sse = 0.0
        pair_z_ref_sse = 0.0
        pair_z_count = 0
        atom_s_sse = 0.0
        atom_s_ref_sse = 0.0
        atom_s_count = 0
        atom_x_sse = 0.0
        atom_x_ref_sse = 0.0
        atom_x_count = 0

        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    sraw = s_raw_b[:Bv, :Lv].float()
                    xb = x_b[:Bv, :Lv].float()
                    aa = atom_attr_b[:Bv, :Lv].float()
                    aw = atom_weights_b[:Bv, :Lv].float()
                    up = update_b[:Bv, :Lv]
                    f = {}
                    for key, value in feats_b.items():
                        if key == "token_bonds":
                            f[key] = value[
                                :Bv, :Lv, :Lv
                            ].float()
                        else:
                            f[key] = value[:Bv, :Lv]

                    s_init = self.s_init(sraw)
                    z_init = (
                        self.z_init_1(sraw)[:, :, None, :]
                        + self.z_init_2(sraw)[:, None, :, :]
                    )
                    z_init = z_init + self.rel_pos(f)
                    z_init = z_init + self.token_bonds(
                        f["token_bonds"].float()
                    )
                    if self.current_state_pair_geometry:
                        ca_idx = 1 if xb.shape[2] > 1 else 0
                        d_ang = torch.cdist(
                            xb[:, :, ca_idx],
                            xb[:, :, ca_idx],
                        ) * float(self.coord_scale_angstrom)
                        z_init = z_init + (
                            self.current_state_dist_embedding(
                                self.distogram_binner(d_ang)
                            ).float()
                        )

                    mask = f["token_pad_mask"].float()
                    pair_mask = (
                        mask[:, :, None] * mask[:, None, :]
                    )

                    sb, zb = self.pairformer(
                        s_init,
                        z_init,
                        mask,
                        pair_mask,
                    )

                    for bi in range(Bv):
                        Li = int(lengths[bi])
                        si, zi = self.pairformer(
                            s_init[bi:bi + 1, :Li],
                            z_init[
                                bi:bi + 1, :Li, :Li
                            ],
                            mask[bi:bi + 1, :Li],
                            pair_mask[
                                bi:bi + 1, :Li, :Li
                            ],
                        )
                        _ds = (
                            sb[bi, :Li].float() - si[0].float()
                        )
                        _dz = (
                            zb[bi, :Li, :Li].float()
                            - zi[0].float()
                        )
                        pair_s_err = max(
                            pair_s_err,
                            float(_ds.abs().max().item()),
                        )
                        pair_z_err = max(
                            pair_z_err,
                            float(_dz.abs().max().item()),
                        )
                        pair_s_sse += float(
                            (_ds * _ds).sum().item()
                        )
                        pair_s_ref_sse += float(
                            (si[0].float() ** 2).sum().item()
                        )
                        pair_s_count += int(_ds.numel())

                        pair_z_sse += float(
                            (_dz * _dz).sum().item()
                        )
                        pair_z_ref_sse += float(
                            (zi[0].float() ** 2).sum().item()
                        )
                        pair_z_count += int(_dz.numel())

                    as_b, ax_b = (
                        self.atom_structure.forward_batched(
                            s=sb,
                            z=zb,
                            coords=xb,
                            atom_attr=aa,
                            atom_weights=aw,
                            residue_update_mask=up,
                            token_valid_mask=mask,
                        )
                    )

                    for bi in range(Bv):
                        Li = int(lengths[bi])
                        as_i, ax_i = self.atom_structure(
                            s=sb[bi, :Li],
                            z=zb[
                                bi:bi + 1, :Li, :Li
                            ],
                            coords=xb[bi, :Li],
                            atom_attr=aa[bi, :Li],
                            atom_weights=aw[bi, :Li],
                            residue_update_mask=up[bi, :Li],
                            token_valid_mask=mask[bi, :Li],
                        )
                        _das = (
                            as_b[bi, :Li].float() - as_i.float()
                        )
                        _dax = (
                            ax_b[bi, :Li].float() - ax_i.float()
                        )
                        atom_s_err = max(
                            atom_s_err,
                            float(_das.abs().max().item()),
                        )
                        atom_x_err = max(
                            atom_x_err,
                            float(_dax.abs().max().item()),
                        )

                        atom_s_sse += float(
                            (_das * _das).sum().item()
                        )
                        atom_s_ref_sse += float(
                            (as_i.float() ** 2).sum().item()
                        )
                        atom_s_count += int(_das.numel())

                        atom_x_sse += float(
                            (_dax * _dax).sum().item()
                        )
                        atom_x_ref_sse += float(
                            (ax_i.float() ** 2).sum().item()
                        )
                        atom_x_count += int(_dax.numel())
        finally:
            self.pairformer.train(pair_training)
            self.atom_structure.train(atom_training)
            if _parity_cuda_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = _parity_cuda_tf32
            if _parity_cudnn_tf32 is not None:
                torch.backends.cudnn.allow_tf32 = _parity_cudnn_tf32
            # The diagnostic performed no-grad linear calls.  Never allow any
            # low-precision no-grad weight casts to leak into the real
            # grad-enabled training pass.
            if torch.is_autocast_enabled():
                torch.clear_autocast_cache()

        def _rms(sse, count):
            return (float(sse) / float(max(1, count))) ** 0.5

        def _rel_l2(sse, ref_sse):
            return (
                float(sse) / max(float(ref_sse), 1e-30)
            ) ** 0.5

        pair_s_rms = _rms(pair_s_sse, pair_s_count)
        pair_z_rms = _rms(pair_z_sse, pair_z_count)
        atom_s_rms = _rms(atom_s_sse, atom_s_count)
        atom_x_rms = _rms(atom_x_sse, atom_x_count)

        pair_s_rel = _rel_l2(pair_s_sse, pair_s_ref_sse)
        pair_z_rel = _rel_l2(pair_z_sse, pair_z_ref_sse)
        atom_s_rel = _rel_l2(atom_s_sse, atom_s_ref_sse)
        atom_x_rel = _rel_l2(atom_x_sse, atom_x_ref_sse)

        hidden_max = max(
            pair_s_err, pair_z_err, atom_s_err
        )
        hidden_rms = max(
            pair_s_rms, pair_z_rms, atom_s_rms
        )
        hidden_rel_l2 = max(
            pair_s_rel, pair_z_rel, atom_s_rel
        )

        line = (
            "[BatchedParity] "
            f"rank={_runtime_trace_rank()} B={Bv} Lmax={Lv} "
            f"pair_s_max={pair_s_err:.3e} "
            f"pair_s_rms={pair_s_rms:.3e} "
            f"pair_s_rel_l2={pair_s_rel:.3e} "
            f"pair_z_max={pair_z_err:.3e} "
            f"pair_z_rms={pair_z_rms:.3e} "
            f"pair_z_rel_l2={pair_z_rel:.3e} "
            f"atom_s_max={atom_s_err:.3e} "
            f"atom_s_rms={atom_s_rms:.3e} "
            f"atom_s_rel_l2={atom_s_rel:.3e} "
            f"atom_x_max={atom_x_err:.3e} "
            f"atom_x_rms={atom_x_rms:.3e} "
            f"atom_x_rel_l2={atom_x_rel:.3e} "
            f"hidden_max_tol={self.batched_parity_hidden_max_tol:.1e} "
            f"hidden_rms_tol={self.batched_parity_hidden_rms_tol:.1e} "
            f"hidden_rel_l2_tol={self.batched_parity_hidden_rel_l2_tol:.1e} "
            f"coord_max_tol={self.batched_parity_coord_max_tol:.1e}"
        )
        if _runtime_trace_rank() == 0:
            print(line, flush=True)
        _runtime_trace_file_line(line)

        parity_ok = (
            hidden_max
                <= float(self.batched_parity_hidden_max_tol)
            and hidden_rms
                <= float(self.batched_parity_hidden_rms_tol)
            and hidden_rel_l2
                <= float(self.batched_parity_hidden_rel_l2_tol)
            and atom_x_err
                <= float(self.batched_parity_coord_max_tol)
        )

        verdict = (
            "[BatchedParityVerdict] "
            f"rank={_runtime_trace_rank()} "
            f"status={'PASS' if parity_ok else 'FAIL'} "
            f"hidden_max={hidden_max:.3e} "
            f"hidden_rms={hidden_rms:.3e} "
            f"hidden_rel_l2={hidden_rel_l2:.3e} "
            f"coord_max={atom_x_err:.3e}"
        )
        if _runtime_trace_rank() == 0:
            print(verdict, flush=True)
        _runtime_trace_file_line(verdict)

        if self.batched_parity_fail_fast and not parity_ok:
            raise RuntimeError(
                "v136 strict-FP32 batched runtime numerical parity failed: "
                f"hidden_max={hidden_max:.6e}, "
                f"hidden_rms={hidden_rms:.6e}, "
                f"hidden_rel_l2={hidden_rel_l2:.6e}, "
                f"coord_max={atom_x_err:.6e}"
            )

    def _forward_batched_runtime(
        self,
        *,
        h0,
        x,
        inter_mask,
        inter_x,
        surf_verts,
        update_mask,
        inter_update_mask,
        aligned_edges,
        epi_index,
        channel_attr,
        channel_weights,
        labels,
        num_graphs,
        required_meta,
        token_bonds,
        token_pad_mask,
        effective_recycling_steps,
    ):
        """Complete batch-first MFDesign runtime for the local DDP batch."""
        N = int(h0.shape[0])
        lengths = [
            int((labels == g).sum().item())
            for g in range(num_graphs)
        ]
        Lmax = max(lengths) if lengths else 0
        B = int(num_graphs)

        if B <= 0 or Lmax <= 0:
            raise RuntimeError(
                "batched runtime received an empty local batch"
            )

        device = x.device
        C = x.shape[1]

        flat_idx = torch.full(
            (B, Lmax), -1, dtype=torch.long, device=device
        )
        for g in range(B):
            gidx = torch.nonzero(
                labels == g, as_tuple=False
            ).reshape(-1)
            flat_idx[g, :gidx.numel()] = gidx

        valid = flat_idx >= 0
        safe_idx = flat_idx.clamp(min=0)

        s_raw_b = h0[safe_idx]
        s_raw_b = s_raw_b * valid.unsqueeze(-1).to(s_raw_b)

        x_b = x[safe_idx]
        x_b = x_b * valid[:, :, None, None].to(x_b)

        attr_b = channel_attr[safe_idx]
        attr_b = attr_b * valid[
            :, :, None, None
        ].to(attr_b)
        weights_b = channel_weights[safe_idx]
        weights_b = weights_b * valid[
            :, :, None
        ].to(weights_b)
        update_b = update_mask[safe_idx].bool() & valid

        feats_b = {}
        for key, value in required_meta.items():
            packed = value[safe_idx]
            packed = torch.where(
                valid,
                packed,
                torch.zeros_like(packed),
            )
            feats_b[key] = packed

        # Preserve the dataset-provided BOA/BOH/BOL masking semantics.
        token_mask_b = torch.zeros(
            (B, Lmax),
            dtype=token_pad_mask.dtype,
            device=device,
        )
        for g, Lg in enumerate(lengths):
            token_mask_b[g, :Lg] = token_pad_mask[g, :Lg]
        feats_b["token_pad_mask"] = token_mask_b

        if token_bonds.ndim != 4:
            raise ValueError(
                "token_bonds must be [B,L,L,Cbond]"
            )
        feats_b["token_bonds"] = token_bonds[
            :B, :Lmax, :Lmax
        ]

        # Replace generated local-shadow coordinates before current-Xt pair
        # geometry and full-complex atom input, exactly as graphwise code.
        local_global = torch.nonzero(
            inter_mask, as_tuple=False
        ).reshape(-1)
        g2shadow = torch.full(
            (N,), -1, dtype=torch.long, device=device
        )
        g2shadow[local_global] = torch.arange(
            local_global.numel(), device=device
        )

        x_state_b = x_b.clone()
        for g, Lg in enumerate(lengths):
            gidx = flat_idx[g, :Lg]
            local_pos = torch.nonzero(
                inter_mask[gidx], as_tuple=False
            ).reshape(-1)
            if local_pos.numel() > 0:
                shadow_idx = g2shadow[gidx[local_pos]]
                x_state_b[g, local_pos] = inter_x[shadow_idx]

        real_tokens = int(sum(lengths))
        padded_total = int(B * Lmax)
        if _env_flag("ABFLOW_PERF_DIAGNOSTICS", False):
            _ATTN_RUNTIME_STATS["real_tokens"] += real_tokens
            _ATTN_RUNTIME_STATS["padded_tokens"] += padded_total

        pad_eff = (
            float(real_tokens) / float(max(1, padded_total))
        )
        self._batched_runtime_log(
            f"B={B} lengths={lengths} Lmax={Lmax} "
            f"pad_eff={pad_eff:.4f} "
            f"recycle={effective_recycling_steps}"
        )

        self._maybe_check_batched_parity(
            s_raw_b=s_raw_b,
            x_b=x_state_b,
            atom_attr_b=attr_b,
            atom_weights_b=weights_b,
            feats_b=feats_b,
            update_b=update_b,
            lengths=lengths,
        )

        s_b, z_b, s_struct_b, pred_b = (
            self._run_batched_complexes(
                s_raw=s_raw_b,
                x=x_state_b,
                atom_attr=attr_b,
                atom_weights=weights_b,
                feats=feats_b,
                residue_update_mask=update_b,
                recycling_steps=effective_recycling_steps,
            )
        )

        # Stitch full-complex outputs back to the exact historical flat API.
        h_out = h0.float().clone()
        pred_x = x.clone()
        for g, Lg in enumerate(lengths):
            gidx = flat_idx[g, :Lg]
            s_g = s_struct_b[g, :Lg]
            if s_g.dtype != h_out.dtype:
                s_g = s_g.to(h_out.dtype)
            h_out[gidx] = s_g
            pred_x[gidx] = pred_b[g, :Lg]

        inter_h_out = h0[inter_mask].float().clone()
        pred_inter_x = inter_x.clone()

        # Pack all dynamic H3/local shadows and run clean-SC AtomStructure
        # batch-first as well.
        infos = []
        for g, Lg in enumerate(lengths):
            gidx = flat_idx[g, :Lg]
            local_pos = torch.nonzero(
                inter_mask[gidx], as_tuple=False
            ).reshape(-1)
            if local_pos.numel() == 0:
                continue
            shadow_idx = g2shadow[gidx[local_pos]]
            infos.append(
                (g, gidx, local_pos, shadow_idx)
            )

        if infos:
            Bh = len(infos)
            Hmax = max(
                int(info[2].numel()) for info in infos
            )
            Ds = s_b.shape[-1]
            Dz = z_b.shape[-1]
            Da = channel_attr.shape[-1]

            s_local_b = h0.new_zeros(
                (Bh, Hmax, Ds)
            )
            z_local_b = z_b.new_zeros(
                (Bh, Hmax, Hmax, Dz)
            )
            coords_local_b = inter_x.new_zeros(
                (Bh, Hmax, C, 3)
            )
            attr_local_b = channel_attr.new_zeros(
                (Bh, Hmax, C, Da)
            )
            weights_local_b = channel_weights.new_zeros(
                (Bh, Hmax, C)
            )
            update_local_b = torch.zeros(
                (Bh, Hmax),
                dtype=torch.bool,
                device=device,
            )
            valid_local_b = torch.zeros(
                (Bh, Hmax),
                dtype=token_mask_b.dtype,
                device=device,
            )

            for j, (g, gidx, local_pos, shadow_idx) in enumerate(infos):
                Lj = int(local_pos.numel())
                s_local_b[j, :Lj] = s_b[g, local_pos]
                z_local_b[j, :Lj, :Lj] = (
                    z_b[g, local_pos][:, local_pos]
                )
                coords_local_b[j, :Lj] = inter_x[shadow_idx]
                attr_local_b[j, :Lj] = channel_attr[
                    gidx[local_pos]
                ]
                weights_local_b[j, :Lj] = channel_weights[
                    gidx[local_pos]
                ]
                update_local_b[j, :Lj] = (
                    inter_update_mask[shadow_idx].bool()
                )
                valid_local_b[j, :Lj] = 1

            active_graph_ids = [int(v[0]) for v in infos]
            (
                s_local_out_b,
                pred_local_b,
                z_local_used_b,
                gates,
            ) = self._clean_self_condition_z_batched(
                s_local=s_local_b,
                z_local=z_local_b,
                current_coords=coords_local_b,
                atom_attr=attr_local_b,
                atom_weights=weights_local_b,
                residue_update_mask=update_local_b,
                token_valid_mask=valid_local_b,
                active_graph_ids=active_graph_ids,
            )

            for j, (g, gidx, local_pos, shadow_idx) in enumerate(infos):
                Lj = int(local_pos.numel())
                s_local = s_local_out_b[j, :Lj]
                if s_local.dtype != inter_h_out.dtype:
                    s_local = s_local.to(inter_h_out.dtype)
                inter_h_out[shadow_idx] = s_local
                pred_inter_x[shadow_idx] = pred_local_b[j, :Lj]

                local_feats = {}
                for key, value in feats_b.items():
                    if key == "token_bonds":
                        local_feats[key] = value[
                            g:g + 1, local_pos
                        ][:, :, local_pos]
                    else:
                        local_feats[key] = value[
                            g:g + 1, local_pos
                        ]

                self._modern_aux_cache.append({
                    "global_idx": gidx[local_pos],
                    "shadow_idx": shadow_idx,
                    "design_mask": update_local_b[j, :Lj],
                    "z_base": z_local_b[
                        j:j + 1, :Lj, :Lj
                    ],
                    "z_used": z_local_used_b[
                        j:j + 1, :Lj, :Lj
                    ],
                    "sc_gate": gates[j],
                    "s_inputs": h0[gidx][local_pos],
                    "confidence_feats": local_feats,
                    "recycling_steps": x.new_tensor(
                        float(effective_recycling_steps)
                    ),
                })

        # Preserve the already-validated AbFlow surface refiner exactly.
        if (
            aligned_edges is not None
            and aligned_edges.numel() > 0
            and surf_verts is not None
            and surf_verts.numel() > 0
        ):
            inter_attr = channel_attr[inter_mask]
            inter_weights = channel_weights[inter_mask]
            inter_h_out, pred_inter_x = self.surface_refiner(
                inter_h_out,
                aligned_edges,
                epi_index,
                pred_inter_x,
                surf_verts,
                inter_attr,
                inter_weights,
            )

        # Same auxiliary/confidence finalization as the graphwise parent.
        pending = self._modern_aux_cache
        self._modern_aux_cache = []
        for meta in pending:
            shadow_idx = meta["shadow_idx"]
            global_idx = meta["global_idx"]
            local_xloss_hint = channel_weights[global_idx]
            final_pred = pred_inter_x[shadow_idx]
            final_s = inter_h_out[shadow_idx]
            self._append_aux_cache(
                global_idx=global_idx,
                design_mask=meta["design_mask"],
                z_base=meta["z_base"],
                pred_coords=final_pred,
                s_inputs=meta["s_inputs"],
                s_for_confidence=final_s,
                xloss_hint=(
                    local_xloss_hint[:, 1]
                    if local_xloss_hint.shape[1] > 1
                    else local_xloss_hint[:, 0]
                ),
                sc_gate=meta["sc_gate"],
                confidence_feats=meta["confidence_feats"],
            )

        h_out = h_out.clone()
        h_out[inter_mask] = inter_h_out
        h_out = self.dropout(h_out)
        h_out = self.linear_out(h_out)
        return h_out, pred_x, pred_inter_x

    def _run_one_complex(
        self,
        s_raw,
        x,
        atom_attr,
        atom_weights,
        feats,
        residue_update_mask=None,
        recycling_steps=None,
    ):
        """MFDesign input -> pair init -> true recycling -> atom structure.

        Pair initialization is now the MFDesign form
            z0 = W1 s_i + W2 s_j + RelPos + TokenBond + Geometry(X_t),
        and recycling exactly follows the MFDesign final-pass-gradient rule.
        """
        s_init = self.s_init(s_raw).unsqueeze(0)
        z_init = (
            self.z_init_1(s_raw)[None, :, None, :]
            + self.z_init_2(s_raw)[None, None, :, :]
        )
        z_init = z_init + self.rel_pos(feats)
        z_init = z_init + self.token_bonds(feats['token_bonds'].float())

        if self.current_state_pair_geometry:
            ca_idx = 1 if x.shape[1] > 1 else 0
            d_ang = torch.cdist(x[:, ca_idx].float(), x[:, ca_idx].float()) \
                * float(self.coord_scale_angstrom)
            geom_bin = self.distogram_binner(d_ang)
            z_init = z_init + self.current_state_dist_embedding(geom_bin).unsqueeze(0)

        mask = feats['token_pad_mask'].to(s_raw.dtype)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        steps = self.recycling_steps if recycling_steps is None else int(recycling_steps)

        for recycle_idx in range(steps + 1):
            final_pass = recycle_idx == steps
            # 1:1 MFDesign training semantics: intermediate recycles are
            # state computation only; the final recycle is gradient-bearing.
            grad_on = bool(
                self.training and final_pass and torch.is_grad_enabled()
            )

            # PyTorch AMP correctness barrier:
            # no-grad recycles may populate the autocast weight cache with
            # low-precision copies that have requires_grad=False.  The final
            # grad-enabled recycle must not reuse those detached copies.
            if (
                grad_on
                and recycle_idx > 0
                and torch.is_autocast_enabled()
            ):
                torch.clear_autocast_cache()

            with torch.set_grad_enabled(grad_on):
                s = s_init + self.s_recycle(self.s_recycle_norm(s))
                z = z_init + self.z_recycle(self.z_recycle_norm(z))
                s, z = self.pairformer(
                    s, z, mask=mask, pair_mask=pair_mask
                )

        # Evaluation/no-grad callers naturally keep all recycle passes detached.
        s_struct, pred_x = self.atom_structure(
            s=s[0], z=z, coords=x,
            atom_attr=atom_attr, atom_weights=atom_weights,
            residue_update_mask=residue_update_mask,
            token_valid_mask=feats['token_pad_mask'][0],
        )
        return s[0], z, s_struct, pred_x

    def _pair_focus_mask(self, valid_residue, design_mask):
        L = valid_residue.shape[0]
        eye = torch.eye(L, dtype=torch.bool, device=valid_residue.device)
        pair = valid_residue[:, None] & valid_residue[None, :] & (~eye)
        if self.pair_distogram_scope == "design":
            pair = pair & (design_mask[:, None] | design_mask[None, :])
        return pair

    def _clean_self_condition_z(
        self,
        s_local,
        z_local,
        current_coords,
        atom_attr,
        atom_weights,
        residue_update_mask,
    ):
        """FoldFlow/AbX-style clean self-conditioning with lazy teacher query.

        Training samples the Bernoulli gate *before* the no-grad clean query.
        If gate=0, the teacher query is skipped entirely; the formal
        gradient-bearing structure query is still executed once.  A zero-valued
        dependency on ``clean_sc_embedding.weight`` keeps DDP parameter usage
        deterministic without paying the teacher-forward cost.

        Expected structure queries per graph at probability p are therefore
            1 + p
        instead of the historical unconditional 2.  At p=0.5 this is 1.5.
        """
        if not self.clean_self_condition:
            s_local_out, pred_local = self.atom_structure(
                s=s_local,
                z=z_local,
                coords=current_coords,
                atom_attr=atom_attr,
                atom_weights=atom_weights,
                residue_update_mask=residue_update_mask,
            )
            return s_local_out, pred_local, z_local, current_coords.new_tensor(0.0)

        if self.training:
            gate = (
                torch.rand((), device=current_coords.device)
                < float(self.clean_self_condition_prob)
            ).to(current_coords.dtype)
        else:
            gate = current_coords.new_tensor(1.0)

        if bool(gate.item() > 0.5):
            with torch.no_grad():
                _, clean0 = self.atom_structure(
                    s=s_local,
                    z=z_local,
                    coords=current_coords,
                    atom_attr=atom_attr,
                    atom_weights=atom_weights,
                    residue_update_mask=residue_update_mask,
                )
                ca_idx = 1 if clean0.shape[1] > 1 else 0
                d_ang = torch.cdist(
                    clean0[:, ca_idx].float(),
                    clean0[:, ca_idx].float(),
                ) * float(self.coord_scale_angstrom)
                sc_bins = self.distogram_binner(d_ang)
            z_sc = z_local + self.clean_sc_embedding(sc_bins).unsqueeze(0)
        else:
            # Keep the SC parameter in the DDP graph with exactly zero gradient.
            z_sc = z_local + 0.0 * self.clean_sc_embedding.weight.sum()

        s_local_out, pred_local = self.atom_structure(
            s=s_local,
            z=z_sc,
            coords=current_coords,
            atom_attr=atom_attr,
            atom_weights=atom_weights,
            residue_update_mask=residue_update_mask,
        )
        return s_local_out, pred_local, z_sc, gate

    def _append_aux_cache(
        self,
        global_idx,
        design_mask,
        z_base,
        pred_coords,
        s_inputs,
        s_for_confidence,
        xloss_hint,
        sc_gate,
        confidence_feats,
    ):
        """Cache pair/confidence inputs; confidence is evaluated after U02 decode.

        The network output on the canonical branch is a carrier, not yet the
        physical clean endpoint.  Running confidence here would calibrate the
        wrong coordinate object.  v132 therefore stores the detached-generator
        inputs and evaluates confidence inside ``compute_auxiliary_losses`` after
        ``AbFlowModel`` supplies the analytically decoded clean endpoint.
        """
        entry = {
            "global_idx": global_idx,
            "design_mask": design_mask.bool(),
            "z_base": z_base,
            "pred_coords": pred_coords,
            "xloss_hint": xloss_hint,
            "sc_gate": sc_gate,
            "s_inputs": s_inputs,
            "s_for_confidence": s_for_confidence,
            "confidence_feats": confidence_feats,
        }
        if self.pair_distogram_enabled:
            entry["distogram_logits"] = self.pair_distogram_head(z_base[0])
        self._modern_aux_cache.append(entry)

    def compute_auxiliary_losses(
        self, true_X, xloss_mask, clean_endpoint_override=None,
        clean_endpoint_override_mask=None, residue_native_gate=None,
    ):
        """Compute pair objective and detached confidence calibration losses.

        Parameters
        ----------
        true_X : [N,C,3]
            Native AbFlow coordinates in raw Å units.
        xloss_mask : [N,C]
            Existing AbFlow resolved-atom mask.

        Returns
        -------
        dict[str, Tensor]
            Scalar losses and diagnostics. Confidence targets are built from
            detached predictions, so their gradients cannot reach the generator.
        """
        zero = next(self.parameters()).sum() * 0.0
        if not self._modern_aux_cache:
            out = {
                "distogram_loss": zero,
                "distogram_loss_raw": zero,
                "confidence_loss": zero,
                "plddt_loss": zero,
                "pde_loss": zero,
                "pae_loss": zero,
                "resolved_loss": zero,
                "self_condition_rate": zero.detach(),
                "confidence_mean_plddt": zero.detach(),
                "confidence_mean_pde": zero.detach(),
                "confidence_mean_pae": zero.detach(),
                "confidence_ptm": zero.detach(),
                "confidence_iptm": zero.detach(),
                "confidence_interface_plddt": zero.detach(),
                "confidence_interface_pde": zero.detach(),
            }
            self.last_modern_auxiliary_losses = {
                k: v.detach() if torch.is_tensor(v) else v for k, v in out.items()
            }
            return out

        dist_losses = []
        dist_losses_raw = []
        plddt_losses, pde_losses, pae_losses, resolved_losses = [], [], [], []
        sc_gates = []
        pred_plddt_means, pred_pde_means, pred_pae_means = [], [], []
        pred_ptm, pred_iptm, pred_iplddt, pred_ipde = [], [], [], []

        for cached_entry in self._modern_aux_cache:
            entry = dict(cached_entry)
            gidx = entry["global_idx"]

            pred_model = entry["pred_coords"]
            if (
                clean_endpoint_override is not None
                and clean_endpoint_override_mask is not None
            ):
                local_override = clean_endpoint_override_mask[gidx].bool()
                if bool(local_override.any()):
                    pred_model = pred_model.clone()
                    pred_model[local_override] = clean_endpoint_override[
                        gidx[local_override]
                    ].to(pred_model)

            # Confidence remains strictly generator-detached inside the module,
            # but is now conditioned on the physical clean-endpoint estimate.
            if self.confidence_enabled:
                valid_residue = entry["xloss_hint"].bool()
                conf = self.confidence_module(
                    s_inputs=entry["s_inputs"],
                    s=entry["s_for_confidence"],
                    z=entry["z_base"],
                    pred_coords=pred_model,
                    feats=entry["confidence_feats"],
                    pred_distogram_logits=entry.get("distogram_logits"),
                    valid_residue_mask=valid_residue,
                )
                entry.update(conf)

            true = true_X[gidx].to(pred_model)
            xmask = xloss_mask[gidx].bool()
            pred_ang = pred_model.detach().float() * float(
                self.coord_scale_angstrom
            )
            true_ang = true.detach().float()

            valid_ca = _ca_valid_from_xloss(xmask, true.shape[1])
            design = entry["design_mask"].to(valid_ca.device)
            pair_focus = self._pair_focus_mask(valid_ca, design)

            ca_idx = 1 if true.shape[1] > 1 else 0
            true_ca = true_ang[:, ca_idx]
            pred_ca = pred_ang[:, ca_idx]
            true_d = torch.cdist(true_ca, true_ca)

            if self.pair_distogram_enabled and "distogram_logits" in entry:
                target = self.distogram_binner(true_d)
                logits = entry["distogram_logits"]
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target.reshape(-1),
                    reduction="none",
                ).reshape_as(target)
                raw_disto = _masked_mean(ce, pair_focus)
                dist_losses_raw.append(raw_disto)
                if residue_native_gate is None:
                    gate_value = raw_disto.new_tensor(1.0)
                else:
                    local_gate = residue_native_gate[gidx].to(raw_disto)
                    focus_gate = local_gate[design] if bool(design.any()) else local_gate
                    gate_value = (
                        focus_gate.mean() if focus_gate.numel() > 0
                        else raw_disto.new_tensor(0.0)
                    )
                dist_losses.append(raw_disto * gate_value)

            if self.confidence_enabled:
                # ----- pLDDT calibration, focused on generated residues.
                lddt_target, has_neighbors = _lddt_target_per_residue(
                    pred_ca, true_ca, valid_ca, cutoff=15.0
                )
                plddt_bin = torch.floor(lddt_target * 50.0).long().clamp(max=49)
                plddt_ce = F.cross_entropy(
                    entry["plddt_logits"],
                    plddt_bin,
                    reduction="none",
                )
                plddt_mask = design & valid_ca & has_neighbors
                plddt_losses.append(_masked_mean(plddt_ce, plddt_mask))

                plddt_probs = torch.softmax(entry["plddt_logits"], dim=-1)
                plddt_centers = (
                    torch.arange(50, device=plddt_probs.device, dtype=plddt_probs.dtype)
                    + 0.5
                ) / 50.0
                pred_plddt = (plddt_probs * plddt_centers).sum(-1)
                if bool(plddt_mask.any()):
                    pred_plddt_means.append(
                        pred_plddt[plddt_mask].mean().detach()
                    )

                # ----- MFDesign resolved/unresolved calibration.
                resolved_target = valid_ca.long()
                resolved_ce = F.cross_entropy(
                    entry['resolved_logits'], resolved_target, reduction='none'
                )
                resolved_losses.append(_masked_mean(
                    resolved_ce, design | valid_ca
                ))

                # Aggregate MFDesign-style confidence diagnostics.
                for key, bucket in (
                    ('ptm', pred_ptm), ('iptm', pred_iptm),
                    ('complex_iplddt', pred_iplddt), ('complex_ipde', pred_ipde),
                ):
                    if key in entry:
                        bucket.append(entry[key].detach())

                # ----- PDE calibration.
                pred_d = torch.cdist(pred_ca, pred_ca)
                pde_target_value = (pred_d - true_d).abs()
                pde_target = self.pde_binner(pde_target_value)
                pde_logits = entry["pde_logits"]
                pde_ce = F.cross_entropy(
                    pde_logits.reshape(-1, pde_logits.shape[-1]),
                    pde_target.reshape(-1),
                    reduction="none",
                ).reshape_as(pde_target)
                pde_losses.append(_masked_mean(pde_ce, pair_focus))

                pde_prob = torch.softmax(pde_logits, dim=-1)
                pde_centers = (
                    torch.arange(64, device=pde_prob.device, dtype=pde_prob.dtype)
                    + 0.5
                ) * (32.0 / 64.0)
                pred_pde = (pde_prob * pde_centers).sum(-1)
                if bool(pair_focus.any()):
                    pred_pde_means.append(
                        pred_pde[pair_focus].mean().detach()
                    )

                # ----- PAE calibration using AbFlow N-CA-C protein frames.
                if self.confidence_pae_enabled and true.shape[1] >= 3:
                    true_frame, true_geom_valid = _express_ca_in_residue_frames(
                        true_ang
                    )
                    pred_frame, pred_geom_valid = _express_ca_in_residue_frames(
                        pred_ang
                    )
                    pae_value = torch.linalg.norm(
                        true_frame - pred_frame, dim=-1
                    )
                    pae_target = self.pae_binner(pae_value)
                    pae_logits = entry["pae_logits"]
                    pae_ce = F.cross_entropy(
                        pae_logits.reshape(-1, pae_logits.shape[-1]),
                        pae_target.reshape(-1),
                        reduction="none",
                    ).reshape_as(pae_target)

                    frame_valid = (
                        _frame_valid_from_xloss(xmask)
                        & true_geom_valid
                        & pred_geom_valid
                    )
                    pae_mask = (
                        frame_valid[:, None]
                        & valid_ca[None, :]
                        & (
                            design[:, None]
                            | design[None, :]
                        )
                    )
                    L = pae_mask.shape[0]
                    pae_mask = pae_mask & (
                        ~torch.eye(L, dtype=torch.bool, device=pae_mask.device)
                    )
                    pae_losses.append(_masked_mean(pae_ce, pae_mask))

                    pae_prob = torch.softmax(pae_logits, dim=-1)
                    pae_centers = (
                        torch.arange(
                            64, device=pae_prob.device, dtype=pae_prob.dtype
                        ) + 0.5
                    ) * (32.0 / 64.0)
                    pred_pae = (pae_prob * pae_centers).sum(-1)
                    if bool(pae_mask.any()):
                        pred_pae_means.append(
                            pred_pae[pae_mask].mean().detach()
                        )

            sc_gates.append(entry["sc_gate"].detach().float())

        def avg(xs):
            return torch.stack(xs).mean() if xs else zero

        distogram_loss = avg(dist_losses)
        distogram_loss_raw = avg(dist_losses_raw)
        plddt_loss = avg(plddt_losses)
        pde_loss = avg(pde_losses)
        pae_loss = avg(pae_losses) if self.confidence_pae_enabled else zero
        resolved_loss = avg(resolved_losses)
        confidence_loss = plddt_loss + pde_loss + pae_loss + resolved_loss

        def diag_avg(xs):
            return torch.stack(xs).mean() if xs else zero.detach()

        out = {
            "distogram_loss": distogram_loss,
            "distogram_loss_raw": distogram_loss_raw,
            "confidence_loss": confidence_loss,
            "plddt_loss": plddt_loss,
            "pde_loss": pde_loss,
            "pae_loss": pae_loss,
            "resolved_loss": resolved_loss,
            "self_condition_rate": diag_avg(sc_gates),
            "confidence_mean_plddt": diag_avg(pred_plddt_means),
            "confidence_mean_pde": diag_avg(pred_pde_means),
            "confidence_mean_pae": diag_avg(pred_pae_means),
            "confidence_ptm": diag_avg(pred_ptm),
            "confidence_iptm": diag_avg(pred_iptm),
            "confidence_interface_plddt": diag_avg(pred_iplddt),
            "confidence_interface_pde": diag_avg(pred_ipde),
        }
        out['mfdesign_recycling_steps'] = zero.detach().new_tensor(float(getattr(self, '_last_effective_recycling_steps', self.recycling_steps)))
        self.last_modern_auxiliary_losses = {
            k: v.detach() if torch.is_tensor(v) else v for k, v in out.items()
        }
        self.last_modern_confidence = {
            k: self.last_modern_auxiliary_losses[k]
            for k in (
                "confidence_mean_plddt",
                "confidence_mean_pde",
                "confidence_mean_pae", "confidence_ptm", "confidence_iptm",
                "confidence_interface_plddt", "confidence_interface_pde",
            )
        }

        # --------------------------------------------------------------
        # v121 exact lifecycle fix.
        #
        # _modern_aux_cache is only an assembly workspace used ABOVE to
        # construct the scalar losses in `out`.  Once those scalars exist,
        # no later code reads the cache.  The returned scalar losses retain
        # every autograd dependency required by backward themselves.
        #
        # Keeping the cache on `self`, however, creates extra module-level
        # roots to z_base/distogram/PDE/PAE tensors carrying grad_fn.  Their
        # direct storage is small, but those roots may keep large upstream
        # checkpoint/autograd graphs alive after backward.
        #
        # Releasing this list changes object lifetime only; it does NOT
        # detach any loss, Tensor used by the returned objective, parameter,
        # RNG state, or mathematical operation.
        # --------------------------------------------------------------
        _released_aux_entries = len(self._modern_aux_cache)
        self._modern_aux_cache = []
        if _runtime_trace_enabled():
            _runtime_cuda_line(
                "aux_loss.cache_released",
                zero.device,
                extra=f"entries={_released_aux_entries}",
            )
        return out

    def forward(
        self,
        h,
        x,
        ctx_edges,
        inter_mask,
        inter_x,
        surf_verts,
        inter_edges,
        update_mask,
        inter_update_mask,
        aligned_edges,
        epi_index,
        channel_attr,
        channel_weights,
        ctx_edge_attr=None,
        inter_edge_attr=None,
        surf_edge_attr=None,
        batch_id=None,
        local_batch_id=None,
        token_index=None,
        residue_index=None,
        asym_id=None,
        entity_id=None,
        sym_id=None,
        token_type=None,
        token_region=None,
        token_bonds=None,
        token_pad_mask=None,
    ):
        # Pair-Time / old edge attributes are intentionally not part of the
        # modern pair authority.  Formal modern runs must keep Pair-Time off.
        del ctx_edge_attr, inter_edge_attr, surf_edge_attr

        # Clear per-forward auxiliary state before any graph is processed.
        self._modern_aux_cache = []
        self.last_modern_auxiliary_losses = {}
        self.last_modern_confidence = {}

        h0 = self.dropout(self.linear_in(h))
        N = h0.shape[0]
        if x.shape[:2] != channel_weights.shape:
            raise ValueError(
                f"x/channel_weights shape mismatch: x={tuple(x.shape)}, "
                f"weights={tuple(channel_weights.shape)}"
            )
        if channel_attr.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"channel_attr/x shape mismatch: attr={tuple(channel_attr.shape)}, "
                f"x={tuple(x.shape)}"
            )

        local_global = torch.nonzero(inter_mask, as_tuple=False).reshape(-1)
        if local_batch_id is not None:
            local_batch_id = torch.as_tensor(
                local_batch_id, device=x.device, dtype=torch.long
            ).reshape(-1)
            if local_batch_id.numel() != local_global.numel():
                raise ValueError(
                    "local_batch_id length mismatch: "
                    f"expected {local_global.numel()}, got {local_batch_id.numel()}"
                )
        if local_global.numel() != inter_x.shape[0]:
            raise ValueError(
                "inter_mask and inter_x disagree: "
                f"{local_global.numel()} vs {inter_x.shape[0]}"
            )

        # Formal AbFlow_model.py passes the exact per-residue batch_id.  This
        # avoids the previous CPU union-find fallback and removes a GPU->CPU
        # synchronization from every modern-backbone forward.  The fallback is
        # retained only for API compatibility with older callers/tests.
        if batch_id is not None:
            labels = torch.as_tensor(
                batch_id, device=x.device, dtype=torch.long
            ).reshape(-1)
            if labels.numel() != N:
                raise ValueError(
                    f"batch_id length mismatch: expected {N}, got {labels.numel()}"
                )
        else:
            labels = _component_labels(
                num_nodes=N,
                ctx_edges=ctx_edges,
                inter_mask=inter_mask,
                inter_edges=inter_edges,
            )
        num_graphs = int(labels.max().item()) + 1 if labels.numel() else 0

        required_meta = {
            'token_index': token_index, 'residue_index': residue_index,
            'asym_id': asym_id, 'entity_id': entity_id, 'sym_id': sym_id,
            'type': token_type, 'region': token_region,
        }
        for name, value in required_meta.items():
            if value is None:
                raise ValueError(f'MFDesign metadata {name} is required in v111')
            value = torch.as_tensor(value, device=x.device, dtype=torch.long).reshape(-1)
            if value.numel() != N:
                raise ValueError(f'{name} length mismatch: {value.numel()} vs {N}')
            required_meta[name] = value
        if token_bonds is None or token_pad_mask is None:
            raise ValueError('token_bonds and token_pad_mask are required in v111')
        token_bonds = torch.as_tensor(token_bonds, device=x.device, dtype=x.dtype)
        token_pad_mask = torch.as_tensor(token_pad_mask, device=x.device)

        if self.training and self.random_recycling:
            effective_recycling_steps = random.randint(0, self.recycling_steps)
        else:
            effective_recycling_steps = self.recycling_steps

        self._last_effective_recycling_steps = int(effective_recycling_steps)

        if self.batched_runtime:
            return self._forward_batched_runtime(
                h0=h0,
                x=x,
                inter_mask=inter_mask,
                inter_x=inter_x,
                surf_verts=surf_verts,
                update_mask=update_mask,
                inter_update_mask=inter_update_mask,
                aligned_edges=aligned_edges,
                epi_index=epi_index,
                channel_attr=channel_attr,
                channel_weights=channel_weights,
                labels=labels,
                num_graphs=num_graphs,
                required_meta=required_meta,
                token_bonds=token_bonds,
                token_pad_mask=token_pad_mask,
                effective_recycling_steps=effective_recycling_steps,
            )

        _trace_id = int(self._runtime_trace_forward_count)
        _trace_forward = (
            _runtime_trace_enabled()
            and self.training
            and _trace_id < max(0, _env_int("ABFLOW_RUNTIME_TRACE_FORWARDS", 2))
        )
        self._runtime_trace_forward_count += 1
        if _trace_forward:
            _runtime_cuda_line(
                "amenc.start",
                x.device,
                extra=f"fwd={_trace_id} N={N} graphs={num_graphs} recycling={effective_recycling_steps}",
            )

        # AMP stitching authority:
        #
        # Under CUDA autocast, linear projections such as ``linear_in`` can
        # produce BF16 ``h0``, while normalization-heavy MFDesign atom/token
        # blocks intentionally return FP32 states (LayerNorm is an FP32-stable
        # autocast op).  PyTorch indexed assignment / index_put does NOT perform
        # autocast and therefore requires exact source/destination dtypes.
        #
        # These tensors are only graph-wise aggregation buffers, not expensive
        # matmul activations.  Keep the hidden-state stitching buffers in FP32:
        #   * preserves the numerically stable normalized MFDesign output;
        #   * avoids silently quantizing FP32 token states back to BF16;
        #   * matches the AMP-off reference representation dtype;
        #   * downstream Linear/attention ops remain governed by autocast.
        #
        # Coordinate buffers keep the coordinate input dtype unchanged.
        h_out = h0.float().clone()
        pred_x = x.clone()
        inter_h_out = h0[inter_mask].float().clone()
        pred_inter_x = inter_x.clone()

        # global index -> local-shadow index
        g2shadow = torch.full((N,), -1, dtype=torch.long, device=x.device)
        g2shadow[local_global] = torch.arange(local_global.numel(), device=x.device)

        for graph_id in range(num_graphs):
            gidx = torch.nonzero(labels == graph_id, as_tuple=False).reshape(-1)
            if gidx.numel() == 0:
                continue

            Lg = int(gidx.numel())
            feats_g = {
                key: value[gidx].view(1, Lg)
                for key, value in required_meta.items()
                if key not in {'type', 'region'}
            }
            feats_g['type'] = required_meta['type'][gidx].view(1, Lg)
            feats_g['region'] = required_meta['region'][gidx].view(1, Lg)
            feats_g['token_pad_mask'] = token_pad_mask[graph_id, :Lg].view(1, Lg)
            feats_g['token_bonds'] = token_bonds[
                graph_id:graph_id + 1, :Lg, :Lg, :
            ]

            # The Pairformer state must be conditioned on the *actual* Score--Flow
            # state X_t.  Replace generated local-shadow coordinates in the graph
            # view before both current-geometry pair encoding and atom input.
            x_state_g = x[gidx].clone()
            local_pos_pre = torch.nonzero(
                inter_mask[gidx], as_tuple=False
            ).reshape(-1)
            if local_pos_pre.numel() > 0:
                shadow_idx_pre = g2shadow[gidx[local_pos_pre]]
                x_state_g[local_pos_pre] = inter_x[shadow_idx_pre]

            if _trace_forward:
                _runtime_cuda_line(
                    "amenc.before_complex",
                    x.device,
                    extra=f"fwd={_trace_id} graph={graph_id}/{num_graphs} L={Lg}",
                )
            s_trunk_g, z_g, s_struct_g, pred_x_g = self._run_one_complex(
                s_raw=h0[gidx],
                x=x_state_g,
                atom_attr=channel_attr[gidx],
                atom_weights=channel_weights[gidx],
                feats=feats_g,
                residue_update_mask=update_mask[gidx],
                recycling_steps=effective_recycling_steps,
            )
            if _trace_forward:
                _runtime_cuda_line(
                    "amenc.after_complex",
                    x.device,
                    extra=(
                        f"fwd={_trace_id} graph={graph_id}/{num_graphs} L={Lg} "
                        f"z={tuple(z_g.shape)} {_runtime_tensor_mib(z_g):.1f}MiB"
                    ),
                )
            # Graph-wise representation stitching is FP32 by contract.
            # ``index_put`` requires exact dtype equality and is not autocast.
            if s_struct_g.dtype != h_out.dtype:
                s_struct_g = s_struct_g.to(dtype=h_out.dtype)
            h_out[gidx] = s_struct_g
            pred_x[gidx] = pred_x_g

            # Shadow dynamic state: reuse exactly the same persistent trunk
            # representation (s,z), but query the atom structure network with
            # the explicit current local Xt carried by AbFlow_model.py.
            #
            # This is the key PCS-RC/Score-Flow separation:
            #   trunk s,z : recurrent proposal/static context
            #   inter_x   : actual dynamic generated state Xt
            local_pos = torch.nonzero(inter_mask[gidx], as_tuple=False).reshape(-1)
            if local_pos.numel() > 0:
                shadow_idx = g2shadow[gidx[local_pos]]
                z_local = z_g[:, local_pos][:, :, local_pos, :]
                local_update = inter_update_mask[shadow_idx].bool()
                local_atom_attr = channel_attr[gidx[local_pos]]
                local_atom_weights = channel_weights[gidx[local_pos]]

                # Module 7: clean-state self-conditioning acts ONLY on the true
                # dynamic shadow Xt, never on the PCS-RC static proposal trunk.
                s_local, pred_local, z_local_used, sc_gate = (
                    self._clean_self_condition_z(
                        s_local=s_trunk_g[local_pos],
                        z_local=z_local,
                        current_coords=inter_x[shadow_idx],
                        atom_attr=local_atom_attr,
                        atom_weights=local_atom_weights,
                        residue_update_mask=local_update,
                    )
                )
                # Same AMP boundary for the dynamic local-shadow state.
                if s_local.dtype != inter_h_out.dtype:
                    s_local = s_local.to(dtype=inter_h_out.dtype)
                inter_h_out[shadow_idx] = s_local
                pred_inter_x[shadow_idx] = pred_local

                # Cache local persistent pair state. Confidence itself is
                # evaluated after the validated surface refinement so it sees
                # the final prediction consumed by AbFlow.
                local_feats = {
                    key: value[:, local_pos] if value.ndim == 2
                    else value[:, local_pos][:, :, local_pos]
                    for key, value in feats_g.items()
                }
                self._modern_aux_cache.append({
                    "global_idx": gidx[local_pos],
                    "shadow_idx": shadow_idx,
                    "design_mask": local_update,
                    "z_base": z_local,
                    "z_used": z_local_used,
                    "sc_gate": sc_gate,
                    "s_inputs": h0[gidx][local_pos],
                    "confidence_feats": local_feats,
                    "recycling_steps": x.new_tensor(
                        float(effective_recycling_steps)
                    ),
                })

        # Retain the original, validated surface-specific Cartesian operator.
        # It acts only on the local shadow state, exactly where historical
        # AMEncoder consumed surf_verts.
        if (
            aligned_edges is not None
            and aligned_edges.numel() > 0
            and surf_verts is not None
            and surf_verts.numel() > 0
        ):
            inter_attr = channel_attr[inter_mask]
            inter_weights = channel_weights[inter_mask]
            inter_h_out, pred_inter_x = self.surface_refiner(
                inter_h_out,
                aligned_edges,
                epi_index,
                pred_inter_x,
                surf_verts,
                inter_attr,
                inter_weights,
            )

        # Finalize Module 6 pair objective / detached confidence caches *after*
        # the existing AbFlow surface refinement, so confidence is calibrated
        # on the actual prediction consumed by downstream loss/sampling.
        pending = self._modern_aux_cache
        self._modern_aux_cache = []
        if _trace_forward:
            _runtime_cuda_line(
                "amenc.before_aux",
                x.device,
                extra=f"fwd={_trace_id} pending={len(pending)}",
            )
        for _aux_i, meta in enumerate(pending):
            shadow_idx = meta["shadow_idx"]
            global_idx = meta["global_idx"]
            local_xloss_hint = channel_weights[global_idx]
            final_pred = pred_inter_x[shadow_idx]
            final_s = inter_h_out[shadow_idx]
            self._append_aux_cache(
                global_idx=global_idx,
                design_mask=meta["design_mask"],
                z_base=meta["z_base"],
                pred_coords=final_pred,
                s_inputs=meta['s_inputs'],
                s_for_confidence=final_s,
                xloss_hint=local_xloss_hint[:, 1]
                    if local_xloss_hint.shape[1] > 1
                    else local_xloss_hint[:, 0],
                sc_gate=meta["sc_gate"],
                confidence_feats=meta['confidence_feats'],
            )
            if _trace_forward:
                _runtime_cuda_line(
                    "amenc.after_aux_entry",
                    x.device,
                    extra=f"fwd={_trace_id} aux={_aux_i}/{len(pending)} cache={len(self._modern_aux_cache)}",
                )

        # Preserve historical shadow -> global semantic synchronization.
        h_out = h_out.clone()
        h_out[inter_mask] = inter_h_out

        h_out = self.dropout(h_out)
        h_out = self.linear_out(h_out)
        if _trace_forward:
            _runtime_cuda_line(
                "amenc.return",
                x.device,
                extra=f"fwd={_trace_id} aux_cache={len(self._modern_aux_cache)}",
            )
        return h_out, pred_x, pred_inter_x
