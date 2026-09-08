#!/usr/bin/env python3
import argparse
import ast
import json
import math
import pathlib
import subprocess
import torch
import torch.nn as nn

EXP = 'R24_R05_GEOMETRY_PAIR_TRIATTN_U02'
R05 = 'R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW'


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def static(root):
    model = root / 'models/AbFlow/AbFlow_model.py'
    launcher = root / 'scripts/train/run_R04_R05_R06_foldflow_noise_v104.sh'
    validator = root / 'scripts/train/validate_R05_v177.py'
    for p in (model, validator):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    subprocess.run(['bash', '-n', str(launcher)], check=True)

    text = model.read_text(encoding='utf-8')
    for mark in [
        'class R05GeometryPair',
        "triangle_mode='multiplication'",
        "self.triangle_mode not in {'multiplication', 'attention'}",
        'mf_geom_triangle_attn_rms',
        'mf_geom_triangle_operator_rms',
        'geom_triangle_multiplication_on',
        'ABFLOW_R05_GEOM_TRIANGLE_MODE',
    ]:
        if mark not in text:
            raise SystemExit('missing model marker: ' + mark)

    launch_text = launcher.read_text(encoding='utf-8')
    for mark in [
        EXP,
        'ABFLOW_R05_GEOM_TRIANGLE_MODE="attention"',
        'validate_R05_v177.py',
        'direct-R05 physics + single-module contract',
    ]:
        if mark not in launch_text:
            raise SystemExit('missing launcher marker: ' + mark)
    if 'R26_R05_GEOMETRY_PAIR_TRIATTN_U02' in launch_text:
        raise SystemExit('v177 formal launcher must not chain triangle attention from R25/R26')
    print('[V177StaticPASS] model AST + shell + direct-R05 R24 attention-only contract')


def config_check(path):
    c = load(path)
    meta = c['_experiment']
    if meta['exp_id'] != EXP:
        raise SystemExit('wrong experiment id')
    if meta['parent'] != R05:
        raise SystemExit(f'R24 must be a direct R05 child: {meta["parent"]}')
    if not bool(meta.get('single_factor_ablation')):
        raise SystemExit('R24 must be marked single-factor')

    for k, v in {
        'batch_size': 56,
        'max_epoch': 200,
        'iter_round': 3,
        'use_ema': True,
    }.items():
        if c.get(k) != v:
            raise SystemExit(f'{k} mismatch: {c.get(k)} != {v}')
    if abs(float(c['ema_decay']) - 0.999) > 1e-12:
        raise SystemExit('EMA mismatch')

    e = meta['runtime_env']
    frozen = {
        'ABFLOW_SEQUENCE_CONTEXT_MODE': 'legacy',
        'ABFLOW_SEQUENCE_PATH_CONTRACT': 'legacy_context',
        'ABFLOW_SEQUENCE_LOSS_SCOPE': 'context_mask',
        'ABFLOW_MF_REPR_CORE': 'off',
        'ABFLOW_MF_PAIR_ATOM_REFINER': 'off',
        'ABFLOW_MF_DISTOGRAM': 'off',
        'ABFLOW_MF_SMOOTH_LDDT': 'off',
        'ABFLOW_LOSS_DISTOGRAM_WEIGHT': '0.0',
        'ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT': '0.0',
        'ABFLOW_R05_ENDPOINT_RELATION': 'off',
        'ABFLOW_R05_GEOM_PAIR': 'on',
        'ABFLOW_PAIR_TIME_SCOPE': 'off',
        'ABFLOW_R05_GEOM_TRIANGLE_MODE': 'attention',
        'ABFLOW_DDP_FIND_UNUSED_PARAMETERS': 'off',
    }
    for k, v in frozen.items():
        if str(e.get(k)) != v:
            raise SystemExit(f'{k} mismatch: {e.get(k)} != {v}')

    if any(float(e.get(k, '0')) != 0.0 for k in [
        'ABFLOW_LOSS_DISTOGRAM_WEIGHT',
        'ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT',
    ]):
        raise SystemExit('R24 replacement must not introduce auxiliary loss weights')

    print('[V177ConfigPASS] direct parent=R05, triangle=attention-only, no new loss')


class MiniTriangleAttention(nn.Module):
    """Math-only zero-start check mirroring the attention operator."""
    def __init__(self, c=16, heads=4):
        super().__init__()
        self.c, self.h, self.d = c, heads, c // heads
        self.norm = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.bias = nn.Linear(c, heads, bias=False)
        self.gate = nn.Linear(c, c)
        self.out = nn.Linear(c, c, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, z):
        x = self.norm(z)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        n = z.shape[0]
        q = q.view(n, n, self.h, self.d)
        k = k.view(n, n, self.h, self.d)
        v = v.view(n, n, self.h, self.d)
        logits = torch.einsum('iqhd,ikhd->ihqk', q.float(), k.float())
        logits = logits / math.sqrt(float(self.d))
        b = self.bias(x).permute(2, 0, 1).float()
        logits = logits + b.unsqueeze(0)
        a = torch.softmax(logits, dim=-1).to(v.dtype)
        y = torch.einsum('ihqk,ikhd->iqhd', a, v).reshape(n, n, self.c)
        y = y * torch.sigmoid(self.gate(x))
        return self.out(y)


def tensor_checks():
    # R05/U02 carrier inversion stays mathematically unchanged.
    torch.manual_seed(7)
    n = 16
    X0 = torch.randn(n, 3)
    X1 = torch.randn(n, 3)
    t = torch.rand(n, 1) * 0.78 + 0.21
    mu = (1 - t) * X0 + t * X1
    r = torch.randn(n, 3) * 0.1
    Xt = mu + r
    Y = X1 + r / (2 * t)
    inv = 2 * Y - (Xt - (1 - t) * X0) / t
    if not torch.allclose(inv, X1, atol=2e-5, rtol=2e-5):
        raise SystemExit('U02 inverse failed')
    print('[TensorPASS] U02 carrier inverse unchanged')

    # Attention residual is exactly zero at initialization, but can learn.
    torch.manual_seed(17)
    attn = MiniTriangleAttention(c=16, heads=4)
    z = torch.randn(5, 5, 16)
    out0 = attn(z)
    if out0.abs().max().item() != 0.0:
        raise SystemExit('triangle attention is not exact zero-start')
    with torch.no_grad():
        attn.out.weight[0, 0] = 0.25
    out1 = attn(z)
    if not (out1.abs().sum().item() > 0.0):
        raise SystemExit('triangle attention cannot become active')
    print('[TensorPASS] direct-R05 triangle attention exact zero-start + learnable activation')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--config')
    ap.add_argument('--runtime', action='store_true')
    ap.add_argument('--tensor', action='store_true')
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    static(root)
    if a.config:
        config_check(a.config)
    if a.tensor:
        tensor_checks()
    if a.runtime:
        print('[V177RuntimePASS] static/config/tensor only; real CUDA/BF16/DDP/data execution is not claimed')


if __name__ == '__main__':
    main()
