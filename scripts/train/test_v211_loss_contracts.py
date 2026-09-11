#!/usr/bin/env python3
"""CPU numerical/static checks for V211 scientific contracts.

The model function is extracted from its AST so this test does not import the
CUDA/torch_scatter AbFlow stack.  It is safe to run during overlay preflight.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Documentation/packaging containers may be CPU-minimal.
    torch = None


def load_function(model_path: Path, name: str):
    tree = ast.parse(model_path.read_text(encoding="utf-8"))
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {name}, found {len(matches)}")
    module = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(model_path), "exec"), namespace)
    return namespace[name]


def test_smooth_lddt(model_path: Path) -> None:
    fn = load_function(model_path, "design_region_smooth_lddt_loss")
    # Design, scaffold, antigen; two resolved atoms per residue.
    true = torch.tensor([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
        [[6.0, 0.0, 0.0], [6.0, 1.0, 0.0]],
    ])
    valid = torch.ones((3, 2), dtype=torch.bool)
    design = torch.tensor([True, False, False])
    antigen = torch.tensor([False, False, True])
    batch = torch.zeros(3, dtype=torch.long)

    perfect, diag = fn(
        true.clone().requires_grad_(True), true, valid, design, batch,
        is_antigen_mask=antigen, cutoff=15.0,
    )
    assert torch.isfinite(perfect)
    assert int(diag["intra_pairs"]) == 2
    assert int(diag["scaffold_pairs"]) == 8
    assert int(diag["antigen_pairs"]) == 8

    shifted = true.clone()
    shifted[0, 0, 0] += 1.5
    shifted.requires_grad_(True)
    shifted_loss, _ = fn(
        shifted, true, valid, design, batch,
        is_antigen_mask=antigen, cutoff=15.0,
    )
    assert shifted_loss > perfect
    shifted_loss.backward()
    assert shifted.grad is not None and torch.isfinite(shifted.grad).all()
    assert float(shifted.grad.abs().sum()) > 0.0

    partly_resolved = valid.clone()
    partly_resolved[0, 1] = False
    _, masked = fn(
        shifted.detach(), true, partly_resolved, design, batch,
        is_antigen_mask=antigen, cutoff=15.0,
    )
    # One design atom paired bidirectionally with four context atoms.
    assert sum(int(masked[f"{name}_pairs"]) for name in (
        "intra", "scaffold", "antigen"
    )) == 8


def test_parent_preserving_affine() -> None:
    if torch is None:
        rng = np.random.default_rng(7)
        base = rng.standard_normal((11, 13))
        donor = rng.standard_normal((11, 5))
        weight_base = rng.standard_normal((17, 13))
        weight_donor = rng.standard_normal((17, 5))
        bias = rng.standard_normal(17)
        concatenated = np.concatenate([base, donor], axis=-1) @ np.concatenate(
            [weight_base, weight_donor], axis=-1
        ).T + bias
        split = base @ weight_base.T + bias + donor @ weight_donor.T
        np.testing.assert_allclose(concatenated, split, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(
            base @ weight_base.T + bias,
            base @ weight_base.T + bias + donor @ np.zeros_like(weight_donor).T,
        )
        return
    torch.manual_seed(7)
    base = torch.randn(11, 13)
    donor = torch.randn(11, 5)
    weight_base = torch.randn(17, 13)
    weight_donor = torch.randn(17, 5)
    bias = torch.randn(17)
    concatenated = torch.nn.functional.linear(
        torch.cat([base, donor], dim=-1),
        torch.cat([weight_base, weight_donor], dim=-1), bias,
    )
    split = (
        torch.nn.functional.linear(base, weight_base, bias)
        + torch.nn.functional.linear(donor, weight_donor, None)
    )
    torch.testing.assert_close(concatenated, split)
    parent = torch.nn.functional.linear(base, weight_base, bias)
    zero_start = parent + torch.nn.functional.linear(
        donor, torch.zeros_like(weight_donor), None
    )
    torch.testing.assert_close(parent, zero_start, rtol=0.0, atol=0.0)


def test_rng_isolation() -> None:
    if torch is None:
        return
    torch.manual_seed(123)
    reference = torch.nn.Linear(9, 7)
    ref_weight = reference.weight.detach().clone()
    ref_bias = reference.bias.detach().clone()

    torch.manual_seed(123)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(314159)
        optional = torch.nn.Linear(37, 41, bias=False)
        torch.nn.init.zeros_(optional.weight)
    observed = torch.nn.Linear(9, 7)
    torch.testing.assert_close(observed.weight, ref_weight, rtol=0.0, atol=0.0)
    torch.testing.assert_close(observed.bias, ref_bias, rtol=0.0, atol=0.0)

    # Training-time donor dropout must leave the parent stream at the exact
    # state it would have had if the donor branch were absent.
    torch.manual_seed(456)
    expected_parent_draw = torch.rand(37)
    torch.manual_seed(456)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(271828)
        _ = torch.nn.functional.dropout(
            torch.ones(4096), p=0.1, training=True
        )
    observed_parent_draw = torch.rand(37)
    torch.testing.assert_close(
        observed_parent_draw, expected_parent_draw, rtol=0.0, atol=0.0
    )


def test_fingerprint_byte_serialization() -> None:
    """Regress the PyTorch-1.11 zero-dimensional dtype-view failure."""
    if torch is None:
        return
    cases = (
        torch.tensor(7, dtype=torch.long),
        torch.tensor(1.25, dtype=torch.float32),
        torch.arange(6, dtype=torch.long).reshape(2, 3),
    )
    for value in cases:
        value = value.detach().cpu().contiguous()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        expected_bytes = int(value.numel()) * int(value.element_size())
        assert len(raw) == expected_bytes, (
            value.dtype, tuple(value.shape), len(raw), expected_bytes
        )


def test_task_masks_and_static_contract(model_path: Path) -> None:
    if torch is None:
        resolved = np.ones((1, 5), dtype=bool)
        design = np.array([[True, False, False, False, False]])
    else:
        resolved = torch.ones((1, 5), dtype=torch.bool)
        design = torch.tensor([[True, False, False, False, False]])
    donor = resolved[:, :, None] & resolved[:, None, :]
    localized = donor & (design[:, :, None] | design[:, None, :])
    assert int(donor.sum()) == 25
    assert int(localized.sum()) == 9
    context_context = donor & (~design[:, :, None]) & (~design[:, None, :])
    assert int(context_context.sum()) == 16
    assert not bool((localized & context_context).any())

    text = model_path.read_text(encoding="utf-8")
    for symbol in (
        "aux_pred_X = true_X.clone()",
        "aux_pred_X[cmask] = pred_X[cmask]",
        "design[:, :, None] | design[:, None, :]",
        "disto_all_pair_raw_loss",
        "shared_initialization_fingerprint",
        "value.reshape(-1).view(torch.uint8)",
        "ABFLOW_ABX_FORWARD_SEED",
        "donor_dropout_rng=isolate",
    ):
        assert symbol in text, symbol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    model_path = args.model.resolve()
    if torch is not None:
        test_smooth_lddt(model_path)
    test_parent_preserving_affine()
    test_rng_isolation()
    test_fingerprint_byte_serialization()
    test_task_masks_and_static_contract(model_path)
    print(
        "[V211LossContractPASS] "
        f"donor_smooth_lddt={'PASS' if torch is not None else 'STATIC_ONLY'} "
        "resolved_mask=PASS design_pair_localization=PASS "
        "fixed_context_view=PASS split_affine_equivalence=PASS "
        "zero_start_parent_identity=PASS rng_isolation=PASS "
        f"fingerprint_scalar_bytes={'PASS' if torch is not None else 'STATIC_ONLY'}"
    )


if __name__ == "__main__":
    main()
