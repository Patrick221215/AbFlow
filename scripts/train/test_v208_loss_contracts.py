#!/usr/bin/env python3
"""CPU-only numerical checks for the V208 donor-loss contracts.

This intentionally extracts the shipped smooth-lDDT function from the model
AST, so the test exercises the release implementation without importing the
full CUDA/torch_scatter AbFlow stack.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch


def load_function(model_path: Path, name: str):
    tree = ast.parse(model_path.read_text(encoding="utf-8"))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {name}, found {len(matches)}")
    module = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(model_path), "exec"), namespace)
    return namespace[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    fn = load_function(args.model.resolve(), "design_region_smooth_lddt_loss")

    # Three residues, two resolved atoms each: design, scaffold, antigen.
    true = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
            [[6.0, 0.0, 0.0], [6.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    pred_perfect = true.clone().requires_grad_(True)
    valid = torch.ones((3, 2), dtype=torch.bool)
    design = torch.tensor([True, False, False])
    antigen = torch.tensor([False, False, True])
    batch = torch.zeros(3, dtype=torch.long)

    perfect, diag = fn(
        pred_perfect, true, valid, design, batch,
        is_antigen_mask=antigen, cutoff=15.0,
    )
    assert torch.isfinite(perfect)
    assert int(diag["intra_pairs"].item()) == 2
    assert int(diag["scaffold_pairs"].item()) == 8
    assert int(diag["antigen_pairs"].item()) == 8

    pred_shifted = true.clone()
    pred_shifted[0, 0, 0] += 1.5
    pred_shifted.requires_grad_(True)
    shifted, _ = fn(
        pred_shifted, true, valid, design, batch,
        is_antigen_mask=antigen, cutoff=15.0,
    )
    assert shifted > perfect, (perfect.item(), shifted.item())
    shifted.backward()
    assert pred_shifted.grad is not None
    assert torch.isfinite(pred_shifted.grad).all()
    assert float(pred_shifted.grad.abs().sum()) > 0.0

    # Authoritative resolved mask must change cardinality: removing one design
    # atom leaves 1 design atom x 4 context atoms in both directions = 8 pairs.
    partly_resolved = valid.clone()
    partly_resolved[0, 1] = False
    _, diag_masked = fn(
        pred_shifted.detach(), true, partly_resolved, design, batch,
        is_antigen_mask=antigen, cutoff=15.0,
    )
    masked_pairs = sum(
        int(diag_masked[f"{name}_pairs"].item())
        for name in ("intra", "scaffold", "antigen")
    )
    assert masked_pairs == 8, masked_pairs

    print(
        "[V208LossContractPASS] smooth_lddt=donor_single_denominator "
        "resolved_mask=authoritative design_pair_localization=PASS gradient=PASS"
    )


if __name__ == "__main__":
    main()
