#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone full-RAbD evaluator sharing the in-Trainer epoch-test core.

Use through ``scripts/test/test_epoch_ddp.sh``.  With torchrun, one checkpoint is
sharded across all listed GPUs by complete logical batches.  The RNG protocol is
identical to the in-Trainer Test phase.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import E2EDataset
from generate import load_model_compat, ensure_model_runtime_compat
from utils.epoch_test import (
    assigned_logical_batches,
    generate_distributed,
    run_cal_metrics_rank0,
    cleanup_structures,
    dist_info,
)


def parse_args():
    p = argparse.ArgumentParser(description="DDP EMA-checkpoint AbFlow test")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_set", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--pep_file", default=None)
    p.add_argument("--surf_file", default=None)
    p.add_argument("--batch_size", type=int, default=20,
                   help="Global logical batch size, identical for 1-GPU and multi-GPU test")
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=2023)
    p.add_argument("--metric_workers", type=int, default=8)
    p.add_argument("--show_sample_progress", action="store_true")
    p.add_argument("--delete_structures_after_metrics", action="store_true")
    return p.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ.get("RANK", "0")) if world_size > 1 else 0
    device = torch.device(
        "cpu" if not torch.cuda.is_available() else f"cuda:{local_rank}"
    )
    return rank, world_size, device


def print_legacy_metric_lines(metrics):
    ordered = [
        ("AAR H3", "AAR_H3_mean"),
        ("CAAR H3", "CAAR_H3_mean"),
        ("AAR", "AAR_mean"),
        ("CAAR", "CAAR_mean"),
        ("RMSD(CA) aligned", "RMSDCA_aligned_mean"),
        ("RMSD(CA) CDRH3", "RMSDCA_CDRH3_mean"),
        ("RMSD(CA) CDRH3 aligned", "RMSDCA_CDRH3_aligned_mean"),
        ("TMscore", "TMscore_mean"),
        ("LDDT", "LDDT_mean"),
        ("DockQ", "DockQ_mean"),
    ]
    for label, key in ordered:
        if key in metrics:
            print(f"{label}: {metrics[key]}")
    if all(k in metrics for k in (
        "DockQ_above_0.23", "DockQ_above_0.49", "DockQ_above_0.8"
    )):
        print(
            "proportion of DockQ above 0.23: "
            f"{metrics['DockQ_above_0.23']}, "
            f"0.49: {metrics['DockQ_above_0.49']}, "
            f"0.8: {metrics['DockQ_above_0.8']}"
        )


def main():
    args = parse_args()
    rank, world_size, device = init_distributed()

    try:
        model = load_model_compat(args.ckpt, map_location="cpu")
        model = ensure_model_runtime_compat(model)
        model.to(device)
        model.eval()

        test_set = E2EDataset(
            args.test_set,
            pep_file=args.pep_file,
            surf_file=args.surf_file,
            cdr=model.cdr_type,
        )

        assignments = assigned_logical_batches(
            len(test_set),
            args.batch_size,
            rank=rank,
            world_size=world_size,
        )
        local_samples = sum(len(indices) for _, indices in assignments)
        print(
            "[StandaloneEpochTestDDP] "
            f"rank={rank}/{world_size} "
            f"device={device} "
            f"logical_batch_size={args.batch_size} "
            f"assigned_batches={len(assignments)} "
            f"assigned_samples={local_samples}",
            flush=True,
        )
        if rank == 0:
            print(
                "[StandaloneEpochTestDDP] cooperative_same_checkpoint=on "
                f"world_size={world_size} "
                "protocol=logical_batch_seeded_v1",
                flush=True,
            )

        generation = generate_distributed(
            model=model,
            dataset=test_set,
            device=device,
            save_dir=args.save_dir,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            base_seed=args.base_seed,
            show_sample_progress=args.show_sample_progress,
        )
        metrics = run_cal_metrics_rank0(
            summary_file=generation.summary_file,
            save_dir=args.save_dir,
            project_root=str(PROJECT_ROOT),
            num_workers=args.metric_workers,
        )

        if rank == 0:
            print("[StandaloneEpochTest] protocol=logical_batch_seeded_v1")
            print(f"[StandaloneEpochTest] world_size={world_size}")
            print_legacy_metric_lines(metrics)

        if args.delete_structures_after_metrics:
            cleanup_structures(args.save_dir)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
