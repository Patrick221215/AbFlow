#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import E2EDataset
from data.pdb_utils import VOCAB, Residue, Peptide, Protein, AgAbComplex
from utils.logger import print_log
from utils.random_seed import setup_seed


def load_model(ckpt_path, map_location="cpu"):
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=map_location)


def to_cplx(ori_cplx, ab_x, ab_s):
    heavy_chain, light_chain = [], []
    chain = None
    for residue, residue_x in zip(ab_s, ab_x):
        residue = VOCAB.idx_to_symbol(residue)
        if residue == VOCAB.BOA:
            continue
        if residue == VOCAB.BOH:
            chain = heavy_chain
            continue
        if residue == VOCAB.BOL:
            chain = light_chain
            continue
        if chain is None:
            continue
        coord = {}
        atoms = VOCAB.backbone_atoms + VOCAB.get_sidechain_info(residue)
        for atom, x in zip(atoms, residue_x):
            coord[atom] = x
        chain.append(Residue(residue, coord, _id=(len(chain), " ")))

    heavy_chain = Peptide(ori_cplx.heavy_chain, heavy_chain)
    light_chain = Peptide(ori_cplx.light_chain, light_chain)
    for res, ori_res in zip(heavy_chain, ori_cplx.get_heavy_chain()):
        res.id = ori_res.id
    for res, ori_res in zip(light_chain, ori_cplx.get_light_chain()):
        res.id = ori_res.id

    antibody = Protein(
        ori_cplx.pdb_id,
        {
            ori_cplx.heavy_chain: heavy_chain,
            ori_cplx.light_chain: light_chain,
        },
    )
    cplx = AgAbComplex(
        ori_cplx.antigen,
        antibody,
        ori_cplx.heavy_chain,
        ori_cplx.light_chain,
        skip_epitope_cal=True,
        skip_validity_check=True,
    )
    cplx.cdr_pos = ori_cplx.cdr_pos
    return cplx


def _init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    return rank, local_rank, world_size, device


def _raw_complex(dataset, logical_idx):
    if hasattr(dataset, "idx_mapping"):
        raw_idx = int(dataset.idx_mapping[logical_idx])
    else:
        raw_idx = int(logical_idx)
    return dataset.data[raw_idx]


def _split_flat_prediction(X, S, batch):
    if "bid" in batch:
        batch_id = batch["bid"]
    else:
        lengths = batch["lengths"]
        batch_id = torch.zeros_like(batch["S"])
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)

    X = X.tolist()
    S = S.tolist()
    X_list, S_list = [], []
    current_bid = -1
    for i, bid in enumerate(batch_id.tolist()):
        if bid != current_bid:
            current_bid = bid
            X_list.append([])
            S_list.append([])
        X_list[-1].append(X[i])
        S_list[-1].append(S[i])
    return X_list, S_list


def generate(config_path, ckpt_override=None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    rank, local_rank, world_size, device = _init_distributed()
    gen = cfg["generation"]
    data = cfg["data"]
    task = data["task"]

    # Reproducible for the fixed formal world-size/GPU protocol.
    setup_seed(int(gen["seed"]) + rank)

    ckpt = ckpt_override or gen["checkpoint"]
    model = load_model(ckpt, map_location="cpu")
    model.to(device)
    model.eval()

    test_cfg = data["test"]
    test_set = E2EDataset(
        test_cfg["set"],
        pep_file=test_cfg["pep"],
        surf_file=test_cfg["surface"],
        cdr=task["cdr"],
        paratope=task["paratope"],
        num_verts=task["num_verts"],
    )

    logical_indices = list(range(rank, len(test_set), world_size))
    local_set = Subset(test_set, logical_indices)
    global_batch_size = int(gen["batch_size"])
    local_batch_size = max(1, global_batch_size // max(1, world_size))

    test_loader = DataLoader(
        local_set,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=int(gen["num_workers"]),
        collate_fn=test_set.collate_fn,
    )

    save_dir = gen["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    local_summary = []
    cursor = 0
    iterator = tqdm(
        test_loader,
        desc=f"Generating rank {rank}",
        dynamic_ncols=True,
        disable=(rank != 0),
    )

    for batch in iterator:
        n_complex = int(batch["lengths"].shape[0])
        batch_indices = logical_indices[cursor:cursor + n_complex]
        cursor += n_complex

        with torch.no_grad():
            for key, value in batch.items():
                if hasattr(value, "to"):
                    batch[key] = value.to(device)

            batch.pop("xloss_mask", None)

            X, S, _ = model.sample(
                **batch,
                n_steps=int(gen["n_steps"]),
                show_progress=bool(gen["show_sample_progress"]) and rank == 0,
            )
            X_list, S_list = _split_flat_prediction(X, S, batch)

        for logical_idx, x, s in zip(batch_indices, X_list, S_list):
            ori_cplx = _raw_complex(test_set, logical_idx)
            cplx = to_cplx(ori_cplx, x, s)
            pdb_id = cplx.get_id().split("(")[0]

            mod_pdb = os.path.join(save_dir, pdb_id + ".pdb")
            ref_pdb = os.path.join(save_dir, pdb_id + "_original.pdb")
            cplx.to_pdb(mod_pdb)
            ori_cplx.to_pdb(ref_pdb)

            local_summary.append({
                "_index": int(logical_idx),
                "mod_pdb": mod_pdb,
                "ref_pdb": ref_pdb,
                "H": cplx.heavy_chain,
                "L": cplx.light_chain,
                "A": cplx.antigen.get_chain_names(),
                "cdr_type": task["cdr"],
                "pdb": pdb_id,
            })

    if world_size > 1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_summary)
        summary = [item for part in gathered for item in part] if rank == 0 else None
    else:
        summary = local_summary

    if rank == 0:
        summary.sort(key=lambda x: x["_index"])
        summary_file = os.path.join(save_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            for item in summary:
                item = dict(item)
                item.pop("_index", None)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print_log(f"Summary written to {summary_file}")

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def parse():
    parser = argparse.ArgumentParser(description="Generate antibodies from modular JSON")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Optional checkpoint override; all scientific generation settings remain JSON-controlled.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    generate(args.config, args.ckpt)
