#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data.dataset import E2EDataset
from data.pdb_utils import VOCAB, Residue, Peptide, Protein, AgAbComplex
from utils.logger import print_log
from utils.random_seed import setup_seed

import sys
import models.AbFlow
import models.AbFlow.AbFlow_model
import models.AbFlow.AbFlowOpt_model

sys.modules['models.isMEAN'] = models.AbFlow
sys.modules['models.dyMEAN'] = models.AbFlow
sys.modules['models.isMEAN.isMEAN_model'] = models.AbFlow.AbFlow_model
sys.modules['models.isMEAN.isMEANOpt_model'] = models.AbFlow.AbFlowOpt_model
sys.modules['models.dyMEAN.dyMEAN_model'] = models.AbFlow.AbFlow_model

models.AbFlow.AbFlow_model.isMEANModel = models.AbFlow.AbFlow_model.AbFlowModel
models.AbFlow.AbFlow_model.dyMEANModel = models.AbFlow.AbFlow_model.AbFlowModel
models.AbFlow.AbFlowOpt_model.isMEANOptModel = models.AbFlow.AbFlowOpt_model.AbFlowOptModel

def load_model_compat(ckpt_path, map_location='cpu'):
    """
    Compatible loader for AbFlow checkpoints saved as full model objects.

    Some newer PyTorch versions support weights_only=False, while the
    current dymean0 environment does not. We first try the newer API and
    fall back to the old API when weights_only is unsupported.
    """
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError as e:
        if 'weights_only' in str(e):
            print_log(
                '[Compatibility] This PyTorch does not support '
                'torch.load(..., weights_only=False). Falling back to torch.load(...).'
            )
            return torch.load(ckpt_path, map_location=map_location)
        raise


def ensure_model_runtime_compat(model):
    """
    Runtime compatibility for old checkpoints loaded under newer AbFlow code.

    Old AbFlow checkpoints were saved as full Python model objects. Loading them
    under a modified AbFlowModel class does not re-run __init__, so newly added
    attributes may be absent. We add safe defaults here.

    For old non-ScoreFM checkpoints, use_scorefm defaults to False so inference
    can keep the original AbFlow sampling behavior, provided AbFlowModel.sample
    checks this flag.
    """
    if not hasattr(model, 'pep_seq'):
        model.pep_seq = True
    if not hasattr(model, 'pep_struct'):
        model.pep_struct = True

    if hasattr(model, '_ensure_scorefm_compat'):
        model._ensure_scorefm_compat()

    defaults = {
        'use_scorefm': False,
        'scorefm_min_sigma': 1e-2,
        'scorefm_eps': 1e-8,
        'scorefm_t_threshold': 0.50,
        'scorefm_loss_weight': 5e-2,
        'scorefm_velocity_weight': 1.0,
        'scorefm_dsm_weight': 1.0,
        'scorefm_x1_weight': 0.25,
        'scorefm_local_dist_weight': 0.05,
        'scorefm_interface_contact_weight': 0.05,
        'scorefm_inter_clash_weight': 0.01,
        'scorefm_intra_clash_weight': 0.005,
        'scorefm_contact_cutoff': 8.0,
        'scorefm_contact_temperature': 1.0,
        'scorefm_inter_clash_cutoff': 2.0,
        'scorefm_intra_clash_cutoff': 1.5,
        'last_scorefm_losses': {},
        'sf2m_t_eps': 0.01,
        'sf2m_score_weight': 1.0,
        'pair_score_feedback': False,
    }
    for name, value in defaults.items():
        if not hasattr(model, name):
            setattr(model, name, value)

    return model

def to_cplx(ori_cplx, ab_x, ab_s) -> AgAbComplex:
    heavy_chain, light_chain = [], []
    chain = None
    for residue, residue_x in zip(ab_s, ab_x):
        residue = VOCAB.idx_to_symbol(residue)
        if residue == VOCAB.BOA:
            continue
        elif residue == VOCAB.BOH:
            chain = heavy_chain
            continue
        elif residue == VOCAB.BOL:
            chain = light_chain
            continue
        if chain is None:  # still in antigen region
            continue
        coord, atoms = {}, VOCAB.backbone_atoms + VOCAB.get_sidechain_info(residue)

        for atom, x in zip(atoms, residue_x):
            coord[atom] = x
        chain.append(Residue(
            residue, coord, _id=(len(chain), ' ')
        ))
    heavy_chain = Peptide(ori_cplx.heavy_chain, heavy_chain)
    light_chain = Peptide(ori_cplx.light_chain, light_chain)
    for res, ori_res in zip(heavy_chain, ori_cplx.get_heavy_chain()):
        res.id = ori_res.id
    for res, ori_res in zip(light_chain, ori_cplx.get_light_chain()):
        res.id = ori_res.id

    peptides = {
        ori_cplx.heavy_chain: heavy_chain,
        ori_cplx.light_chain: light_chain
    }
    antibody = Protein(ori_cplx.pdb_id, peptides)
    cplx = AgAbComplex(
        ori_cplx.antigen, antibody, ori_cplx.heavy_chain,
        ori_cplx.light_chain, skip_epitope_cal=True,
        skip_validity_check=True
    )
    cplx.cdr_pos = ori_cplx.cdr_pos
    return cplx


def generate(args):

    # load model
    # load model
    model = load_model_compat(args.ckpt, map_location='cpu')
    model = ensure_model_runtime_compat(model)

    if getattr(args, 'sampler_mode', 'checkpoint') != 'checkpoint':
        model.scorefm_sampler_mode = args.sampler_mode
        print_log(
            f'[Sampler override] scorefm_sampler_mode={args.sampler_mode}'
        )

    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    
    model.to(device)
    model.eval()

    # model_type
    print_log(f'Model type: {type(model)}')

    # cdr type
    cdr_type = model.cdr_type
    print_log(f'CDR type: {cdr_type}')
    print_log(f'Paratope definition: {model.paratope}')

    # load test set
    test_set = E2EDataset(args.test_set, pep_file=args.pep_file, surf_file=args.surf_file, cdr=cdr_type)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             collate_fn=E2EDataset.collate_fn)


    # create save dir
    if args.save_dir is None:
        save_dir = '.'.join(args.ckpt.split('.')[:-1]) + '_results'
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    idx = 0
    summary_items = []
    batch_pbar = tqdm(
        test_loader,
        total=len(test_loader),
        desc='Generating batches',
        dynamic_ncols=True
    )

    for batch_idx, batch in enumerate(batch_pbar):
        batch_pbar.set_postfix(batch=f'{batch_idx + 1}/{len(test_loader)}')
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            # generate
            if 'xloss_mask' in batch:
                del batch['xloss_mask']

            X, S, pmets = model.sample(
                **batch,
                n_steps=args.n_steps,
                show_progress=args.show_sample_progress
            )

            X, S, pmets = X.tolist(), S.tolist(), pmets.tolist()
            X_list, S_list = [], []
            cur_bid = -1
            if 'bid' in batch:
                batch_id = batch['bid']
            else:
                lengths = batch['lengths']
                batch_id = torch.zeros_like(batch['S'])  # [N]
                batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
                batch_id.cumsum_(dim=0)  # [N], item idx in the batch
            for i, bid in enumerate(batch_id):
                if bid != cur_bid:
                    cur_bid = bid
                    X_list.append([])
                    S_list.append([])
                X_list[-1].append(X[i])
                S_list[-1].append(S[i])
                
        for i, (x, s) in enumerate(zip(X_list, S_list)):
            ori_cplx = test_set.data[idx]
            cplx = to_cplx(ori_cplx, x, s)
            pdb_id = cplx.get_id().split('(')[0]
            mod_pdb = os.path.join(save_dir, pdb_id + '.pdb')
            cplx.to_pdb(mod_pdb)
            ref_pdb = os.path.join(save_dir, pdb_id + '_original.pdb')
            ori_cplx.to_pdb(ref_pdb)
            summary_items.append({
                'mod_pdb': mod_pdb,
                'ref_pdb': ref_pdb,
                'H': cplx.heavy_chain,
                'L': cplx.light_chain,
                'A': cplx.antigen.get_chain_names(),
                'cdr_type': cdr_type,
                'pdb': pdb_id,
                'pmetric': None
            })
            idx += 1

    # write done the summary
    summary_file = os.path.join(save_dir, 'summary.json')
    with open(summary_file, 'w') as fout:
        fout.writelines(list(map(lambda item: json.dumps(item) + '\n', summary_items)))
    print_log(f'Summary of generated complexes written to {summary_file}')


def parse():
    parser = argparse.ArgumentParser(description='Generate antibodies given epitopes')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--test_set', type=str, required=True, help='Path to test set')
    parser.add_argument('--surf_file', type=str, default='all_data/RAbD/test_surf.pkl', help='pep file')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated antibodies')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--n_steps', type=int, default=10,
                        help='Number of flow sampling steps')
    parser.add_argument('--show_sample_progress', action='store_true',
                        help='Show inner progress bar for flow sampling steps')
    parser.add_argument(
        '--sampler_mode',
        type=str,
        default='checkpoint',
        choices=[
            'checkpoint',
            'sf2m_ode',
            'sf2m_sde',
            'bridge',
            'r3_scoreflow',
            'residual',
        ],
        help=(
            'checkpoint: use sampler stored in checkpoint; '
            'sf2m_ode/sf2m_sde are same-checkpoint v69 diagnostics.'
        ),
    )

    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use, -1 for cpu')
    
    parser.add_argument('--pep_file', type=str, nargs='?', const='all_data/RAbD/test.pkl', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(2023)
    generate(parse())
