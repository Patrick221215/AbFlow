#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import pickle
import argparse
from typing import List

import numpy as np
import torch

from utils.logger import print_log

########## import your packages below ##########
from tqdm import tqdm

from .pdb_utils import AgAbComplex, VOCAB, Protein
from .surface import get_epi_surf, _cal_verts, pad_sample
from .framework_templates import ConserveTemplateGenerator


# ============================================================================
# Module 1: explicit AbFlow data contract
# ----------------------------------------------------------------------------
# IMPORTANT:
#   This is a semantic/engineering refactor only. It deliberately preserves
#   the existing AbFlow scientific state:
#     - fixed residue-level full-atom Cartesian tensor [N, 14, 3];
#     - existing atom slot/order and xloss_mask;
#     - existing cmask/smask/paratope_mask;
#     - existing PCS-RC proposal X_pep/S_pep;
#     - existing surface/template/common-center behavior;
#     - no MSA/profile/deletion features;
#     - no new model parameters and no new RNG calls.
#
# The useful MFDesign/Boltz idea absorbed here is data organization:
# one explicit schema, one owner for each field, and fail-fast validation at
# the dataset/collate boundary. Model-facing keys and tensor values are kept
# unchanged so existing trainers/models/configs continue to work.
# ============================================================================

ABFLOW_BATCH_KEYS = (
    'X',
    'S',
    'smask',
    'cmask',
    'paratope_mask',
    'residue_pos',
    'template',
    'xloss_mask',
    'X_pep',
    'S_pep',
    'surface',
)

ABFLOW_BATCH_DTYPES = (
    torch.float,
    torch.long,
    torch.bool,
    torch.bool,
    torch.bool,
    torch.long,
    torch.float,
    torch.bool,
    torch.float,
    torch.long,
    torch.float,
)

ABFLOW_BATCH_DTYPE_MAP = dict(zip(ABFLOW_BATCH_KEYS, ABFLOW_BATCH_DTYPES))


# ============================================================================
# v111: MFDesign/Boltz token metadata contract
# ----------------------------------------------------------------------------
# These are genuine model inputs, not diagnostic annotations.  The names mirror
# the MFDesign/Boltz trunk contract so RelativePositionEncoder, sequence
# conditioning, recycling and confidence all consume one authoritative schema.
# AbFlow remains one-token-per-residue, therefore token_index is 0 for every
# real protein residue; global BOA/BOH/BOL bookkeeping nodes are masked out.
# ============================================================================
MFDESIGN_FLAT_KEYS = (
    'token_index', 'residue_index', 'asym_id', 'entity_id', 'sym_id',
    'type', 'region',
)
MFDESIGN_FLAT_DTYPES = (
    torch.long, torch.long, torch.long, torch.long, torch.long,
    torch.long, torch.long,
)
MFDESIGN_FLAT_DTYPE_MAP = dict(zip(MFDESIGN_FLAT_KEYS, MFDESIGN_FLAT_DTYPES))

MFDESIGN_PADDED_KEYS = ('token_pad_mask', 'token_bonds')

# Exact MFDesign antibody semantic IDs.
MF_TYPE_PAD = 0
MF_TYPE_HEAVY = 1
MF_TYPE_LIGHT = 2
MF_TYPE_ANTIGEN = 3

MF_REGION_PAD = 0
MF_REGION_FR1 = 1
MF_REGION_CDR1 = 2
MF_REGION_FR2 = 3
MF_REGION_CDR2 = 4
MF_REGION_FR3 = 5
MF_REGION_CDR3 = 6
MF_REGION_FR4 = 7
MF_REGION_AG_NON_EPITOPE = 8
MF_REGION_AG_EPITOPE = 9

# Scientific role of each EXISTING field. These are metadata only and are not
# added to the model input dictionary.
ABFLOW_FIELD_ROLES = {
    'X': 'native/full-complex Cartesian coordinates in the existing AbFlow fixed atom-channel layout',
    'S': 'native/full-complex residue tokens',
    'smask': 'sequence-generation residue mask',
    'cmask': 'coordinate-generation residue mask',
    'paratope_mask': 'interface/paratope residue mask used by the current AbFlow state',
    'residue_pos': 'existing residue-position encoding',
    'template': 'existing conserved-framework coordinate template',
    'xloss_mask': 'existing valid/resolved atom-channel mask',
    'X_pep': 'existing PCS/PCS-RC proposal coordinates for the paratope',
    'S_pep': 'existing PCS/PCS-RC proposal residue tokens for the paratope',
    'surface': 'existing antigen/epitope surface vertices',
    'lengths': 'number of residue/global-node tokens per complex after collate',
    'token_index': 'MFDesign token index within one residue; 0 for AbFlow protein one-token-per-residue',
    'residue_index': 'MFDesign sequential residue index within the physical chain',
    'asym_id': 'MFDesign chain/asymmetric-unit id; antigen chains, heavy and light are distinct',
    'entity_id': 'MFDesign entity id; equals asym_id for the current non-symmetry-expanded RAbD representation',
    'sym_id': 'MFDesign symmetry-copy id; 0 for the current RAbD representation',
    'type': 'MFDesign antibody type: 1 heavy, 2 light, 3 antigen, 0 bookkeeping/pad',
    'region': 'MFDesign region: FR/CDR 1..7, antigen non-epitope/epitope 8/9',
    'token_pad_mask': 'MFDesign token-valid mask [B,Lmax]; BOA/BOH/BOL are excluded from the trunk',
    'token_bonds': 'MFDesign token covalent-adjacency pair feature [B,Lmax,Lmax,1]',
}


def _contract_error(sample_name: str, message: str) -> ValueError:
    return ValueError(f'AbFlow data contract violation [{sample_name}]: {message}')


def _shape_string(x) -> str:
    try:
        return str(tuple(x.shape))
    except Exception:
        return f'<no shape: {type(x)}>'


def _validate_numpy_item_contract(data: dict, sample_name: str = '<unknown>') -> None:
    """Validate one existing E2EDataset item without changing it.

    This function is validation-only:
      * no tensor/array is mutated;
      * no random number is sampled;
      * no new feature is created;
      * atom order/state semantics are not redefined.
    """
    missing = [k for k in ABFLOW_BATCH_KEYS if k not in data]
    if missing:
        raise _contract_error(sample_name, f'missing required fields: {missing}')

    X = np.asarray(data['X'])
    S = np.asarray(data['S'])
    smask = np.asarray(data['smask'])
    cmask = np.asarray(data['cmask'])
    paratope_mask = np.asarray(data['paratope_mask'])
    residue_pos = np.asarray(data['residue_pos'])
    template = np.asarray(data['template'])
    xloss_mask = np.asarray(data['xloss_mask'])
    X_pep = np.asarray(data['X_pep'])
    S_pep = np.asarray(data['S_pep'])
    surface = data['surface']

    if X.ndim != 3 or X.shape[-1] != 3:
        raise _contract_error(
            sample_name,
            f'X must be [N,n_channel,3], got {_shape_string(X)}',
        )

    if X.shape[1] != VOCAB.MAX_ATOM_NUMBER:
        raise _contract_error(
            sample_name,
            f'X channel dimension must equal VOCAB.MAX_ATOM_NUMBER='
            f'{VOCAB.MAX_ATOM_NUMBER}, got {X.shape[1]}',
        )

    n_token = X.shape[0]
    one_dim_fields = {
        'S': S,
        'smask': smask,
        'cmask': cmask,
        'paratope_mask': paratope_mask,
        'residue_pos': residue_pos,
    }
    for key, value in one_dim_fields.items():
        if value.ndim != 1 or value.shape[0] != n_token:
            raise _contract_error(
                sample_name,
                f'{key} must be [N] with N={n_token}, got {_shape_string(value)}',
            )

    n_coord_generate = int(cmask.astype(np.int64).sum())
    expected_template_shape = (n_coord_generate, X.shape[1], 3)
    if template.shape != expected_template_shape:
        raise _contract_error(
            sample_name,
            f'template must match X[cmask] with expected shape '
            f'{expected_template_shape}, got {_shape_string(template)}',
        )

    if xloss_mask.shape != X.shape[:2]:
        raise _contract_error(
            sample_name,
            f'xloss_mask must be [N,n_channel]={X.shape[:2]}, '
            f'got {_shape_string(xloss_mask)}',
        )

    n_paratope = int(paratope_mask.astype(np.int64).sum())

    if X_pep.ndim != 3 or X_pep.shape[-1] != 3:
        raise _contract_error(
            sample_name,
            f'X_pep must be [N_paratope,n_channel,3], got {_shape_string(X_pep)}',
        )
    if X_pep.shape[0] != n_paratope:
        raise _contract_error(
            sample_name,
            f'X_pep first dimension must equal paratope residue count '
            f'{n_paratope}, got {X_pep.shape[0]}',
        )
    if X_pep.shape[1] != X.shape[1]:
        raise _contract_error(
            sample_name,
            f'X_pep channel count must match X ({X.shape[1]}), got {X_pep.shape[1]}',
        )

    if S_pep.ndim != 1 or S_pep.shape[0] != n_paratope:
        raise _contract_error(
            sample_name,
            f'S_pep must be [N_paratope] with N_paratope={n_paratope}, '
            f'got {_shape_string(S_pep)}',
        )

    # Historical AbFlow surface semantics are intentionally preserved:
    # at __getitem__ time ``surface`` is a per-epitope-residue mapping
    #
    #     residue_id -> array[num_surface_vertices_for_this_residue, 3]
    #
    # and only collate_fn/pad_sample converts that ragged mapping into a dense
    # Cartesian tensor.  Therefore treating the dict itself with
    # ``np.asarray(surface)`` is incorrect: NumPy produces a 0-D object array.
    #
    # Empty per-residue arrays are valid and explicitly produced by
    # data.surface._cal_verts when a residue has no assigned surface vertex.
    # The validator is read-only and MUST NOT call pad_sample here because
    # pad_sample may randomly subsample vertices and would change dataset RNG.
    if not isinstance(surface, dict):
        raise _contract_error(
            sample_name,
            f'surface must be the historical per-residue dict before collate, '
            f'got {type(surface).__name__}',
        )
    if len(surface) == 0:
        raise _contract_error(
            sample_name,
            'surface dict must contain at least one epitope-residue entry',
        )
    for residue_key, residue_surface in surface.items():
        verts = np.asarray(residue_surface)
        if verts.size == 0:
            # Historical _cal_verts uses np.array([]) for the empty case.
            continue
        if verts.ndim != 2 or verts.shape[-1] != 3:
            raise _contract_error(
                sample_name,
                f'surface[{residue_key!r}] must be [V,3] or empty, '
                f'got {_shape_string(verts)}',
            )
        if not np.issubdtype(verts.dtype, np.number):
            raise _contract_error(
                sample_name,
                f'surface[{residue_key!r}] must contain numeric Cartesian '
                f'coordinates, got dtype={verts.dtype}',
            )
        if not np.isfinite(verts).all():
            raise _contract_error(
                sample_name,
                f'surface[{residue_key!r}] contains non-finite coordinates',
            )


    # MFDesign token metadata validation.
    for key in MFDESIGN_FLAT_KEYS:
        if key not in data:
            raise _contract_error(sample_name, f'missing MFDesign field {key}')
        value = np.asarray(data[key])
        if value.ndim != 1 or value.shape[0] != n_token:
            raise _contract_error(
                sample_name,
                f'{key} must be [N={n_token}], got {_shape_string(value)}',
            )
    token_pad_mask = np.asarray(data.get('token_pad_mask'))
    if token_pad_mask.ndim != 1 or token_pad_mask.shape[0] != n_token:
        raise _contract_error(sample_name, 'token_pad_mask must be [N]')
    token_bonds = np.asarray(data.get('token_bonds'))
    if token_bonds.shape != (n_token, n_token, 1):
        raise _contract_error(
            sample_name,
            f'token_bonds must be [N,N,1], got {_shape_string(token_bonds)}',
        )
    # MFDesign semantic IDs: real tokens are exactly Heavy/Light/Ag.
    real = token_pad_mask.astype(bool)
    if np.any(~np.isin(np.asarray(data['type'])[real], [1, 2, 3])):
        raise _contract_error(sample_name, 'real-token type must be 1/2/3')
    if np.any(~np.isin(np.asarray(data['region'])[real], np.arange(1, 10))):
        raise _contract_error(sample_name, 'real-token region must be in 1..9')


def _validate_collated_contract(batch: dict) -> None:
    """Validate the exact dictionary returned to the current AbFlow model."""
    required = list(ABFLOW_BATCH_KEYS) + ['lengths']
    missing = [k for k in required if k not in batch]
    if missing:
        raise ValueError(f'AbFlow collated batch is missing required fields: {missing}')

    n_token = int(batch['S'].shape[0])
    if batch['X'].ndim != 3 or batch['X'].shape[0] != n_token:
        raise ValueError(
            'AbFlow collated X/S mismatch: '
            f'X={tuple(batch["X"].shape)}, S={tuple(batch["S"].shape)}'
        )

    if batch['X'].shape[1] != VOCAB.MAX_ATOM_NUMBER:
        raise ValueError(
            'AbFlow collated X uses an unexpected atom-channel count: '
            f'{batch["X"].shape[1]} vs VOCAB.MAX_ATOM_NUMBER='
            f'{VOCAB.MAX_ATOM_NUMBER}'
        )

    # After historical pad_sample(), surface has finally become a dense
    # Cartesian tensor [sum_epitope_residues, num_verts, 3].
    if batch['surface'].ndim != 3 or batch['surface'].shape[-1] != 3:
        raise ValueError(
            'AbFlow collated surface must be '
            '[sum_epitope_residues,num_verts,3], got '
            f'{tuple(batch["surface"].shape)}'
        )
    if not torch.isfinite(batch['surface']).all():
        raise ValueError('AbFlow collated surface contains non-finite coordinates')

    for key in ('smask', 'cmask', 'paratope_mask', 'residue_pos'):
        if batch[key].ndim != 1 or batch[key].shape[0] != n_token:
            raise ValueError(
                f'AbFlow collated {key} must be [N={n_token}], '
                f'got {tuple(batch[key].shape)}'
            )

    n_coord_generate = int(batch['cmask'].sum().item())
    expected_template_shape = (n_coord_generate, batch['X'].shape[1], 3)
    if tuple(batch['template'].shape) != expected_template_shape:
        raise ValueError(
            'AbFlow collated template must match X[cmask]: '
            f'{tuple(batch["template"].shape)} vs expected '
            f'{expected_template_shape}'
        )

    if batch['xloss_mask'].shape != batch['X'].shape[:2]:
        raise ValueError(
            'AbFlow collated xloss_mask/X mismatch: '
            f'{tuple(batch["xloss_mask"].shape)} vs {tuple(batch["X"].shape[:2])}'
        )

    n_paratope = int(batch['paratope_mask'].sum().item())
    if batch['X_pep'].shape[0] != n_paratope:
        raise ValueError(
            'AbFlow collated PCS-RC proposal length mismatch: '
            f'X_pep={batch["X_pep"].shape[0]}, paratope={n_paratope}'
        )
    if batch['S_pep'].shape[0] != n_paratope:
        raise ValueError(
            'AbFlow collated PCS-RC proposal sequence length mismatch: '
            f'S_pep={batch["S_pep"].shape[0]}, paratope={n_paratope}'
        )

    if batch['lengths'].ndim != 1:
        raise ValueError(f'lengths must be [B], got {tuple(batch["lengths"].shape)}')
    if int(batch['lengths'].sum().item()) != n_token:
        raise ValueError(
            'AbFlow collated lengths do not sum to total token count: '
            f'sum(lengths)={int(batch["lengths"].sum().item())}, N={n_token}'
        )


    for key in MFDESIGN_FLAT_KEYS:
        if key not in batch or batch[key].ndim != 1 or batch[key].shape[0] != n_token:
            raise ValueError(f'MFDesign flat field {key} must be [N={n_token}]')
    B = int(batch['lengths'].shape[0])
    Lmax = int(batch['lengths'].max().item())
    if tuple(batch['token_pad_mask'].shape) != (B, Lmax):
        raise ValueError(
            f'token_pad_mask must be [B,Lmax]={(B,Lmax)}, '
            f'got {tuple(batch["token_pad_mask"].shape)}'
        )
    if tuple(batch['token_bonds'].shape) != (B, Lmax, Lmax, 1):
        raise ValueError(
            'token_bonds must be [B,Lmax,Lmax,1], got '
            f'{tuple(batch["token_bonds"].shape)}'
        )


def _generate_pep_data(cplx: AgAbComplex, workspace='workspace/',
                       pdb_path=os.getcwd() + '/all_data/SAb-23-H2-Ab/pdb/'):
    name = cplx.pdb_id[:4]

    if not os.path.exists(workspace):
        os.mkdir(workspace)
    if not os.path.exists(workspace + name):
        os.mkdir(workspace + name)

    workspace = workspace + name + '/'

    pocket = []
    pocket_chain = cplx.antigen.get_chain_names()[0]
    for epi in cplx.epitope:
        res_pose = list(epi[0].id)
        poc = [pocket_chain, res_pose]
        pocket.append(poc)

    json_data = json.dumps(pocket)
    with open(workspace + name + '_pocket.json', 'w') as f:
        f.write(json_data)

    workspace = '../' + workspace
    os.chdir('PepGLAD')

    cdr_length = len(cplx.get_cdr().seq)
    pdb_file = pdb_path + name + '.pdb'
    pocket_file = workspace + name + '_pocket.json'

    cmd = 'CUDA_VISIBLE_DEVICES=0 python -m api.run --mode codesign --pdb ' + pdb_file + ' --pocket ' + pocket_file + ' --out_dir ' + workspace + ' --length_min ' + str(cdr_length) + ' --length_max ' + str(cdr_length + 1) + ' --n_samples 1'
    X = np.array([])
    S = []
    try:
        if os.path.exists(workspace + 'summary.jsonl') and os.path.exists(workspace + name + '_0.pdb'):
            pass
        else:
            os.system(cmd)

        with open(workspace + 'summary.jsonl', 'r') as f:
            pep = json.load(f)
            pep_seq = pep['pep_seq']
            pep_chain = pep['pep_chain']
        for s in pep_seq:
            S.append(VOCAB.symbol_to_idx(s))
        pep_structure = Protein.from_pdb(workspace + name + '_0.pdb').peptides[pep_chain]
        RES = []
        for res in pep_structure.residues:
            coord_map = res.get_coord_map()
            full_atom_coords = np.array([coord_map[key] for key in coord_map.keys() if key != 'OXT'])
            repeat_ca = np.tile(full_atom_coords[1], (VOCAB.MAX_ATOM_NUMBER - len(full_atom_coords), 1))
            full_atom_coords = np.row_stack((full_atom_coords, repeat_ca))
            RES.append(full_atom_coords)
        X = np.stack(RES, axis=0)
    except Exception as e:
        print(name, ':', e)

    os.chdir('../')
    return X, S


def load_pep(pkl_file: str, pdb: str):
    with open(pkl_file, 'rb') as f:
        pep = pickle.load(f)

    pep_X = pep[pdb]['X']
    pep_S = pep[pdb]['S']
    return pep_X, pep_S


def load_surf(pkl_file: str, pdb: str):
    with open(pkl_file, 'rb') as f:
        surf = pickle.load(f)

    return surf[pdb]



def _mf_antibody_regions(item, chain_type: str, chain_len: int):
    """MFDesign FR/CDR labels 1..7 for one antibody chain.

    MFDesign's training code derives alternating framework/CDR regions.  AbFlow
    already owns authoritative CDR index ranges, so use those ranges directly
    and label the intervening framework segments with the exact MFDesign IDs.
    """
    if chain_type not in {'H', 'L'}:
        raise ValueError(f'chain_type must be H or L, got {chain_type!r}')
    labels = np.full(chain_len, MF_REGION_FR1, dtype=np.int64)
    ranges = []
    for cdr_no, region_id in ((1, MF_REGION_CDR1), (2, MF_REGION_CDR2), (3, MF_REGION_CDR3)):
        name = f'{chain_type}{cdr_no}'
        try:
            lo, hi = item.get_cdr_pos(name)
            lo, hi = int(lo), int(hi)
        except Exception:
            continue
        lo = max(0, min(chain_len, lo))
        hi = max(-1, min(chain_len - 1, hi))
        if hi >= lo:
            ranges.append((lo, hi, region_id))

    ranges.sort(key=lambda x: x[0])
    framework_id = MF_REGION_FR1
    cursor = 0
    for lo, hi, cdr_region in ranges:
        if lo > cursor:
            labels[cursor:lo] = framework_id
        labels[lo:hi + 1] = cdr_region
        framework_id = {
            MF_REGION_CDR1: MF_REGION_FR2,
            MF_REGION_CDR2: MF_REGION_FR3,
            MF_REGION_CDR3: MF_REGION_FR4,
        }[cdr_region]
        cursor = hi + 1
    if cursor < chain_len:
        labels[cursor:] = framework_id
    return labels


def _mf_build_token_metadata(item, ag_records, n_ag, n_h, n_l):
    """Build the exact token metadata family consumed by MFDesign modules.

    ``n_ag/n_h/n_l`` include the historical BOA/BOH/BOL bookkeeping token.
    ``ag_records`` contains tuples ``(chain_name, chain_residue_index,
    is_epitope)`` for every real antigen token in the order used by X/S.
    """
    total = int(n_ag + n_h + n_l)
    meta = {
        'token_index': np.zeros(total, dtype=np.int64),
        'residue_index': np.zeros(total, dtype=np.int64),
        'asym_id': np.zeros(total, dtype=np.int64),
        'entity_id': np.zeros(total, dtype=np.int64),
        'sym_id': np.zeros(total, dtype=np.int64),
        'type': np.zeros(total, dtype=np.int64),
        'region': np.zeros(total, dtype=np.int64),
        'token_pad_mask': np.zeros(total, dtype=np.bool_),
    }
    token_bonds = np.zeros((total, total, 1), dtype=np.float32)

    # Stable per-complex asym ids.  0 is reserved for bookkeeping/pad.
    antigen_chain_names = []
    for chain_name, _, _ in ag_records:
        if chain_name not in antigen_chain_names:
            antigen_chain_names.append(chain_name)
    asym_map = {name: i + 1 for i, name in enumerate(antigen_chain_names)}
    heavy_asym = len(asym_map) + 1
    light_asym = heavy_asym + 1

    # Antigen tokens. ag_data[0] is BOA and is deliberately masked out.
    prev_by_chain = {}
    for local_i, (chain_name, chain_res_idx, is_epitope) in enumerate(ag_records, start=1):
        gi = local_i
        asym = asym_map[chain_name]
        meta['residue_index'][gi] = int(chain_res_idx)
        meta['asym_id'][gi] = asym
        meta['entity_id'][gi] = asym
        meta['type'][gi] = MF_TYPE_ANTIGEN
        meta['region'][gi] = (
            MF_REGION_AG_EPITOPE if bool(is_epitope)
            else MF_REGION_AG_NON_EPITOPE
        )
        meta['token_pad_mask'][gi] = True
        if chain_name in prev_by_chain:
            prev_gi, prev_res_idx = prev_by_chain[chain_name]
            # Only true adjacent residues carry the MFDesign token-bond feature.
            if int(chain_res_idx) == int(prev_res_idx) + 1:
                token_bonds[prev_gi, gi, 0] = 1.0
                token_bonds[gi, prev_gi, 0] = 1.0
        prev_by_chain[chain_name] = (gi, chain_res_idx)

    # Heavy chain: BOH is the first token in hc_data and is masked out.
    h_start = n_ag
    h_real = n_h - 1
    h_regions = _mf_antibody_regions(item, 'H', h_real)
    for r in range(h_real):
        gi = h_start + 1 + r
        meta['residue_index'][gi] = r
        meta['asym_id'][gi] = heavy_asym
        meta['entity_id'][gi] = heavy_asym
        meta['type'][gi] = MF_TYPE_HEAVY
        meta['region'][gi] = int(h_regions[r])
        meta['token_pad_mask'][gi] = True
        if r > 0:
            token_bonds[gi - 1, gi, 0] = 1.0
            token_bonds[gi, gi - 1, 0] = 1.0

    # Light chain.
    l_start = n_ag + n_h
    l_real = n_l - 1
    l_regions = _mf_antibody_regions(item, 'L', l_real)
    for r in range(l_real):
        gi = l_start + 1 + r
        meta['residue_index'][gi] = r
        meta['asym_id'][gi] = light_asym
        meta['entity_id'][gi] = light_asym
        meta['type'][gi] = MF_TYPE_LIGHT
        meta['region'][gi] = int(l_regions[r])
        meta['token_pad_mask'][gi] = True
        if r > 0:
            token_bonds[gi - 1, gi, 0] = 1.0
            token_bonds[gi, gi - 1, 0] = 1.0

    meta['token_bonds'] = token_bonds
    return meta


def _generate_chain_data(residues, start):
    backbone_atoms = VOCAB.backbone_atoms
    # Coords, Sequence, residue positions, mask for loss calculation (exclude missing coordinates)
    X, S, res_pos, xloss_mask = [], [], [], []
    # global node
    # coordinates will be set to the center of the chain
    X.append([[0, 0, 0] for _ in range(VOCAB.MAX_ATOM_NUMBER)])
    S.append(VOCAB.symbol_to_idx(start))
    res_pos.append(0)
    xloss_mask.append([0 for _ in range(VOCAB.MAX_ATOM_NUMBER)])
    # other nodes
    for residue in residues:
        residue_xloss_mask = [0 for _ in range(VOCAB.MAX_ATOM_NUMBER)]
        bb_atom_coord = residue.get_backbone_coord_map()
        sc_atom_coord = residue.get_sidechain_coord_map()
        if 'CA' not in bb_atom_coord:
            for atom in bb_atom_coord:
                ca_x = bb_atom_coord[atom]
                print_log(f'no ca, use {atom}', level='DEBUG')
                break
        else:
            ca_x = bb_atom_coord['CA']
        x = [ca_x for _ in range(VOCAB.MAX_ATOM_NUMBER)]

        i = 0
        for atom in backbone_atoms:
            if atom in bb_atom_coord:
                x[i] = bb_atom_coord[atom]
                residue_xloss_mask[i] = 1
            i += 1
        for atom in residue.sidechain:
            if atom in sc_atom_coord:
                x[i] = sc_atom_coord[atom]
                residue_xloss_mask[i] = 1
            i += 1

        X.append(x)
        S.append(VOCAB.symbol_to_idx(residue.get_symbol()))
        res_pos.append(residue.get_id()[0])
        xloss_mask.append(residue_xloss_mask)
    X = np.array(X)
    center = np.mean(X[1:].reshape(-1, 3), axis=0)
    X[0] = center  # set center
    if start == VOCAB.BOA:  # epitope does not have position encoding
        res_pos = [0 for _ in res_pos]
    data = {'X': X, 'S': S, 'residue_pos': res_pos, 'xloss_mask': xloss_mask}
    return data


# use this class to splice the dataset and maintain only one part of it in RAM
# Antibody-Antigen Complex dataset
class E2EDataset(torch.utils.data.Dataset):
    # Centralized existing model-facing data schema.
    BATCH_KEYS = ABFLOW_BATCH_KEYS
    BATCH_DTYPES = ABFLOW_BATCH_DTYPES
    FIELD_ROLES = ABFLOW_FIELD_ROLES

    def __init__(self, file_path, save_dir=None, pep_file=None, surf_file=None,
                 cdr=None, paratope='H3', num_verts=75, full_antigen=False,
                 num_entry_per_file=-1, random=False):
        '''
        file_path: path to the dataset
        save_dir: directory to save the processed data
        cdr: which cdr to generate (L1/2/3, H1/2/3) (can be list), None for all including framework
        paratope: which cdr to use as paratope (L1/2/3, H1/2/3) (can be list)
        full_antigen: whether to use the full antigen information
        num_entry_per_file: number of entries in a single file. -1 to save all data into one file
                            (In-memory dataset)

        Module-1 contract:
          - returned scientific fields stay identical to original AbFlow;
          - no MSA/profile features are introduced;
          - coordinate/sequence/source semantics are not redefined.
        '''
        super().__init__()

        self.pep_file = pep_file
        self.use_pep = pep_file is not None
        self.surf_file = surf_file
        self.cdr = cdr
        self.paratope = paratope
        self.num_verts = num_verts
        self.full_antigen = full_antigen
        if save_dir is None:
            if not os.path.isdir(file_path):
                save_dir = os.path.split(file_path)[0]
            else:
                save_dir = file_path
            prefix = os.path.split(file_path)[1]
            if '.' in prefix:
                prefix = prefix.split('.')[0]
            save_dir = os.path.join(save_dir, f'{prefix}_processed')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        metainfo_file = os.path.join(save_dir, '_metainfo')
        self.data: List[AgAbComplex] = []  # list of ABComplex

        # try loading preprocessed files
        need_process = False
        try:
            with open(metainfo_file, 'r') as fin:
                metainfo = json.load(fin)
                self.num_entry = metainfo['num_entry']
                self.file_names = metainfo['file_names']
                self.file_num_entries = metainfo['file_num_entries']
        except FileNotFoundError:
            print_log('No meta-info file found, start processing', level='INFO')
            need_process = True
        except Exception as e:
            print_log(f'Faild to load file {metainfo_file}, error: {e}', level='WARN')
            need_process = True

        if need_process:
            # preprocess
            self.file_names, self.file_num_entries = [], []
            self.preprocess(file_path, save_dir, num_entry_per_file)
            self.num_entry = sum(self.file_num_entries)

            metainfo = {
                'num_entry': self.num_entry,
                'file_names': self.file_names,
                'file_num_entries': self.file_num_entries
            }
            with open(metainfo_file, 'w') as fout:
                json.dump(metainfo, fout)

        self.random = random
        self.cur_file_idx, self.cur_idx_range = 0, (0, self.file_num_entries[0])  # left close, right open
        self._load_part()

        # user defined variables
        self.idx_mapping = [i for i in range(self.num_entry)]
        self.mode = '111'  # H/L/Antigen, 1 for include, 0 for exclude

    @classmethod
    def contract(cls):
        """Return the complete AbFlow + MFDesign model-facing data contract."""
        out = {
            key: {
                'dtype': str(ABFLOW_BATCH_DTYPE_MAP[key]),
                'role': cls.FIELD_ROLES[key],
            }
            for key in cls.BATCH_KEYS
        }
        for key in MFDESIGN_FLAT_KEYS:
            out[key] = {
                'dtype': str(MFDESIGN_FLAT_DTYPE_MAP[key]),
                'role': cls.FIELD_ROLES[key],
            }
        out['token_pad_mask'] = {'dtype': 'torch.bool', 'role': cls.FIELD_ROLES['token_pad_mask']}
        out['token_bonds'] = {'dtype': 'torch.float', 'role': cls.FIELD_ROLES['token_bonds']}
        return out

    def _save_part(self, save_dir, num_entry):
        file_name = os.path.join(save_dir, f'part_{len(self.file_names)}.pkl')
        print_log(f'Saving {file_name} ...')
        file_name = os.path.abspath(file_name)
        if num_entry == -1:
            end = len(self.data)
        else:
            end = min(num_entry, len(self.data))
        with open(file_name, 'wb') as fout:
            pickle.dump(self.data[:end], fout)
        self.file_names.append(file_name)
        self.file_num_entries.append(end)
        self.data = self.data[end:]

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        print_log(f'Loading preprocessed file {f}, {self.cur_file_idx + 1}/{len(self.file_names)}')
        with open(f, 'rb') as fin:
            del self.data
            self.data = pickle.load(fin)
        self.access_idx = [i for i in range(len(self.data))]
        if self.random:
            np.random.shuffle(self.access_idx)

    def _check_load_part(self, idx):
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        idx = self.access_idx[idx - self.cur_idx_range[0]]
        return idx

    def __len__(self):
        return self.num_entry

    ########### load data from file_path and add to self.data ##########
    def preprocess(self, file_path, save_dir, num_entry_per_file):
        '''
        Load data from file_path and add processed data entries to self.data.
        Remember to call self._save_data(num_entry_per_file) to control the number
        of items in self.data (this function will save the first num_entry_per_file
        data and release them from self.data) e.g. call it when len(self.data) reaches
        num_entry_per_file.
        '''
        with open(file_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
        # line_id = 0
        for line in tqdm(lines):
            # if line_id < 206:
            #     line_id += 1
            #     continue
            item = json.loads(line)
            try:
                cplx = AgAbComplex.from_pdb(
                    item['pdb_data_path'], item['heavy_chain'], item['light_chain'],
                    item['antigen_chains'])
            except AssertionError as e:
                print_log(e, level='ERROR')
                print_log(f'parse {item["pdb"]} pdb failed, skip', level='ERROR')
                continue

            self.data.append(cplx)
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)

    ########## override get item ##########
    def __getitem__(self, idx):
        '''
        an example of the returned data
        {
            'X': [n, n_channel, 3],
            'S': [n],
            'cmask': [n],
            'smask': [n],
            'paratope_mask': [n], when cdr=paratope='H3', smask=paratope_mask
            'xloss_mask': [n, n_channel], can refer to residue type
            'template': [n, n_channel, 3]
        }
        '''
        idx = self.idx_mapping[idx]
        idx = self._check_load_part(idx)
        item = self.data[idx]

        # pdb name
        self.pdb_name = item.pdb_id[:4]

        # antigen + MFDesign chain/index/epitope metadata
        ag_residues = []
        ag_records = []  # (chain_name, sequential residue index, is_epitope)
        epitope_keys = set()
        for residue, chain_name, chain_i in item.get_epitope():
            epitope_keys.add((str(chain_name), int(chain_i)))

        if self.full_antigen:
            ag = item.get_antigen()
            for chain_name in ag.get_chain_names():
                chain_obj = ag.get_chain(chain_name)
                for i in range(len(chain_obj)):
                    residue = chain_obj.get_residue(i)
                    ag_residues.append(residue)
                    ag_records.append((
                        str(chain_name), int(i),
                        (str(chain_name), int(i)) in epitope_keys,
                    ))
        else:
            # Epitope-only AbFlow context: every retained antigen residue is an
            # epitope token, but original chain/index is preserved.
            for residue, chain_name, i in item.get_epitope():
                ag_residues.append(residue)
                ag_records.append((str(chain_name), int(i), True))

        # generate antigen data
        ag_data = _generate_chain_data(ag_residues, VOCAB.BOA)

        hc, lc = item.get_heavy_chain(), item.get_light_chain()
        hc_residues, lc_residues = [], []

        # generate heavy chain data
        for i in range(len(hc)):
            hc_residues.append(hc.get_residue(i))
        hc_data = _generate_chain_data(hc_residues, VOCAB.BOH)

        # generate light chain data
        for i in range(len(lc)):
            lc_residues.append(lc.get_residue(i))
        lc_data = _generate_chain_data(lc_residues, VOCAB.BOL)

        data = {key: np.concatenate([ag_data[key], hc_data[key], lc_data[key]], axis=0)
                for key in hc_data}

        # smask (sequence) and cmask (coordinates): 0 for fixed, 1 for generate
        # not generate coordinates of global node and antigen
        cmask = [0 for _ in ag_data['S']] + [0] + [1 for _ in hc_data['S'][1:]] + [0] + [1 for _ in lc_data['S'][1:]]
        # epitope mask: 1 for epitope, 0 for others
        emask = [0] + [1 for _ in ag_data['S'][1:]] + [0 for _ in hc_data['S']] + [0 for _ in lc_data['S']]
        # according to the setting of cdr
        if self.cdr is None:
            smask = cmask
        else:
            smask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
            cdrs = [self.cdr] if type(self.cdr) == str else self.cdr
            for cdr in cdrs:
                cdr_range = item.get_cdr_pos(cdr)
                offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
                for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                    smask[idx] = 1

        data['cmask'], data['smask'] = cmask, smask
        data['emask'] = emask

        paratope_mask = [0 for _ in range(len(ag_data['S']) + len(hc_data['S']) + len(lc_data['S']))]
        paratope = [self.paratope] if type(self.paratope) == str else self.paratope
        for cdr in paratope:
            cdr_range = item.get_cdr_pos(cdr)
            offset = len(ag_data['S']) + 1 + (0 if cdr[0] == 'H' else len(hc_data['S']))
            for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                paratope_mask[idx] = 1
        data['paratope_mask'] = paratope_mask

        # v111: authoritative MFDesign token metadata.  This is generated from
        # the same complex object that produced X/S and is available identically
        # in train, validation and test; no native endpoint coordinate is used.
        mf_meta = _mf_build_token_metadata(
            item=item,
            ag_records=ag_records,
            n_ag=len(ag_data['S']),
            n_h=len(hc_data['S']),
            n_l=len(lc_data['S']),
        )
        data.update(mf_meta)

        template = ConserveTemplateGenerator().construct_template(item, align=False)
        data['template'] = template
        if self.use_pep:
            data['X_pep'], data['S_pep'] = load_pep(self.pep_file, self.pdb_name)
        else:
            n_paratope = int(sum(paratope_mask))
            data['X_pep'] = np.zeros((n_paratope, VOCAB.MAX_ATOM_NUMBER, 3), dtype=np.float32)
            data['S_pep'] = np.zeros(n_paratope, dtype=np.int64)

        if self.surf_file:
            data['surface'] = load_surf(self.surf_file, self.pdb_name)

        data['name'] = self.pdb_name

        # Module-1 addition: validation only; returned data are unchanged.
        _validate_numpy_item_contract(data, sample_name=self.pdb_name)
        return data

    @classmethod
    def collate_fn(cls, batch):
        # Preserve all historical AbFlow tensors exactly.
        res = {}
        for key, _type in zip(cls.BATCH_KEYS, cls.BATCH_DTYPES):
            val = []
            for item in batch:
                if key == 'surface':
                    v = pad_sample(item[key])
                    val.append(torch.tensor(v, dtype=_type))
                else:
                    val.append(torch.tensor(item[key], dtype=_type))
            res[key] = torch.cat(val, dim=0)

        lengths = [len(item['S']) for item in batch]
        res['lengths'] = torch.tensor(lengths, dtype=torch.long)

        # MFDesign flat metadata follow the same flattened token order as X/S.
        for key, dtype in zip(MFDESIGN_FLAT_KEYS, MFDESIGN_FLAT_DTYPES):
            res[key] = torch.cat([
                torch.tensor(item[key], dtype=dtype) for item in batch
            ], dim=0)

        # Pairwise metadata remain graph-batched and padded, matching MFDesign.
        B = len(batch)
        Lmax = max(lengths)
        token_pad_mask = torch.zeros((B, Lmax), dtype=torch.bool)
        token_bonds = torch.zeros((B, Lmax, Lmax, 1), dtype=torch.float)
        for b, item in enumerate(batch):
            L = lengths[b]
            token_pad_mask[b, :L] = torch.tensor(
                item['token_pad_mask'], dtype=torch.bool
            )
            token_bonds[b, :L, :L] = torch.tensor(
                item['token_bonds'], dtype=torch.float
            )
        res['token_pad_mask'] = token_pad_mask
        res['token_bonds'] = token_bonds

        _validate_collated_contract(res)
        return res


def parse():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to save processed data')
    parser.add_argument('--pep_file', type=str, default=None, help='Path to save processed data')
    parser.add_argument('--surf_file', type=str, default=None, help='Path to save processed data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    dataset = E2EDataset(args.dataset, args.save_dir, pep_file=args.pep_file,
                         surf_file=args.surf_file, cdr='H3', num_entry_per_file=-1)
    print(len(dataset))
