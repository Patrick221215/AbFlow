#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import copy
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



# ---------------------------------------------------------------------------
# Full-object checkpoint runtime compatibility
# ---------------------------------------------------------------------------
#
# Historical AbFlow checkpoints were serialized as full Python model objects.
# When such an object is loaded under a newer AbFlowModel class, Python/PyTorch
# restores the old instance state but DOES NOT re-run the new __init__().
# Therefore a newer method may legally exist while the old instance lacks the
# attributes that method expects.
#
# Compatibility policy:
#   1) Never overwrite an attribute already stored in the checkpoint.
#   2) Missing historical optional features mean the historical behavior
#      ("feature off"), not the current environment setting.
#   3) Never create new trainable modules for an old checkpoint at evaluation.
#   4) If a partially migrated checkpoint claims a feature is enabled but lacks
#      the trained module/state required to realize it, fail fast instead of
#      silently changing the scientific model.
#   5) Prefer a model-owned compatibility hook when future AbFlow versions add
#      one.  This keeps future migrations next to the code that owns the new
#      attributes rather than growing generate.py forever.
#
# This is deliberately a migration layer, not an experiment override layer.
# ---------------------------------------------------------------------------

RUNTIME_COMPAT_SCHEMA_VERSION = 3


def _compat_log(message):
    print_log(f"[RuntimeCompat] {message}")


def _set_missing_attr(model, name, value, patched, reason=""):
    """Set a checkpoint-missing plain attribute without overwriting stored state."""
    if hasattr(model, name):
        return False

    # Avoid sharing mutable default objects between model instances.
    setattr(model, name, copy.deepcopy(value))
    patched.append(name)
    if reason:
        _compat_log(f"added {name}={value!r} ({reason})")
    else:
        _compat_log(f"added {name}={value!r}")
    return True


def _ensure_empty_buffer(model, name, patched):
    """Register an empty non-persistent buffer only when the old object lacks it."""
    if hasattr(model, name):
        return False

    # nn.Module.register_buffer is preferable to setattr because later .to(device)
    # then handles the tensor exactly like buffers created by the current __init__.
    if hasattr(model, "register_buffer"):
        model.register_buffer(name, torch.empty(0), persistent=False)
    else:
        setattr(model, name, torch.empty(0))
    patched.append(name)
    _compat_log(f"added empty buffer {name} (historical feature-off state)")
    return True


def _call_model_compat_hooks(model):
    """Run compatibility hooks implemented by the CURRENT model class.

    A historical full-object checkpoint still resolves methods on the current
    class, so this is the preferred future-proof extension point.  Going
    forward, newly introduced inference-time attributes should be initialized
    in AbFlowModel._ensure_runtime_compat(), using only hasattr/getattr-safe
    logic and historical feature-off defaults.
    """
    # New general hook: future versions should put new migrations here.
    hook = getattr(model, "_ensure_runtime_compat", None)
    if callable(hook):
        hook()

    # Existing ScoreFM-specific compatibility hook retained for old revisions.
    score_hook = getattr(model, "_ensure_scorefm_compat", None)
    if callable(score_hook):
        score_hook()


def _migrate_structure_sequence_readout(model, patched):
    """v90/U07 migration for checkpoints created before the readout existed.

    Historical checkpoints (U02/U03/U05 and earlier) never contained this
    module, so missing state must map to U07=OFF.  Existing U07+ checkpoints
    keep their serialized adapter and mode untouched.
    """
    had_mode = hasattr(model, "structure_seq_readout_mode")
    had_adapter = hasattr(model, "structure_seq_adapter")

    # A transitional/new checkpoint may already carry a trained adapter but miss
    # only the string mode.  In that narrow case the adapter itself is strong
    # evidence that the feature was enabled, so preserve rather than disable it.
    if not had_mode:
        adapter = getattr(model, "structure_seq_adapter", None)
        inferred_mode = "interface_geometry" if adapter is not None else "off"
        _set_missing_attr(
            model,
            "structure_seq_readout_mode",
            inferred_mode,
            patched,
            reason=(
                "inferred from serialized adapter"
                if inferred_mode == "interface_geometry"
                else "pre-U07 checkpoint => feature off"
            ),
        )

    _set_missing_attr(
        model,
        "structure_seq_feature_dim",
        6,
        patched,
        reason="v90 interface-geometry feature width",
    )

    if not had_adapter:
        _set_missing_attr(
            model,
            "structure_seq_adapter",
            None,
            patched,
            reason="pre-U07 checkpoint => no untrained module is created",
        )

    _set_missing_attr(
        model,
        "_last_structure_seq_diagnostics",
        {},
        patched,
        reason="diagnostics-only cache",
    )

    mode = getattr(model, "structure_seq_readout_mode", "off")
    adapter = getattr(model, "structure_seq_adapter", None)

    if mode not in {"off", "interface_geometry"}:
        raise RuntimeError(
            "Unsupported historical structure_seq_readout_mode="
            f"{mode!r}. Refusing to guess checkpoint semantics."
        )

    if mode == "off":
        # Historical U02/U03/U05 path: no geometry feature is evaluated.
        _ensure_empty_buffer(model, "_structure_seq_rbf_centers", patched)
    else:
        # Never silently manufacture a fresh adapter for an evaluation checkpoint.
        if adapter is None:
            raise RuntimeError(
                "Checkpoint says structure_seq_readout_mode='interface_geometry' "
                "but contains no trained structure_seq_adapter. Refusing to "
                "silently evaluate a different model."
            )
        if not hasattr(model, "_structure_seq_rbf_centers"):
            raise RuntimeError(
                "Checkpoint enables structure-conditioned sequence readout but "
                "is missing _structure_seq_rbf_centers. This is not a safe "
                "historical-feature-off migration; inspect the checkpoint/code "
                "version instead of guessing the RBF basis."
            )


def _validate_runtime_compat(model):
    """Fail early on combinations that would silently change experiment meaning."""
    coordinate_authority = getattr(model, "coordinate_authority", "legacy_dual")
    if coordinate_authority not in {"legacy_dual", "carrier_single"}:
        raise RuntimeError(
            "Invalid coordinate_authority in checkpoint: "
            f"{coordinate_authority!r}"
        )

    mode = getattr(model, "structure_seq_readout_mode", "off")
    adapter = getattr(model, "structure_seq_adapter", None)
    if mode == "off" and adapter is not None:
        # Current message-passing code may key directly on adapter is None.
        # An off-mode + live adapter is therefore ambiguous and unsafe.
        raise RuntimeError(
            "Inconsistent checkpoint: structure_seq_readout_mode='off' but "
            "structure_seq_adapter is not None. Refusing silent behavior change."
        )

    # Record the compatibility schema on the in-memory object only.  This does
    # not alter the checkpoint file and is useful for diagnostic logs.
    setattr(model, "_runtime_compat_schema_version", RUNTIME_COMPAT_SCHEMA_VERSION)


def ensure_model_runtime_compat(model):
    """Upgrade a historical full-object checkpoint to the CURRENT runtime API.

    This function preserves checkpoint scientific semantics.  It never enables
    a feature that did not exist in the serialized model and never overwrites a
    stored attribute.  Future model versions should preferably implement
    ``AbFlowModel._ensure_runtime_compat()``; this loader automatically calls it.

    Important limitation
    --------------------
    No loader can safely guess the value/type/weights of an arbitrary future
    attribute.  A newly added inference-time feature still needs ONE explicit
    backward rule (normally "missing => feature off") in the model-owned hook.
    What this framework prevents is scattering ad-hoc fixes throughout test
    scripts or silently evaluating old checkpoints with new untrained modules.
    """
    patched = []

    # Let the current class repair model-owned state first.
    _call_model_compat_hooks(model)

    # Very old AbFlow defaults.
    _set_missing_attr(model, "pep_seq", True, patched, "historical AbFlow default")
    _set_missing_attr(model, "pep_struct", True, patched, "historical AbFlow default")

    # Legacy ScoreFM fields read by newer runtime methods.  These are only
    # injected when absent; checkpoints that already store their experiment
    # values remain untouched.
    historical_defaults = {
        "use_scorefm": False,
        "scorefm_min_sigma": 1e-2,
        "scorefm_eps": 1e-8,
        "scorefm_t_threshold": 0.50,
        "scorefm_loss_weight": 5e-2,
        "scorefm_velocity_weight": 1.0,
        "scorefm_dsm_weight": 1.0,
        "scorefm_x1_weight": 0.25,
        "scorefm_local_dist_weight": 0.05,
        "scorefm_interface_contact_weight": 0.05,
        "scorefm_inter_clash_weight": 0.01,
        "scorefm_intra_clash_weight": 0.005,
        "scorefm_contact_cutoff": 8.0,
        "scorefm_contact_temperature": 1.0,
        "scorefm_inter_clash_cutoff": 2.0,
        "scorefm_intra_clash_cutoff": 1.5,
        "last_scorefm_losses": {},
        "sf2m_t_eps": 0.01,
        "sf2m_score_weight": 1.0,

        # v81: old checkpoints predate pair-score feedback.
        "pair_score_feedback": False,

        # v88: U02/U03/U05 and earlier use the historical dual coordinate
        # supervision/readout semantics.  Missing must NOT inherit a current
        # ABFLOW_COORDINATE_AUTHORITY environment variable.
        "coordinate_authority": "legacy_dual",
    }
    for name, value in historical_defaults.items():
        _set_missing_attr(model, name, value, patched)

    # v90/U07 migration.
    _migrate_structure_sequence_readout(model, patched)

    _validate_runtime_compat(model)

    if patched:
        _compat_log(
            "patched historical checkpoint with "
            f"{len(patched)} runtime field(s): {', '.join(patched)}"
        )
    else:
        _compat_log("checkpoint already satisfies current runtime schema")

    _compat_log(
        "effective compatibility state: "
        f"schema={RUNTIME_COMPAT_SCHEMA_VERSION}, "
        f"coordinate_authority={getattr(model, 'coordinate_authority', '<missing>')}, "
        f"structure_seq_readout_mode="
        f"{getattr(model, 'structure_seq_readout_mode', '<missing>')}, "
        f"structure_seq_adapter="
        f"{'present' if getattr(model, 'structure_seq_adapter', None) is not None else 'none'}"
    )
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
