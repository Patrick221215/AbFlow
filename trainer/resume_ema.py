#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import random
from contextlib import contextmanager

import numpy as np
import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def cfg_get(cfg, name, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {}
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.shadow:
                    self.shadow[name] = param.detach().clone()
                    continue
                old = self.shadow[name].to(param.device)
                self.shadow[name] = old.mul(decay).add(param.detach(), alpha=1.0 - decay).clone()

    def apply_shadow(self):
        self.backup = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name].to(param.device))

    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.backup:
                    param.copy_(self.backup[name].to(param.device))
        self.backup = {}

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = int(state.get("num_updates", 0))
        shadow = state.get("shadow", state)

        loaded = 0
        model_keys = {n for n, p in self.model.named_parameters() if p.requires_grad}
        for key in model_keys:
            if key in shadow:
                self.shadow[key] = shadow[key].detach().clone()
                loaded += 1
        if loaded == 0:
            raise RuntimeError("EMA load failed: no matching parameter was loaded.")


def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _as_cpu_byte_tensor(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.uint8)
    if isinstance(x, (bytes, bytearray)):
        return torch.ByteTensor(list(x))
    if isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=torch.uint8)
    return None


def set_rng_state(state):
    if not state:
        return

    if "python" in state:
        try:
            random.setstate(state["python"])
        except Exception as e:
            print(f"[Resume] skip python RNG restore: {e}")

    if "numpy" in state:
        try:
            np.random.set_state(state["numpy"])
        except Exception as e:
            print(f"[Resume] skip numpy RNG restore: {e}")

    if "torch" in state:
        try:
            torch_state = _as_cpu_byte_tensor(state["torch"])
            if torch_state is not None:
                torch.set_rng_state(torch_state)
            else:
                print(f"[Resume] skip torch RNG restore: unsupported type {type(state['torch'])}")
        except Exception as e:
            print(f"[Resume] skip torch RNG restore: {e}")

    cuda_state = state.get("cuda", None)
    if torch.cuda.is_available() and cuda_state is not None:
        try:
            if isinstance(cuda_state, (list, tuple)):
                cuda_state = [
                    _as_cpu_byte_tensor(s) for s in cuda_state
                    if _as_cpu_byte_tensor(s) is not None
                ]
                if cuda_state:
                    torch.cuda.set_rng_state_all(cuda_state)
            else:
                one_state = _as_cpu_byte_tensor(cuda_state)
                if one_state is not None:
                    torch.cuda.set_rng_state(one_state)
        except Exception as e:
            print(f"[Resume] skip cuda RNG restore: {e}")

def init_resume_ema(trainer):
    if cfg_get(trainer.config, "use_ema", False):
        trainer.ema = EMA(unwrap_model(trainer.model), decay=cfg_get(trainer.config, "ema_decay", 0.999))
    else:
        trainer.ema = None


def maybe_resume(trainer, device):
    path = cfg_get(trainer.config, "resume_checkpoint", "")
    if not path:
        return

    if not os.path.isfile(path):
        raise FileNotFoundError(f"resume_checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    model_state = ckpt.get("model", ckpt.get("state_dict", None))
    if model_state is None:
        raise KeyError("Checkpoint has no 'model' or 'state_dict' field.")

    missing, unexpected = unwrap_model(trainer.model).load_state_dict(model_state, strict=False)

    trainer.global_step = int(ckpt.get("global_step", getattr(trainer, "global_step", 0)))
    trainer.epoch = int(ckpt.get("epoch", getattr(trainer, "epoch", 0)))

    trainer_state = ckpt.get("trainer_state", {})
    trainer.valid_global_step = int(trainer_state.get("valid_global_step", getattr(trainer, "valid_global_step", 0)))
    trainer.last_valid_metric = trainer_state.get("last_valid_metric", getattr(trainer, "last_valid_metric", None))
    trainer.patience = int(trainer_state.get("patience", getattr(trainer, "patience", cfg_get(trainer.config, "patience", 3))))

    if ckpt.get("optimizer") is not None:
        trainer.optimizer.load_state_dict(ckpt["optimizer"])

    if getattr(trainer, "scheduler", None) is not None and ckpt.get("scheduler") is not None:
        trainer.scheduler.load_state_dict(ckpt["scheduler"])

    if getattr(trainer, "ema", None) is not None and ckpt.get("ema") is not None:
        trainer.ema.load_state_dict(ckpt["ema"])

    set_rng_state(ckpt.get("rng_state"))

    local_rank = getattr(trainer, "local_rank", -1)
    if local_rank in (-1, 0):
        print(f"[Resume] loaded checkpoint: {path}")
        print(f"[Resume] epoch={trainer.epoch}, global_step={trainer.global_step}")
        print(f"[Resume] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")


def after_optimizer_step(trainer):
    if getattr(trainer, "ema", None) is not None:
        trainer.ema.update()


@contextmanager
def validation_ema(trainer):
    use_ema = getattr(trainer, "ema", None) is not None
    if use_ema:
        trainer.ema.apply_shadow()
    try:
        yield
    finally:
        if use_ema:
            trainer.ema.restore()


def build_checkpoint_payload(trainer, metric=None):
    payload = {
        "format_version": 2,
        "epoch": int(getattr(trainer, "epoch", 0)),
        "global_step": int(getattr(trainer, "global_step", 0)),
        "metric": metric,
        "model": unwrap_model(trainer.model).state_dict(),
        "optimizer": trainer.optimizer.state_dict() if getattr(trainer, "optimizer", None) is not None else None,
        "scheduler": trainer.scheduler.state_dict() if getattr(trainer, "scheduler", None) is not None else None,
        "rng_state": get_rng_state(),
        "config": vars(trainer.config) if hasattr(trainer.config, "__dict__") else {},
        "trainer_state": {
            "valid_global_step": int(getattr(trainer, "valid_global_step", 0)),
            "last_valid_metric": getattr(trainer, "last_valid_metric", None),
            "patience": int(getattr(trainer, "patience", 0)),
        },
    }

    if getattr(trainer, "ema", None) is not None:
        payload["ema"] = trainer.ema.state_dict()

    return payload


def save_checkpoint(trainer, path, metric=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(build_checkpoint_payload(trainer, metric=metric), path)
    print(f"[Checkpoint] saved: {path}")