# llmresfi/hooks.py
"""Reusable low-level helpers shared by the injector's hook implementations."""
from __future__ import annotations

import torch

from . import fault_models


def has_2d_weight(module_info) -> bool:
    w = getattr(module_info.module, "weight", None)
    return isinstance(w, torch.nn.Parameter) and w.ndim == 2


def apply_fault_to_scalar(value: torch.Tensor, fault_model_cfg, rng):
    """Apply a fault to a scalar tensor; return (corrupted_value, bit_idxs)."""
    return fault_models.apply_fault(
        value,
        fault_model_cfg.type,
        rng,
        bit_idx=fault_model_cfg.bit_idx,
        min_val=fault_model_cfg.min_val,
        max_val=fault_model_cfg.max_val,
        factor=fault_model_cfg.factor,
    )


def flip_flat(tensor: torch.Tensor, flat_idx: int, fault_model_cfg, rng):
    """Flip one element of a flattened view of ``tensor`` in place.

    Returns the bit indices that were flipped (or None for value faults).
    """
    flat = tensor.view(-1)
    corrupted, bits = apply_fault_to_scalar(flat[flat_idx], fault_model_cfg, rng)
    flat[flat_idx] = corrupted
    return bits


class TimingGate:
    """Decide whether a hook should fire, based on a timing config.

    - transient:    fire on the first eligible trigger only (classic single fault).
    - permanent:    fire on the first ``steps`` eligible triggers (persistent fault).
    - intermittent: fire on every ``period``-th eligible trigger (periodic fault).
    """

    def __init__(self, timing_cfg) -> None:
        self.cfg = timing_cfg
        self._triggers = 0

    def should_fire(self) -> bool:
        t = self.cfg.type
        if t == "transient":
            return self._triggers == 0
        if t == "permanent":
            return self._triggers < self.cfg.steps
        if t == "intermittent":
            return self._triggers % self.cfg.period == 0
        return True

    def mark_fired(self) -> None:
        self._triggers += 1
