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

    Training hooks pass the current training step, so timing is *step based*:

    - transient:    fire once in the whole run.
    - permanent:    fire on each of the first ``steps`` eligible training steps.
    - intermittent: fire on every ``period``-th eligible training step.

    At most one fault is emitted per step (``one_per_step``), which matters for
    communication hooks: a DDP backward fires the hook once per gradient bucket,
    but a "permanent" fault should still inject only once per training step.

    When ``step`` is ``None`` (inference prefill/first-token) the legacy
    trigger-count semantics are used so that a single phase emits one fault.
    """

    def __init__(self, timing_cfg) -> None:
        self.cfg = timing_cfg
        self._triggers = 0
        self._fired_any = False
        self._start_step = None
        self._last_fired_step = None

    def should_fire(self, step=None, one_per_step: bool = True) -> bool:
        t = self.cfg.type
        if step is not None:
            if one_per_step and self._last_fired_step == step:
                return False
            if t == "transient":
                return not self._fired_any
            if t == "permanent":
                if self._start_step is None:
                    return True
                return step < self._start_step + self.cfg.steps
            if t == "intermittent":
                if self._start_step is None:
                    return True
                return (step - self._start_step) % self.cfg.period == 0
            return True

        # Legacy trigger-count mode (inference prefill / first_token).
        if t == "transient":
            return self._triggers == 0
        if t == "permanent":
            return self._triggers < self.cfg.steps
        if t == "intermittent":
            return self._triggers % self.cfg.period == 0
        return True

    def mark_fired(self, step=None) -> None:
        self._triggers += 1
        self._fired_any = True
        if step is not None:
            if self._start_step is None:
                self._start_step = step
            self._last_fired_step = step
