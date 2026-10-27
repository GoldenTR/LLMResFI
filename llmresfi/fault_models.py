# llmresfi/fault_models.py
"""Fault model library: bit flips, value perturbation, and stuck-at faults."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .bitflip import BitFlipper

BIT_FLIP_TYPES = {"single", "double", "multi", "sign", "exponent", "mantissa", "random"}
VALUE_TYPES = {"random_value", "zero", "nan", "inf", "scale"}
STUCK_TYPES = {"stuck_at_0", "stuck_at_1"}


def pick_bit_idxs(fault_type: str, dtype: torch.dtype, rng) -> List[int]:
    """Select bit index/indices to flip for a bit-flip fault type."""
    total = BitFlipper.total_bits(dtype)
    if fault_type == "single":
        return [rng.randrange(total)]
    if fault_type == "double":
        return rng.sample(range(total), 2)
    if fault_type == "multi":
        return rng.sample(range(total), rng.randint(1, 4))
    if fault_type == "random":
        return rng.sample(range(total), rng.choice([1, 2]))
    if fault_type in ("sign", "exponent", "mantissa"):
        pool = BitFlipper.region_bits(dtype, fault_type)
        return [rng.choice(pool)]
    raise ValueError(f"Unknown bit-flip type: {fault_type}")


def apply_fault(
    value: torch.Tensor,
    fault_type: str,
    rng,
    bit_idx: Optional[int] = None,
    min_val: float = -1.0,
    max_val: float = 1.0,
    factor: float = 2.0,
) -> Tuple[torch.Tensor, Optional[List[int]]]:
    """Apply a fault to a scalar tensor ``value``.

    Returns ``(corrupted_value, bit_idxs)`` where ``bit_idxs`` is ``None`` for
    value-perturbation faults (no bit-level location).
    """
    if fault_type in BIT_FLIP_TYPES:
        bits = pick_bit_idxs(fault_type, value.dtype, rng)
        return BitFlipper.flip(value, bits), bits

    if fault_type in STUCK_TYPES:
        total = BitFlipper.total_bits(value.dtype)
        bit = bit_idx if bit_idx is not None else rng.randrange(total)
        val = 0 if fault_type == "stuck_at_0" else 1
        return BitFlipper.stuck_at(value, bit, val), [bit]

    if fault_type in VALUE_TYPES:
        return _value_fault(value, fault_type, rng, min_val, max_val, factor), None

    raise ValueError(f"Unknown fault model type: {fault_type}")


def _value_fault(value, fault_type, rng, min_val, max_val, factor):
    if fault_type == "random_value":
        return value.new_tensor(rng.uniform(min_val, max_val))
    if fault_type == "zero":
        return torch.zeros_like(value)
    if fault_type == "nan":
        return value.new_tensor(float("nan"))
    if fault_type == "inf":
        return value.new_tensor(float("inf"))
    if fault_type == "scale":
        return value * factor
    raise ValueError(f"Unknown value fault type: {fault_type}")
