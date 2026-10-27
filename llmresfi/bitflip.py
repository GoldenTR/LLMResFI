# llmresfi/bitflip.py
"""IEEE-754 bit-level manipulation engine.

Unlike PyTorchFI's quantization-based flip, LLMResFI performs true IEEE bit
flips by reinterpreting the floating-point tensor as its integer twin and
XOR-ing a mask, exactly as done by NVBitFI at the register level.
"""
from __future__ import annotations

from typing import List

import torch

_BIT_FLIP_TYPES = {"single", "double", "multi", "sign", "exponent", "mantissa", "random"}


class BitFlipper:
    # float dtype -> integer view dtype (same bit width)
    _DTYPE_VIEW = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
        torch.float64: torch.int64,
    }

    # (sign, exponent, mantissa) bit counts, total = bit width
    _IEEE_LAYOUT = {
        torch.float16: (1, 5, 10),
        torch.bfloat16: (1, 8, 7),
        torch.float32: (1, 8, 23),
        torch.float64: (1, 11, 52),
    }

    @classmethod
    def view_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        dv = cls._DTYPE_VIEW.get(dtype)
        if dv is None:
            raise ValueError(f"Unsupported dtype for bit flip: {dtype}")
        return dv

    @classmethod
    def layout(cls, dtype: torch.dtype):
        try:
            return cls._IEEE_LAYOUT[dtype]
        except KeyError:
            raise ValueError(f"Unsupported dtype for bit flip: {dtype}")

    @classmethod
    def total_bits(cls, dtype: torch.dtype) -> int:
        sign, exp, man = cls.layout(dtype)
        return sign + exp + man

    @classmethod
    def region_bits(cls, dtype: torch.dtype, region: str):
        """Return the list of bit indices belonging to a region.

        Bit numbering: 0 = LSB (fraction bit 0), top bit = sign (MSB),
        matching the in-memory IEEE-754 layout when viewed as an integer.
        """
        sign, exp, man = cls.layout(dtype)
        if region == "mantissa":
            return list(range(0, man))
        if region == "exponent":
            return list(range(man, man + exp))
        if region == "sign":
            return list(range(man + exp, man + exp + sign))
        raise ValueError(f"Unknown region '{region}' (expected sign/exponent/mantissa)")

    @classmethod
    def flip(cls, tensor_elem: torch.Tensor, bit_idxs: List[int]) -> torch.Tensor:
        """Flip the given bit indices in a scalar tensor element."""
        dv = cls._DTYPE_VIEW.get(tensor_elem.dtype, torch.int32)
        with torch.no_grad():
            iview = tensor_elem.view(dv)
            mask = 0
            for b in bit_idxs:
                mask ^= 1 << b
            iview ^= mask
            return iview.view(tensor_elem.dtype)

    @classmethod
    def stuck_at(cls, tensor_elem: torch.Tensor, bit_idx: int, value: int) -> torch.Tensor:
        """Force a single bit to 0 or 1."""
        dv = cls._DTYPE_VIEW.get(tensor_elem.dtype, torch.int32)
        with torch.no_grad():
            iview = tensor_elem.view(dv)
            if value == 0:
                iview &= ~(1 << bit_idx)
            else:
                iview |= 1 << bit_idx
            return iview.view(tensor_elem.dtype)
