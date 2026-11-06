import random

import pytest
import torch

from llmresfi.bitflip import BitFlipper
from llmresfi.fault_models import apply_fault


def test_layout_fp32():
    assert BitFlipper.layout(torch.float32) == (1, 8, 23)
    assert BitFlipper.total_bits(torch.float32) == 32
    assert BitFlipper.region_bits(torch.float32, "mantissa") == list(range(23))
    assert BitFlipper.region_bits(torch.float32, "exponent") == list(range(23, 31))
    assert BitFlipper.region_bits(torch.float32, "sign") == [31]


def test_flip_exponent_bit_float32():
    # 1.0f = 0x3F800000; flipping bit 23 -> 0x3F000000 = 0.5f
    x = torch.tensor(1.0, dtype=torch.float32)
    y = BitFlipper.flip(x, [23])
    assert float(y.item()) == 0.5


def test_flip_sign_bit_float32():
    # flipping bit 31 -> negative
    x = torch.tensor(1.0, dtype=torch.float32)
    y = BitFlipper.flip(x, BitFlipper.region_bits(torch.float32, "sign"))
    assert float(y.item()) == -1.0


def test_flip_mantissa_bit_float32():
    # 1.0f = 0x3F800000; flipping bit 0 -> 0x3F800001
    x = torch.tensor(1.0, dtype=torch.float32)
    y = BitFlipper.flip(x, [0])
    assert float(y.item()) == float(torch.tensor(1.0 + 2 ** -23, dtype=torch.float32))


def test_stuck_at_sign_bit():
    x = torch.tensor(1.0, dtype=torch.float32)
    y = BitFlipper.stuck_at(x, 31, 1)
    assert float(y.item()) == -1.0


def test_apply_fault_honors_bit_idx():
    x = torch.tensor(1.0, dtype=torch.float32)
    y, bits = apply_fault(x, "exponent", random.Random(0), bit_idx=23)
    assert bits == [23]
    assert float(y.item()) == 0.5


def test_apply_fault_rejects_bit_outside_region():
    x = torch.tensor(1.0, dtype=torch.float32)
    with pytest.raises(ValueError, match="not in the exponent region"):
        apply_fault(x, "exponent", random.Random(0), bit_idx=0)
