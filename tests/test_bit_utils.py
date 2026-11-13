import random

import pytest
import torch

from llmresfi.bit_utils import (
    BitType,
    bit_positions_for_type,
    coerce_bit_type,
    get_bit_type,
    get_float_format,
    pattern_to_binary,
    random_bit_position,
    resolve_float_dtype,
    tensor_bit_pattern,
)
from llmresfi.exceptions import InvalidBitPositionError, UnsupportedDTypeError


LAYOUTS = [
    (torch.float16, "float16", 1, 5, 10, torch.int16, 0x3C00),
    (torch.bfloat16, "bfloat16", 1, 8, 7, torch.int16, 0x3F80),
    (torch.float32, "float32", 1, 8, 23, torch.int32, 0x3F800000),
]


@pytest.mark.parametrize("dtype,name,sign,exponent,mantissa,view_dtype,one_pattern", LAYOUTS)
def test_float_format_layout(dtype, name, sign, exponent, mantissa, view_dtype, one_pattern):
    fmt = get_float_format(dtype)
    assert fmt.name == name
    assert fmt.dtype == dtype
    assert fmt.sign_bits == sign
    assert fmt.exponent_bits == exponent
    assert fmt.mantissa_bits == mantissa
    assert fmt.total_bits == sign + exponent + mantissa
    assert fmt.int_view_dtype == view_dtype
    assert fmt.bit_mask == (1 << fmt.total_bits) - 1
    assert tensor_bit_pattern(torch.tensor(1.0, dtype=dtype)) == one_pattern


@pytest.mark.parametrize("dtype,_name,_s,_e,_m,_view,_one", LAYOUTS)
def test_bit_type_boundaries(dtype, _name, _s, _e, _m, _view, _one):
    fmt = get_float_format(dtype)
    assert get_bit_type(dtype, 0) is BitType.MANTISSA
    assert get_bit_type(dtype, fmt.mantissa_msb) is BitType.MANTISSA
    assert get_bit_type(dtype, fmt.exponent_lsb) is BitType.EXPONENT
    assert get_bit_type(dtype, fmt.exponent_msb) is BitType.EXPONENT
    assert get_bit_type(dtype, fmt.sign_bit) is BitType.SIGN
    assert len(bit_positions_for_type(dtype, BitType.MANTISSA)) == fmt.mantissa_bits
    assert len(bit_positions_for_type(dtype, BitType.EXPONENT)) == fmt.exponent_bits
    assert bit_positions_for_type(dtype, "sign") == (fmt.total_bits - 1,)


def test_expected_half_layout_ranges():
    assert bit_positions_for_type(torch.float16, "mantissa") == tuple(range(0, 10))
    assert bit_positions_for_type(torch.float16, "exponent") == tuple(range(10, 15))
    assert bit_positions_for_type(torch.float16, "sign") == (15,)
    assert bit_positions_for_type(torch.bfloat16, "mantissa") == tuple(range(0, 7))
    assert bit_positions_for_type(torch.bfloat16, "exponent") == tuple(range(7, 15))
    assert bit_positions_for_type(torch.float32, "exponent") == tuple(range(23, 31))


@pytest.mark.parametrize("dtype,_name,_s,_e,_m,_view,_one", LAYOUTS)
def test_invalid_bit_positions(dtype, _name, _s, _e, _m, _view, _one):
    total = get_float_format(dtype).total_bits
    for bad in (-1, total):
        with pytest.raises(InvalidBitPositionError):
            get_bit_type(dtype, bad)
    with pytest.raises(InvalidBitPositionError):
        get_bit_type(dtype, True)
    with pytest.raises(InvalidBitPositionError):
        get_bit_type(dtype, 1.0)


def test_unsupported_dtypes_are_explicit():
    with pytest.raises(UnsupportedDTypeError):
        get_float_format(torch.float64)
    with pytest.raises(UnsupportedDTypeError):
        get_float_format(torch.int32)
    with pytest.raises(UnsupportedDTypeError):
        resolve_float_dtype("complex64")


def test_dtype_aliases():
    assert resolve_float_dtype("fp16") is torch.float16
    assert resolve_float_dtype("BF16") is torch.bfloat16
    assert resolve_float_dtype("torch.float32") is torch.float32
    assert resolve_float_dtype(torch.float16) is torch.float16


def test_bit_type_coercion():
    assert coerce_bit_type("SIGN") is BitType.SIGN
    assert coerce_bit_type(BitType.EXPONENT) is BitType.EXPONENT
    with pytest.raises(ValueError):
        coerce_bit_type("leading")


def test_tensor_bit_pattern_requires_scalar():
    with pytest.raises(ValueError):
        tensor_bit_pattern(torch.ones(2, dtype=torch.float16))
    with pytest.raises(TypeError):
        tensor_bit_pattern([1.0])  # type: ignore[arg-type]


@pytest.mark.parametrize("dtype,_name,_s,_e,_m,_view,_one", LAYOUTS)
def test_special_values_have_stable_patterns(dtype, _name, _s, _e, _m, _view, _one):
    values = [0.0, -0.0, float("inf"), float("-inf"), float("nan")]
    patterns = [tensor_bit_pattern(torch.tensor(value, dtype=dtype)) for value in values]
    assert len(set(patterns)) == len(patterns)
    assert patterns[0] == 0
    assert patterns[1] == (1 << (get_float_format(dtype).total_bits - 1))


def test_pattern_to_binary_width_and_validation():
    assert pattern_to_binary(0, torch.float16) == "0" * 16
    assert pattern_to_binary(0x3C00, torch.float16) == "0011110000000000"
    with pytest.raises(ValueError):
        pattern_to_binary(1 << 16, torch.float16)


def test_random_bit_position_is_local_and_reproducible():
    random.seed(12345)
    expected_global_next = random.random()
    random.seed(12345)
    first = [random_bit_position(torch.float16, rng=random.Random(7)) for _ in range(8)]
    actual_global_next = random.random()
    second = [random_bit_position(torch.float16, rng=random.Random(7)) for _ in range(8)]
    assert first == second
    assert actual_global_next == expected_global_next
    assert all(0 <= bit < 16 for bit in first)


def test_random_bit_position_region_restriction():
    rng = random.Random(11)
    positions = [random_bit_position(torch.bfloat16, "exponent", rng) for _ in range(50)]
    assert set(positions) <= set(range(7, 15))

