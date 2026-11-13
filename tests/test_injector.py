import random

import pytest
import torch

from llmresfi.bit_utils import BitType, get_float_format, tensor_bit_pattern
from llmresfi.exceptions import (
    InvalidBitPositionError,
    InvalidFlatIndexError,
    UnsafeInjectionError,
    UnsupportedDTypeError,
)
from llmresfi.injector import (
    BitFlipSpec,
    inject_bit_flip,
    inject_bit_flips,
    inject_random_bit_flips,
    plan_random_injections,
)


DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def representative_bits(dtype):
    fmt = get_float_format(dtype)
    return {
        BitType.MANTISSA: (0, fmt.mantissa_msb),
        BitType.EXPONENT: (fmt.exponent_lsb, fmt.exponent_msb),
        BitType.SIGN: (fmt.sign_bit,),
    }


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, -2.5, float("inf"), float("nan")])
def test_each_bit_type_flips_exactly_one_bit(dtype, value, device):
    tensor = torch.tensor([value, 3.0], dtype=dtype, device=device)
    original_pattern = tensor_bit_pattern(tensor[0])
    bits = representative_bits(dtype)
    for expected_type, positions in bits.items():
        for bit_position in positions:
            result = inject_bit_flip(tensor, 0, bit_position)
            mask = 1 << bit_position
            assert result.injection_success
            assert result.bit_type is expected_type
            assert result.original_bit_pattern == original_pattern
            assert result.corrupted_bit_pattern == original_pattern ^ mask
            assert result.original_bit_pattern ^ result.corrupted_bit_pattern == mask
            assert result.only_target_bit_changed
            assert result.target_bit_changed
            assert result.dtype == dtype
            assert result.tensor.device == tensor.device
            assert result.tensor.shape == tensor.shape
            assert tensor_bit_pattern(tensor[0]) == original_pattern


@pytest.mark.parametrize("dtype", DTYPES)
def test_flipping_same_bit_restores_pattern(dtype, device):
    tensor = torch.tensor([1.2345, -7.5], dtype=dtype, device=device)
    fmt = get_float_format(dtype)
    for bit in (0, fmt.mantissa_msb, fmt.exponent_lsb, fmt.sign_bit):
        first = inject_bit_flip(tensor, 1, bit)
        second = inject_bit_flip(first.tensor, 1, bit)
        assert second.corrupted_bit_pattern == first.original_bit_pattern
        assert tensor_bit_pattern(second.tensor[1]) == tensor_bit_pattern(tensor[1])


@pytest.mark.parametrize("dtype", DTYPES)
def test_inplace_and_non_inplace_semantics(dtype, device):
    tensor = torch.tensor([1.0, 2.0, 4.0], dtype=dtype, device=device)
    original_patterns = [tensor_bit_pattern(item) for item in tensor]

    result = inject_bit_flip(tensor, 1, 0, inplace=False)
    assert result.tensor is not tensor
    assert result.inplace is False
    assert [tensor_bit_pattern(item) for item in tensor] == original_patterns
    assert result.tensor.data_ptr() != tensor.data_ptr()

    inplace_result = inject_bit_flip(tensor, 1, 0, inplace=True)
    assert inplace_result.tensor is tensor
    assert inplace_result.inplace is True
    assert tensor_bit_pattern(tensor[1]) == original_patterns[1] ^ 1
    assert tensor_bit_pattern(tensor[0]) == original_patterns[0]
    assert tensor_bit_pattern(tensor[2]) == original_patterns[2]


@pytest.mark.parametrize("dtype", DTYPES)
def test_boundary_bit_positions(dtype, device):
    total = get_float_format(dtype).total_bits
    tensor = torch.tensor([1.0], dtype=dtype, device=device)
    for bit in (0, total - 1):
        result = inject_bit_flip(tensor, 0, bit)
        assert result.only_target_bit_changed


@pytest.mark.parametrize("dtype", DTYPES)
def test_invalid_indices_and_bits(dtype, device):
    tensor = torch.ones(3, dtype=dtype, device=device)
    for index in (-1, 3, 100):
        with pytest.raises(InvalidFlatIndexError):
            inject_bit_flip(tensor, index, 0)
    with pytest.raises(InvalidFlatIndexError):
        inject_bit_flip(tensor, True, 0)
    total = get_float_format(dtype).total_bits
    for bit in (-1, total):
        with pytest.raises(InvalidBitPositionError):
            inject_bit_flip(tensor, 0, bit)
    with pytest.raises(InvalidBitPositionError):
        inject_bit_flip(tensor, 0, 0.5)  # type: ignore[arg-type]


def test_unsupported_tensor_dtypes(device):
    for tensor in (
        torch.ones(3, dtype=torch.float64, device=device),
        torch.ones(3, dtype=torch.int32, device=device),
        torch.ones(3, dtype=torch.uint8, device=device),
    ):
        with pytest.raises(UnsupportedDTypeError):
            inject_bit_flip(tensor, 0, 0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_noncontiguous_tensor_inplace_updates_original_storage(dtype, device):
    base = torch.arange(12, dtype=dtype, device=device).reshape(3, 4)
    view = base.t()
    assert not view.is_contiguous()
    before_base = base.clone()
    expected_logical_original = view.flatten()[5].clone()
    result = inject_bit_flip(view, 5, 0, inplace=True)
    assert result.tensor is view
    assert torch.equal(view.flatten()[5], result.tensor.flatten()[5])
    assert not torch.equal(view.flatten()[5], expected_logical_original)
    assert int((base != before_base).sum().item()) == 1


@pytest.mark.parametrize("dtype", DTYPES)
def test_noncontiguous_non_inplace_does_not_modify_base(dtype, device):
    base = torch.arange(12, dtype=dtype, device=device).reshape(3, 4)
    view = base[:, ::2]
    before = base.clone()
    result = inject_bit_flip(view, 3, 1, inplace=False)
    assert torch.equal(base, before)
    assert result.tensor.shape == view.shape
    assert result.tensor.dtype == dtype
    assert result.original_bit_pattern == tensor_bit_pattern(before[:, ::2].flatten()[3])


@pytest.mark.parametrize("dtype", DTYPES)
def test_slice_shared_storage_inplace(dtype, device):
    base = torch.arange(20, dtype=dtype, device=device).reshape(4, 5)
    view = base[1:3, 1:5:2]
    before = base.clone()
    result = inject_bit_flip(view, 2, 0, inplace=True)
    changed = (base != before).nonzero(as_tuple=False)
    assert changed.shape[0] == 1
    assert result.flat_index == 2


def test_scalar_tensor_and_event_context(device):
    scalar = torch.tensor(1.0, dtype=torch.float16, device=device)
    result = inject_bit_flip(
        scalar,
        0,
        15,
        parameter_version=3,
        injection_phase="before_forward",
    )
    assert result.tensor.shape == ()
    assert result.corrupted_value == -1.0
    assert result.parameter_version == 3
    assert result.injection_phase == "before_forward"
    assert result.injection_timestamp
    assert result.tensor_version_before is not None
    assert result.tensor_version_after is not None
    with pytest.raises(InvalidFlatIndexError):
        inject_bit_flip(scalar, 1, 0)


def test_multi_bit_injection_records_each_event(device):
    tensor = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32, device=device)
    specs = [BitFlipSpec(0, 0), (1, 23), BitFlipSpec(2, 31)]
    result = inject_bit_flips(tensor, specs, inplace=False)
    assert len(result) == 3
    for item, spec in zip(result, specs):
        flat_index, bit_position = (spec.flat_index, spec.bit_position) if isinstance(spec, BitFlipSpec) else spec
        assert item.flat_index == flat_index
        assert item.bit_position == bit_position
        assert item.only_target_bit_changed
    assert torch.equal(
        tensor, torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32, device=device)
    )
    with pytest.raises(Exception):
        inject_bit_flips(tensor, [(0, 0), (99, 0)])


@pytest.mark.parametrize("dtype", DTYPES)
def test_random_plan_is_deterministic_and_unique(dtype, device):
    tensor = torch.ones(256, dtype=dtype, device=device)
    first = plan_random_injections(tensor, count=20, seed=42)
    second = plan_random_injections(tensor, count=20, seed=42)
    assert first == second
    assert len({spec.flat_index for spec in first}) == 20
    assert all(0 <= spec.flat_index < tensor.numel() for spec in first)


def test_random_plan_restricts_bit_type_and_validates_count(device):
    tensor = torch.ones(8, dtype=torch.bfloat16, device=device)
    plan = plan_random_injections(tensor, count=8, seed=5, bit_type="exponent")
    assert all(7 <= spec.bit_position <= 14 for spec in plan)
    with pytest.raises(ValueError):
        plan_random_injections(tensor, count=9, seed=5)
    with pytest.raises(ValueError):
        plan_random_injections(tensor, count=-1, seed=5)
    assert plan_random_injections(tensor, count=0, seed=5) == []


def test_random_injection_is_deterministic_and_local(device):
    tensor = torch.arange(64, dtype=torch.float32, device=device)
    random.seed(123)
    expected = random.random()
    random.seed(123)
    first = inject_random_bit_flips(tensor, count=5, seed=9)
    actual = random.random()
    second = inject_random_bit_flips(tensor, count=5, seed=9)
    assert [(r.flat_index, r.bit_position) for r in first] == [
        (r.flat_index, r.bit_position) for r in second
    ]
    assert actual == expected


def test_inplace_injection_refuses_live_nonleaf_autograd_tensor(device):
    x = torch.tensor(
        [1.0, 2.0], dtype=torch.float32, device=device, requires_grad=True
    )
    activation = x * 2.0
    with pytest.raises(UnsafeInjectionError):
        inject_bit_flip(activation, 0, 0, inplace=True)
    result = inject_bit_flip(activation, 0, 0, inplace=False)
    assert result.tensor is not activation


def test_inplace_leaf_parameter_is_supported(device):
    parameter = torch.nn.Parameter(
        torch.tensor([1.0], dtype=torch.float32, device=device)
    )
    result = inject_bit_flip(parameter, 0, 0, inplace=True)
    assert result.tensor is parameter
    assert parameter.requires_grad
