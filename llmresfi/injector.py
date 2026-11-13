"""Tensor-level IEEE-754 bit-flip injection.

This module performs *software data corruption* on PyTorch tensors.  It does
not model physical GPU SRAM/cache/register upsets.  Fault locations and timing
must be chosen by the caller or by the deterministic helpers in this module.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

import torch

from .bit_utils import (
    BitType,
    FloatFormat,
    coerce_bit_type,
    get_bit_type,
    get_float_format,
)
from .exceptions import (
    InjectionVerificationError,
    InvalidBitPositionError,
    InvalidFlatIndexError,
    UnsafeInjectionError,
)


@dataclass(frozen=True)
class BitFlipSpec:
    """One logical fault location: flat element index plus bit index."""

    flat_index: int
    bit_position: int


@dataclass(frozen=True)
class InjectionResult:
    """Structured result of one independently traceable bit-flip event."""

    tensor: torch.Tensor
    flat_index: int
    bit_position: int
    bit_type: BitType
    original_value: float
    corrupted_value: float
    original_bit_pattern: int
    corrupted_bit_pattern: int
    dtype: torch.dtype
    device: torch.device
    tensor_shape: tuple[int, ...]
    inplace: bool
    injection_success: bool
    execution_time_ns: int
    injection_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tensor_version_before: int | None = None
    tensor_version_after: int | None = None
    parameter_version: int | str | None = None
    injection_phase: str | None = None
    tensor_name: str | None = None
    step: int | None = None
    random_seed: int | None = None

    @property
    def target_bit_changed(self) -> bool:
        return bool(
            (self.original_bit_pattern ^ self.corrupted_bit_pattern)
            & (1 << self.bit_position)
        )

    @property
    def only_target_bit_changed(self) -> bool:
        return (self.original_bit_pattern ^ self.corrupted_bit_pattern) == (
            1 << self.bit_position
        )


def _validate_flat_index(tensor: torch.Tensor, flat_index: int) -> None:
    if isinstance(flat_index, bool) or not isinstance(flat_index, int):
        raise InvalidFlatIndexError(
            f"flat_index must be an integer, got {type(flat_index).__name__}"
        )
    numel = tensor.numel()
    if flat_index < 0 or flat_index >= numel:
        raise InvalidFlatIndexError(
            f"flat_index {flat_index} is out of range for tensor with numel={numel}; "
            "negative indexing is not supported"
        )


def _validate_bit_position(fmt: FloatFormat, bit_position: int) -> None:
    if isinstance(bit_position, bool) or not isinstance(bit_position, int):
        raise InvalidBitPositionError(
            f"bit_position must be an integer, got {type(bit_position).__name__}"
        )
    if bit_position < 0 or bit_position >= fmt.total_bits:
        raise InvalidBitPositionError(
            f"bit_position {bit_position} is out of range for {fmt.name}; "
            f"valid range is [0, {fmt.total_bits - 1}]"
        )


def _logical_flat_coordinates(tensor: torch.Tensor, flat_index: int) -> tuple[int, ...]:
    """Convert a row-major logical flat index to a multidimensional index."""

    if tensor.dim() == 0:
        return ()
    remaining = flat_index
    coordinates = [0] * tensor.dim()
    for axis in range(tensor.dim() - 1, -1, -1):
        size = int(tensor.shape[axis])
        coordinates[axis] = remaining % size
        remaining //= size
    return tuple(coordinates)


def _select_flat_element(tensor: torch.Tensor, flat_index: int) -> torch.Tensor:
    """Return a zero-dimensional *view* of the logical flat element.

    Basic integer indexing returns a view, so an in-place bit operation reaches
    the original storage instead of a copied, contiguous reshape.
    """

    coordinates = _logical_flat_coordinates(tensor, flat_index)
    return tensor[coordinates]


def _signed_mask(mask: int, total_bits: int) -> int:
    """Represent an unsigned XOR mask as a two's-complement signed integer."""

    sign_threshold = 1 << (total_bits - 1)
    return mask if mask < sign_threshold else mask - (1 << total_bits)


def _flip_pattern(element: torch.Tensor, fmt: FloatFormat, bit_position: int) -> None:
    mask = 1 << bit_position
    view = element.view(fmt.int_view_dtype)
    # Signed integer XOR operates on the same two's-complement bit pattern as
    # the unsigned IEEE-754 word.  Using a signed mask avoids overflow when the
    # sign bit or FP32 exponent MSB is targeted.
    view.bitwise_xor_(_signed_mask(mask, fmt.total_bits))


def _sync_if_requested(tensor: torch.Tensor, synchronize: bool) -> None:
    if synchronize and tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor._version)
    except (AttributeError, RuntimeError):
        return None


def _inject_one(
    work: torch.Tensor,
    spec: BitFlipSpec,
    *,
    inplace: bool,
    tensor_name: str | None,
    step: int | None,
    random_seed: int | None,
    synchronize: bool,
    parameter_version: int | str | None,
    injection_phase: str | None,
) -> InjectionResult:
    _validate_flat_index(work, spec.flat_index)
    fmt = get_float_format(work)
    _validate_bit_position(fmt, spec.bit_position)

    if inplace and work.requires_grad and work.grad_fn is not None:
        raise UnsafeInjectionError(
            "Refusing an in-place fault injection into a non-leaf tensor with a live "
            "autograd graph because it would invalidate backward(). Inject after the "
            "relevant tensor has been consumed, use a persistent leaf parameter, or "
            "use inplace=False and feed the returned tensor into subsequent computation."
        )

    _sync_if_requested(work, synchronize)
    started = time.perf_counter_ns()
    injection_timestamp = datetime.now(timezone.utc).isoformat()
    version_before = _tensor_version(work)
    target = _select_flat_element(work, spec.flat_index)
    original_pattern = int(target.detach().reshape(1).view(fmt.int_view_dtype).item())
    original_pattern &= fmt.bit_mask
    original_value = float(target.detach().item())

    mask = 1 << spec.bit_position
    with torch.no_grad():
        _flip_pattern(target, fmt, spec.bit_position)

    corrupted_pattern = int(target.detach().reshape(1).view(fmt.int_view_dtype).item())
    corrupted_pattern &= fmt.bit_mask
    corrupted_value = float(target.detach().item())
    version_after = _tensor_version(work)
    expected_pattern = original_pattern ^ mask

    if corrupted_pattern != expected_pattern or (original_pattern ^ corrupted_pattern) != mask:
        # Do not leave an unverified in-place mutation behind.
        if inplace:
            with torch.no_grad():
                _flip_pattern(target, fmt, spec.bit_position)
        raise InjectionVerificationError(
            f"Bit-flip verification failed for dtype={fmt.name}, flat_index="
            f"{spec.flat_index}, bit_position={spec.bit_position}: expected pattern "
            f"0x{expected_pattern:x}, observed 0x{corrupted_pattern:x}"
        )

    _sync_if_requested(work, synchronize)
    elapsed_ns = time.perf_counter_ns() - started
    return InjectionResult(
        tensor=work,
        flat_index=spec.flat_index,
        bit_position=spec.bit_position,
        bit_type=get_bit_type(fmt.dtype, spec.bit_position),
        original_value=original_value,
        corrupted_value=corrupted_value,
        original_bit_pattern=original_pattern,
        corrupted_bit_pattern=corrupted_pattern,
        dtype=work.dtype,
        device=work.device,
        tensor_shape=tuple(work.shape),
        inplace=inplace,
        injection_success=True,
        execution_time_ns=elapsed_ns,
        injection_timestamp=injection_timestamp,
        tensor_version_before=version_before,
        tensor_version_after=version_after,
        parameter_version=parameter_version,
        injection_phase=injection_phase,
        tensor_name=tensor_name,
        step=step,
        random_seed=random_seed,
    )

def inject_bit_flip(
    tensor: torch.Tensor,
    flat_index: int,
    bit_position: int,
    inplace: bool = False,
    *,
    tensor_name: str | None = None,
    step: int | None = None,
    random_seed: int | None = None,
    parameter_version: int | str | None = None,
    injection_phase: str | None = None,
    synchronize: bool = False,
) -> InjectionResult:
    """Flip one IEEE-754 bit of one logical tensor element.

    Parameters
    ----------
    tensor:
        FP16, BF16, or FP32 tensor on CPU or CUDA.
    flat_index:
        Non-negative row-major logical element index, as produced by
        ``tensor.flatten()[index]``.
    bit_position:
        ``0`` is the LSB; the dtype's MSB is the sign bit.
    inplace:
        If ``False``, operate on a clone and leave ``tensor`` unchanged.  If
        ``True``, mutate the original storage, including for slices and other
        non-contiguous views.

    Returns
    -------
    InjectionResult
        The affected tensor plus exact original/corrupted values and bit
        patterns.  The target bit is always verified after mutation.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)!r}")
    fmt = get_float_format(tensor)
    _validate_flat_index(tensor, flat_index)
    _validate_bit_position(fmt, bit_position)
    work = tensor if inplace else tensor.clone()
    return _inject_one(
        work,
        BitFlipSpec(flat_index=flat_index, bit_position=bit_position),
        inplace=inplace,
        tensor_name=tensor_name,
        step=step,
        random_seed=random_seed,
        synchronize=synchronize,
        parameter_version=parameter_version,
        injection_phase=injection_phase,
    )
def inject_bit_flips(
    tensor: torch.Tensor,
    specs: Sequence[BitFlipSpec | tuple[int, int]],
    inplace: bool = False,
    *,
    tensor_name: str | None = None,
    step: int | None = None,
    random_seed: int | None = None,
    parameter_version: int | str | None = None,
    injection_phase: str | None = None,
    synchronize: bool = False,
) -> list[InjectionResult]:
    """Apply multiple independently recorded bit flips to one tensor.

    All injections share one working tensor.  With ``inplace=False`` the
    returned results reference the same combined clone.  Every specification
    is validated before mutation, and every event is verified separately.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)!r}")
    fmt = get_float_format(tensor)
    normalized: list[BitFlipSpec] = []
    for item in specs:
        if isinstance(item, BitFlipSpec):
            spec = item
        else:
            try:
                flat_index, bit_position = item
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Each specification must be BitFlipSpec or (flat_index, bit_position)"
                ) from exc
            spec = BitFlipSpec(int(flat_index), int(bit_position))
        _validate_flat_index(tensor, spec.flat_index)
        _validate_bit_position(fmt, spec.bit_position)
        normalized.append(spec)

    if not normalized:
        return []

    work = tensor if inplace else tensor.clone()
    return [
        _inject_one(
            work,
            spec,
            inplace=inplace,
            tensor_name=tensor_name,
            step=step,
            random_seed=random_seed,
            synchronize=synchronize,
            parameter_version=parameter_version,
            injection_phase=injection_phase,
        )
        for spec in normalized
    ]


def plan_random_injections(
    tensor: torch.Tensor,
    count: int = 1,
    seed: int | None = None,
    bit_type: BitType | str | None = None,
) -> list[BitFlipSpec]:
    """Create a reproducible random fault plan without touching the tensor.

    One logical element is selected at most once.  The RNG is local and no
    Python/PyTorch global random state is read or modified.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)!r}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"count must be a non-negative integer, got {count!r}")
    if count > tensor.numel():
        raise ValueError(
            f"Cannot select {count} unique elements from a tensor with numel={tensor.numel()}"
        )
    if count == 0:
        return []

    fmt = get_float_format(tensor)
    positions = (
        tuple(range(fmt.total_bits))
        if bit_type is None
        else fmt.bit_positions(coerce_bit_type(bit_type))
    )
    rng = random.Random(seed)
    flat_indices = rng.sample(range(tensor.numel()), count)
    return [
        BitFlipSpec(flat_index=int(index), bit_position=int(rng.choice(positions)))
        for index in flat_indices
    ]


def inject_random_bit_flips(
    tensor: torch.Tensor,
    count: int = 1,
    seed: int | None = None,
    bit_type: BitType | str | None = None,
    inplace: bool = False,
    *,
    tensor_name: str | None = None,
    step: int | None = None,
    parameter_version: int | str | None = None,
    injection_phase: str | None = None,
    synchronize: bool = False,
) -> list[InjectionResult]:
    """Plan and apply deterministic random bit flips to distinct elements."""

    specs = plan_random_injections(tensor, count=count, seed=seed, bit_type=bit_type)
    return inject_bit_flips(
        tensor,
        specs,
        inplace=inplace,
        tensor_name=tensor_name,
        step=step,
        random_seed=seed,
        synchronize=synchronize,
        parameter_version=parameter_version,
        injection_phase=injection_phase,
    )
