"""IEEE-754 format metadata and bit-region utilities.

Bit numbering is uniform across all supported dtypes:

* bit ``0`` is the least-significant bit (LSB) of the stored word;
* the mantissa occupies the low bits;
* the exponent occupies the next bits;
* the sign bit is the most-significant bit (MSB).

The functions in this module only inspect metadata and integer bit patterns;
they never convert floating-point *values* to integers for fault injection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Union

import torch

from .exceptions import InvalidBitPositionError, UnsupportedDTypeError


class BitType(str, Enum):
    """IEEE-754 field containing a bit position."""

    SIGN = "sign"
    EXPONENT = "exponent"
    MANTISSA = "mantissa"


@dataclass(frozen=True)
class FloatFormat:
    """Description of a supported floating-point storage format."""

    name: str
    dtype: torch.dtype
    sign_bits: int
    exponent_bits: int
    mantissa_bits: int
    int_view_dtype: torch.dtype

    @property
    def total_bits(self) -> int:
        return self.sign_bits + self.exponent_bits + self.mantissa_bits

    @property
    def bit_mask(self) -> int:
        return (1 << self.total_bits) - 1

    @property
    def sign_bit(self) -> int:
        return self.total_bits - 1

    @property
    def mantissa_lsb(self) -> int:
        return 0

    @property
    def mantissa_msb(self) -> int:
        return self.mantissa_bits - 1

    @property
    def exponent_lsb(self) -> int:
        return self.mantissa_bits

    @property
    def exponent_msb(self) -> int:
        return self.mantissa_bits + self.exponent_bits - 1

    def bit_type(self, bit_position: int) -> BitType:
        _validate_bit_position(self, bit_position)
        if bit_position >= self.exponent_lsb:
            return BitType.EXPONENT if bit_position < self.sign_bit else BitType.SIGN
        return BitType.MANTISSA

    def bit_positions(self, bit_type: Union["BitType", str]) -> tuple[int, ...]:
        kind = coerce_bit_type(bit_type)
        if kind is BitType.MANTISSA:
            return tuple(range(0, self.mantissa_bits))
        if kind is BitType.EXPONENT:
            return tuple(range(self.exponent_lsb, self.exponent_msb + 1))
        return (self.sign_bit,)


_FORMATS: dict[torch.dtype, FloatFormat] = {
    torch.float16: FloatFormat(
        name="float16",
        dtype=torch.float16,
        sign_bits=1,
        exponent_bits=5,
        mantissa_bits=10,
        int_view_dtype=torch.int16,
    ),
    torch.bfloat16: FloatFormat(
        name="bfloat16",
        dtype=torch.bfloat16,
        sign_bits=1,
        exponent_bits=8,
        mantissa_bits=7,
        int_view_dtype=torch.int16,
    ),
    torch.float32: FloatFormat(
        name="float32",
        dtype=torch.float32,
        sign_bits=1,
        exponent_bits=8,
        mantissa_bits=23,
        int_view_dtype=torch.int32,
    ),
}

SUPPORTED_FLOAT_DTYPES: tuple[torch.dtype, ...] = tuple(_FORMATS)

_DTYPE_NAMES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "torch.float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "torch.float32": torch.float32,
}


def resolve_float_dtype(value: torch.dtype | str) -> torch.dtype:
    """Resolve a supported dtype from a torch dtype or common string alias."""

    if isinstance(value, torch.dtype):
        get_float_format(value)
        return value
    if isinstance(value, str):
        try:
            return _DTYPE_NAMES[value.strip().lower()]
        except KeyError as exc:
            aliases = ", ".join(sorted(_DTYPE_NAMES))
            raise UnsupportedDTypeError(
                f"Unsupported dtype name {value!r}; accepted aliases: {aliases}"
            ) from exc
    raise UnsupportedDTypeError(
        f"dtype must be a torch.dtype or string, got {type(value)!r}"
    )


def coerce_bit_type(bit_type: Union[BitType, str]) -> BitType:
    """Convert a string such as ``"exponent"`` to :class:`BitType`."""

    if isinstance(bit_type, BitType):
        return bit_type
    if not isinstance(bit_type, str):
        raise TypeError(f"bit_type must be BitType or str, got {type(bit_type)!r}")
    try:
        return BitType(bit_type.lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in BitType)
        raise ValueError(f"Unknown bit type {bit_type!r}; expected one of: {valid}") from exc


def _coerce_dtype(dtype_or_tensor: Union[torch.dtype, torch.Tensor]) -> torch.dtype:
    if isinstance(dtype_or_tensor, torch.Tensor):
        return dtype_or_tensor.dtype
    if not isinstance(dtype_or_tensor, torch.dtype):
        raise UnsupportedDTypeError(
            f"Expected a torch.dtype or Tensor, got {type(dtype_or_tensor)!r}"
        )
    return dtype_or_tensor


def get_float_format(dtype_or_tensor: Union[torch.dtype, torch.Tensor]) -> FloatFormat:
    """Return IEEE-754 layout metadata for FP16, BF16, or FP32.

    Parameters
    ----------
    dtype_or_tensor:
        A supported ``torch.dtype`` or any tensor whose dtype is supported.
    """

    dtype = _coerce_dtype(dtype_or_tensor)
    try:
        return _FORMATS[dtype]
    except KeyError as exc:
        supported = ", ".join(str(item) for item in SUPPORTED_FLOAT_DTYPES)
        raise UnsupportedDTypeError(
            f"Unsupported dtype {dtype}; supported floating dtypes are: {supported}"
        ) from exc


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


def get_bit_type(
    dtype_or_tensor: Union[torch.dtype, torch.Tensor], bit_position: int
) -> BitType:
    """Classify ``bit_position`` as ``sign``, ``exponent``, or ``mantissa``."""

    fmt = get_float_format(dtype_or_tensor)
    return fmt.bit_type(bit_position)


def bit_positions_for_type(
    dtype_or_tensor: Union[torch.dtype, torch.Tensor],
    bit_type: Union[BitType, str],
) -> tuple[int, ...]:
    """Return all bit indices belonging to one IEEE-754 field."""

    return get_float_format(dtype_or_tensor).bit_positions(bit_type)


def random_bit_position(
    dtype_or_tensor: Union[torch.dtype, torch.Tensor],
    bit_type: Union[BitType, str, None] = None,
    rng: random.Random | None = None,
) -> int:
    """Choose a deterministic bit index using the supplied local RNG.

    Passing an explicit :class:`random.Random` instance keeps experiment
    reproducibility independent from Python's global random state.
    """

    local_rng = rng if rng is not None else random.Random()
    fmt = get_float_format(dtype_or_tensor)
    positions = (
        tuple(range(fmt.total_bits))
        if bit_type is None
        else fmt.bit_positions(bit_type)
    )
    return int(local_rng.choice(positions))


def tensor_bit_pattern(tensor: torch.Tensor) -> int:
    """Return the unsigned integer bit pattern of a one-element float tensor."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor)!r}")
    if tensor.numel() != 1:
        raise ValueError(f"tensor_bit_pattern expects one element, got {tensor.numel()}")
    fmt = get_float_format(tensor)
    # A one-element reshape is a view even for unusual strides.  Viewing as an
    # integer preserves the exact in-memory IEEE-754 representation.
    int_view = tensor.detach().reshape(1).view(fmt.int_view_dtype)
    return int(int_view.item()) & fmt.bit_mask


def pattern_to_binary(pattern: int, dtype_or_tensor: Union[torch.dtype, torch.Tensor]) -> str:
    """Format an unsigned bit pattern as a fixed-width binary string."""

    fmt = get_float_format(dtype_or_tensor)
    if not isinstance(pattern, int) or pattern < 0 or pattern > fmt.bit_mask:
        raise ValueError(f"pattern must fit in {fmt.total_bits} bits, got {pattern!r}")
    return f"{pattern:0{fmt.total_bits}b}"

