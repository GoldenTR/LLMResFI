"""Selection of model parameters or caller-provided tensors for injection."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable

import torch

from .bit_utils import resolve_float_dtype
from .exceptions import TargetSelectionError


@dataclass(frozen=True)
class TargetTensor:
    """A selected tensor plus the metadata needed for reproducible audits.

    ``name`` is a logical model path and is deliberately kept separate from
    storage identity fields (``storage_ptr``/``storage_offset``).
    """

    name: str
    tensor: torch.Tensor

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def numel(self) -> int:
        return self.tensor.numel()

    @property
    def stride(self) -> tuple[int, ...]:
        return tuple(self.tensor.stride())

    @property
    def storage_offset(self) -> int:
        return int(self.tensor.storage_offset())

    @property
    def storage_ptr(self) -> int:
        try:
            return int(self.tensor.untyped_storage().data_ptr())
        except (RuntimeError, NotImplementedError):
            return 0

    @property
    def is_contiguous(self) -> bool:
        return bool(self.tensor.is_contiguous())

    @property
    def requires_grad(self) -> bool:
        return bool(self.tensor.requires_grad)

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "numel": self.numel,
            "stride": self.stride,
            "storage_offset": self.storage_offset,
            "storage_ptr": self.storage_ptr,
            "is_contiguous": self.is_contiguous,
            "requires_grad": self.requires_grad,
        }


def _iter_named_parameters(model: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    try:
        return model.named_parameters()
    except AttributeError as exc:
        raise TargetSelectionError(
            "model must be a torch.nn.Module, or pass a torch.Tensor directly"
        ) from exc


def list_target_tensors(
    model: torch.nn.Module | torch.Tensor,
    dtype: torch.dtype | str | None = None,
    name_pattern: str | None = None,
) -> list[TargetTensor]:
    """Enumerate eligible tensors without random selection."""

    resolved_dtype = resolve_float_dtype(dtype) if dtype is not None else None
    if isinstance(model, torch.Tensor):
        candidates = [TargetTensor(name="<tensor>", tensor=model)]
    else:
        candidates = [
            TargetTensor(name=name, tensor=tensor)
            for name, tensor in _iter_named_parameters(model)
        ]

    if resolved_dtype is not None:
        candidates = [item for item in candidates if item.dtype == resolved_dtype]
    if name_pattern is not None:
        try:
            pattern = re.compile(name_pattern)
        except re.error as exc:
            raise TargetSelectionError(f"Invalid name_pattern {name_pattern!r}: {exc}") from exc
        candidates = [item for item in candidates if pattern.search(item.name)]
    return candidates


def select_target_tensor(
    model: torch.nn.Module | torch.Tensor,
    target_name: str | None = None,
    dtype: torch.dtype | str | None = None,
    name_pattern: str | None = None,
    seed: int | None = None,
) -> TargetTensor:
    """Select exactly one tensor using name, dtype, regex, and local RNG.

    ``target_name`` is an exact logical parameter name.  If multiple eligible
    tensors remain, a local ``random.Random(seed)`` chooses one; Python's
    global random state is neither read nor modified.
    """

    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TargetSelectionError(f"seed must be an integer or None, got {seed!r}")

    direct_tensor = isinstance(model, torch.Tensor)
    if target_name is not None and not isinstance(target_name, str):
        raise TargetSelectionError("target_name must be a string or None")

    if direct_tensor:
        if target_name is not None:
            candidates = [TargetTensor(name=target_name, tensor=model)]
        else:
            candidates = [TargetTensor(name="<tensor>", tensor=model)]
        # Apply filters explicitly so a failed direct-tensor request is clear.
        if dtype is not None:
            resolved = resolve_float_dtype(dtype)
            if model.dtype != resolved:
                candidates = []
        if name_pattern is not None:
            try:
                matches = re.search(name_pattern, candidates[0].name) is not None
            except re.error as exc:
                raise TargetSelectionError(
                    f"Invalid name_pattern {name_pattern!r}: {exc}"
                ) from exc
            if not matches:
                candidates = []
    else:
        if target_name is not None:
            try:
                named = dict(model.named_parameters())
            except AttributeError as exc:
                raise TargetSelectionError(
                    "model must provide named_parameters(); alternatively pass a Tensor"
                ) from exc
            if target_name not in named:
                close = [name for name in named if target_name in name]
                hint = f" Similar names: {close[:5]}" if close else ""
                raise TargetSelectionError(
                    f"Model parameter {target_name!r} was not found.{hint}"
                )
            candidates = [TargetTensor(name=target_name, tensor=named[target_name])]
            if dtype is not None and candidates[0].dtype != resolve_float_dtype(dtype):
                candidates = []
            if name_pattern is not None:
                try:
                    matches = re.search(name_pattern, target_name) is not None
                except re.error as exc:
                    raise TargetSelectionError(
                        f"Invalid name_pattern {name_pattern!r}: {exc}"
                    ) from exc
                if not matches:
                    candidates = []
        else:
            candidates = list_target_tensors(model, dtype=dtype, name_pattern=name_pattern)

    if not candidates:
        filters = []
        if dtype is not None:
            filters.append(f"dtype={dtype}")
        if name_pattern is not None:
            filters.append(f"name_pattern={name_pattern!r}")
        if target_name is not None:
            filters.append(f"target_name={target_name!r}")
        detail = ", ".join(filters) if filters else "no filters"
        raise TargetSelectionError(f"No tensor matched selection criteria ({detail})")

    if len(candidates) == 1:
        return candidates[0]
    rng = random.Random(seed)
    return candidates[rng.randrange(len(candidates))]

