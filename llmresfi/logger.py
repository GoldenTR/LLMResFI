"""Audit-only fault event logging.

The logger is intentionally a separate component from injection execution and
from any future detection/recovery code.  Detection algorithms must consume
only model/training observables, never these ground-truth records or the
``InjectionResult`` returned to the injection driver.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ._version import __version__
from .bit_utils import coerce_bit_type
from .exceptions import LoggingError
from .ids import generate_experiment_id
from .injector import InjectionResult


REQUIRED_EVENT_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "fault_id",
    "step",
    "tensor_name",
    "tensor_shape",
    "tensor_dtype",
    "tensor_device",
    "flat_index",
    "bit_position",
    "bit_type",
    "original_value",
    "corrupted_value",
    "injection_timestamp",
    "random_seed",
    "injection_success",
    "failure_reason",
    "original_bit_pattern",
    "corrupted_bit_pattern",
    "injection_time_ns",
    "injection_timestamp_precise",
    "parameter_version",
    "injection_phase",
    "tensor_version_before",
    "tensor_version_after",
    "tensor_backup_path",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_float(value: float | None) -> float | str | None:
    if value is None:
        return None
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return float(value)


@dataclass
class FaultEvent:
    """One independently auditable fault event, successful or failed."""

    experiment_id: str
    fault_id: str
    step: int | None
    tensor_name: str | None
    tensor_shape: Sequence[int] | None
    tensor_dtype: str | None
    tensor_device: str | None
    flat_index: int | None
    bit_position: int | None
    bit_type: str | None
    original_value: float | None
    corrupted_value: float | None
    injection_timestamp: str = field(default_factory=_now_iso)
    random_seed: int | None = None
    injection_success: bool = True
    failure_reason: str | None = None
    original_bit_pattern: int | None = None
    corrupted_bit_pattern: int | None = None
    injection_time_ns: int | None = None
    injection_timestamp_precise: str | None = None
    parameter_version: int | str | None = None
    injection_phase: str | None = None
    tensor_version_before: int | None = None
    tensor_version_after: int | None = None
    tensor_backup_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_injection_result(
        cls,
        result: InjectionResult,
        *,
        experiment_id: str,
        fault_id: str,
        tensor_name: str | None = None,
        step: int | None = None,
        random_seed: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        parameter_version: int | str | None = None,
        injection_phase: str | None = None,
    ) -> "FaultEvent":
        return cls(
            experiment_id=experiment_id,
            fault_id=fault_id,
            step=result.step if step is None else step,
            tensor_name=tensor_name if tensor_name is not None else result.tensor_name,
            tensor_shape=result.tensor_shape,
            tensor_dtype=str(result.dtype),
            tensor_device=str(result.device),
            flat_index=result.flat_index,
            bit_position=result.bit_position,
            bit_type=result.bit_type.value,
            original_value=result.original_value,
            corrupted_value=result.corrupted_value,
            random_seed=result.random_seed if random_seed is None else random_seed,
            injection_success=result.injection_success,
            original_bit_pattern=result.original_bit_pattern,
            corrupted_bit_pattern=result.corrupted_bit_pattern,
            injection_time_ns=result.execution_time_ns,
            injection_timestamp=result.injection_timestamp,
            injection_timestamp_precise=result.injection_timestamp,
            parameter_version=(
                result.parameter_version
                if parameter_version is None
                else parameter_version
            ),
            injection_phase=(
                result.injection_phase if injection_phase is None else injection_phase
            ),
            tensor_version_before=result.tensor_version_before,
            tensor_version_after=result.tensor_version_after,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "fault_id": self.fault_id,
            "step": self.step,
            "tensor_name": self.tensor_name,
            "tensor_shape": list(self.tensor_shape) if self.tensor_shape is not None else None,
            "tensor_dtype": self.tensor_dtype,
            "tensor_device": self.tensor_device,
            "flat_index": self.flat_index,
            "bit_position": self.bit_position,
            "bit_type": self.bit_type,
            "original_value": _json_float(self.original_value),
            "corrupted_value": _json_float(self.corrupted_value),
            "injection_timestamp": self.injection_timestamp,
            "random_seed": self.random_seed,
            "injection_success": bool(self.injection_success),
            "failure_reason": self.failure_reason,
            "original_bit_pattern": self.original_bit_pattern,
            "corrupted_bit_pattern": self.corrupted_bit_pattern,
            "injection_time_ns": self.injection_time_ns,
            "injection_timestamp_precise": self.injection_timestamp_precise,
            "parameter_version": self.parameter_version,
            "injection_phase": self.injection_phase,
            "tensor_version_before": self.tensor_version_before,
            "tensor_version_after": self.tensor_version_after,
            "tensor_backup_path": self.tensor_backup_path,
            "metadata": dict(self.metadata),
        }


def _normalize_formats(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [value]
    else:
        raw = list(value)
    normalized: list[str] = []
    for item in raw:
        lowered = str(item).lower()
        if lowered == "both":
            for fmt in ("jsonl", "csv"):
                if fmt not in normalized:
                    normalized.append(fmt)
        elif lowered in {"jsonl", "csv"}:
            if lowered not in normalized:
                normalized.append(lowered)
        else:
            raise LoggingError(f"Unsupported log format {item!r}; use jsonl, csv, or both")
    if not normalized:
        raise LoggingError("At least one log format is required")
    return tuple(normalized)


def _environment_metadata(experiment_id: str, random_seed: int | None) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "experiment_id": experiment_id,
        "random_seed": random_seed,
        "tool": "LLMResFI",
        "tool_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": devices,
        "platform": platform.platform(),
    }


class EventLogger:
    """Append-only JSONL/CSV ground-truth logger for fault events."""

    def __init__(
        self,
        experiment_id: str | None,
        output_dir: str | Path,
        format: str | Sequence[str] = "jsonl",
        *,
        save_tensor_backup: bool = False,
        random_seed: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if experiment_id is None:
            experiment_id = generate_experiment_id()
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise LoggingError("experiment_id must be a non-empty string")
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = _normalize_formats(format)
        self.save_tensor_backup = bool(save_tensor_backup)
        self.random_seed = random_seed
        self.jsonl_path = self.output_dir / "fault_events.jsonl"
        self.csv_path = self.output_dir / "fault_events.csv"
        self.metadata_path = self.output_dir / "run_metadata.json"
        self._events: list[dict[str, Any]] = []
        self._fault_ids: set[str] = set()
        self._next_fault_number = 1
        self._lock = threading.Lock()

        run_metadata = _environment_metadata(experiment_id, random_seed)
        if metadata:
            run_metadata["caller_metadata"] = dict(metadata)
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(run_metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    def _next_fault_id(self) -> str:
        while True:
            fault_id = f"{self.experiment_id}-f{self._next_fault_number:06d}"
            self._next_fault_number += 1
            if fault_id not in self._fault_ids:
                return fault_id

    def _event_from_mapping(self, mapping: Mapping[str, Any]) -> FaultEvent:
        payload = {field_name: mapping.get(field_name) for field_name in REQUIRED_EVENT_FIELDS}
        payload["experiment_id"] = payload["experiment_id"] or self.experiment_id
        if payload["experiment_id"] != self.experiment_id:
            raise LoggingError(
                f"Event experiment_id {payload['experiment_id']!r} does not match logger "
                f"experiment_id {self.experiment_id!r}"
            )
        payload["fault_id"] = payload["fault_id"] or self._next_fault_id()
        payload["injection_timestamp"] = payload["injection_timestamp"] or _now_iso()
        if payload["bit_type"] is not None:
            payload["bit_type"] = coerce_bit_type(str(payload["bit_type"])).value
        if payload["tensor_shape"] is not None:
            payload["tensor_shape"] = tuple(payload["tensor_shape"])
        if payload["random_seed"] is None:
            payload["random_seed"] = self.random_seed
        if payload["injection_success"] is None:
            payload["injection_success"] = True
        if not bool(payload["injection_success"]) and not payload["failure_reason"]:
            raise LoggingError("Failed events must include a non-empty failure_reason")
        return FaultEvent(
            **payload,
            metadata=dict(mapping.get("metadata") or {}),
        )

    def _write_jsonl(self, event_dict: Mapping[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            json.dump(event_dict, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()

    def _write_csv(self, event_dict: Mapping[str, Any]) -> None:
        needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        row = dict(event_dict)
        row["tensor_shape"] = json.dumps(row.get("tensor_shape"), ensure_ascii=False)
        row["metadata"] = json.dumps(row.get("metadata") or {}, ensure_ascii=False, sort_keys=True)
        fields = list(REQUIRED_EVENT_FIELDS) + ["metadata"]
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()

    def log(
        self,
        event: FaultEvent | InjectionResult | Mapping[str, Any],
        *,
        tensor_name: str | None = None,
        step: int | None = None,
        random_seed: int | None = None,
        fault_id: str | None = None,
        original_tensor: torch.Tensor | None = None,
        corrupted_tensor: torch.Tensor | None = None,
        metadata: Mapping[str, Any] | None = None,
        parameter_version: int | str | None = None,
        injection_phase: str | None = None,
    ) -> dict[str, Any]:
        """Persist one event and return its fully normalized dictionary."""

        with self._lock:
            if isinstance(event, InjectionResult):
                assigned_id = fault_id or self._next_fault_id()
                fault_event = FaultEvent.from_injection_result(
                    event,
                    experiment_id=self.experiment_id,
                    fault_id=assigned_id,
                    tensor_name=tensor_name,
                    step=step,
                    random_seed=random_seed,
                    metadata=metadata,
                    parameter_version=parameter_version,
                    injection_phase=injection_phase,
                )
            elif isinstance(event, FaultEvent):
                fault_event = event
                if fault_event.experiment_id != self.experiment_id:
                    raise LoggingError(
                        "FaultEvent experiment_id does not match EventLogger experiment_id"
                    )
                if not fault_event.fault_id:
                    fault_event.fault_id = fault_id or self._next_fault_id()
            elif isinstance(event, Mapping):
                payload = dict(event)
                if fault_id is not None:
                    payload["fault_id"] = fault_id
                if tensor_name is not None:
                    payload["tensor_name"] = tensor_name
                if step is not None:
                    payload["step"] = step
                if random_seed is not None:
                    payload["random_seed"] = random_seed
                if parameter_version is not None:
                    payload["parameter_version"] = parameter_version
                if injection_phase is not None:
                    payload["injection_phase"] = injection_phase
                if metadata:
                    payload["metadata"] = {**dict(payload.get("metadata") or {}), **dict(metadata)}
                fault_event = self._event_from_mapping(payload)
            else:
                raise LoggingError(
                    f"event must be FaultEvent, InjectionResult, or Mapping, got {type(event)!r}"
                )

            if fault_event.fault_id in self._fault_ids:
                raise LoggingError(f"Duplicate fault_id {fault_event.fault_id!r}")
            self._fault_ids.add(fault_event.fault_id)

            if self.save_tensor_backup and original_tensor is not None:
                backup_dir = self.output_dir / "tensor_backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / f"{fault_event.fault_id}.pt"
                payload = {"original_tensor": original_tensor.detach().cpu().clone()}
                if corrupted_tensor is not None:
                    payload["corrupted_tensor"] = corrupted_tensor.detach().cpu().clone()
                torch.save(payload, backup_path)
                fault_event.tensor_backup_path = str(backup_path)

            event_dict = fault_event.to_dict()
            if "jsonl" in self.formats:
                self._write_jsonl(event_dict)
            if "csv" in self.formats:
                self._write_csv(event_dict)
            self._events.append(event_dict)
            return dict(event_dict)

    def log_failure(
        self,
        *,
        step: int | None,
        tensor_name: str | None,
        failure_reason: str,
        tensor_shape: Sequence[int] | None = None,
        tensor_dtype: str | None = None,
        tensor_device: str | None = None,
        flat_index: int | None = None,
        bit_position: int | None = None,
        bit_type: str | None = None,
        random_seed: int | None = None,
        parameter_version: int | str | None = None,
        injection_phase: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an attempted injection that did not complete successfully."""

        if not failure_reason:
            raise LoggingError("failure_reason must be a non-empty string")
        payload = {
            "experiment_id": self.experiment_id,
            "fault_id": None,
            "step": step,
            "tensor_name": tensor_name,
            "tensor_shape": tensor_shape,
            "tensor_dtype": tensor_dtype,
            "tensor_device": tensor_device,
            "flat_index": flat_index,
            "bit_position": bit_position,
            "bit_type": bit_type,
            "original_value": None,
            "corrupted_value": None,
            "injection_timestamp": _now_iso(),
            "random_seed": random_seed if random_seed is not None else self.random_seed,
            "injection_success": False,
            "failure_reason": failure_reason,
            "original_bit_pattern": None,
            "corrupted_bit_pattern": None,
            "injection_time_ns": None,
            "injection_timestamp_precise": None,
            "parameter_version": parameter_version,
            "injection_phase": injection_phase,
            "tensor_version_before": None,
            "tensor_version_after": None,
            "tensor_backup_path": None,
            "metadata": dict(metadata or {}),
        }
        return self.log(payload)

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._events]

    def to_dicts(self) -> list[dict[str, Any]]:
        return self.events

    def save_config_copy(self, config_path: str | Path) -> Path:
        source = Path(config_path)
        if not source.is_file():
            raise LoggingError(f"Configuration file does not exist: {source}")
        destination = self.output_dir / "config.yaml"
        shutil.copy2(source, destination)
        return destination

