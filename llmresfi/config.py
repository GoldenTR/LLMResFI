"""YAML experiment configuration and reproducibility snapshots."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from ._version import __version__
from .bit_utils import get_float_format, resolve_float_dtype
from .exceptions import (
    ConfigurationError,
    InvalidBitPositionError,
    UnsupportedDTypeError,
)
from .ids import generate_experiment_id
from .scheduler import FaultScheduler


def _require_mapping(value: Any, section: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Configuration section {section!r} must be a mapping")
    return value


def _check_unknown_keys(section: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown key(s) in {section}: {', '.join(unknown)}; allowed keys: "
            f"{', '.join(sorted(allowed))}"
        )


def _validate_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FaultConfig:
    enabled: bool = True
    target_type: str = "model_parameter"
    target_name: str | None = None
    dtype: torch.dtype = torch.float16
    injection_mode: str = "single_bit"
    bit_type: str | None = None
    injection_steps: tuple[int, ...] = ()
    step_range: tuple[int, int] | None = None
    random_injection_steps: int | None = None
    injection_count: int | None = None
    selection_mode: str = "random_element"
    name_pattern: str | None = None
    step_interval: int = 1
    flat_index: int | None = None
    bit_position: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dtype"] = str(self.dtype)
        payload["injection_steps"] = list(self.injection_steps)
        payload["step_range"] = list(self.step_range) if self.step_range else None
        return payload


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool = True
    format: str = "jsonl"
    output_dir: str = "logs"
    save_tensor_backup: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMResFIConfig:
    experiment: ExperimentConfig
    fault: FaultConfig
    logging: LoggingConfig
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment.to_dict(),
            "fault": self.fault.to_dict(),
            "logging": self.logging.to_dict(),
        }

    def build_scheduler(self) -> FaultScheduler:
        if not self.fault.enabled:
            return FaultScheduler(seed=self.experiment.seed)
        return FaultScheduler(
            injection_steps=self.fault.injection_steps or None,
            max_faults=self.fault.injection_count,
            seed=self.experiment.seed,
            step_range=self.fault.step_range,
            random_steps=self.fault.random_injection_steps,
            step_interval=self.fault.step_interval,
        )


_TOP_KEYS = {"experiment", "fault", "logging"}
_EXPERIMENT_KEYS = {"experiment_id", "seed"}
_FAULT_KEYS = {
    "enabled",
    "target_type",
    "target_name",
    "dtype",
    "injection_mode",
    "bit_type",
    "injection_step",
    "injection_steps",
    "step_range",
    "random_injection_steps",
    "injection_count",
    "selection_mode",
    "name_pattern",
    "step_interval",
    "flat_index",
    "bit_position",
}
_LOGGING_KEYS = {"enabled", "format", "output_dir", "save_tensor_backup"}


def load_config(path: str | Path) -> LLMResFIConfig:
    """Load, validate, and normalize an experiment YAML file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigurationError("Top-level configuration must be a mapping")
    _check_unknown_keys("top level", raw, _TOP_KEYS)

    experiment_raw = _require_mapping(raw.get("experiment"), "experiment")
    fault_raw = _require_mapping(raw.get("fault"), "fault")
    logging_raw = _require_mapping(raw.get("logging"), "logging")
    _check_unknown_keys("experiment", experiment_raw, _EXPERIMENT_KEYS)
    _check_unknown_keys("fault", fault_raw, _FAULT_KEYS)
    _check_unknown_keys("logging", logging_raw, _LOGGING_KEYS)

    experiment_id = experiment_raw.get("experiment_id") or generate_experiment_id()
    if not isinstance(experiment_id, str):
        raise ConfigurationError("experiment.experiment_id must be a string")
    seed = _validate_int(experiment_raw.get("seed", 0), "experiment.seed")
    experiment = ExperimentConfig(experiment_id=experiment_id, seed=seed)

    try:
        dtype = resolve_float_dtype(fault_raw.get("dtype", "float16"))
    except UnsupportedDTypeError as exc:
        raise ConfigurationError(str(exc)) from exc
    explicit_steps: list[int] = []
    for key in ("injection_step", "injection_steps"):
        if key not in fault_raw or fault_raw[key] is None:
            continue
        value = fault_raw[key]
        if isinstance(value, int) and not isinstance(value, bool):
            explicit_steps.extend([_validate_int(value, key)])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            explicit_steps.extend(_validate_int(item, key) for item in value)
        else:
            raise ConfigurationError(f"fault.{key} must be an integer or list of integers")
    injection_steps = tuple(sorted(set(explicit_steps)))

    step_range_raw = fault_raw.get("step_range")
    step_range: tuple[int, int] | None = None
    if step_range_raw is not None:
        if (
            not isinstance(step_range_raw, Sequence)
            or isinstance(step_range_raw, (str, bytes))
            or len(step_range_raw) != 2
        ):
            raise ConfigurationError("fault.step_range must be [start, end]")
        start = _validate_int(step_range_raw[0], "fault.step_range start")
        end = _validate_int(step_range_raw[1], "fault.step_range end")
        if start > end:
            raise ConfigurationError("fault.step_range start must not exceed end")
        step_range = (start, end)

    if injection_steps and (
        step_range is not None or fault_raw.get("random_injection_steps") is not None
    ):
        raise ConfigurationError(
            "Explicit fault.injection_step(s) cannot be combined with step_range or "
            "random_injection_steps"
        )

    injection_count_raw = fault_raw.get("injection_count")
    injection_count = (
        None
        if injection_count_raw is None
        else _validate_int(injection_count_raw, "fault.injection_count")
    )
    random_injection_steps_raw = fault_raw.get("random_injection_steps")
    random_injection_steps = (
        None
        if random_injection_steps_raw is None
        else _validate_int(random_injection_steps_raw, "fault.random_injection_steps")
    )
    if random_injection_steps is not None and step_range is None:
        raise ConfigurationError("fault.random_injection_steps requires fault.step_range")
    target_type = str(fault_raw.get("target_type", "model_parameter"))
    if target_type not in {"model_parameter", "user_tensor"}:
        raise ConfigurationError(
            "fault.target_type must be 'model_parameter' or 'user_tensor' for v0.1"
        )
    injection_mode = str(fault_raw.get("injection_mode", "single_bit"))
    if injection_mode not in {"single_bit", "multi_bit"}:
        raise ConfigurationError("fault.injection_mode must be single_bit or multi_bit")
    bit_type_raw = fault_raw.get("bit_type")
    bit_type = None if bit_type_raw is None else str(bit_type_raw).lower()
    if bit_type not in {None, "sign", "exponent", "mantissa", "random"}:
        raise ConfigurationError(
            "fault.bit_type must be sign, exponent, mantissa, random, or null"
        )
    selection_mode = str(fault_raw.get("selection_mode", "random_element"))
    if selection_mode not in {"random_element", "fixed_element"}:
        raise ConfigurationError(
            "fault.selection_mode must be 'random_element' or 'fixed_element'"
        )
    step_interval = _validate_int(
        fault_raw.get("step_interval", 1), "fault.step_interval", minimum=1
    )

    flat_index_raw = fault_raw.get("flat_index")
    bit_position_raw = fault_raw.get("bit_position")
    flat_index = (
        None if flat_index_raw is None else _validate_int(flat_index_raw, "fault.flat_index")
    )
    bit_position = (
        None
        if bit_position_raw is None
        else _validate_int(bit_position_raw, "fault.bit_position")
    )
    if bit_position is not None:
        try:
            get_float_format(dtype).bit_type(bit_position)
        except InvalidBitPositionError as exc:
            raise ConfigurationError(str(exc)) from exc
    if selection_mode == "fixed_element" and (flat_index is None or bit_position is None):
        raise ConfigurationError(
            "fault.selection_mode='fixed_element' requires flat_index and bit_position"
        )

    fault = FaultConfig(
        enabled=bool(fault_raw.get("enabled", True)),
        target_type=target_type,
        target_name=fault_raw.get("target_name"),
        dtype=dtype,
        injection_mode=injection_mode,
        bit_type=bit_type,
        injection_steps=injection_steps,
        step_range=step_range,
        random_injection_steps=random_injection_steps,
        injection_count=injection_count,
        selection_mode=selection_mode,
        name_pattern=fault_raw.get("name_pattern"),
        step_interval=step_interval,
        flat_index=flat_index,
        bit_position=bit_position,
    )

    log_format = str(logging_raw.get("format", "jsonl")).lower()
    if log_format not in {"jsonl", "csv", "both"}:
        raise ConfigurationError("logging.format must be jsonl, csv, or both")
    logging_config = LoggingConfig(
        enabled=bool(logging_raw.get("enabled", True)),
        format=log_format,
        output_dir=str(logging_raw.get("output_dir", f"logs/{experiment_id}")),
        save_tensor_backup=bool(logging_raw.get("save_tensor_backup", False)),
    )
    return LLMResFIConfig(
        experiment=experiment,
        fault=fault,
        logging=logging_config,
        raw=dict(raw),
        source_path=config_path,
    )


def write_run_metadata(
    config: LLMResFIConfig,
    output_dir: str | Path,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write the exact resolved config plus runtime versions to a run directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cuda_available = bool(torch.cuda.is_available())
    gpu_devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpu_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    metadata: dict[str, Any] = {
        "experiment_id": config.experiment.experiment_id,
        "seed": config.experiment.seed,
        "tool": "LLMResFI",
        "tool_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": gpu_devices,
        "platform": platform.platform(),
        "resolved_config": config.to_dict(),
        "scheduler_plan": config.build_scheduler().to_dict(),
    }
    if extra:
        metadata["extra"] = dict(extra)
    path = destination / "run_metadata.json"
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if config.source_path is not None:
        (destination / "source_config.yaml").write_text(
            config.source_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return path
