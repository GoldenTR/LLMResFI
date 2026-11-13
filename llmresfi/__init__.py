"""LLMResFI: controllable tensor-level soft-error injection for LLM research."""

from ._version import __version__
from .bit_utils import (
    BitType,
    FloatFormat,
    SUPPORTED_FLOAT_DTYPES,
    bit_positions_for_type,
    coerce_bit_type,
    get_bit_type,
    get_float_format,
    pattern_to_binary,
    random_bit_position,
    resolve_float_dtype,
    tensor_bit_pattern,
)
from .config import (
    ExperimentConfig,
    FaultConfig,
    LLMResFIConfig,
    LoggingConfig,
    load_config,
    write_run_metadata,
)
from .exceptions import (
    ConfigurationError,
    InjectionVerificationError,
    InvalidBitPositionError,
    InvalidFlatIndexError,
    LLMResFIError,
    LoggingError,
    SchedulerError,
    TargetSelectionError,
    UnsafeInjectionError,
    UnsupportedDTypeError,
)
from .injector import (
    BitFlipSpec,
    InjectionResult,
    inject_bit_flip,
    inject_bit_flips,
    inject_random_bit_flips,
    plan_random_injections,
)
from .ids import generate_experiment_id
from .logger import REQUIRED_EVENT_FIELDS, EventLogger, FaultEvent
from .scheduler import FaultScheduler, SchedulePlan
from .target_selector import TargetTensor, list_target_tensors, select_target_tensor

__all__ = [
    "__version__",
    "BitType",
    "FloatFormat",
    "SUPPORTED_FLOAT_DTYPES",
    "bit_positions_for_type",
    "coerce_bit_type",
    "get_bit_type",
    "get_float_format",
    "pattern_to_binary",
    "random_bit_position",
    "resolve_float_dtype",
    "tensor_bit_pattern",
    "ExperimentConfig",
    "FaultConfig",
    "LLMResFIConfig",
    "LoggingConfig",
    "load_config",
    "write_run_metadata",
    "ConfigurationError",
    "InjectionVerificationError",
    "InvalidBitPositionError",
    "InvalidFlatIndexError",
    "LLMResFIError",
    "LoggingError",
    "SchedulerError",
    "TargetSelectionError",
    "UnsafeInjectionError",
    "UnsupportedDTypeError",
    "BitFlipSpec",
    "InjectionResult",
    "inject_bit_flip",
    "inject_bit_flips",
    "inject_random_bit_flips",
    "plan_random_injections",
    "generate_experiment_id",
    "REQUIRED_EVENT_FIELDS",
    "EventLogger",
    "FaultEvent",
    "FaultScheduler",
    "SchedulePlan",
    "TargetTensor",
    "list_target_tensors",
    "select_target_tensor",
]
