"""Exception hierarchy for LLMResFI.

The exceptions intentionally distinguish user-input errors from verification
failures.  A verification failure means the requested bit pattern was not
observed after mutation; callers should treat it as a tool error rather than a
valid fault event.
"""


class LLMResFIError(Exception):
    """Base class for all LLMResFI-specific errors."""


class UnsupportedDTypeError(LLMResFIError, TypeError):
    """Raised when a dtype cannot be represented by the supported formats."""


class InvalidFlatIndexError(LLMResFIError, IndexError):
    """Raised when a logical flat element index is out of bounds."""


class InvalidBitPositionError(LLMResFIError, ValueError):
    """Raised when a bit index is outside the dtype's IEEE-754 representation."""


class InjectionVerificationError(LLMResFIError, RuntimeError):
    """Raised when the injected bit pattern or target bit cannot be verified."""


class UnsafeInjectionError(LLMResFIError, RuntimeError):
    """Raised when an in-place mutation would invalidate a live autograd graph."""


class TargetSelectionError(LLMResFIError, ValueError):
    """Raised when a model tensor cannot be selected unambiguously."""


class SchedulerError(LLMResFIError, ValueError):
    """Raised for invalid scheduling configuration or step values."""


class ConfigurationError(LLMResFIError, ValueError):
    """Raised for malformed or inconsistent experiment configuration."""


class LoggingError(LLMResFIError, RuntimeError):
    """Raised when an audit event cannot be serialized or persisted."""

