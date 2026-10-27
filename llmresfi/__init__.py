# llmresfi/__init__.py
"""LLMResFI: a fault injection framework for LLM training and inference.

Public API
----------
- FIConfig: YAML-based configuration (``FIConfig.load("config.yml")``)
- ModelInspector: registry of layers/modules/parameters
- StageTracker: unified inference/training phase tracker
- FaultInjector: the injection engine (``arm()`` / ``disarm()``)
"""
from .analysis import Outcome, bit_position_sensitivity, classify_text, summarize
from .bitflip import BitFlipper
from .config import FIConfig
from .injector import FaultInjector
from .inspector import ModelInspector
from .scheduler import StageTracker

__all__ = [
    "FIConfig",
    "ModelInspector",
    "StageTracker",
    "FaultInjector",
    "BitFlipper",
    "Outcome",
    "classify_text",
    "summarize",
    "bit_position_sensitivity",
]

__version__ = "0.1.0"
