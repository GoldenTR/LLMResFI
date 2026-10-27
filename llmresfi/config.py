# llmresfi/config.py
"""Configuration loader and schema validation for LLMResFI."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


# ---------------- model ----------------
class ModelCfg(BaseModel):
    repo: str
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device: Literal["cuda", "cpu", "auto"] = "cuda"


# ---------------- distributed ----------------
class DistributedCfg(BaseModel):
    backend: Literal["none", "ddp", "fsdp"] = "none"
    world_size: int = 1
    inject_rank: Optional[int] = None  # None -> all ranks


# ---------------- fault ----------------
class StageCfg(BaseModel):
    type: str
    token_step: Optional[int] = None


class FaultModelCfg(BaseModel):
    type: Literal[
        "single",
        "double",
        "multi",
        "sign",
        "exponent",
        "mantissa",
        "random",
        "random_value",
        "zero",
        "nan",
        "inf",
        "scale",
        "stuck_at_0",
        "stuck_at_1",
    ] = "single"
    bit_idx: Optional[int] = None
    min_val: float = -1.0
    max_val: float = 1.0
    factor: float = 2.0


class TimingCfg(BaseModel):
    type: Literal["transient", "permanent", "intermittent"] = "transient"
    steps: int = Field(1, ge=1)
    period: int = Field(1, ge=1)


class FaultCfg(BaseModel):
    target: Literal[
        "weight",
        "activation",
        "kv_cache",
        "loss",
        "gradient",
        "optimizer",
        "master_weight",
        "comm",
    ]
    locate: Literal["random", "manual"] = "random"
    layer_idx: Optional[int] = None
    module_regex: str = ".*"
    stage: StageCfg
    fault_model: FaultModelCfg = FaultModelCfg()
    timing: TimingCfg = TimingCfg()


# ---------------- execution ----------------
class ExecCfg(BaseModel):
    random_seed: int = 42
    one_fault_per_run: bool = True
    log_dir: Path = Path("./runs/default")
    save_text: bool = True
    verbosity: int = Field(1, ge=0, le=2)


# ---------------- root ----------------
_INFERENCE_STAGES = {"prefill", "first_token", "decode"}
_TRAINING_STAGES = {"forward", "backward", "optimizer_step", "comm"}


class FIConfig(BaseModel):
    run_mode: Literal["inference", "training"]
    model: ModelCfg
    distributed: DistributedCfg = DistributedCfg()
    fault: FaultCfg
    execution: ExecCfg = ExecCfg()

    @model_validator(mode="after")
    def _check_stage(self) -> "FIConfig":
        allowed = _INFERENCE_STAGES if self.run_mode == "inference" else _TRAINING_STAGES
        if self.fault.stage.type not in allowed:
            raise ValueError(
                f"stage.type '{self.fault.stage.type}' is invalid for run_mode "
                f"'{self.run_mode}'. Allowed: {sorted(allowed)}"
            )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "FIConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        try:
            return cls.model_validate(raw)
        except ValidationError as e:
            raise SystemExit(f"[LLMResFI Config] Validation failed:\n{e}")

    @property
    def log_dir_resolved(self) -> Path:
        self.execution.log_dir.mkdir(parents=True, exist_ok=True)
        return self.execution.log_dir
