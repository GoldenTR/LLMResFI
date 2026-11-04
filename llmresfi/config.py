# llmresfi/config.py
"""Configuration loader and schema validation for LLMResFI."""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


# ---------------- model ----------------
class ModelCfg(BaseModel):
    repo: str
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device: Literal["cuda", "cpu", "auto"] = "cuda"
    trust_remote_code: bool = False


# ---------------- distributed ----------------
class DistributedCfg(BaseModel):
    """Distributed communication fault-injection policy.

    Rank selection (mutually exclusive):
    - ``inject_rank``:  a single rank.
    - ``inject_ranks``: an explicit subset of ranks.
    - ``inject_rank_fraction``: deterministic random subset (seeded by
      ``execution.random_seed``), 1.0 = every rank, 0.0 = none.

    ``injection_point`` selects whether the corruption is applied to a local
    gradient bucket *before* the collective (a rank-local silent error that is
    then averaged into every rank) or *after* the collective completes (an
    asymmetric error that makes one rank diverge).
    """

    backend: Literal["none", "ddp", "fsdp"] = "none"
    world_size: int = 1
    inject_rank: Optional[int] = Field(None, ge=0)
    inject_ranks: Optional[List[int]] = None
    inject_rank_fraction: float = Field(1.0, ge=0.0, le=1.0)
    injection_point: Literal["before_reduce", "after_reduce"] = "before_reduce"
    bucket_index: Optional[int] = Field(None, ge=0)
    # Legacy FSDP (< 2.0) has no gradient comm hook; choose the shard-level
    # injection point used by the fallback implementation instead.
    fsdp_injection: Literal["param_shard", "grad_shard", "both"] = "grad_shard"
    divide_by_world_size: bool = True
    verify_comm: bool = False
    verify_atol: float = Field(1e-6, gt=0.0)  # relative to sum(|grad|)

    @model_validator(mode="after")
    def _check_rank_policy(self) -> "DistributedCfg":
        if self.inject_rank is not None and self.inject_ranks is not None:
            raise ValueError("inject_rank and inject_ranks are mutually exclusive")
        if self.inject_ranks is not None:
            if self.inject_rank_fraction != 1.0:
                raise ValueError(
                    "inject_rank_fraction cannot be combined with inject_ranks"
                )
            if len(set(self.inject_ranks)) != len(self.inject_ranks):
                raise ValueError("inject_ranks contains duplicates")
        return self


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



# ---------------- mitigation ----------------
class DetectorCfg(BaseModel):
    """Runtime anomaly checks used to trigger recovery."""

    non_finite: bool = True
    check_parameters: bool = True
    grad_norm_ratio: float = Field(5.0, ge=0.0)  # 0 disables the ratio check
    loss_ratio: float = Field(3.0, ge=0.0)       # 0 disables the ratio check
    warmup_steps: int = Field(5, ge=0)


class RecoveryCfg(BaseModel):
    """Recovery policy applied when the detector fires.

    - ``sanitize``: zero/clamp non-finite gradients and continue (cheap, not
      lossless for large value faults).
    - ``step_rollback``: restore a per-step in-memory snapshot and replay the
      step.  With deterministic data/RNG this is lossless for transient
      faults; the cost is one extra forward/backward on the affected step.
    - ``checkpoint_rollback``: restore the latest on-disk checkpoint (lossless
      up to the checkpoint interval, but more expensive).
    """

    mode: Literal["none", "sanitize", "step_rollback", "checkpoint_rollback"] = "none"
    max_retries: int = Field(1, ge=0)
    sanitize: Literal["zero", "clamp"] = "zero"
    clip_norm: Optional[float] = Field(None, gt=0.0)
    snapshot_every: int = Field(1, ge=1)
    checkpoint_every: int = Field(0, ge=0)
    checkpoint_dir: Optional[Path] = None
    keep_checkpoints: int = Field(2, ge=1)


class MitigationCfg(BaseModel):
    enabled: bool = False
    detector: DetectorCfg = DetectorCfg()
    recovery: RecoveryCfg = RecoveryCfg()


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
    mitigation: MitigationCfg = MitigationCfg()

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
