# llmresfi/scheduler.py
"""StageTracker: a unified execution-phase tracker for inference & training.

Inference stages:  prefill -> first_token -> decode (per token step)
Training stages:   forward -> backward -> optimizer_step -> comm (per step)
"""
from __future__ import annotations

from typing import Optional

INFERENCE_STAGES = ("prefill", "first_token", "decode")
TRAINING_STAGES = ("forward", "backward", "optimizer_step", "comm")


class StageTracker:
    def __init__(self, run_mode: str) -> None:
        if run_mode not in ("inference", "training"):
            raise ValueError(f"Unknown run_mode: {run_mode}")
        self.run_mode = run_mode
        self.stage = "prefill" if run_mode == "inference" else "forward"
        self.token_step: int = -1  # valid only when stage == "decode"
        self.step: int = 0         # training step counter
        self._rand_decode_target: Optional[int] = None

    # ---------------- inference transitions ----------------
    def on_prefill_done(self) -> None:
        self.stage = "first_token"

    def on_first_token_emitted(self) -> None:
        self.stage = "decode"
        self.token_step = 0

    def inc_decode(self) -> None:
        if self.stage != "decode":
            raise RuntimeError("inc_decode() called while not in decode phase")
        self.token_step += 1

    def set_random_decode_target(self, max_steps: int, rng) -> None:
        self._rand_decode_target = rng.randrange(max_steps) if max_steps > 0 else None

    # ---------------- training transitions ----------------
    def on_forward_done(self) -> None:
        self.stage = "backward"

    def on_backward_done(self) -> None:
        self.stage = "optimizer_step"

    def on_optimizer_done(self) -> None:
        self.stage = "comm"

    def on_comm_done(self) -> None:
        self.stage = "forward"
        self.step += 1

    # ---------------- helpers ----------------
    def matches(self, spec_stage: str, token_step=None) -> bool:
        if spec_stage != self.stage:
            return False
        if spec_stage != "decode":
            return True
        if token_step is None:
            return True
        if token_step == "random":
            return self._rand_decode_target is None or self.token_step == self._rand_decode_target
        return token_step == self.token_step
