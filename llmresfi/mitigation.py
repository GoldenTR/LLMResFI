# llmresfi/mitigation.py
"""Detection and recovery primitives for soft-error resilience studies.

The module provides three independent building blocks:

- :class:`AnomalyDetector` finds non-finite gradients/parameters, gradient-norm
  outliers, and loss spikes.  It also consumes checksum mismatches reported by
  the distributed layer.
- :class:`CheckpointManager` saves/restores model+optimizer+RNG state.
- :class:`RecoveryManager` coordinates the above and implements three policies:
  cheap gradient sanitisation, lossless per-step rollback, and checkpoint
  rollback.

A typical training loop::

    recovery = RecoveryManager(cfg, model, optimizer, tracker)
    while not done:
        recovery.begin_step()
        optimizer.zero_grad()
        loss = train_step(batch)
        loss.backward()
        signals = recovery.check(loss, injector=injector)
        action = recovery.decide(signals)
        if action == "retry_step":
            continue                      # replay the same batch
        if action == "abort":
            raise RuntimeError("unrecoverable fault")
        optimizer.step()
        recovery.end_step(loss, clean=not signals)

Step rollback is lossless for *transient* faults when the caller replays the
same batch and the model/RNG state is deterministic.  It costs an in-memory
copy of parameters and optimizer state at the start of every snapshot step.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from .config import FIConfig
from .distributed import dist_ready, global_grad_norm


# ---------------------------------------------------------------------------
# signals / detector
# ---------------------------------------------------------------------------
@dataclass
class Signal:
    kind: str
    detail: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    step: int = -1

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "rank": self.rank,
            "step": self.step,
            **self.detail,
        }


class AnomalyDetector:
    """Cheap runtime checks used to decide whether a step needs recovery."""

    def __init__(self, cfg, warmup_steps: Optional[int] = None) -> None:
        self.cfg = cfg
        self.warmup_steps = cfg.warmup_steps if warmup_steps is None else warmup_steps
        self._loss_ema: Optional[float] = None
        self._grad_ema: Optional[float] = None
        self.last_grad_norm: Optional[float] = None

    def reset(self) -> None:
        self._loss_ema = None
        self._grad_ema = None
        self.last_grad_norm = None

    # -- individual checks ------------------------------------------------
    def check_loss(self, loss, step: int) -> List[Signal]:
        signals: List[Signal] = []
        value = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)
        if not torch.isfinite(torch.tensor(value)):
            signals.append(Signal("loss_non_finite", {"loss": value}, step=step))
            return signals
        if (
            self.cfg.loss_ratio > 0
            and self._loss_ema is not None
            and step >= self.warmup_steps
        ):
            ratio = abs(value - self._loss_ema) / max(abs(self._loss_ema), 1e-12)
            if ratio > self.cfg.loss_ratio:
                signals.append(
                    Signal(
                        "loss_spike",
                        {"loss": value, "ema": self._loss_ema, "ratio": ratio},
                        step=step,
                    )
                )
        return signals

    def check_gradients(self, model, step: int) -> List[Signal]:
        signals: List[Signal] = []
        nonfinite = 0
        grad_norm_sq = 0.0
        for _, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if self.cfg.non_finite:
                nonfinite += int((~torch.isfinite(g)).sum().item())
            grad_norm_sq += float(g.double().pow(2).sum().item())

        grad_norm = grad_norm_sq ** 0.5
        self.last_grad_norm = grad_norm
        if nonfinite:
            signals.append(
                Signal("grad_non_finite", {"count": nonfinite}, step=step)
            )
        elif not torch.isfinite(torch.tensor(grad_norm)):
            signals.append(Signal("grad_norm_non_finite", {}, step=step))
        elif (
            self.cfg.grad_norm_ratio > 0
            and self._grad_ema is not None
            and step >= self.warmup_steps
            and self._grad_ema > 0
        ):
            ratio = grad_norm / max(self._grad_ema, 1e-12)
            if ratio > self.cfg.grad_norm_ratio:
                signals.append(
                    Signal(
                        "grad_norm_spike",
                        {"grad_norm": grad_norm, "ema": self._grad_ema, "ratio": ratio},
                        step=step,
                    )
                )
        return signals

    def check_parameters(self, model, step: int) -> List[Signal]:
        if not self.cfg.check_parameters:
            return []
        nonfinite = 0
        for _, p in model.named_parameters():
            nonfinite += int((~torch.isfinite(p.detach())).sum().item())
        if nonfinite:
            return [Signal("param_non_finite", {"count": nonfinite}, step=step)]
        return []

    def observe(self, loss, step: int) -> None:
        """Update EMAs after a clean step (call only when no fault fired)."""
        value = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)
        if not torch.isfinite(torch.tensor(value)):
            return
        self._loss_ema = value if self._loss_ema is None else 0.9 * self._loss_ema + 0.1 * value
        if self.last_grad_norm is not None and torch.isfinite(torch.tensor(self.last_grad_norm)):
            self._grad_ema = (
                self.last_grad_norm
                if self._grad_ema is None
                else 0.9 * self._grad_ema + 0.1 * self.last_grad_norm
            )


# ---------------------------------------------------------------------------
# gradient sanitisation
# ---------------------------------------------------------------------------
def sanitize_gradients(model, mode: str = "zero", clip_norm: Optional[float] = None) -> dict:
    """Remove non-finite gradients (and optionally clip the global norm)."""
    if mode not in ("zero", "clamp"):
        raise ValueError(f"Unknown sanitize mode: {mode}")
    replaced = 0
    for _, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        bad = ~torch.isfinite(g)
        count = int(bad.sum().item())
        if count:
            replaced += count
            g[bad] = 0.0
    clipped_norm = None
    if clip_norm is not None:
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        clipped_norm = float(total)
    return {"replaced": replaced, "clip_norm": clip_norm, "total_norm": clipped_norm}


# ---------------------------------------------------------------------------
# snapshots / checkpoints
# ---------------------------------------------------------------------------
def _clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state_dict.items()}


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def capture_rng_state() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass
class StateSnapshot:
    step: int
    model_state: Dict[str, torch.Tensor]
    optimizer_state: dict
    rng_state: dict

    def restore(self, model, optimizer) -> None:
        model.load_state_dict(self.model_state)
        if optimizer is not None:
            optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))
        restore_rng_state(self.rng_state)


def capture_snapshot(model, optimizer, step: int, to_cpu: bool = False) -> StateSnapshot:
    model_state = _clone_state_dict(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else {}
    rng_state = capture_rng_state()
    if to_cpu:
        model_state = _to_cpu(model_state)
        optimizer_state = _to_cpu(optimizer_state)
        rng_state = _to_cpu(rng_state)
    return StateSnapshot(step, model_state, optimizer_state, rng_state)


class CheckpointManager:
    """Atomic, rank-local checkpoint storage for rollback experiments."""

    def __init__(self, directory, keep: int = 2, rank: Optional[int] = None) -> None:
        base = Path(directory)
        if rank is None and dist_ready():
            rank = dist.get_rank()
        self.rank = 0 if rank is None else rank
        self.directory = base / f"rank{self.rank}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep = keep

    def save(self, step: int, model, optimizer, extra: Optional[dict] = None) -> Path:
        path = self.directory / f"step_{step:08d}.pt"
        tmp = path.with_suffix(".tmp")
        payload = {
            "step": step,
            "model": _to_cpu(model.state_dict()),
            "optimizer": _to_cpu(optimizer.state_dict()) if optimizer is not None else {},
            "rng": _to_cpu(capture_rng_state()),
            "extra": _to_cpu(extra or {}),
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)
        latest = self.directory / "latest"
        latest_tmp = self.directory / "latest.tmp"
        latest_tmp.write_text(path.name, encoding="utf-8")
        os.replace(latest_tmp, latest)
        self._prune()
        return path

    def _prune(self) -> None:
        files = sorted(self.directory.glob("step_*.pt"))
        for old in files[: max(0, len(files) - self.keep)]:
            try:
                old.unlink()
            except OSError:
                pass

    def latest(self) -> Optional[Path]:
        marker = self.directory / "latest"
        if marker.exists():
            name = marker.read_text(encoding="utf-8").strip()
            candidate = self.directory / name
            if candidate.exists():
                return candidate
        files = sorted(self.directory.glob("step_*.pt"))
        return files[-1] if files else None

    def load(self, path=None) -> Optional[dict]:
        path = Path(path) if path is not None else self.latest()
        if path is None or not Path(path).exists():
            return None
        kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
        return torch.load(path, **kwargs)

    def restore(self, payload: dict, model, optimizer) -> None:
        model.load_state_dict(payload["model"])
        if optimizer is not None:
            optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
        restore_rng_state(payload.get("rng", {}))


# ---------------------------------------------------------------------------
# recovery controller
# ---------------------------------------------------------------------------
class RecoveryManager:
    """Detect faults, agree across ranks, and apply the configured policy."""

    def __init__(
        self,
        cfg: FIConfig,
        model,
        optimizer=None,
        tracker=None,
        log_dir=None,
    ) -> None:
        self.cfg = cfg.mitigation
        self.model = model
        self.optimizer = optimizer
        self.tracker = tracker
        self.detector = AnomalyDetector(cfg.mitigation.detector)
        self.rank = dist.get_rank() if dist_ready() else 0

        root = Path(log_dir) if log_dir is not None else cfg.execution.log_dir
        root.mkdir(parents=True, exist_ok=True)
        self.log_path = root / "recovery.jsonl"

        recovery = cfg.mitigation.recovery
        checkpoint_dir = recovery.checkpoint_dir or (root / "checkpoints")
        self.checkpoints = CheckpointManager(
            checkpoint_dir, keep=recovery.keep_checkpoints, rank=self.rank
        )

        self._snapshot: Optional[StateSnapshot] = None
        self._retries = 0
        self._last_signals: List[Signal] = []
        self._last_action = "continue"
        self.last_restored_step: Optional[int] = None
        self.events: List[dict] = []

    # -- loop hooks -------------------------------------------------------
    def begin_step(self) -> None:
        if not self.cfg.enabled:
            return
        mode = self.cfg.recovery.mode
        if mode == "checkpoint_rollback":
            if self.checkpoints.latest() is None:
                self.save_checkpoint(self._step())
            return
        if mode != "step_rollback" or self._snapshot is not None:
            return
        step = self._step()
        if step % self.cfg.recovery.snapshot_every == 0:
            self._snapshot = capture_snapshot(self.model, self.optimizer, step)

    def check(self, loss=None, injector=None) -> List[Signal]:
        if not self.cfg.enabled:
            return []
        step = self._step()
        signals: List[Signal] = []
        if loss is not None:
            signals.extend(self.detector.check_loss(loss, step))
        signals.extend(self.detector.check_gradients(self.model, step))
        signals.extend(self.detector.check_parameters(self.model, step))
        if injector is not None:
            for raw in injector.drain_comm_signals():
                signals.append(Signal("comm_checksum_mismatch", raw, self.rank, step))

        # All ranks must make the same recovery decision or DDP/FSDP deadlocks.
        any_fault = self._agree_any(bool(signals))
        if any_fault and not signals:
            signals.append(
                Signal("peer_signal", {"note": "another rank detected a fault"}, self.rank, step)
            )
        for sig in signals:
            sig.rank = self.rank
            sig.step = step
        self._last_signals = signals
        return signals

    def decide(self, signals: List[Signal]) -> str:
        if not self.cfg.enabled:
            return "continue"
        if not signals:
            self._last_action = "continue"
            return "continue"

        mode = self.cfg.recovery.mode
        if mode == "none":
            self._log("observed", signals)
            self._last_action = "continue"
            return "continue"

        if mode == "sanitize":
            summary = sanitize_gradients(
                self.model,
                mode=self.cfg.recovery.sanitize,
                clip_norm=self.cfg.recovery.clip_norm,
            )
            self._log("sanitized", signals, summary)
            self._last_action = "continue"
            return "continue"

        if mode == "step_rollback":
            if self._snapshot is None or self._retries >= self.cfg.recovery.max_retries:
                self._log("abort", signals, {"reason": "no snapshot or retries exhausted"})
                self._last_action = "abort"
                return "abort"
            self._snapshot.restore(self.model, self.optimizer)
            self._retries += 1
            self._log("step_rollback", signals, {"retry": self._retries})
            self._last_action = "retry_step"
            return "retry_step"

        if mode == "checkpoint_rollback":
            if self._retries >= self.cfg.recovery.max_retries:
                self._log("abort", signals, {"reason": "checkpoint retries exhausted"})
                self._last_action = "abort"
                return "abort"
            payload = self.checkpoints.load()
            if payload is None:
                self._log("abort", signals, {"reason": "no checkpoint available"})
                self._last_action = "abort"
                return "abort"
            self.checkpoints.restore(payload, self.model, self.optimizer)
            self._retries += 1
            self.last_restored_step = int(payload.get("step", 0))
            self._log(
                "checkpoint_rollback",
                signals,
                {"checkpoint_step": payload.get("step"), "retry": self._retries},
            )
            self._last_action = "restored_checkpoint"
            return "restored_checkpoint"

        raise ValueError(f"Unknown recovery mode: {mode}")

    def end_step(self, loss=None, clean: bool = True) -> None:
        if not self.cfg.enabled:
            return
        step = self._step()
        if clean:
            if loss is not None:
                self.detector.observe(loss, step)
            self._retries = 0
            self._snapshot = None
            self.last_restored_step = None
            if (
                self.cfg.recovery.checkpoint_every > 0
                and (step + 1) % self.cfg.recovery.checkpoint_every == 0
            ):
                self.save_checkpoint(step + 1)

    # -- checkpoint helpers ----------------------------------------------
    def save_checkpoint(self, step: int, extra: Optional[dict] = None) -> Path:
        return self.checkpoints.save(step, self.model, self.optimizer, extra)

    def load_checkpoint(self) -> Optional[int]:
        payload = self.checkpoints.load()
        if payload is None:
            return None
        self.checkpoints.restore(payload, self.model, self.optimizer)
        return int(payload.get("step", -1))

    # -- internal ---------------------------------------------------------
    def _step(self) -> int:
        return int(getattr(self.tracker, "step", 0)) if self.tracker is not None else 0

    def _agree_any(self, local: bool) -> bool:
        if not dist_ready():
            return local
        device = torch.device("cpu")
        if dist.get_backend() == "nccl":
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cuda")
        flag = torch.tensor([1 if local else 0], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _log(self, action: str, signals: List[Signal], extra: Optional[dict] = None) -> None:
        event = {
            "time": time.time(),
            "rank": self.rank,
            "step": self._step(),
            "action": action,
            "signals": [s.to_dict() for s in signals],
        }
        if extra:
            event.update(extra)
        self.events.append(event)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
