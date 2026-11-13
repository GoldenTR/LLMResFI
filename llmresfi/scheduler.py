"""Deterministic, non-repeating fault-injection schedules.

Training steps are zero-based throughout LLMResFI, matching
``enumerate(dataloader)`` without an explicit start offset.  A scheduled step
is claimed by the first ``should_inject(step)`` call that returns ``True``;
subsequent calls for the same step return ``False``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .exceptions import SchedulerError


def _validate_step(step: int, name: str = "step") -> None:
    if isinstance(step, bool) or not isinstance(step, int):
        raise SchedulerError(f"{name} must be an integer, got {type(step).__name__}")
    if step < 0:
        raise SchedulerError(f"{name} must be non-negative, got {step}")


def _validate_optional_positive_int(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchedulerError(f"{name} must be a non-negative integer or None, got {value!r}")


@dataclass(frozen=True)
class SchedulePlan:
    """Immutable audit view of a scheduler's deterministic plan."""

    injection_steps: tuple[int, ...]
    seed: int | None
    max_faults: int | None
    step_range: tuple[int, int] | None
    random_steps: int | None
    step_interval: int

    def to_dict(self) -> dict[str, object]:
        return {
            "injection_steps": list(self.injection_steps),
            "seed": self.seed,
            "max_faults": self.max_faults,
            "step_range": list(self.step_range) if self.step_range else None,
            "random_steps": self.random_steps,
            "step_interval": self.step_interval,
            "step_base": 0,
        }


class FaultScheduler:
    """Plan and claim one injection event per selected training step.

    Parameters
    ----------
    injection_steps:
        Explicit zero-based steps.  Duplicates are collapsed and the plan is
        stored chronologically.
    max_faults:
        Maximum number of events for the experiment.  If it is smaller than
        the candidate set, a local seeded RNG selects a subset.
    seed:
        Seed used only by this scheduler's local RNG.
    step_range:
        Inclusive ``(start, end)`` range used to generate candidates.  It is
        mutually exclusive with explicit ``injection_steps``.
    random_steps:
        Number of distinct random steps sampled from ``step_range``.  It is
        mutually exclusive with explicit ``injection_steps``.
    step_interval:
        Spacing used when generating candidates from ``step_range`` without
        ``random_steps``.
    """

    def __init__(
        self,
        injection_steps: Sequence[int] | int | None = None,
        max_faults: int | None = None,
        seed: int | None = None,
        *,
        step_range: tuple[int, int] | Sequence[int] | None = None,
        random_steps: int | None = None,
        step_interval: int = 1,
    ) -> None:
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise SchedulerError(f"seed must be an integer or None, got {seed!r}")
        _validate_optional_positive_int(max_faults, "max_faults")
        _validate_optional_positive_int(random_steps, "random_steps")
        if isinstance(step_interval, bool) or not isinstance(step_interval, int) or step_interval <= 0:
            raise SchedulerError(f"step_interval must be a positive integer, got {step_interval!r}")

        normalized_range: tuple[int, int] | None = None
        if step_range is not None:
            if len(step_range) != 2:
                raise SchedulerError(f"step_range must contain exactly two values, got {step_range!r}")
            start, end = int(step_range[0]), int(step_range[1])
            _validate_step(start, "step_range start")
            _validate_step(end, "step_range end")
            if start > end:
                raise SchedulerError(f"step_range start {start} exceeds end {end}")
            normalized_range = (start, end)

        explicit_steps = injection_steps is not None
        if explicit_steps:
            if isinstance(injection_steps, int) and not isinstance(injection_steps, bool):
                raw_steps = [injection_steps]
            else:
                try:
                    raw_steps = list(injection_steps)  # type: ignore[arg-type]
                except TypeError as exc:
                    raise SchedulerError(
                        "injection_steps must be an integer, a sequence of integers, or None"
                    ) from exc
            for value in raw_steps:
                _validate_step(value, "injection step")
            candidates = sorted(set(raw_steps))
            if normalized_range is not None or random_steps is not None:
                raise SchedulerError(
                    "Explicit injection_steps cannot be combined with step_range or random_steps"
                )
        elif random_steps is not None:
            if normalized_range is None:
                raise SchedulerError("random_steps requires an explicit step_range")
            candidates = list(range(normalized_range[0], normalized_range[1] + 1, step_interval))
            if random_steps > len(candidates):
                raise SchedulerError(
                    f"random_steps={random_steps} exceeds candidate count {len(candidates)}"
                )
            if random_steps > 0:
                rng = random.Random(seed)
                candidates = sorted(rng.sample(candidates, random_steps))
            else:
                candidates = []
        elif normalized_range is not None:
            candidates = list(range(normalized_range[0], normalized_range[1] + 1, step_interval))
        else:
            candidates = []

        if max_faults is not None and max_faults == 0:
            candidates = []
        elif max_faults is not None and len(candidates) > max_faults:
            rng = random.Random(seed)
            candidates = sorted(rng.sample(candidates, max_faults))

        self._seed = seed
        self._max_faults = max_faults
        self._step_range = normalized_range
        self._random_steps = random_steps
        self._step_interval = step_interval
        self._plan = tuple(candidates)
        self._pending: set[int] = set(self._plan)
        self._triggered: set[int] = set()

    @property
    def plan(self) -> SchedulePlan:
        return SchedulePlan(
            injection_steps=self._plan,
            seed=self._seed,
            max_faults=self._max_faults,
            step_range=self._step_range,
            random_steps=self._random_steps,
            step_interval=self._step_interval,
        )

    @property
    def injection_steps(self) -> tuple[int, ...]:
        return self._plan

    @property
    def remaining_steps(self) -> tuple[int, ...]:
        return tuple(sorted(self._pending))

    @property
    def triggered_steps(self) -> tuple[int, ...]:
        return tuple(sorted(self._triggered))

    @property
    def fired_count(self) -> int:
        return len(self._triggered)

    @property
    def is_exhausted(self) -> bool:
        return not self._pending

    def should_inject(self, step: int) -> bool:
        """Claim and return ``True`` at most once for each planned step."""

        _validate_step(step)
        if step not in self._pending:
            return False
        self._pending.remove(step)
        self._triggered.add(step)
        return True

    def was_triggered(self, step: int) -> bool:
        _validate_step(step)
        return step in self._triggered

    def reset(self) -> None:
        """Restore the original deterministic plan without re-sampling."""

        self._pending = set(self._plan)
        self._triggered.clear()

    def to_dict(self) -> dict[str, object]:
        return self.plan.to_dict()
