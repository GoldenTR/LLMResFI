import random

import pytest

from llmresfi.exceptions import SchedulerError
from llmresfi.scheduler import FaultScheduler


def test_explicit_steps_trigger_once():
    scheduler = FaultScheduler(injection_steps=[100, 200], max_faults=2, seed=42)
    assert scheduler.injection_steps == (100, 200)
    assert scheduler.should_inject(100) is True
    assert scheduler.should_inject(100) is False
    assert scheduler.should_inject(200) is True
    assert scheduler.should_inject(200) is False
    assert scheduler.fired_count == 2
    assert scheduler.triggered_steps == (100, 200)
    assert scheduler.is_exhausted


def test_non_scheduled_steps_do_not_consume_events():
    scheduler = FaultScheduler(injection_steps=[10], seed=1)
    for step in range(10):
        assert scheduler.should_inject(step) is False
    assert scheduler.should_inject(10) is True


def test_duplicate_explicit_steps_are_collapsed():
    scheduler = FaultScheduler(injection_steps=[5, 5, 2, 5], seed=1)
    assert scheduler.injection_steps == (2, 5)


def test_max_faults_subset_is_seeded():
    first = FaultScheduler(step_range=(0, 99), max_faults=7, seed=8)
    second = FaultScheduler(step_range=(0, 99), max_faults=7, seed=8)
    assert first.injection_steps == second.injection_steps
    assert len(first.injection_steps) == 7
    assert first.injection_steps == tuple(sorted(first.injection_steps))


def test_random_steps_are_reproducible():
    first = FaultScheduler(step_range=(50, 80), random_steps=4, seed=99)
    second = FaultScheduler(step_range=(50, 80), random_steps=4, seed=99)
    assert first.injection_steps == second.injection_steps
    assert len(first.injection_steps) == 4
    assert all(50 <= step <= 80 for step in first.injection_steps)


def test_range_and_interval_include_end_when_aligned():
    scheduler = FaultScheduler(step_range=(10, 20), step_interval=5)
    assert scheduler.injection_steps == (10, 15, 20)
    scheduler = FaultScheduler(step_range=(10, 21), step_interval=5)
    assert scheduler.injection_steps == (10, 15, 20)


def test_zero_budget_has_empty_plan():
    scheduler = FaultScheduler(injection_steps=[1, 2, 3], max_faults=0)
    assert scheduler.injection_steps == ()
    assert scheduler.should_inject(1) is False
    assert scheduler.is_exhausted


def test_reset_restores_plan_without_resampling():
    scheduler = FaultScheduler(step_range=(0, 20), random_steps=3, seed=4)
    plan = scheduler.injection_steps
    for step in plan:
        assert scheduler.should_inject(step)
    scheduler.reset()
    assert scheduler.injection_steps == plan
    assert scheduler.fired_count == 0
    assert scheduler.should_inject(plan[0])


def test_scheduler_does_not_touch_global_random_state():
    random.seed(1234)
    expected = random.random()
    random.seed(1234)
    FaultScheduler(step_range=(0, 100), random_steps=10, seed=44)
    FaultScheduler(step_range=(0, 100), max_faults=10, seed=45)
    assert random.random() == expected


@pytest.mark.parametrize("bad_step", [-1, 1.5, True, "10"])
def test_invalid_step_values(bad_step):
    scheduler = FaultScheduler(injection_steps=[1])
    with pytest.raises(SchedulerError):
        scheduler.should_inject(bad_step)


def test_invalid_constructor_arguments():
    with pytest.raises(SchedulerError):
        FaultScheduler(injection_steps=[-1])
    with pytest.raises(SchedulerError):
        FaultScheduler(injection_steps=[1], step_range=(0, 2))
    with pytest.raises(SchedulerError):
        FaultScheduler(random_steps=2)
    with pytest.raises(SchedulerError):
        FaultScheduler(step_range=(5, 4))
    with pytest.raises(SchedulerError):
        FaultScheduler(step_range=(0, 3), random_steps=5)
    with pytest.raises(SchedulerError):
        FaultScheduler(step_range=(0, 3), step_interval=0)
    with pytest.raises(SchedulerError):
        FaultScheduler(max_faults=-1)


def test_plan_metadata_documents_zero_based_steps():
    scheduler = FaultScheduler(injection_steps=[100], seed=2)
    payload = scheduler.to_dict()
    assert payload["injection_steps"] == [100]
    assert payload["step_base"] == 0
