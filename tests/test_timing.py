import pytest

from llmresfi.config import TimingCfg
from llmresfi.hooks import TimingGate


def test_transient_fires_once_across_steps():
    gate = TimingGate(TimingCfg(type="transient"))
    assert gate.should_fire(0)
    gate.mark_fired(0)
    assert not gate.should_fire(0)
    assert not gate.should_fire(1)


def test_permanent_window_is_step_based():
    gate = TimingGate(TimingCfg(type="permanent", steps=3))
    for step in (0, 1, 2):
        assert gate.should_fire(step)
        gate.mark_fired(step)
    assert not gate.should_fire(3)
    assert not gate.should_fire(4)


def test_one_fault_per_step_dedupes_buckets():
    gate = TimingGate(TimingCfg(type="permanent", steps=10))
    assert gate.should_fire(0)
    gate.mark_fired(0)
    # A second bucket in the same backward must not fire another fault.
    assert not gate.should_fire(0)
    assert gate.should_fire(1)


def test_intermittent_period():
    gate = TimingGate(TimingCfg(type="intermittent", period=2))
    assert gate.should_fire(0)
    gate.mark_fired(0)
    assert not gate.should_fire(1)
    assert gate.should_fire(2)
    gate.mark_fired(2)
    assert not gate.should_fire(3)
    assert gate.should_fire(4)
