import pytest
import torch.nn as nn
from pydantic import ValidationError

from llmresfi.config import (
    DistributedCfg,
    ExecCfg,
    FaultCfg,
    ModelCfg,
    StageCfg,
)
from llmresfi.distributed import DDPCommInjector, resolve_inject_ranks


def _cfg(distributed: DistributedCfg):
    from llmresfi.config import FIConfig

    return FIConfig(
        run_mode="training",
        model=ModelCfg(repo="tiny"),
        distributed=distributed,
        fault=FaultCfg(target="comm", stage=StageCfg(type="backward")),
        execution=ExecCfg(random_seed=7),
    )


def test_resolve_single_rank():
    cfg = _cfg(DistributedCfg(backend="ddp", inject_rank=2))
    assert resolve_inject_ranks(cfg, world_size=4) == {2}


def test_resolve_explicit_ranks():
    cfg = _cfg(DistributedCfg(backend="ddp", inject_ranks=[0, 3]))
    assert resolve_inject_ranks(cfg, world_size=4) == {0, 3}


def test_resolve_fraction_is_deterministic():
    cfg = _cfg(DistributedCfg(backend="ddp", inject_rank_fraction=0.5))
    first = resolve_inject_ranks(cfg, world_size=8)
    second = resolve_inject_ranks(cfg, world_size=8)
    assert first == second
    assert len(first) == 4


def test_resolve_fraction_zero():
    cfg = _cfg(DistributedCfg(backend="ddp", inject_rank_fraction=0.0))
    assert resolve_inject_ranks(cfg, world_size=4) == set()


def test_rank_policy_mutual_exclusion():
    with pytest.raises(ValidationError):
        DistributedCfg(inject_rank=0, inject_ranks=[1])


def test_invalid_injection_point_rejected():
    with pytest.raises(ValidationError):
        DistributedCfg(injection_point="during_reduce")


def test_ddp_injector_requires_ddp_wrapper():
    cfg = _cfg(DistributedCfg(backend="ddp"))
    injector = type("DummyInjector", (), {})()
    comm = DDPCommInjector(cfg, injector)
    with pytest.raises(TypeError):
        comm.arm(nn.Linear(2, 2))
