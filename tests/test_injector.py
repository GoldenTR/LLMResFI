import random

import torch
import torch.nn as nn

from llmresfi import FIConfig, FaultInjector, ModelInspector, StageTracker
from llmresfi.config import ExecCfg, FaultCfg, FaultModelCfg, ModelCfg, StageCfg


def _cfg(target="weight", stage="prefill", fault_model="single", timing="transient"):
    return FIConfig(
        run_mode="inference" if target in ("weight", "activation") else "training",
        model=ModelCfg(repo="x", dtype="float32", device="cpu"),
        fault=FaultCfg(
            target=target,
            stage=StageCfg(type=stage),
            fault_model=FaultModelCfg(type=fault_model),
        ),
        execution=ExecCfg(random_seed=42),
    )


def test_weight_injection():
    model = nn.Sequential(nn.Linear(4, 4))
    cfg = _cfg(target="weight", stage="prefill")
    rng = random.Random(42)
    inspector = ModelInspector(model)
    tracker = StageTracker("inference")
    injector = FaultInjector(cfg, inspector, tracker, rng)
    injector.arm()
    model(torch.randn(1, 4))
    assert injector.record is not None
    assert injector.record["target"] == "weight"
    injector.disarm()


def test_activation_injection():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    cfg = _cfg(target="activation", stage="prefill")
    rng = random.Random(42)
    inspector = ModelInspector(model)
    tracker = StageTracker("inference")
    injector = FaultInjector(cfg, inspector, tracker, rng)
    injector.arm()
    model(torch.randn(1, 4))
    assert injector.record is not None
    assert injector.record["target"] == "activation"
    injector.disarm()


def test_gradient_injection():
    model = nn.Linear(4, 4)
    cfg = _cfg(target="gradient", stage="backward")
    rng = random.Random(42)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    inspector = ModelInspector(model, optimizer)
    tracker = StageTracker("training")
    injector = FaultInjector(cfg, inspector, tracker, rng, optimizer)
    injector.arm()
    tracker.on_forward_done()  # stage == backward
    (model(torch.randn(4))).sum().backward()
    assert injector.record is not None
    assert injector.record["target"] == "gradient"
    injector.disarm()
