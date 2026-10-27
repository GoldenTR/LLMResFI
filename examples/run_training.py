# examples/run_training.py
"""Single-GPU training fault injection example (tiny MLP on synthetic data).

Demonstrates the training-only fault targets that existing tools do not cover:
gradient, optimizer (Adam state), master_weight, and loss.

Usage:
    python examples/run_training.py --target gradient --stage backward
    python examples/run_training.py --target optimizer --stage optimizer_step
    python examples/run_training.py --target loss --stage backward
"""
from __future__ import annotations

import argparse
import random

import torch
import torch.nn as nn

from llmresfi import FIConfig, FaultInjector, ModelInspector, StageTracker
from llmresfi.config import (
    DistributedCfg,
    ExecCfg,
    FaultCfg,
    FaultModelCfg,
    ModelCfg,
    StageCfg,
    TimingCfg,
)


class MLP(nn.Module):
    def __init__(self, dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_cfg(target: str, stage: str, fault_model: str, timing: str) -> FIConfig:
    return FIConfig(
        run_mode="training",
        model=ModelCfg(repo="mlp", dtype="float32", device="cpu"),
        distributed=DistributedCfg(backend="none"),
        fault=FaultCfg(
            target=target,
            stage=StageCfg(type=stage),
            fault_model=FaultModelCfg(type=fault_model),
            timing=TimingCfg(type=timing, steps=3, period=2),
        ),
        execution=ExecCfg(random_seed=42),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gradient",
                    choices=["gradient", "optimizer", "master_weight", "loss", "weight", "activation"])
    ap.add_argument("--stage", default=None,
                    help="override stage (default is derived from target)")
    ap.add_argument("--fault_model", default="single")
    ap.add_argument("--timing", default="transient", choices=["transient", "permanent", "intermittent"])
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    stage_defaults = {
        "gradient": "backward",
        "optimizer": "optimizer_step",
        "master_weight": "optimizer_step",
        "loss": "backward",
        "weight": "forward",
        "activation": "forward",
    }
    stage = args.stage or stage_defaults[args.target]

    cfg = build_cfg(args.target, stage, args.fault_model, args.timing)

    torch.manual_seed(cfg.execution.random_seed)
    rng = random.Random(cfg.execution.random_seed)

    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    inspector = ModelInspector(model, optimizer)
    tracker = StageTracker(cfg.run_mode)
    injector = FaultInjector(cfg, inspector, tracker, rng, optimizer)
    injector.arm()

    x = torch.randn(16, 64)
    y = torch.randint(0, 10, (16,))

    for step in range(args.steps):
        optimizer.zero_grad()
        logits = model(x)                       # stage == forward
        tracker.on_forward_done()               # stage == backward
        loss = criterion(logits, y)
        injector.inject_loss(loss)              # fires only if target == loss
        loss.backward()                         # gradient hook fires (stage == backward)
        tracker.on_backward_done()              # stage == optimizer_step
        optimizer.step()                        # optimizer/master_weight wrapper fires
        tracker.on_optimizer_done()             # stage == comm
        tracker.on_comm_done()                  # stage == forward, step += 1
        if cfg.execution.verbosity >= 1:
            print(f"[step {step}] loss={loss.item():.4f} stage={tracker.stage}")

    injector.disarm()

    print("\nInjection record:")
    if injector.record:
        for k, v in injector.record.items():
            print(f"  {k}: {v}")
    else:
        print("  (no fault injected)")


if __name__ == "__main__":
    main()
