# examples/run_ddp.py
"""DDP (data-parallel) communication fault injection example.

Run with torchrun:
    torchrun --nproc_per_node=2 examples/run_ddp.py --target comm --stage backward

Note: DDP performs gradient all-reduce *during* ``loss.backward()``, so the
communication hook fires while the tracker is in the ``backward`` stage.
"""
from __future__ import annotations

import argparse
import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

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
from llmresfi.distributed import DDPCommInjector


class MLP(nn.Module):
    def __init__(self, dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_cfg(target: str, stage: str, inject_rank: int) -> FIConfig:
    return FIConfig(
        run_mode="training",
        model=ModelCfg(repo="mlp", dtype="float32", device="cpu"),
        distributed=DistributedCfg(backend="ddp", inject_rank=inject_rank),
        fault=FaultCfg(
            target=target,
            stage=StageCfg(type=stage),
            fault_model=FaultModelCfg(type="single"),
            timing=TimingCfg(type="transient"),
        ),
        execution=ExecCfg(random_seed=42),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="comm")
    ap.add_argument("--stage", default="backward")
    ap.add_argument("--inject_rank", type=int, default=0)
    args = ap.parse_args()

    if "RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
        raise SystemExit(
            "This example must be launched via `torchrun`, which sets the "
            "RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT environment variables.\n"
            "Run:\n"
            "  torchrun --nproc_per_node=2 examples/run_ddp.py --target comm --stage backward\n"
            "Note: on Windows CPU-only torch builds, torchrun's rendezvous store may "
            "fail (PyTorch built without libuv); on such machines launch the ranks "
            "manually with init_method=tcp://... instead."
        )

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()

    cfg = build_cfg(args.target, args.stage, args.inject_rank)

    torch.manual_seed(cfg.execution.random_seed)
    rng = random.Random(cfg.execution.random_seed)

    model = MLP()
    ddp_model = DDP(model)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    inspector = ModelInspector(model, optimizer)
    tracker = StageTracker(cfg.run_mode)
    injector = FaultInjector(cfg, inspector, tracker, rng, optimizer)

    if cfg.fault.target == "comm":
        comm_inj = DDPCommInjector(cfg, injector)
        ddp_model.register_comm_hook(comm_inj.comm_hook)
    else:
        injector.arm()

    x = torch.randn(16, 64)
    y = torch.randint(0, 10, (16,))

    for step in range(3):
        optimizer.zero_grad()
        logits = ddp_model(x)
        tracker.on_forward_done()          # stage == backward
        loss = criterion(logits, y)
        loss.backward()                    # all-reduce fires here (comm hook)
        tracker.on_backward_done()
        optimizer.step()
        tracker.on_optimizer_done()
        tracker.on_comm_done()
        if rank == 0:
            print(f"[rank{rank} step{step}] loss={loss.item():.4f}")

    if rank == 0:
        print("\nInjection record:", injector.record)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
