"""FSDP shard-injection test for torch < 2.0 (no FSDP comm hook).

Skipped unless two CUDA devices are visible. Run with:
    CUDA_VISIBLE_DEVICES=0,1 pytest -q tests/test_fsdp_gpu.py
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or hasattr(FSDP, "register_comm_hook"),
    reason="legacy FSDP shard injection needs >=2 CUDA devices and torch < 2.0",
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)


def _cfg(mode: str, log_dir: Path) -> FIConfig:
    return FIConfig(
        run_mode="training",
        model=ModelCfg(repo="tiny", dtype="float32", device="cuda"),
        distributed=DistributedCfg(
            backend="fsdp", inject_rank=0, fsdp_injection=mode
        ),
        fault=FaultCfg(
            target="comm",
            stage=StageCfg(type="comm"),
            fault_model=FaultModelCfg(type="single"),
            timing=TimingCfg(type="transient"),
        ),
        execution=ExecCfg(random_seed=42, log_dir=log_dir),
    )


def _worker(rank: int, world_size: int, tmpdir: str) -> None:
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{Path(tmpdir) / 'store'}",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    results = {}
    try:
        x = torch.randn(8, 8, device=rank)
        y = torch.randint(0, 3, (8,), device=rank)
        for mode in ("param_shard", "grad_shard"):
            torch.manual_seed(42)
            model = TinyNet().cuda()
            fsdp = FSDP(model, device_id=rank)
            optimizer = torch.optim.Adam(fsdp.parameters(), lr=1e-3)
            cfg = _cfg(mode, Path(tmpdir) / mode / f"rank{rank}")
            injector = FaultInjector(
                cfg,
                ModelInspector(fsdp, optimizer),
                StageTracker("training"),
                random.Random(42),
                optimizer,
            )
            tracker = injector.tracker
            injector.arm(fsdp)

            loss = nn.CrossEntropyLoss()(fsdp(x), y)
            tracker.on_forward_done()
            loss.backward()
            tracker.on_backward_done()
            injector.inject_distributed_now()
            optimizer.step()
            results[mode] = {
                "records": list(injector.records),
                "loss": float(loss.item()),
            }
            injector.disarm()
            dist.barrier()
    finally:
        Path(tmpdir, f"result_rank{rank}.json").write_text(
            json.dumps(results), encoding="utf-8"
        )
        dist.destroy_process_group()


def test_legacy_fsdp_shard_injection():
    tmpdir = tempfile.mkdtemp(prefix="llmresfi_fsdp_test_")
    mp.spawn(_worker, nprocs=2, args=(2, tmpdir), join=True, start_method="spawn")

    r0 = json.loads(Path(tmpdir, "result_rank0.json").read_text(encoding="utf-8"))
    r1 = json.loads(Path(tmpdir, "result_rank1.json").read_text(encoding="utf-8"))

    for mode, expected_point in (
        ("param_shard", "fsdp_param_shard"),
        ("grad_shard", "fsdp_grad_shard"),
    ):
        records0 = r0[mode]["records"]
        records1 = r1[mode]["records"]
        assert len(records0) == 1
        assert len(records1) == 0
        rec = records0[0]
        assert rec["target"] == "comm"
        assert rec["backend"] == "fsdp"
        assert rec["legacy_fsdp"] is True
        assert rec["injection_point"] == expected_point
        assert rec["rank"] == 0
        assert rec["bucket_numel"] > 0
