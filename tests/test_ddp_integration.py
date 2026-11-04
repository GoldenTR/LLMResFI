"""Two-process DDP integration tests (CPU + gloo).

These tests exercise the full path: comm hook -> fault record -> checksum
signal -> cross-rank agreement -> snapshot rollback -> replay.
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from llmresfi import FIConfig, FaultInjector, ModelInspector, RecoveryManager, StageTracker
from llmresfi.config import (
    DetectorCfg,
    DistributedCfg,
    ExecCfg,
    FaultCfg,
    FaultModelCfg,
    MitigationCfg,
    ModelCfg,
    RecoveryCfg,
    StageCfg,
    TimingCfg,
)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)


SCENARIOS = ("before_nan", "after_bit", "silent_value")


def _cfg_for(scenario: str, log_dir: Path) -> FIConfig:
    if scenario == "before_nan":
        fault_model, point, verify, mode = "nan", "before_reduce", True, "step_rollback"
    elif scenario == "after_bit":
        fault_model, point, verify, mode = "single", "after_reduce", True, "step_rollback"
    else:
        fault_model, point, verify, mode = "random_value", "after_reduce", False, "none"

    mitigation = MitigationCfg(
        enabled=mode != "none",
        detector=DetectorCfg(
            non_finite=True,
            check_parameters=True,
            warmup_steps=0,
            grad_norm_ratio=0.0,
            loss_ratio=0.0,
        ),
        recovery=RecoveryCfg(mode=mode, max_retries=1),
    )
    return FIConfig(
        run_mode="training",
        model=ModelCfg(repo="tiny", dtype="float32", device="cpu"),
        distributed=DistributedCfg(
            backend="ddp",
            inject_rank=0,
            injection_point=point,
            verify_comm=verify,
        ),
        fault=FaultCfg(
            target="comm",
            stage=StageCfg(type="backward"),
            fault_model=FaultModelCfg(type=fault_model, min_val=-1.0, max_val=1.0),
            timing=TimingCfg(type="transient"),
        ),
        execution=ExecCfg(random_seed=123, log_dir=log_dir),
        mitigation=mitigation,
    )


def _train_with_recovery(ddp, optimizer, tracker, injector, recovery, criterion, x, y):
    recovery.begin_step()
    optimizer.zero_grad()
    logits = ddp(x)
    tracker.on_forward_done()
    loss = criterion(logits, y)
    injector.inject_loss(loss)
    loss.backward()

    signals = recovery.check(loss, injector=injector)
    action = recovery.decide(signals)
    assert action in ("continue", "retry_step")
    trigger_signals = list(signals)

    if action == "retry_step":
        tracker.on_retry()
        optimizer.zero_grad()
        logits = ddp(x)
        tracker.on_forward_done()
        loss = criterion(logits, y)
        loss.backward()
        signals = recovery.check(loss, injector=injector)
        retry_action = recovery.decide(signals)
        assert retry_action == "continue", retry_action
        action = "retry_step"

    tracker.on_backward_done()
    optimizer.step()
    tracker.on_optimizer_done()
    tracker.on_comm_done()
    recovery.end_step(loss, clean=not signals)
    return loss, trigger_signals, action


def _worker(rank: int, world_size: int, tmpdir: str) -> None:
    store = f"file://{Path(tmpdir) / 'store'}"
    dist.init_process_group(
        backend="gloo", init_method=store, rank=rank, world_size=world_size
    )
    results = {}
    try:
        for scenario in SCENARIOS:
            torch.manual_seed(123)
            random.seed(123)

            model = TinyMLP()
            reference = TinyMLP()
            reference.load_state_dict(model.state_dict())

            ddp = DDP(model)
            ref_ddp = DDP(reference)
            optimizer = torch.optim.Adam(ddp.parameters(), lr=1e-2)
            ref_optimizer = torch.optim.Adam(ref_ddp.parameters(), lr=1e-2)

            cfg = _cfg_for(scenario, Path(tmpdir) / scenario / f"rank{rank}")
            inspector = ModelInspector(model, optimizer)
            tracker = StageTracker("training")
            injector = FaultInjector(
                cfg, inspector, tracker, random.Random(123), optimizer
            )
            injector.arm(ddp)
            recovery = RecoveryManager(cfg, ddp, optimizer, tracker)
            criterion = nn.CrossEntropyLoss()

            torch.manual_seed(1000)
            x = torch.randn(8, 8)
            y = torch.randint(0, 3, (8,))

            torch_state = torch.get_rng_state()
            py_state = random.getstate()

            _, signals, action = _train_with_recovery(
                ddp, optimizer, tracker, injector, recovery, criterion, x, y
            )

            # Clean reference step from the same RNG state.
            torch.set_rng_state(torch_state)
            random.setstate(py_state)
            ref_optimizer.zero_grad()
            ref_loss = criterion(ref_ddp(x), y)
            ref_loss.backward()
            ref_optimizer.step()

            local = dict(ddp.module.named_parameters())
            ref = dict(ref_ddp.module.named_parameters())
            max_diff = max(
                float((local[name] - ref[name]).abs().max().item()) for name in local
            )
            results[scenario] = {
                "max_diff": max_diff,
                "action": action,
                "n_records": len(injector.records),
                "n_signals": len(signals),
                "signal_kinds": [s.kind for s in signals],
            }
            injector.disarm()
            dist.barrier()
    finally:
        Path(tmpdir, f"result_rank{rank}.json").write_text(
            json.dumps(results), encoding="utf-8"
        )
        dist.destroy_process_group()


def test_ddp_fault_scenarios():
    tmpdir = tempfile.mkdtemp(prefix="llmresfi_ddp_test_")
    mp.spawn(_worker, nprocs=2, args=(2, tmpdir), join=True, start_method="spawn")

    r0 = json.loads(Path(tmpdir, "result_rank0.json").read_text(encoding="utf-8"))
    r1 = json.loads(Path(tmpdir, "result_rank1.json").read_text(encoding="utf-8"))

    # A rank-local NaN before all-reduce must be detected and rolled back.
    assert r0["before_nan"]["action"] == "retry_step"
    assert r0["before_nan"]["n_records"] == 1
    assert r0["before_nan"]["max_diff"] < 1e-6
    assert r1["before_nan"]["max_diff"] < 1e-6

    # A silent bit flip after all-reduce is caught by cross-rank consistency.
    assert r0["after_bit"]["action"] == "retry_step"
    assert "comm_checksum_mismatch" in r0["after_bit"]["signal_kinds"]
    assert r0["after_bit"]["max_diff"] < 1e-6
    assert r1["after_bit"]["max_diff"] < 1e-6

    # Without verification the same fault is a silent data corruption (SDC):
    # rank 0 diverges while rank 1 stays correct.
    assert r0["silent_value"]["action"] == "continue"
    assert r0["silent_value"]["n_signals"] == 0
    assert r0["silent_value"]["max_diff"] > 1e-8
    assert r1["silent_value"]["max_diff"] < 1e-6
