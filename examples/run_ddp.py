# examples/run_ddp.py
"""DDP fault-injection + mitigation demo (CPU/gloo or GPU/nccl).

Examples
--------
Baseline (no fault):

    torchrun --nproc_per_node=2 examples/run_ddp.py --baseline

Rank-local gradient corruption before all-reduce, with checksum detection and
lossless step rollback:

    torchrun --nproc_per_node=2 examples/run_ddp.py \
        --target comm --fault_model nan --inject_rank 0 \
        --mitigation step_rollback --verify

Asymmetric corruption after all-reduce (only rank 0 diverges):

    torchrun --nproc_per_node=2 examples/run_ddp.py \
        --target comm --fault_model single --injection_point after_reduce \
        --inject_rank 0 --verify
"""
from __future__ import annotations

import argparse
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from llmresfi import (
    FIConfig,
    FaultInjector,
    ModelInspector,
    RecoveryManager,
    StageTracker,
    build_summary,
    param_signature,
    write_summary,
)
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
from llmresfi.distributed import all_gather_metrics, all_gather_object, gather_records


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


def build_cfg(args) -> FIConfig:
    mitigation = MitigationCfg(enabled=args.mitigation != "none")
    if mitigation.enabled:
        mitigation.detector = DetectorCfg(
            non_finite=True,
            check_parameters=True,
            warmup_steps=1,
            loss_ratio=0.0,
            grad_norm_ratio=0.0,
        )
        mitigation.recovery = RecoveryCfg(
            mode=args.mitigation,
            max_retries=args.max_retries,
            snapshot_every=1,
            checkpoint_every=args.checkpoint_every,
        )
    return FIConfig(
        run_mode="training",
        model=ModelCfg(repo="mlp", dtype="float32", device=args.device),
        distributed=DistributedCfg(
            backend="ddp",
            inject_rank=args.inject_rank,
            injection_point=args.injection_point,
            bucket_index=args.bucket_index,
            verify_comm=args.verify,
            divide_by_world_size=True,
        ),
        fault=FaultCfg(
            target=args.target,
            stage=StageCfg(type=args.stage),
            fault_model=FaultModelCfg(type=args.fault_model),
            timing=TimingCfg(type=args.timing),
        ),
        execution=ExecCfg(random_seed=args.seed, log_dir=args.log_dir),
        mitigation=mitigation,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="comm", choices=["comm", "gradient", "weight", "activation", "loss", "optimizer", "master_weight"])
    ap.add_argument("--stage", default="backward")
    ap.add_argument("--inject_rank", type=int, default=0)
    ap.add_argument("--injection_point", default="before_reduce", choices=["before_reduce", "after_reduce"])
    ap.add_argument("--bucket_index", type=int, default=None)
    ap.add_argument("--fault_model", default="single")
    ap.add_argument("--timing", default="transient", choices=["transient", "permanent", "intermittent"])
    ap.add_argument("--mitigation", default="none", choices=["none", "sanitize", "step_rollback", "checkpoint_rollback"])
    ap.add_argument("--max_retries", type=int, default=1)
    ap.add_argument("--checkpoint_every", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="enable DDP checksum verification")
    ap.add_argument("--baseline", action="store_true", help="run without injection")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_dir", default="./runs/ddp_demo")
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()

    if "RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
        raise SystemExit(
            "This example must be launched via `torchrun`.\n"
            "  torchrun --nproc_per_node=2 examples/run_ddp.py --baseline"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available.")
    backend = "nccl" if args.device == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.device == "cuda":
        torch.cuda.set_device(rank)
    cfg = build_cfg(args)

    torch.manual_seed(cfg.execution.random_seed + rank)
    rng = random.Random(cfg.execution.random_seed + rank)

    device = torch.device(rank if args.device == "cuda" else "cpu")
    model = MLP(args.dim).to(device)
    ddp_model = DDP(model, device_ids=[rank]) if args.device == "cuda" else DDP(model)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    inspector = ModelInspector(model, optimizer)
    tracker = StageTracker(cfg.run_mode)
    injector = FaultInjector(cfg, inspector, tracker, rng, optimizer)
    if not args.baseline:
        injector.arm(ddp_model)

    recovery = RecoveryManager(cfg, ddp_model, optimizer, tracker) if cfg.mitigation.enabled else None

    # Different data per rank (normal data parallelism).
    x = torch.randn(args.batch_size, args.dim, device=device)
    y = torch.randint(0, 10, (args.batch_size,), device=device)

    step = 0
    step_times = []
    loss_history = []
    run_start = time.perf_counter()
    step_start = time.perf_counter()
    while step < args.steps:
        if recovery is not None:
            recovery.begin_step()
        optimizer.zero_grad()
        logits = ddp_model(x)
        tracker.on_forward_done()
        loss = criterion(logits, y)
        injector.inject_loss(loss)
        loss.backward()

        signals = recovery.check(loss, injector=injector) if recovery is not None else []
        action = recovery.decide(signals) if recovery is not None else "continue"

        if action == "retry_step":
            print(f"[rank{rank}] step {step}: {action} after {[s.kind for s in signals]}")
            tracker.on_retry()
            continue
        if action == "restored_checkpoint":
            restored = int(recovery.last_restored_step or 0)
            print(
                f"[rank{rank}] step {step}: restored checkpoint at step {restored} "
                f"after {[s.kind for s in signals]}"
            )
            tracker.on_retry()
            tracker.step = restored
            step = restored
            loss_history = loss_history[:restored]
            step_times = step_times[:restored]
            continue
        if action == "abort":
            print(f"[rank{rank}] step {step}: ABORT after {[s.kind for s in signals]}")
            dist.destroy_process_group()
            raise SystemExit(2)

        tracker.on_backward_done()
        optimizer.step()
        tracker.on_optimizer_done()
        tracker.on_comm_done()

        if recovery is not None:
            recovery.end_step(loss, clean=not signals)
        if cfg.execution.verbosity >= 1:
            print(f"[rank{rank} step{step}] loss={loss.item():.4f}")
        loss_history.append(float(loss.detach().item()))
        step_times.append(time.perf_counter() - step_start)
        step_start = time.perf_counter()
        step += 1

    injector.disarm()

    metrics = all_gather_metrics(ddp_model, loss)
    records = gather_records(injector)
    signature = param_signature(ddp_model)
    local_payload = {
        "rank": rank,
        "loss_history": {"rank": rank, "losses": list(loss_history)},
        "step_times": {"rank": rank, "times": list(step_times)},
        "peak_memory_gb": {
            "rank": rank,
            "gb": torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else None,
        },
        "param_signature": {"rank": rank, **signature},
        "recovery_events": list(recovery.events) if recovery is not None else [],
    }
    rank_payloads = all_gather_object(local_payload)
    wall_time_s = time.perf_counter() - run_start
    if rank == 0:
        run_id = args.run_id or f"ddp-{args.device}-seed{args.seed}-{int(time.time())}"
        summary = build_summary(
            run_id=run_id,
            cfg=cfg,
            baseline=args.baseline,
            steps=args.steps,
            records=records,
            per_rank_metrics=metrics,
            recovery_events=[
                event for payload in rank_payloads for event in payload["recovery_events"]
            ],
            loss_history=[payload["loss_history"] for payload in rank_payloads],
            step_times=[payload["step_times"] for payload in rank_payloads],
            peak_memory_gb=[payload["peak_memory_gb"] for payload in rank_payloads],
            param_signatures=[payload["param_signature"] for payload in rank_payloads],
            device=args.device,
            world_size=world_size,
            wall_time_s=wall_time_s,
        )
        summary_path = write_summary(args.log_dir, summary)

    if rank == 0:
        print("\nPer-rank metrics:")
        for m in metrics:
            print(f"  rank{m['rank']}: loss={m['loss']:.6f} grad_norm={m['grad_norm']:.6f}")
        print(f"\nInjection records ({len(records)}):")
        for rec in records:
            print(" ", rec)
        if injector.comm_signals:
            print(f"Unconsumed comm signals: {injector.comm_signals}")
        if recovery is not None:
            print(f"Recovery events: {summary['rollback_events']}")
        if step_times:
            print(f"Mean step time: {1000 * sum(step_times) / len(step_times):.2f} ms")
        if args.device == "cuda":
            print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
        print(f"Summary: {summary_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
