# examples/resnet18/run_resnet18.py
"""ResNet-18 / CIFAR-10 distributed fault-injection experiment.

The default output root is the ``ResNet-18`` directory next to this repository.
Every scenario writes ``summary.json``, ``recovery.jsonl``, and
``training_metrics.csv`` below its own run directory.

Examples
--------
Baseline::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        examples/resnet18/run_resnet18.py --scenario baseline --baseline

After-reduce fault without verification (SDC)::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        examples/resnet18/run_resnet18.py --scenario fault_sdc \
        --fault_model exponent --injection_point after_reduce

Detected and losslessly rolled back::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        examples/resnet18/run_resnet18.py --scenario fault_recovered \
        --fault_model exponent --injection_point after_reduce \
        --verify --mitigation step_rollback
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet18

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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT.parent / "ResNet-18"


def make_model() -> nn.Module:
    """CIFAR-style ResNet-18: 3x3 stem and no max-pool."""
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def make_transforms():
    # No random augmentation: the same batch must be exactly replayable after a
    # rollback. CIFAR-10 still provides a real image-classification workload.
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    return train_transform, test_transform


def build_cfg(args, log_dir: Path) -> FIConfig:
    mitigation = MitigationCfg(enabled=args.mitigation != "none")
    if mitigation.enabled:
        mitigation.detector = DetectorCfg(
            non_finite=True,
            check_parameters=True,
            warmup_steps=0,
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
        model=ModelCfg(repo="resnet18-cifar10", dtype="float32", device="cuda"),
        distributed=DistributedCfg(
            backend="ddp",
            inject_rank=args.inject_rank,
            injection_point=args.injection_point,
            bucket_index=args.bucket_index,
            verify_comm=args.verify,
            verify_atol=args.verify_atol,
            divide_by_world_size=True,
        ),
        fault=FaultCfg(
            target="comm",
            stage=StageCfg(type="backward"),
            fault_model=FaultModelCfg(type=args.fault_model, bit_idx=args.bit_idx),
            timing=TimingCfg(type=args.timing),
        ),
        execution=ExecCfg(random_seed=args.seed, log_dir=log_dir),
        mitigation=mitigation,
    )


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def optimizer_state_sha256(optimizer) -> str:
    """Hash per-parameter optimizer tensors in a stable parameter order."""
    digest = hashlib.sha256()
    for group_index, group in enumerate(optimizer.param_groups):
        digest.update(f"group:{group_index}".encode("ascii"))
        for key in sorted(group):
            if key == "params":
                continue
            digest.update(key.encode("utf-8"))
            digest.update(repr(group[key]).encode("utf-8"))
        for param_index, param in enumerate(group["params"]):
            state = optimizer.state.get(param, {})
            digest.update(f"param:{group_index}:{param_index}".encode("ascii"))
            for name in sorted(state):
                digest.update(name.encode("utf-8"))
                value = state[name]
                if torch.is_tensor(value):
                    tensor = value.detach().cpu().contiguous()
                    digest.update(str(tensor.dtype).encode("ascii"))
                    digest.update(str(tuple(tensor.shape)).encode("ascii"))
                    digest.update(tensor.numpy().tobytes())
                else:
                    digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def make_train_loader(dataset, args, world_size: int, rank: int):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def infinite_batches(loader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def resume_batches(loader, sampler, skip: int):
    iterator = infinite_batches(loader, sampler)
    for _ in range(skip):
        next(iterator)
    return iterator, next(iterator)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    local_loss = torch.zeros(1, dtype=torch.float64, device=device)
    local_correct = torch.zeros(1, dtype=torch.float64, device=device)
    local_count = torch.zeros(1, dtype=torch.float64, device=device)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        local_loss += loss.detach().double() * targets.numel()
        local_correct += (logits.argmax(dim=1) == targets).sum().double()
        local_count += targets.numel()
    stats = torch.cat((local_loss, local_correct, local_count))
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    model.train()
    count = max(1.0, float(stats[2].item()))
    return float(stats[0].item()) / count, float(stats[1].item()) / count


def write_metrics_csv(path: Path, rank_payloads) -> None:
    rows = []
    for payload in rank_payloads:
        rank = payload["rank"]
        losses = payload["loss_history"]["losses"]
        times = payload["step_times"]["times"]
        for step, (loss, elapsed) in enumerate(zip(losses, times)):
            rows.append((rank, step, loss, elapsed))
    rows.sort(key=lambda row: (row[1], row[0]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "step", "loss", "step_time_s"])
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="resnet18")
    ap.add_argument("--experiment_root", default=str(DEFAULT_EXPERIMENT_ROOT))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--no_download", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--fault_model", default="exponent")
    ap.add_argument(
        "--bit_idx",
        type=int,
        default=None,
        help="Pin the flipped bit (e.g. 25 for a strong float32 exponent fault).",
    )
    ap.add_argument("--timing", default="transient", choices=["transient", "permanent", "intermittent"])
    ap.add_argument("--injection_point", default="after_reduce", choices=["before_reduce", "after_reduce"])
    ap.add_argument("--inject_rank", type=int, default=0)
    ap.add_argument("--bucket_index", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify_atol", type=float, default=1e-6)
    ap.add_argument("--mitigation", default="none", choices=["none", "sanitize", "step_rollback", "checkpoint_rollback"])
    ap.add_argument("--max_retries", type=int, default=1)
    ap.add_argument("--checkpoint_every", type=int, default=0)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()

    if "RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
        raise SystemExit("Launch this example with torchrun (two processes recommended).")
    if not torch.cuda.is_available():
        raise SystemExit("ResNet-18 experiment requires CUDA.")

    experiment_root = Path(args.experiment_root)
    log_dir = Path(args.log_dir) if args.log_dir else experiment_root / "runs" / args.scenario
    data_root = Path(args.data_root) if args.data_root else experiment_root / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    # Always start with a clean event log for this run. RecoveryManager appends
    # events as they occur; runs without mitigation leave the file empty.
    (log_dir / "recovery.jsonl").write_text("", encoding="utf-8")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_transform, test_transform = make_transforms()
    cifar_dir = data_root / "cifar-10-batches-py"
    if not cifar_dir.exists():
        if args.no_download:
            raise SystemExit(f"CIFAR-10 not found at {data_root}; rerun without --no_download.")
        if rank == 0:
            datasets.CIFAR10(root=str(data_root), train=True, download=True)
            datasets.CIFAR10(root=str(data_root), train=False, download=True)
        dist.barrier()
    train_set = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        transform=train_transform,
        download=False,
    )
    test_set = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        transform=test_transform,
        download=False,
    )

    # Identical initialization on every rank and in every scenario.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = make_model().to(device)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        ddp_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    train_loader, train_sampler = make_train_loader(train_set, args, world_size, rank)
    test_sampler = DistributedSampler(
        test_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    cfg = build_cfg(args, log_dir)
    tracker = StageTracker("training")
    rng = random.Random(args.seed + rank)
    inspector = ModelInspector(model, optimizer)
    injector = FaultInjector(cfg, inspector, tracker, rng, optimizer)
    if not args.baseline:
        injector.arm(ddp_model)
    recovery = RecoveryManager(cfg, ddp_model, optimizer, tracker, log_dir=log_dir) if cfg.mitigation.enabled else None

    torch.cuda.reset_peak_memory_stats(device)
    batch_iterator, current_batch = resume_batches(train_loader, train_sampler, 0)
    step = 0
    attempts = 0
    retries = 0
    losses = []
    step_times = []
    run_start = time.perf_counter()
    step_start = time.perf_counter()

    while step < args.steps:
        attempts += 1
        if recovery is not None:
            recovery.begin_step()
        images, targets = current_batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = ddp_model(images)
        tracker.on_forward_done()
        loss = criterion(logits, targets)
        loss.backward()

        signals = recovery.check(loss, injector=injector) if recovery is not None else []
        action = recovery.decide(signals) if recovery is not None else "continue"

        if action == "retry_step":
            retries += 1
            print(
                f"[rank{rank}] step {step}: retry_step after "
                f"{[signal.kind for signal in signals]}",
                flush=True,
            )
            tracker.on_retry()
            continue

        if action == "restored_checkpoint":
            restored = int(recovery.last_restored_step or 0)
            retries += 1
            print(
                f"[rank{rank}] step {step}: restored checkpoint at step {restored}",
                flush=True,
            )
            tracker.on_retry()
            tracker.step = restored
            step = restored
            losses = losses[:restored]
            step_times = step_times[:restored]
            batch_iterator, current_batch = resume_batches(
                train_loader, train_sampler, restored
            )
            step_start = time.perf_counter()
            continue

        if action == "abort":
            print(
                f"[rank{rank}] step {step}: ABORT after "
                f"{[signal.kind for signal in signals]}",
                flush=True,
            )
            dist.destroy_process_group()
            raise SystemExit(2)

        tracker.on_backward_done()
        optimizer.step()
        tracker.on_optimizer_done()
        tracker.on_comm_done()
        if recovery is not None:
            recovery.end_step(loss, clean=not signals)

        losses.append(float(loss.detach().item()))
        step_times.append(time.perf_counter() - step_start)
        print(f"[rank{rank} step{step}] loss={losses[-1]:.4f}", flush=True)
        step += 1
        current_batch = next(batch_iterator)
        step_start = time.perf_counter()

    injector.disarm()
    test_loss, test_accuracy = evaluate(ddp_model, test_loader, criterion, device)
    final_metrics = all_gather_metrics(ddp_model, loss)
    records = gather_records(injector)
    signature = param_signature(ddp_model)
    model_hash = state_sha256(ddp_model)
    optimizer_hash = optimizer_state_sha256(optimizer)
    wall_time_s = time.perf_counter() - run_start
    local_payload = {
        "rank": rank,
        "loss_history": {"rank": rank, "losses": losses},
        "step_times": {"rank": rank, "times": step_times},
        "peak_memory_gb": {
            "rank": rank,
            "gb": torch.cuda.max_memory_allocated(device) / 1e9,
        },
        "param_signature": {"rank": rank, **signature},
        "state_sha256": {"rank": rank, "sha256": model_hash},
        "optimizer_state_sha256": {"rank": rank, "sha256": optimizer_hash},
        "attempts": attempts,
        "recovery_events": list(recovery.events) if recovery is not None else [],
    }
    rank_payloads = all_gather_object(local_payload)

    if rank == 0:
        run_id = args.run_id or f"{args.scenario}-seed{args.seed}"
        summary = build_summary(
            run_id=run_id,
            cfg=cfg,
            baseline=args.baseline,
            steps=args.steps,
            records=records,
            per_rank_metrics=final_metrics,
            recovery_events=[
                event for payload in rank_payloads for event in payload["recovery_events"]
            ],
            loss_history=[payload["loss_history"] for payload in rank_payloads],
            step_times=[payload["step_times"] for payload in rank_payloads],
            peak_memory_gb=[payload["peak_memory_gb"] for payload in rank_payloads],
            param_signatures=[payload["param_signature"] for payload in rank_payloads],
            device="cuda",
            world_size=world_size,
            wall_time_s=wall_time_s,
            extra={
                "dataset": "CIFAR-10",
                "model_name": "resnet18-cifar10",
                "effective_batch_size": args.batch_size * world_size,
                "optimizer": {"name": "SGD", "lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay},
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "attempts": max(payload["attempts"] for payload in rank_payloads),
                "attempts_all_ranks": sum(payload["attempts"] for payload in rank_payloads),
                "retried_steps": retries,
                "state_sha256": [payload["state_sha256"] for payload in rank_payloads],
                "optimizer_state_sha256": [
                    payload["optimizer_state_sha256"] for payload in rank_payloads
                ],
            },
        )
        write_summary(log_dir, summary)
        write_metrics_csv(log_dir / "training_metrics.csv", rank_payloads)
        print("\nResNet-18 result", flush=True)
        print(f"  run_id: {run_id}", flush=True)
        print(f"  final_loss: {losses[-1]:.6f}", flush=True)
        print(f"  test_loss: {test_loss:.6f}", flush=True)
        print(f"  test_accuracy: {test_accuracy:.4f}", flush=True)
        print(f"  injected_faults: {len(records)}", flush=True)
        print(f"  recovery_events: {summary['rollback_events']}", flush=True)
        print(f"  attempts: {summary['attempts']}", flush=True)
        print(f"  wall_time_s: {wall_time_s:.3f}", flush=True)
        print(f"  summary: {log_dir / 'summary.json'}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
