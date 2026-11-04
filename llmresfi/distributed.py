# llmresfi/distributed.py
"""Distributed fault injection for DDP and FSDP.

The layer has three jobs:

1. Select *where* a communication fault is injected (rank subset + bucket).
2. Select *when* it is injected (before or after the gradient collective).
3. Optionally verify the collective with a cheap checksum so that a corrupted
   reduce can be detected and handed to :mod:`llmresfi.mitigation`.

DDP hooks receive a gradient bucket immediately before DDP's all-reduce.
FSDP hooks receive the full (unsharded) gradient before its reduce-scatter (or
all-reduce with ``NO_SHARD``).  Both paths mirror PyTorch's default reduction
semantics, including gradient averaging by ``world_size``.
"""
from __future__ import annotations

import math
import random
import warnings
from typing import Iterable, List, Optional, Sequence, Set

import torch
import torch.distributed as dist

from .config import FIConfig
from .hooks import flip_flat


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def resolve_inject_ranks(cfg: FIConfig, world_size: Optional[int] = None) -> Set[int]:
    """Resolve the configured rank policy deterministically on every rank.

    ``inject_rank_fraction`` is sampled from a seed derived from
    ``execution.random_seed``, so every rank independently computes the same
    subset without an extra collective.
    """
    if world_size is None:
        world_size = dist.get_world_size() if dist_ready() else 1
    d = cfg.distributed

    if d.inject_ranks is not None:
        return {r for r in d.inject_ranks if 0 <= r < world_size}
    if d.inject_rank is not None:
        return {d.inject_rank} if 0 <= d.inject_rank < world_size else set()

    fraction = float(d.inject_rank_fraction)
    if fraction >= 1.0:
        return set(range(world_size))
    if fraction <= 0.0 or world_size == 0:
        return set()
    count = max(1, int(round(world_size * fraction)))
    rng = random.Random(int(cfg.execution.random_seed) ^ 0x5EED)
    return set(rng.sample(range(world_size), count))


def tensor_checksum(tensor: torch.Tensor) -> float:
    """Sum a tensor in float64 (cheap, and sensitive to most bit flips)."""
    return float(tensor.detach().to(torch.float64).sum().item())


def tensor_checksum_pair(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``[signed_sum, abs_sum]`` in float64."""
    flat = tensor.detach().to(torch.float64)
    return torch.stack((flat.sum(), flat.abs().sum()))


def _checksum_device(tensor: torch.Tensor) -> torch.device:
    if dist.get_backend() == "nccl":
        return tensor.device
    return torch.device("cpu")


def _all_gather_values(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """All-gather a per-rank float64 vector; returns shape ``[world, K]``."""
    if not dist_ready():
        return values.detach().to(torch.float64).reshape(1, -1).cpu()
    world_size = dist.get_world_size()
    local = values.detach().to(torch.float64).to(device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return torch.stack(gathered)


def all_gather_object(obj):
    """Return the object from every rank (or ``[obj]`` when not distributed)."""
    if not dist_ready():
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def gather_records(injector) -> List[dict]:
    """Aggregate ``injector.records`` from all ranks onto every rank."""
    return [rec for part in all_gather_object(list(injector.records)) for rec in (part or [])]


def global_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().double().pow(2).sum().item())
    return total ** 0.5


def all_gather_metrics(model, loss: Optional[float] = None) -> List[dict]:
    """Collect per-rank loss/gradient metrics (useful for divergence studies)."""
    local = {
        "rank": dist.get_rank() if dist_ready() else 0,
        "loss": None if loss is None else float(loss),
        "grad_norm": global_grad_norm(model),
    }
    return all_gather_object(local)


# ---------------------------------------------------------------------------
# base interface
# ---------------------------------------------------------------------------
class CommInjector:
    """Base interface for injecting faults into distributed communication."""

    def __init__(self, cfg: FIConfig, injector) -> None:
        self.cfg = cfg
        self.injector = injector
        self._enabled = False
        self._ranks: Set[int] = set()
        self._world_size = 1
        self._local_rank = 0
        self._last_step = None
        self._bucket_counter = 0

    def arm(self, model=None):
        raise NotImplementedError

    def disarm(self):
        self._enabled = False

    def inject_now(self) -> None:
        """Inject outside a collective hook (legacy FSDP gradient shard)."""
        return None

    # -- common helpers ---------------------------------------------------
    def _rank_allowed(self) -> bool:
        return self._local_rank in self._ranks

    def _next_bucket_index(self) -> int:
        step = getattr(self.injector.tracker, "step", 0)
        if step != self._last_step:
            self._last_step = step
            self._bucket_counter = 0
        idx = self._bucket_counter
        self._bucket_counter += 1
        return idx

    def _should_inject(self, bucket_index: int) -> bool:
        if not self._enabled:
            return False
        if not self._rank_allowed():
            return False
        target_bucket = self.cfg.distributed.bucket_index
        if target_bucket is not None and bucket_index != target_bucket:
            return False
        return bool(self.injector._should_fire_comm())

    def _inject(self, tensor: torch.Tensor, bucket_index: int, point: str, **extra) -> None:
        if tensor.numel() == 0:
            return
        flat_idx = self.injector.rng.randrange(tensor.numel())
        bits = flip_flat(tensor, flat_idx, self.cfg.fault.fault_model, self.injector.rng)
        rec = self.injector._make_record(
            "comm",
            injection_point=point,
            bucket_index=bucket_index,
            bucket_numel=int(tensor.numel()),
            dtype=str(tensor.dtype),
            flat_idx=flat_idx,
            bit_idxs=bits,
            **extra,
        )
        self.injector.record = rec

    def _verify(self, tensor: torch.Tensor, bucket_index: int, pre_checksums: torch.Tensor) -> None:
        post = tensor_checksum_pair(tensor).to(pre_checksums.device)
        scale = float(self._world_size) if self.cfg.distributed.divide_by_world_size else 1.0
        expected_sum = float(pre_checksums[:, 0].sum().item())
        expected_abs = float(pre_checksums[:, 1].sum().item())
        actual_sum = float(post[0].item()) * scale
        actual_abs = float(post[1].item()) * scale

        # (a) reduction consistency: the reduced tensor should equal the sum
        # of the per-rank pre-reduction checksums.
        try:
            eps = float(torch.finfo(tensor.dtype).eps)
        except TypeError:
            eps = 1e-6
        tolerance = max(
            1e-12,
            self.cfg.distributed.verify_atol * max(1.0, abs(expected_abs)),
            4.0 * eps * max(1.0, abs(expected_abs)),
        )
        sum_error = abs(actual_sum - expected_sum)
        reduction_bad = (
            not math.isfinite(actual_sum)
            or sum_error > tolerance
        )

        # (b) cross-rank consistency: after a correct all-reduce every rank
        # holds the same value. This catches asymmetric (after-reduce) faults
        # even when the value change is tiny.
        post_all = _all_gather_values(post, _checksum_device(tensor))
        mean_sum = float(post_all[:, 0].mean().item())
        mean_abs = float(post_all[:, 1].mean().item())
        cross_tol = max(1e-12, 1e-12 * max(1.0, abs(mean_abs)))
        cross_bad = bool(
            (post_all[:, 0] - mean_sum).abs().max().item() > cross_tol
            or (post_all[:, 1] - mean_abs).abs().max().item() > cross_tol
        )

        if reduction_bad or cross_bad:
            self.injector.add_comm_signal(
                {
                    "kind": "comm_checksum_mismatch",
                    "backend": self.cfg.distributed.backend,
                    "rank": self._local_rank,
                    "bucket_index": bucket_index,
                    "expected_sum": expected_sum,
                    "actual_sum": actual_sum,
                    "expected_abs": expected_abs,
                    "actual_abs": actual_abs,
                    "sum_error": sum_error,
                    "tolerance": tolerance,
                    "reduction_mismatch": reduction_bad,
                    "cross_rank_mismatch": cross_bad,
                }
            )


# ---------------------------------------------------------------------------
# DDP
# ---------------------------------------------------------------------------
class DDPCommInjector(CommInjector):
    """Inject into DDP gradient buckets.

    ``before_reduce`` corrupts the local bucket first, so the error is averaged
    into every rank (a silent-error propagation scenario).
    ``after_reduce`` corrupts the reduced bucket on the selected rank only, so
    that rank silently diverges from its peers.
    """

    def __init__(self, cfg: FIConfig, injector) -> None:
        super().__init__(cfg, injector)
        self._model = None

    def arm(self, model=None):
        from torch.nn.parallel import DistributedDataParallel as DDP

        if self._enabled:
            return self
        model = model if model is not None else self._model
        if model is None:
            raise ValueError(
                "DDP comm injection requires the DDP-wrapped model: "
                "call injector.arm(ddp_model)."
            )
        if not isinstance(model, DDP):
            raise TypeError(
                "DDP comm injection expects a DistributedDataParallel instance, "
                f"got {type(model).__name__}."
            )
        if not dist_ready():
            raise RuntimeError(
                "DDP comm injection requires an initialized process group. "
                "Call torch.distributed.init_process_group() first."
            )
        self._model = model
        self._world_size = dist.get_world_size()
        self._local_rank = dist.get_rank()
        self._ranks = resolve_inject_ranks(self.cfg, self._world_size)
        self._enabled = True
        model.register_comm_hook(self, self.comm_hook)
        return self

    def disarm(self):
        # PyTorch does not expose a public unregister API. The hook stays
        # registered but becomes a pass-through reduction.
        super().disarm()

    def comm_hook(self, state, bucket):
        tensor = bucket.buffer()
        bucket_index = self._next_bucket_index()
        fire = self._should_inject(bucket_index)
        point = self.cfg.distributed.injection_point

        verify = bool(self.cfg.distributed.verify_comm) and dist_ready()
        pre_checksums = None
        if verify:
            pre_checksums = _all_gather_values(
                tensor_checksum_pair(tensor), _checksum_device(tensor)
            )

        if fire and point == "before_reduce":
            self._inject(tensor, bucket_index, point, backend="ddp")

        fut = dist.all_reduce(tensor, async_op=True).get_future()

        def _after(_fut):
            _fut.wait()
            if self.cfg.distributed.divide_by_world_size:
                tensor.div_(self._world_size)
            if fire and point == "after_reduce":
                self._inject(tensor, bucket_index, point, backend="ddp")
            if verify:
                self._verify(tensor, bucket_index, pre_checksums)
            return tensor

        return fut.then(_after)


# ---------------------------------------------------------------------------
# FSDP
# ---------------------------------------------------------------------------
class FSDPCommInjector(CommInjector):
    """Inject FSDP faults.

    Modern PyTorch (>= 2.0) exposes ``register_comm_hook``; the hook receives
    the full unsharded gradient and we inject before/after reduce-scatter (or
    all-reduce for ``NO_SHARD``).

    PyTorch 1.12 has no FSDP comm hook.  The fallback injects into a local
    shard, selected by ``distributed.fsdp_injection``:

    - ``param_shard``: flat-parameter shard before all-gather (forward phase).
    - ``grad_shard``: sharded gradient before ``optimizer.step``.

    These are genuine rank-local shard faults; they are not identical to
    injecting a full gradient before reduce-scatter, so records are marked
    ``legacy_fsdp=True``.
    """

    def __init__(self, cfg: FIConfig, injector) -> None:
        super().__init__(cfg, injector)
        self._root = None
        self._pg = None
        self._previous_hooks = []
        self._legacy_modules = []
        self._legacy_handles = []
        self._legacy_original_step = None
        self._legacy = False

    def arm(self, model=None):
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except ImportError as exc:  # pragma: no cover - depends on torch build
            raise RuntimeError("FSDP is not available in this PyTorch build") from exc

        if self._enabled:
            return self
        model = model if model is not None else self._root
        if model is None:
            raise ValueError(
                "FSDP comm injection requires the FSDP-wrapped model: "
                "call injector.arm(fsdp_model)."
            )
        if not dist_ready():
            raise RuntimeError(
                "FSDP comm injection requires an initialized process group."
            )

        root = model if isinstance(model, FSDP) else None
        if root is None:
            roots = [m for m in model.modules() if isinstance(m, FSDP)]
            if not roots:
                raise TypeError(
                    "FSDP comm injection expects a model containing "
                    "FullyShardedDataParallel modules."
                )
            root = roots[0]

        self._root = root
        self._pg = getattr(root, "process_group", None) or dist.group.WORLD
        self._world_size = dist.get_world_size(self._pg)
        self._local_rank = dist.get_rank()
        self._ranks = resolve_inject_ranks(self.cfg, self._world_size)

        if self.cfg.distributed.verify_comm:
            warnings.warn(
                "verify_comm is currently implemented for DDP only; "
                "FSDP injection will run without checksum verification.",
                RuntimeWarning,
                stacklevel=2,
            )

        if hasattr(root, "register_comm_hook"):
            self._arm_modern(root)
        else:
            self._arm_legacy(model)
        self._enabled = True
        return self

    def _arm_modern(self, root) -> None:
        try:
            from torch.distributed.fsdp._traversal_utils import _get_fsdp_states

            states = list(_get_fsdp_states(root))
            for fsdp_state in states:
                if getattr(fsdp_state, "_comm_hook", None) is not None:
                    raise RuntimeError("An FSDP communication hook is already registered")
        except ImportError:
            states = []
        self._previous_hooks = [
            (s, getattr(s, "_comm_hook", None), getattr(s, "_comm_hook_state", None))
            for s in states
        ]
        root.register_comm_hook(self, self._fsdp_hook)

    def _arm_legacy(self, model) -> None:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        self._legacy = True
        self._legacy_modules = [m for m in model.modules() if isinstance(m, FSDP)]
        if not self._legacy_modules:
            raise TypeError("No FSDP modules found for legacy injection")

        mode = self.cfg.distributed.fsdp_injection
        if mode in ("param_shard", "both"):
            for module in self._legacy_modules:
                self._legacy_handles.append(
                    module.register_forward_pre_hook(self._param_shard_hook)
                )
        if mode in ("grad_shard", "both") and self.injector.optimizer is None:
            raise ValueError("FSDP grad_shard injection requires an optimizer")

    def disarm(self):
        super().disarm()
        for handle in self._legacy_handles:
            handle.remove()
        self._legacy_handles = []
        for fsdp_state, hook, hook_state in self._previous_hooks:
            fsdp_state._comm_hook = hook
            fsdp_state._comm_hook_state = hook_state
        self._previous_hooks = []
        self._legacy_modules = []

    # -- legacy shard injection (torch < 2.0) ----------------------------
    @staticmethod
    def _flat_param(module):
        flat = getattr(module, "_flat_param", None)
        if flat is None:
            flat = getattr(module, "flat_param", None)
        return flat

    def _legacy_phase_ok(self, kind: str) -> bool:
        spec = self.cfg.fault.stage
        if spec.type == "comm":
            expected = "forward" if kind == "param_shard" else "optimizer_step"
            return self.injector.tracker.stage == expected
        return self.injector.tracker.matches(spec.type, spec.token_step)

    def _legacy_should_inject(self, kind: str, bucket_index: int) -> bool:
        if not self._enabled or not self._rank_allowed():
            return False
        target_bucket = self.cfg.distributed.bucket_index
        if target_bucket is not None and bucket_index != target_bucket:
            return False
        return bool(
            self.injector._should_fire_with_phase(self._legacy_phase_ok(kind))
        )

    def _param_shard_hook(self, module, _inputs):
        if not self._enabled:
            return
        bucket_index = self._next_bucket_index()
        if not self._legacy_should_inject("param_shard", bucket_index):
            return
        flat = self._flat_param(module)
        if flat is None or flat.data.numel() == 0:
            return
        self._inject(
            flat.data,
            bucket_index,
            "fsdp_param_shard",
            backend="fsdp",
            sharded=True,
            legacy_fsdp=True,
            module_path=type(module).__name__,
        )

    def _maybe_inject_grad_shard(self) -> None:
        if not self._enabled:
            return
        bucket_index = self._next_bucket_index()
        if not self._legacy_should_inject("grad_shard", bucket_index):
            return
        candidates = []
        for module in self._legacy_modules:
            flat = self._flat_param(module)
            if flat is not None and flat.grad is not None and flat.grad.numel() > 0:
                candidates.append((module, flat))
        if not candidates:
            return
        module, flat = self.injector.rng.choice(candidates)
        self._inject(
            flat.grad,
            bucket_index,
            "fsdp_grad_shard",
            backend="fsdp",
            sharded=True,
            legacy_fsdp=True,
            module_path=type(module).__name__,
        )

    def inject_now(self) -> None:
        """Run the legacy gradient-shard injection at the start of a step.

        Call this after ``backward``/``on_backward_done`` and before
        ``optimizer.step`` so that the recovery detector can observe the fault.
        """
        if self._legacy and self.cfg.distributed.fsdp_injection in ("grad_shard", "both"):
            self._maybe_inject_grad_shard()

    def _fsdp_hook(self, state, *args):
        bucket_index = self._next_bucket_index()
        fire = self._should_inject(bucket_index)
        point = self.cfg.distributed.injection_point

        if len(args) == 1:
            # NO_SHARD: the hook receives the full gradient and must all-reduce.
            grad = args[0]
            if fire and point == "before_reduce":
                self._inject(grad, bucket_index, point, backend="fsdp", sharded=False)
            dist.all_reduce(grad, group=self._pg)
            if self.cfg.distributed.divide_by_world_size:
                grad.div_(self._world_size)
            if fire and point == "after_reduce":
                self._inject(grad, bucket_index, point, backend="fsdp", sharded=False)
            return None

        if len(args) == 2:
            # Sharded strategies: (full_grad, sharded_grad_out).
            full_grad, sharded_grad = args
            if fire and point == "before_reduce":
                self._inject(full_grad, bucket_index, point, backend="fsdp", sharded=True)
            dist.reduce_scatter_tensor(sharded_grad, full_grad, group=self._pg)
            if self.cfg.distributed.divide_by_world_size:
                sharded_grad.div_(self._world_size)
            if fire and point == "after_reduce":
                self._inject(sharded_grad, bucket_index, point, backend="fsdp", sharded=True)
            return None

        raise TypeError(
            f"Unexpected FSDP comm-hook signature: got {len(args)} tensor arguments"
        )


def make_comm_injector(cfg: FIConfig, injector) -> CommInjector:
    backend = cfg.distributed.backend
    if backend == "ddp":
        return DDPCommInjector(cfg, injector)
    if backend == "fsdp":
        return FSDPCommInjector(cfg, injector)
    raise ValueError(f"Unknown distributed backend for comm injection: {backend}")
