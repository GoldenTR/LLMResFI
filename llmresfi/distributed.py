# llmresfi/distributed.py
"""Distributed fault injection layer (DDP + FSDP).

v0.1 implements a clean ``CommInjector`` interface and a minimal DDP
implementation via ``register_comm_hook``. FSDP is a documented stub and will
be implemented in a later phase.
"""
from __future__ import annotations

from typing import Optional

import torch.distributed as dist

from .config import FIConfig


class CommInjector:
    """Base interface for injecting faults into distributed-communication tensors."""

    def __init__(self, cfg: FIConfig, injector) -> None:
        self.cfg = cfg
        self.injector = injector

    def arm(self):
        raise NotImplementedError

    def disarm(self):
        raise NotImplementedError


class DDPCommInjector(CommInjector):
    """Inject into gradient buckets before/after the all-reduce in DDP.

    ``inject_rank`` (config) controls which rank(s) perform the injection,
    enabling studies of inconsistent gradients across data-parallel ranks.
    """

    def __init__(self, cfg: FIConfig, injector) -> None:
        super().__init__(cfg, injector)
        self._handle = None

    def _rank_allowed(self) -> bool:
        rank = self.cfg.distributed.inject_rank
        if rank is None:
            return True
        return dist.is_initialized() and dist.get_rank() == rank

    def arm(self):
        # The model must be a DDP-wrapped module; the comm hook is registered
        # by the caller on the DDP wrapper. Here we return the hook function.
        pass

    def comm_hook(self, state, bucket):
        """A DDP communication hook: inject before the all-reduce."""
        tensor = bucket.buffer()
        if self._rank_allowed() and self.injector._should_fire():
            from .hooks import flip_flat

            flat_idx = self.injector.rng.randrange(tensor.numel())
            bits = flip_flat(tensor, flat_idx, self.cfg.fault.fault_model, self.injector.rng)
            self.injector.record = self.injector._make_record(
                "comm", None, "ddp", flat_idx=flat_idx, bit_idxs=bits, backend="ddp"
            )
        return dist.all_reduce(tensor, async_op=True).get_future()

    def disarm(self):
        self._handle = None


class FSDPCommInjector(CommInjector):
    """Placeholder for FSDP (ZeRO) fault injection.

    Planned hook points: post-backward gradient reduction, reshard, and
    parameter gather. Not implemented in v0.1.
    """

    def arm(self):
        raise NotImplementedError(
            "FSDP comm injection is not implemented in v0.1. "
            "See docs/design.md for the planned hook points."
        )

    def disarm(self):
        pass


def make_comm_injector(cfg: FIConfig, injector) -> CommInjector:
    backend = cfg.distributed.backend
    if backend == "ddp":
        return DDPCommInjector(cfg, injector)
    if backend == "fsdp":
        return FSDPCommInjector(cfg, injector)
    raise ValueError(f"Unknown distributed backend for comm injection: {backend}")
