# llmresfi/injector.py
"""FaultInjector: unified injection engine for inference and training.

Handles *where* (target/layer/module/param), *when* (stage + timing), and
*how* (fault model) to inject a soft error.

Usage
-----
    rng = random.Random(seed)
    inspector = ModelInspector(model, optimizer)
    tracker = StageTracker(cfg.run_mode)
    injector = FaultInjector(cfg, inspector, tracker, rng, optimizer)

    injector.arm()          # register hooks / patch optimizer / comm hooks
    # ... run training or inference, advancing `tracker` at each stage ...
    injector.disarm()       # remove hooks, restore weights

Timing semantics (see ``hooks.TimingGate``):
    transient:    fire once (default; one fault per run)
    permanent:    re-fire on the first ``steps`` eligible triggers
    intermittent: fire on every ``period``-th eligible trigger

Weight faults are single-shot by nature: they flip once and remain corrupted
until ``disarm()`` (persistence is achieved by keeping the flip, not by
re-flipping).
"""
from __future__ import annotations

import random
import re
from typing import List, Optional, Tuple

import torch

from .config import FIConfig
from .hooks import TimingGate, apply_fault_to_scalar, flip_flat, has_2d_weight
from .inspector import ModelInspector
from .scheduler import StageTracker


class FaultInjector:
    def __init__(
        self,
        cfg: FIConfig,
        inspector: ModelInspector,
        tracker: StageTracker,
        rng: random.Random,
        optimizer=None,
    ):
        self.cfg = cfg
        self.insp = inspector
        self.tracker = tracker
        self.rng = rng
        self.optimizer = optimizer

        self._hook_handles: List = []
        self._weight_backup: Optional[Tuple[torch.nn.Module, Tuple[int, int], torch.Tensor]] = None
        self._weight_fired = False
        self._original_step = None
        self._timing = TimingGate(cfg.fault.timing)
        self.record: Optional[dict] = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def arm(self):
        target = self.cfg.fault.target
        if target == "weight":
            self._arm_weight()
        elif target == "activation":
            self._arm_activation()
        elif target == "kv_cache":
            self._arm_kv_cache()
        elif target == "gradient":
            self._arm_gradient()
        elif target == "optimizer":
            self._arm_optimizer()
        elif target == "master_weight":
            self._arm_master_weight()
        elif target == "loss":
            pass  # loss is injected explicitly via inject_loss() during the loop
        elif target == "comm":
            self._arm_comm()
        else:
            raise ValueError(f"Unknown fault target: {target}")

    def disarm(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        if self._weight_backup is not None:
            module, (r, c), orig = self._weight_backup
            with torch.no_grad():
                module.weight.data[r, c] = orig
            self._weight_backup = None
        if self._original_step is not None and self.optimizer is not None:
            self.optimizer.step = self._original_step
            self._original_step = None
        self.disarm_comm()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _phase_ok(self) -> bool:
        s = self.cfg.fault.stage
        return self.tracker.matches(s.type, s.token_step)

    def _should_fire(self) -> bool:
        if self._phase_ok() and self._timing.should_fire():
            self._timing.mark_fired()
            return True
        return False

    def _resolve_target(self):
        f = self.cfg.fault
        if f.locate == "manual":
            if f.layer_idx is None or f.module_regex is None:
                raise ValueError("layer_idx and module_regex must be set when locate == 'manual'")
            return f.layer_idx, f.module_regex
        layer_idx = self.insp.random_layer(self.rng)
        return layer_idx, ".*"

    def _pick_weightable_module(self, layer_idx: int, mod_regex: str):
        pattern = re.compile(mod_regex)
        layer_list = self.insp.modules_by_layer.get(layer_idx, [])
        by_regex = [m for m in layer_list if pattern.fullmatch(m.name)]
        filtered = [m for m in by_regex if has_2d_weight(m)]
        if not filtered:
            filtered = [m for m in layer_list if has_2d_weight(m)]
        if not filtered:
            raise ValueError(
                f"No 2D-weight module found in layer {layer_idx} for regex='{mod_regex}'."
            )
        return self.rng.choice(filtered)

    # ------------------------------------------------------------------
    # weight (inference & training forward) - single-shot
    # ------------------------------------------------------------------
    def _arm_weight(self):
        layer_idx, mod_regex = self._resolve_target()
        mod_info = self._pick_weightable_module(layer_idx, mod_regex)

        def pre_hook(module, _inputs):
            if self._weight_fired or not self._phase_ok():
                return
            W = module.weight.data
            row = self.rng.randrange(W.shape[0])
            col = self.rng.randrange(W.shape[1])
            orig = W[row, col].clone()
            corrupted, bits = apply_fault_to_scalar(W[row, col], self.cfg.fault.fault_model, self.rng)
            W[row, col] = corrupted
            self._weight_backup = (module, (row, col), orig)
            self.record = self._make_record(
                "weight",
                layer_idx,
                mod_info.name,
                row=row,
                col=col,
                bit_idxs=bits,
            )
            self._weight_fired = True
            # weight stays corrupted until disarm; never re-fire
            handle = self._hook_handles[0]
            handle.remove()

        self._hook_handles.append(mod_info.module.register_forward_pre_hook(pre_hook))

    # ------------------------------------------------------------------
    # activation (inference & training forward) - ephemeral
    # ------------------------------------------------------------------
    def _arm_activation(self):
        layer_idx, mod_regex = self._resolve_target()
        mod_info = self.insp.pick_module(layer_idx, mod_regex, self.rng)

        def hook(_, __, output):
            if not self._should_fire():
                return output
            tensor = output[0] if isinstance(output, tuple) else output
            flat_idx = self.rng.randrange(tensor.numel())
            bits = flip_flat(tensor, flat_idx, self.cfg.fault.fault_model, self.rng)
            self.record = self._make_record(
                "activation", layer_idx, mod_info.name, flat_idx=flat_idx, bit_idxs=bits
            )
            return output

        self._hook_handles.append(mod_info.module.register_forward_hook(hook))

    # ------------------------------------------------------------------
    # kv_cache (inference)
    # ------------------------------------------------------------------
    def _arm_kv_cache(self):
        layer_idx, mod_regex = self._resolve_target()
        attn_mod = self.insp.pick_module(layer_idx, mod_regex, self.rng).module

        def hook(module, _, output):
            if not self._should_fire():
                return output
            pkv = output[-1]
            lyr = layer_idx % len(pkv)
            K, V = pkv[lyr]
            tensor = K if self.rng.random() < 0.5 else V
            flat_idx = self.rng.randrange(tensor.numel())
            bits = flip_flat(tensor, flat_idx, self.cfg.fault.fault_model, self.rng)
            self.record = self._make_record(
                "kv_cache",
                layer_idx,
                attn_mod.__class__.__name__,
                flat_idx=flat_idx,
                bit_idxs=bits,
                kv="K" if tensor is K else "V",
            )
            return output

        self._hook_handles.append(attn_mod.register_forward_hook(hook))

    # ------------------------------------------------------------------
    # loss (training) - injected explicitly during the loop
    # ------------------------------------------------------------------
    def inject_loss(self, loss: torch.Tensor) -> Optional[dict]:
        if self.cfg.fault.target != "loss":
            return None
        if not self._should_fire():
            return None
        flat_idx = self.rng.randrange(loss.numel())
        bits = flip_flat(loss, flat_idx, self.cfg.fault.fault_model, self.rng)
        self.record = self._make_record("loss", None, None, flat_idx=flat_idx, bit_idxs=bits)
        return self.record

    # ------------------------------------------------------------------
    # gradient (training) - via parameter autograd hook
    # ------------------------------------------------------------------
    def _arm_gradient(self):
        param_info = self.insp.random_param(self.rng)
        p = param_info.param

        def grad_hook(grad):
            if grad is None or not self._should_fire():
                return grad
            flat_idx = self.rng.randrange(grad.numel())
            bits = flip_flat(grad, flat_idx, self.cfg.fault.fault_model, self.rng)
            self.record = self._make_record(
                "gradient", None, param_info.name, flat_idx=flat_idx, bit_idxs=bits
            )
            return grad

        self._hook_handles.append(p.register_hook(grad_hook))

    # ------------------------------------------------------------------
    # optimizer state (training) - patch optimizer.step()
    # ------------------------------------------------------------------
    def _arm_optimizer(self):
        if self.optimizer is None:
            raise ValueError("Optimizer injection requires an optimizer.")
        param_info = self.insp.random_param(self.rng)
        p = param_info.param
        original_step = self.optimizer.step

        def wrapped_step(*args, **kwargs):
            if self._should_fire():
                self._inject_optimizer_state(p, param_info.name)
            return original_step(*args, **kwargs)

        self.optimizer.step = wrapped_step
        self._original_step = original_step

    def _inject_optimizer_state(self, p, param_path: str):
        state = self.optimizer.state.get(p, {})
        if "exp_avg" in state:
            tensor, key = state["exp_avg"], "exp_avg"
        elif "exp_avg_sq" in state:
            tensor, key = state["exp_avg_sq"], "exp_avg_sq"
        else:
            # first Adam step: state not initialized yet -> corrupt param
            tensor, key = p.data, "param"
        flat_idx = self.rng.randrange(tensor.numel())
        bits = flip_flat(tensor, flat_idx, self.cfg.fault.fault_model, self.rng)
        self.record = self._make_record(
            "optimizer", None, param_path, flat_idx=flat_idx, bit_idxs=bits, state_key=key
        )

    # ------------------------------------------------------------------
    # master weight (training) - corrupt param.data at optimizer step
    # ------------------------------------------------------------------
    def _arm_master_weight(self):
        if self.optimizer is None:
            raise ValueError("master_weight injection requires an optimizer.")
        param_info = self.insp.random_param(self.rng)
        p = param_info.param
        original_step = self.optimizer.step

        def wrapped_step(*args, **kwargs):
            if self._should_fire():
                flat_idx = self.rng.randrange(p.data.numel())
                bits = flip_flat(p.data, flat_idx, self.cfg.fault.fault_model, self.rng)
                self.record = self._make_record(
                    "master_weight", None, param_info.name, flat_idx=flat_idx, bit_idxs=bits
                )
            return original_step(*args, **kwargs)

        self.optimizer.step = wrapped_step
        self._original_step = original_step

    # ------------------------------------------------------------------
    # comm (distributed) - delegate to the distributed layer
    # ------------------------------------------------------------------
    def _arm_comm(self):
        from .distributed import make_comm_injector

        self._comm_injector = make_comm_injector(self.cfg, self)
        self._comm_injector.arm()

    def disarm_comm(self):
        if getattr(self, "_comm_injector", None) is not None:
            self._comm_injector.disarm()

    # ------------------------------------------------------------------
    # record helper
    # ------------------------------------------------------------------
    def _make_record(self, target, layer_idx, module_path, **extra) -> dict:
        rec = {
            "target": target,
            "layer_idx": layer_idx,
            "module_path": module_path,
            "fault_model": self.cfg.fault.fault_model.type,
            "timing": self.cfg.fault.timing.type,
            "stage": self.tracker.stage,
            "step": self.tracker.step,
            "token_step": self.tracker.token_step,
        }
        rec.update(extra)
        return rec
