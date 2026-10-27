# llmresfi/inspector.py
"""ModelInspector: registry of layers, sub-modules, and parameters for targeting.

Layer index is extracted as the first all-numeric segment of a module's dotted
path, making discovery agnostic to naming conventions (model.layers.10,
transformer.h.10, blocks.10, ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class ModuleInfo:
    name: str
    module: object
    layer_idx: int


@dataclass
class ParamInfo:
    name: str
    param: torch.nn.Parameter


class ModelInspector:
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.layer_ids: List[int] = []
        self.modules_by_layer: Dict[int, List[ModuleInfo]] = {}
        self.params: List[ParamInfo] = []
        self._build_registry()

    # ------------------------------------------------------------------
    def random_layer(self, rng) -> int:
        if not self.layer_ids:
            raise RuntimeError("No layers found - inspector registry is empty.")
        return rng.choice(self.layer_ids)

    def pick_module(self, layer_idx: int, regex: str, rng) -> ModuleInfo:
        import re as _re

        if layer_idx not in self.modules_by_layer:
            raise ValueError(f"Layer {layer_idx} not found in model.")
        pattern = _re.compile(regex)
        candidates = [m for m in self.modules_by_layer[layer_idx] if pattern.fullmatch(m.name)]
        if not candidates:
            raise ValueError(
                f"No module matches regex='{regex}' in layer {layer_idx} "
                f"(available: {[m.name for m in self.modules_by_layer[layer_idx]]})"
            )
        return rng.choice(candidates)

    def random_param(self, rng) -> ParamInfo:
        if not self.params:
            raise RuntimeError("No parameters found - inspector registry is empty.")
        return rng.choice(self.params)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_layer_idx(module_name: str) -> Optional[int]:
        for part in module_name.split("."):
            if part.isdigit():
                return int(part)
        return None

    def _build_registry(self):
        for full_name, submod in self.model.named_modules():
            idx = self._extract_layer_idx(full_name)
            if idx is None:
                continue
            if idx not in self.modules_by_layer:
                self.modules_by_layer[idx] = []
                self.layer_ids.append(idx)
            # register only parameter-bearing modules (Linear/Conv/Embedding/...)
            # so container modules (ModuleList/Sequential parents) are skipped
            if any(True for _ in submod.parameters()):
                self.modules_by_layer[idx].append(ModuleInfo(full_name, submod, idx))

        for name, param in self.model.named_parameters():
            self.params.append(ParamInfo(name, param))

        self.layer_ids.sort()
