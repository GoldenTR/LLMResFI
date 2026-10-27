# LLMResFI

<p align="center">
  <b>LLMResFI = LLM Resilience Fault Injection</b><br/>
  A fault injection framework for studying soft-error resilience in large language model (LLM) <b>training</b> and <b>inference</b>, with first-class support for multi-GPU distributed training (DDP/FSDP).
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="#"><img src="https://img.shields.io/badge/LLMResFI-v0.1-orange.svg" alt="LLMResFI v0.1"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
</p>

---

## Why LLMResFI?

Existing fault injection tools leave large gaps:

| Capability | LLMFI | NVBitFI | PyTorchFI | **LLMResFI** |
|---|---|---|---|---|
| LLM inference (phase-aware) | ✅ | ❌ | ❌ | ✅ |
| **Training support** | ❌ | ❌ | ❌ | ✅ |
| **Gradient / optimizer-state faults** | ❌ | ❌ | ❌ | ✅ |
| **Distributed communication faults** | ❌ | ❌ | ❌ | ✅ |
| True IEEE bit-flip (sign/exp/mantissa) | ⚠️ partial | ✅ | ❌ | ✅ |
| Fault timing (transient/permanent/intermittent) | ❌ | ⚠️ | ❌ | ✅ |
| Abstraction level | model | architecture | model | model |

LLMResFI unifies **training and inference** under one API and targets the states that prior tools cannot reach: gradients, Adam moments, mixed-precision master weights, and all-reduce communication tensors.

## Features

- **Unified training + inference**: a single `run_mode` + `stage` + `target` abstraction.
  - Inference stages: `prefill → first_token → decode`
  - Training stages: `forward → backward → optimizer_step → comm`
- **True IEEE bit-flip** via `view(int) ^ mask`, with `sign` / `exponent` / `mantissa` targeting.
- **Training-only fault targets**: `gradient`, `optimizer` (Adam `exp_avg`/`exp_avg_sq`), `master_weight`, `loss`.
- **Distributed**: DDP gradient-bucket injection via `register_comm_hook` (FSDP planned).
- **Fault timing**: `transient`, `permanent` (persist N steps), `intermittent` (every k steps).
- **Reproducible**: seed-driven, deterministic greedy decoding, JSONL logging.
- **Result analysis**: `masked` / `DUE` / `SDC` classification + bit-position sensitivity.

## Installation

```bash
git clone https://github.com/GoldenTR/LLMResFI.git
cd LLMResFI
pip install -e .
# optional dev / dataset support
pip install -e ".[dev,datasets]"
```

GPU is recommended for practical LLM workloads; CPU works for the toy training example.

## Quick Start

### 1. Inference (single GPU)

```bash
python examples/run_inference.py --config config.yml --prompt "Explain transformers briefly." --new_tokens 16
```

The `fault` section of `config.yml` controls the target/phase/fault model:

```yaml
fault:
  target: "weight"        # weight | activation | kv_cache
  stage: { type: "prefill" }   # prefill | first_token | decode
  fault_model: { type: "exponent" }   # single|double|sign|exponent|mantissa|...
  timing: { type: "transient" }       # transient | permanent | intermittent
```

### 2. Training (single GPU)

Demonstrates the training-only targets on a tiny MLP (no GPU/HF required):

```bash
python examples/run_training.py --target gradient --stage backward
python examples/run_training.py --target optimizer --stage optimizer_step
python examples/run_training.py --target loss --stage backward
python examples/run_training.py --target master_weight --stage optimizer_step
```

### 3. Distributed (DDP)

```bash
torchrun --nproc_per_node=2 examples/run_ddp.py --target comm --stage backward
```

> Note: DDP performs gradient all-reduce *during* `loss.backward()`, so the comm hook fires in the `backward` stage.

## Configuration

The full schema is defined in [`llmresfi/config.py`](llmresfi/config.py) (pydantic). Key sections:

| Section | Fields | Description |
|---|---|---|
| `run_mode` | `inference` / `training` | selects the stage schedule |
| `model` | `repo`, `dtype`, `device` | HF model id or local path |
| `distributed` | `backend`, `world_size`, `inject_rank` | `none` / `ddp` / `fsdp` |
| `fault.target` | `weight/activation/kv_cache/loss/gradient/optimizer/master_weight/comm` | what to corrupt |
| `fault.stage` | depends on `run_mode` | when to inject |
| `fault.fault_model` | `single/double/multi/sign/exponent/mantissa/random/random_value/zero/nan/inf/scale/stuck_at_0/stuck_at_1` | fault model |
| `fault.timing` | `transient/permanent/intermittent` + `steps`/`period` | persistence |
| `execution` | `random_seed`, `log_dir`, `verbosity` | reproducibility & logging |

## Programmatic API

```python
from llmresfi import FIConfig, ModelInspector, StageTracker, FaultInjector

cfg = FIConfig.load("config.yml")
inspector = ModelInspector(model, optimizer)
tracker = StageTracker(cfg.run_mode)
injector = FaultInjector(cfg, inspector, tracker, rng, optimizer)

injector.arm()
# ... run training or inference, advancing `tracker` at each stage ...
injector.disarm()
print(injector.record)  # fault metadata (None if never fired)
```

## Repository Layout

```text
LLMResFI/
├── README.md
├── LICENSE
├── pyproject.toml
├── config.yml                     # default experiment config
├── llmresfi/
│   ├── __init__.py                # public API
│   ├── config.py                  # YAML + pydantic schema
│   ├── bitflip.py                 # true IEEE bit-flip engine
│   ├── fault_models.py            # fault model library
│   ├── scheduler.py               # StageTracker (inference/training)
│   ├── inspector.py               # layer/module/parameter registry
│   ├── hooks.py                   # low-level fault application helpers
│   ├── injector.py                # FaultInjector (arm/disarm)
│   ├── distributed.py             # DDP/FSDP comm injection
│   └── analysis.py                # SDC/DUE/masked classification
├── examples/
│   ├── run_inference.py
│   ├── run_training.py
│   └── run_ddp.py
├── tests/
│   ├── test_bitflip.py
│   ├── test_config.py
│   ├── test_scheduler.py
│   └── test_injector.py
└── docs/
    └── design.md                  # full design document
```

## Testing

```bash
pytest
```

## Roadmap

- **Phase 1** (done in v0.1): single-GPU inference with true bit-flips and timing.
- **Phase 2** (done in v0.1): single-GPU training (gradient/optimizer/master_weight/loss).
- **Phase 3**: full distributed support (DDP + FSDP comm hooks).
- **Phase 4**: fault propagation/temporal modeling + campaign orchestration + sensitivity reports.

See [`docs/design.md`](docs/design.md) for the full design.

## License

[Apache License 2.0](./LICENSE)
