# LLMResFI

<p align="center">
  <b>LLMResFI = LLM Resilience Fault Injection</b><br/>
  A fault injection framework for studying soft-error resilience in large language model (LLM) <b>training</b> and <b>inference</b>, with first-class support for multi-GPU distributed training (DDP/FSDP).
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="#"><img src="https://img.shields.io/badge/LLMResFI-v0.2-orange.svg" alt="LLMResFI v0.2"></a>
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
- **Distributed**: DDP gradient-bucket injection before/after all-reduce, FSDP gradient-reduction hooks, rank-subset policies, and an optional checksum guard.
- **Detection + recovery**: non-finite/loss/grad-norm detectors, gradient sanitisation, lossless step-level rollback, and checkpoint rollback.
- **Fault timing**: `transient`, `permanent` (persist N steps), `intermittent` (every k steps).
- **Reproducible**: seed-driven, deterministic greedy decoding, JSONL logging.
- **Result analysis**: per-run `summary.json`, baseline-relative outcome classification
  (`recovered` / `mitigated` / `SDC` / `masked` / `DUE`), bit-region sensitivity,
  and time/memory overhead tables.

## Installation

```bash
git clone https://github.com/GoldenTR/LLMResFI.git
cd LLMResFI
pip install -e .
# optional dev / dataset support
pip install -e ".[dev,datasets]"
```

GPU is recommended for practical LLM workloads; CPU works for the toy training example.

The framework is tested on Python 3.9+ and PyTorch 1.12 (CUDA 11.3) as well as
PyTorch 2.x.  With PyTorch 1.12, use `numpy<2`.

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

Example output (gradient injection):

```text
[step 0] loss=2.3141 stage=forward
[step 1] loss=2.2390 stage=forward
[step 2] loss=2.1681 stage=forward
[step 3] loss=2.1005 stage=forward
[step 4] loss=2.0349 stage=forward

Injection record:
  target: gradient
  param_path: net.4.bias        # parameter targeted (layer_idx is None for param targets)
  fault_model: single
  timing: transient
  stage: backward               # injected during the backward pass of step 0
  step: 0
  token_step: -1
  flat_idx: 1
  bit_idxs: [1]                 # bit 1 flipped in element 1 of the bias tensor
```

### 3. Distributed (DDP)

```bash
# CPU / gloo smoke test
torchrun --nproc_per_node=2 examples/run_ddp.py --target comm --fault_model nan --mitigation step_rollback

# two V100 GPUs / NCCL
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model nan --inject_rank 0 \
  --mitigation step_rollback --verify --steps 3
```

> Note: DDP performs gradient all-reduce *during* `loss.backward()`, so the comm hook fires in the `backward` stage.

Injection semantics:

- `injection_point: before_reduce` corrupts a rank-local bucket **before**
  all-reduce; the error is averaged into every rank (silent propagation).
- `injection_point: after_reduce` corrupts the reduced bucket on the selected
  rank only; that rank silently diverges from its peers.
- `inject_rank` (single), `inject_ranks` (list), or `inject_rank_fraction`
  (deterministic random subset) control which ranks are faulty.
- `verify_comm: true` gathers a tiny per-rank checksum before the reduction and
  checks both reduction consistency and cross-rank agreement afterwards.

### 4. Distributed (FSDP)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_fsdp.py \
  --fsdp_injection grad_shard --fault_model nan --mitigation step_rollback
```

- PyTorch >= 2.0: uses FSDP's gradient communication hook (before/after reduce).
- PyTorch 1.12: falls back to shard-level injection (`param_shard` before
  all-gather, `grad_shard` before the optimizer step).  Such records carry
  `legacy_fsdp: true`.

## Fault Detection and Recovery

Enable the mitigation layer with:

```yaml
mitigation:
  enabled: true
  detector:
    non_finite: true
    grad_norm_ratio: 5.0
    loss_ratio: 3.0
  recovery:
    mode: step_rollback   # none | sanitize | step_rollback | checkpoint_rollback
    max_retries: 1
```

- `sanitize`: zero (or clamp) non-finite gradients and continue.  Cheap, but
  not lossless for large value faults.
- `step_rollback`: snapshot parameters, optimizer state, and RNG at the start
  of a step; on detection restore and replay the same batch.  This is lossless
  for transient faults when replay is deterministic; the cost is one extra
  forward/backward and an in-memory copy of model+optimizer state.
- `checkpoint_rollback`: restore the latest on-disk checkpoint.  Lossless up to
  the checkpoint interval when the caller rewinds the training step/data
  iterator and replays every step after the checkpoint; more expensive than a
  step snapshot.  The first `begin_step()` writes an initial checkpoint;
  `--checkpoint_every N` bounds replay distance for later faults.

All ranks agree on the recovery decision with a MAX all-reduce, avoiding the
deadlock that would occur if only one rank rolled back.  Events are appended to
`<log_dir>/recovery.jsonl`.

**Detection limits.**  A single-bit flip before reduce can be smaller than the
float32 reduction noise, so a scalar sum checksum may miss it.  Asymmetric
(after-reduce) corruption is caught exactly by cross-rank consistency.  For
pre-reduce silent errors, use non-finite/large-value faults, gradient-norm
detectors, or redundant recomputation.  This limitation is fundamental without
replicated gradients.

## Experiment Artifacts and Result Analysis

`run_ddp.py` and `run_fsdp.py` write `<log_dir>/summary.json` in addition to
`recovery.jsonl`.  The summary contains the resolved config, every rank's
injection records, loss curve, accepted-step times, peak memory, recovery
events, final metrics, and an approximate parameter signature.

Run a matched baseline and fault case in separate directories, then aggregate:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model nan --inject_rank 0 \
  --mitigation step_rollback --verify --steps 20 \
  --log_dir runs/ddp_nan/seed42/baseline --baseline

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model nan --inject_rank 0 \
  --mitigation step_rollback --verify --steps 20 \
  --log_dir runs/ddp_nan/seed42/fault

python examples/analyze_runs.py --root runs/ddp_nan/seed42
python examples/analyze_runs.py --root runs/ddp_nan/seed42 --json \
  --output runs/ddp_nan/report.json
```

The text report prints detection/recovery rates, outcomes grouped by fault
model, injection point, bit region, rank, bucket, timing, and mitigation, plus
an overhead table.  A matching baseline is identified by backend, model,
device, world size, seed, and step count.  Fault-specific groups exclude
baseline runs; `by_backend` includes them.

| Outcome | Meaning |
|---|---|
| `baseline` | fault-free reference run |
| `no_fault` | a non-baseline run in which no injection record was produced |
| `recovered` | detected, rollback succeeded, and loss/parameter signature match the baseline within tolerance |
| `mitigated` | detected and no measurable divergence, but no rollback action (for example `sanitize`) |
| `detected_unrecovered` | detected, but the final state still diverges from the baseline |
| `sdc` | no detection and measurable divergence: silent data corruption |
| `masked` | no detection and no measurable divergence |
| `due` | the run aborted after detection |
| `detected` / `undetected` | fallback classification when no matching baseline exists |

`time_ovh%` and `mem_ovh%` are relative to the matching baseline row for the
same backend (preferring the same mitigation mode).  Accepted-step time already
includes any replay cost on a retried step.  `retried` counts unique steps that
were rolled back; `events` counts per-rank recovery events, so a two-rank
step rollback normally contributes two events.  `param_signature` combines
signed/absolute/L2/weighted checksums and is a compact approximation, not a
bitwise state-dict comparison; use an explicit reference model/state dict when
an experiment requires exact losslessness.

## Configuration

The full schema is defined in [`llmresfi/config.py`](llmresfi/config.py) (pydantic). Key sections:

| Section | Fields | Description |
|---|---|---|
| `run_mode` | `inference` / `training` | selects the stage schedule |
| `model` | `repo`, `dtype`, `device` | HF model id or local path |
| `distributed` | `backend`, `inject_rank(s)`, `inject_rank_fraction`, `injection_point`, `bucket_index`, `fsdp_injection`, `verify_comm` | `none` / `ddp` / `fsdp`; rank policy and comm-detection controls |
| `fault.target` | `weight/activation/kv_cache/loss/gradient/optimizer/master_weight/comm` | what to corrupt |
| `fault.stage` | depends on `run_mode` | when to inject |
| `fault.fault_model` | `single/double/multi/sign/exponent/mantissa/random/random_value/zero/nan/inf/scale/stuck_at_0/stuck_at_1` | fault model |
| `fault.timing` | `transient/permanent/intermittent` + `steps`/`period` | persistence |
| `execution` | `random_seed`, `log_dir`, `verbosity` | reproducibility & logging |
| `mitigation` | `enabled`, `detector`, `recovery` | anomaly detection + sanitise/rollback policies |

## Programmatic API

```python
from llmresfi import FIConfig, ModelInspector, StageTracker, FaultInjector, RecoveryManager

cfg = FIConfig.load("config.yml")
inspector = ModelInspector(model, optimizer)
tracker = StageTracker(cfg.run_mode)
injector = FaultInjector(cfg, inspector, tracker, rng, optimizer)
recovery = RecoveryManager(cfg, model, optimizer, tracker)

injector.arm(ddp_model)          # comm targets need the wrapped module
recovery.begin_step()
# ... forward / backward, advancing `tracker` ...
injector.inject_distributed_now()  # legacy FSDP only; no-op for DDP
signals = recovery.check(loss, injector=injector)
action = recovery.decide(signals)
if action == "retry_step":
    tracker.on_retry()
    # replay the same batch
optimizer.step()
recovery.end_step(loss, clean=not signals)
injector.disarm()
print(injector.records)          # every injected fault (all ranks via gather_records)
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
│   ├── mitigation.py              # detection, snapshots, rollback, checkpoints
│   ├── reporting.py               # summary.json schema, outcome aggregation, overhead
│   └── analysis.py                # text-level SDC/DUE/masked classification
├── examples/
│   ├── run_inference.py
│   ├── run_training.py
│   ├── run_ddp.py
│   ├── run_fsdp.py
│   └── analyze_runs.py
├── tests/
│   ├── test_bitflip.py
│   ├── test_config.py
│   ├── test_scheduler.py
│   ├── test_injector.py
│   ├── test_timing.py
│   ├── test_mitigation.py
│   ├── test_distributed_unit.py
│   ├── test_ddp_integration.py
│   ├── test_fsdp_gpu.py
│   └── test_reporting.py
└── docs/
    ├── design.md                  # full design document
    └── experiments.md             # experiment protocol and metrics
```

## Testing

```bash
pytest
CUDA_VISIBLE_DEVICES=0,1 pytest -q   # also runs the 2-GPU FSDP test
```

## Roadmap

- **Phase 1** (done in v0.1): single-GPU inference with true bit-flips and timing.
- **Phase 2** (done in v0.1): single-GPU training (gradient/optimizer/master_weight/loss).
- **Phase 3** (done in v0.2): DDP before/after-reduce injection, rank policies,
  checksum verification, FSDP comm hook + torch 1.12 shard fallback.
- **Phase 4** (partial in v0.2): anomaly detection, gradient sanitisation,
  lossless step rollback, checkpoint rollback, per-rank metrics; campaign
  orchestration and sensitivity reports remain future work.

See [`docs/design.md`](docs/design.md) for the full design and
[`docs/experiments.md`](docs/experiments.md) for the experiment protocol.

## License

[Apache License 2.0](./LICENSE)
