# Distributed Soft-Error Experiments

This document describes a reproducible protocol for studying how soft errors in
distributed LLM training affect convergence, whether they can be recovered
losslessly, and what the recovery overhead is.

## 1. Environment

```bash
conda activate llmresfi                 # Python 3.10 + torch 1.12.1+cu113
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Only the two Tesla V100s; GPU 2 is an unsupported GT 710.
export CUDA_VISIBLE_DEVICES=0,1
```

Install the package (already done in this workspace) and the test dependency:

```bash
pip install -e . --no-deps
pip install pytest 'numpy<2'            # torch 1.12 is incompatible with NumPy 2.x
```

Run the full test suite:

```bash
pytest -q                               # CPU/gloo DDP integration tests
CUDA_VISIBLE_DEVICES=0,1 pytest -q      # adds the 2-GPU FSDP shard test
```

## 2. Research questions

1. **Impact**: how far does a single rank-local bit flip propagate through
   all-reduce / reduce-scatter, and how long does the effect persist?
2. **Detection**: which faults are visible via non-finite checks, gradient-norm
   outliers, loss spikes, or communication checksums?
3. **Lossless recovery**: can a transient fault be rolled back so that the
   final parameters match a fault-free run bit-for-bit (or within tolerance)?
4. **Overhead**: what is the cost of detection and recovery per step, in extra
   time, communication, and memory?

## 3. Fault dimensions

| Dimension | Values | Why it matters |
|---|---|---|
| Backend | `ddp`, `fsdp` | all-reduce vs reduce-scatter; sharded state |
| Injection point | `before_reduce`, `after_reduce` (DDP/modern FSDP); `param_shard`, `grad_shard` (torch 1.12 FSDP) | rank-local propagation vs asymmetric divergence |
| Faulty ranks | `inject_rank`, `inject_ranks`, `inject_rank_fraction` | one bad GPU vs a correlated multi-rank event |
| Gradient bucket | `bucket_index` | early vs late buckets, layer sensitivity |
| Fault model | `single`, `double`, `sign`, `exponent`, `mantissa`, `random_value`, `nan`, `inf`, `scale` | silent bit flips vs detectable value corruption |
| Timing | `transient`, `permanent`, `intermittent` | one-shot vs persistent hardware damage |
| Mitigation | `none`, `sanitize`, `step_rollback`, `checkpoint_rollback` | detection-only vs cheap mitigation vs lossless recovery |

## 4. Core experiments

### E1. Asymmetric SDC (after all-reduce)

```bash
# Detected by cross-rank checksum consistency -> rollback.
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model single --injection_point after_reduce \
  --inject_rank 0 --verify --mitigation step_rollback --steps 20

# Same fault without verification -> rank 0 silently diverges (SDC).
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model single --injection_point after_reduce \
  --inject_rank 0 --mitigation none --steps 20
```

Compare `Injection records`, per-rank losses, and per-rank gradient norms.  The
second run should show rank 0 diverging while rank 1 stays close to the
fault-free baseline.

### E2. Symmetric propagation (before all-reduce)

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model nan --injection_point before_reduce \
  --inject_rank 0 --verify --mitigation step_rollback --steps 20
```

A NaN before all-reduce becomes NaN on every rank: both the non-finite detector
and the reduction checksum fire.  Record how many retries are needed and compare
the post-recovery loss curve with a baseline run.

### E3. Rank-fraction and bucket sensitivity

```bash
# Half the ranks, one specific bucket.
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model exponent --inject_rank_fraction 0.5 \
  --bucket_index 2 --verify --mitigation none --steps 50
```

Repeat across `bucket_index` values and bit regions (`sign`, `exponent`,
`mantissa`).  The `bit_idxs` in each record identifies the exact bit; aggregate
SDC counts by bit position with `analysis.bit_position_sensitivity`.

### E4. FSDP shard faults (torch 1.12)

```bash
# Parameter shard before all-gather: NaN reaches the full parameter on every rank.
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_fsdp.py \
  --fsdp_injection param_shard --fault_model nan --mitigation step_rollback --steps 20

# Gradient shard before the optimizer update.
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_fsdp.py \
  --fsdp_injection grad_shard --fault_model nan --mitigation step_rollback --steps 20
```

FSDP records contain `sharded: true` and `legacy_fsdp: true` on torch 1.12.
`param_shard` faults are injected in the forward phase; `grad_shard` faults are
injected by `injector.inject_distributed_now()` after backward.

### E5. Cost of mitigation

For each mode (`none`, `sanitize`, `step_rollback`, `checkpoint_rollback`),
measure:

- wall-clock time per step (excluding the first warm-up steps),
- peak GPU memory (`torch.cuda.max_memory_allocated`),
- number of retries and extra forward/backward passes,
- checkpoint I/O bytes and time.

Use the same seeds and fault schedule across modes.  `step_rollback` should add
one extra forward/backward only on faulty steps; its fixed cost is the snapshot
memory (parameters + optimizer state) and the snapshot copy time.

### E6. Automated collection and analysis

Both distributed examples write a self-contained `summary.json` per run.  Keep
each configuration in its own directory so summaries are not overwritten:

```bash
# Fault-free reference.
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model single --injection_point after_reduce \
  --inject_rank 0 --verify --mitigation step_rollback --steps 20 \
  --log_dir runs/e1/seed42/baseline --baseline

# Fault case with identical model, world size, seed, and step count.
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/run_ddp.py \
  --device cuda --target comm --fault_model single --injection_point after_reduce \
  --inject_rank 0 --verify --mitigation step_rollback --steps 20 \
  --log_dir runs/e1/seed42/fault

python examples/analyze_runs.py --root runs/e1/seed42
python examples/analyze_runs.py --root runs/e1/seed42 --json \
  --output runs/e1/report.json
```

`summary.json` contains:

| Field | Meaning |
|---|---|
| `config`, `run_id`, `world_size`, `steps` | resolved experiment identity and scale |
| `records` | all-rank injected faults: rank, bucket, tensor dtype/index, bit indices, injection point |
| `loss_history` | per-rank accepted-step loss curve; a retried attempt is not appended twice |
| `per_rank_metrics` | final loss and gradient norm per rank |
| `recovery_events` | detection signals, action, step, and retry metadata from every rank |
| `step_times` | per-rank accepted-step wall time, including replay time on faulty steps |
| `peak_memory_gb` | per-rank peak allocated CUDA memory |
| `param_signatures` | per-rank signed/absolute/L2/weighted parameter checksums |

A baseline is matched by backend, model, device, world size, random seed, and
step count.  Run the same seed for both baseline and fault variants; repeat the
pair across seeds for aggregate statistics.  For a clean overhead comparison,
run the baseline with the same mitigation mode as the fault case.  To isolate
the total mitigation cost as well as the fault-replay cost, include a second
baseline with `--mitigation none`.

The analyzer reports detection and recovery rates, then groups outcomes by
fault model, injection point, bit region, rank, bucket, timing, and mitigation.
The overhead table reports mean accepted-step time, peak memory, percentage
overhead relative to a matching baseline, unique retried steps, and per-rank
recovery events.  A successful two-rank step rollback normally produces two
events but one retried step.

Fault-specific groups exclude baseline runs so that counts answer "of the runs
that exercised this fault dimension, how many were recovered/SDC/masked/DUE?"
`by_backend` includes baselines to show the reference population.

### E7. Interpreting outcomes

| Outcome | Interpretation |
|---|---|
| `recovered` | Detection fired, rollback completed, and final loss/parameter signature agree with the baseline within `--loss_atol` / `--param_atol`. |
| `mitigated` | Detection fired and there is no measurable divergence, but the policy did not roll back (typically `sanitize`). |
| `detected_unrecovered` | The fault was observed, but the final state still differs from the baseline. Investigate incorrect recovery, checkpoint replay, or persistent faults. |
| `sdc` | No detection and measurable divergence. This is the dangerous silent-corruption class and should drive detector/redundancy improvements. |
| `masked` | No detection and no measurable divergence. The fault may have hit an irrelevant element, been numerically absorbed, or been hidden by coarse metrics. |
| `due` | The run aborted after detection, for example retries were exhausted or no snapshot/checkpoint was available. |
| `no_fault` | No injection record was produced; check fault schedule, stage, rank policy, and bucket selection before treating this as a result. |

`param_signature` is deliberately compact so summaries remain small for LLM
experiments.  It is not a proof of bitwise equality.  A `recovered` result means
"lossless within the configured tolerances"; exact losslessness should be
checked by comparing a saved final state dict or by the deterministic reference
protocol in section 5.  Loss curves can expose divergence that a final scalar
hides, so inspect `curve_max_diff` and `first_divergence_step` in the JSON
report for failing or borderline runs.

## 5. Lossless recovery check

The integration test (`tests/test_ddp_integration.py`) implements the strongest
check: after a transient fault and rollback, replay the same batch and compare
the final parameters with an independently trained fault-free reference model.
A difference below `1e-6` is treated as lossless for the CPU/gloo setup.

For GPU experiments, use the same recipe:

1. Save the RNG state before the faulty step.
2. Run the faulty step, detect, restore the step snapshot, and replay the same
   batch.
3. Restore the saved RNG state and run one clean reference step on a separate
   model with identical initial parameters.
4. Compare parameters element-wise and report the maximum absolute difference.

Caveats:

- `step_rollback` is lossless only if the batch and data order are replayed and
  all stochastic operations are seeded/deterministic.
- `checkpoint_rollback` restores model/optimizer/RNG at a completed-step
  boundary.  Rewind the tracker and data iterator to `checkpoint_step` and
  replay every subsequent step; merely continuing from the current step is not
  lossless.  The first `begin_step()` creates an initial checkpoint, and the
  distributed examples handle the tracker/history rewind, but a real dataset
  loader must also seek back to the checkpoint position.
- GPU all-reduce reduction order can introduce small numerical differences; use
  a tolerance rather than bitwise equality.
- A snapshot doubles model+optimizer memory unless you checkpoint to CPU/disk.

## 6. What is detectable?

- **After-reduce corruption**: cross-rank checksum consistency detects any
  deviation, even below float32 reduction noise.
- **Before-reduce corruption**: a scalar sum checksum competes with float32
  accumulation noise; small mantissa flips can be missed.  Non-finite and
  large-value faults are reliably detected by the value/norm detectors.
- **Exact pre-reduce single-bit detection** requires redundancy (recompute the
  gradient, replicate the bucket, or run a second independent reduction of
  replicated data).  This is a fundamental observability limit of a single
  all-reduce, not just an implementation gap.

## 7. Recommended reporting

For each configuration report:

- fault location (`rank`, `bucket_index`, `flat_idx`, `bit_idxs`),
- detection signal(s) and detection latency,
- recovery action and number of retries,
- final loss / gradient-norm divergence vs baseline,
- max parameter difference vs the fault-free reference,
- per-step time and peak memory overhead.

Aggregate across at least 5 seeds and report the fraction of runs classified as
masked / detected / SDC / DUE.  The machine-readable form is produced by:

```bash
python examples/analyze_runs.py --root runs/campaign --json \
  --output runs/campaign-report.json
```

Record the tolerances used for baseline comparison.  Do not describe a run as
bitwise lossless based only on `summary.json`; use the exact state-dict check
when that claim matters.
