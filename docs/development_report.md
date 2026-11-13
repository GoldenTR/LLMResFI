# LLMResFI v0.1 Development Report

## Environment

- OS: Ubuntu 22.04.5 LTS, Linux 6.8.0-124-generic, x86_64.
- CPU: Intel Xeon Silver 4410Y; 2 sockets, 12 cores/socket, 2 threads/core,
  48 logical CPUs.
- GPU: 8 × NVIDIA GeForce RTX 4090, 49140 MiB each, compute capability 8.9.
- NVIDIA driver: 535.309.01.
- `nvidia-smi`/NVML: available outside the command sandbox; 8 CUDA devices.
- `torch.cuda.is_available()`: `True`; device count 8.
- System NVCC: CUDA 12.2 (V12.2.140).
- PyTorch CUDA runtime: 12.1; cuDNN 90100.

Full report: [environment.md](environment.md).

## Conda and Python

- Environment: `llmresfi`.
- Prefix: `/data2/lijinshu/miniconda3/envs/llmresfi`.
- Python: 3.10.22.
- PyTorch: 2.5.1+cu121.
- Test: pytest 9.2.0.
- YAML: PyYAML 6.0.3.
- Other existing relevant packages include NumPy 2.2.6, pandas 2.3.3,
  transformers 5.20.2, triton 3.1.0.

## Installed Dependencies

No new third-party Python package was required. The existing environment already
had PyTorch, PyYAML, pytest, and the optional scientific packages listed above.
LLMResFI itself was re-installed as an editable package into `llmresfi`:

```bash
conda run -n llmresfi python -m pip install -e . --no-deps --no-build-isolation
```

PyTorch, CUDA runtime, driver, and NVIDIA packages were not upgraded.

## Implemented Scope

- IEEE-754 FP16/BF16/FP32 bit flip through integer views, with bit verification.
- Sign/exponent/mantissa classification and boundary validation.
- Single-bit, multi-event, and seeded random injection APIs.
- Named model parameter selection with dtype/regex/random filters.
- Zero-based deterministic scheduler with claim-once semantics.
- JSONL/CSV/dict audit logging, failure records, optional Tensor backup.
- YAML validation, configuration snapshots, environment/version metadata.
- CPU/GPU-compatible code path; real CUDA tests completed on GPU 1.
- Runnable examples for all dtypes, model parameters, and three training paths.

Design and experiment boundaries: [design.md](design.md).

## Test Results

Command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n llmresfi python -m pytest -q
```

Result:

```text
123 passed, 0 skipped, 0 failed
```

Total collected cases: 123. CUDA execution used `CUDA_VISIBLE_DEVICES=1` on an
otherwise idle RTX 4090.

CPU fallback command:

```bash
conda run -n llmresfi python -m pytest -q --device=cpu
```

Result:

```text
117 passed, 6 skipped, 0 failed
```

The six skips are the CUDA-specific dtype and non-contiguous storage tests.

CUDA-only command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n llmresfi python -m pytest -q -m cuda
```

Result:

```text
6 passed, 117 deselected
```

## Runnable Commands

```bash
# Examples default to CUDA.
conda run -n llmresfi python examples/basic_bit_flip.py
conda run -n llmresfi python examples/fp16_injection.py
conda run -n llmresfi python examples/bf16_injection.py
conda run -n llmresfi python examples/fp32_injection.py
conda run -n llmresfi python examples/model_parameter_injection.py
conda run -n llmresfi python examples/training_loop_integration.py \
  --mode persistent_low_precision --dtype bfloat16 --steps 6 --injection-steps 3
CUDA_VISIBLE_DEVICES=1 conda run -n llmresfi python -m pytest -q

# CPU fallback is available for every example.
conda run -n llmresfi python examples/basic_bit_flip.py --device cpu
conda run -n llmresfi python examples/training_loop_integration.py \
  --mode persistent_low_precision --device cpu --dtype bfloat16 --steps 6 --injection-steps 3
```

The CUDA and CPU fallback paths were both executed successfully in the audited
environment.

## Known Limitations

- Software Tensor corruption is not a physical GPU SEU model.
- FP64/FP8/quantized/sparse tensors and distributed consistency are not covered.
- CUDA was validated on GPU 1 only; multi-GPU/DDP consistency is not implemented.
- Multi-bit plan integration into an automatic experiment runner is basic.
- Scalar reads for exact ground-truth values may synchronize CUDA execution.

## Recommended Next Steps

1. Add operator-level forward/backward injection hooks.
2. Add GPT-2/Llama semantic selectors and AdamW state injection.
3. Extend CUDA validation and injection consistency to multi-GPU/DDP.
4. Add asynchronous logging and CUDA-event timing.
5. Build a blind detector evaluation interface that cannot access audit logs.

