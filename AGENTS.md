# Repository Guidelines

## Project Structure & Module Organization

- `llmresfi/` contains the source package. Public API is exported from `llmresfi/__init__.py`.
- `llmresfi/config.py` defines the YAML/Pydantic configuration schema; `bitflip.py`, `fault_models.py`, `scheduler.py`, `injector.py`, and `distributed.py` implement injection primitives, scheduling, and DDP/FSDP support.
- `examples/` contains runnable inference, single-GPU training, and DDP demos.
- `tests/` contains the pytest suite; `docs/design.md` holds the detailed architecture reference.
- Default experiment settings live in `config.yml`.

## Build, Test, and Development Commands

- `pip install -e .` installs the package and runtime dependencies.
- `pip install -e ".[dev,datasets]"` adds test and dataset extras.
- `pytest` runs the test suite from `tests/`.
- `python examples/run_inference.py --config config.yml --prompt "..." --new_tokens 16` runs a single-GPU inference fault-injection demo.
- `python examples/run_training.py --target gradient --stage backward` runs the CPU toy training demo.
- `torchrun --nproc_per_node=2 examples/run_ddp.py --target comm --stage backward` runs the distributed demo.

## Coding Style & Naming Conventions

- Use Python 3.9+, PEP 8, and four-space indentation.
- Keep public names consistent with `llmresfi/__init__.py`; prefer descriptive module, function, and variable names.
- Use `from __future__ import annotations` and type hints where practical.
- When adding configuration fields, update `FIConfig` and `config.yml` together; document supported values in `README.md`.

## Testing Guidelines

- Use `pytest`. Name test modules `test_<area>.py` and functions `test_<behavior>`.
- Keep tests fast and deterministic by using CPU tensors, small models, and fixed seeds.
- Cover new fault models, stage transitions, and configuration validation alongside bug fixes.

## Commit & Pull Request Guidelines

- Follow the repository's conventional commits: `feat:`, `fix:`, or `refactor:` as the prefix.
- Keep commits focused and reference the relevant `docs/design.md` or configuration field when behavior changes.
- Pull requests should include a short description, tests or example commands used for validation, and any linked issue.

## Configuration & Safety Tips

- Use `trust_remote_code: false` unless a model genuinely requires custom remote code.
- Preserve seeded, reproducible runs through `execution.random_seed`; log outputs under `execution.log_dir`.
