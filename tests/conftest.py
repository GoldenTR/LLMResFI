"""Shared pytest options and device selection.

Tests default to CUDA.  Pass ``--device=cpu`` to run the CPU path explicitly.
"""

from __future__ import annotations

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        action="store",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Tensor device for tests (default: cuda; use --device=cpu for CPU)",
    )


@pytest.fixture(scope="session")
def device_name(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--device"))


@pytest.fixture(scope="session")
def device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.fail(
            "Tests default to CUDA, but torch.cuda.is_available() is False. "
            "Run with --device=cpu to use the CPU path."
        )
    return torch.device(device_name)


@pytest.fixture(scope="session")
def cuda_device(device_name: str) -> torch.device:
    if device_name != "cuda":
        pytest.skip("CUDA-specific test skipped because --device=cpu was requested")
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA-specific test skipped because torch.cuda.is_available() is False"
        )
    return torch.device("cuda")
