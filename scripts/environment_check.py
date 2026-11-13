"""Audit the runtime and emit a machine-readable environment report."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from llmresfi import __version__


def run_command(command: list[str], timeout: int = 15) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "error": "executable not found"}
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "command": command, "error": str(exc)}
    return {
        "available": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {"unknown": "true"}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value.strip().strip('"')
    return result


def collect_cpu() -> dict[str, Any]:
    lscpu = run_command(["lscpu"])
    fields: dict[str, str] = {}
    if lscpu.get("available"):
        for line in lscpu["stdout"].splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
    return {
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "model_name": fields.get("Model name", "unknown"),
        "sockets": fields.get("Socket(s)", "unknown"),
        "cores_per_socket": fields.get("Core(s) per socket", "unknown"),
        "threads_per_core": fields.get("Thread(s) per core", "unknown"),
        "raw_lscpu_available": bool(lscpu.get("available")),
    }


def collect_gpu() -> dict[str, Any]:
    nvidia_smi = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    gpus: list[dict[str, str]] = []
    if nvidia_smi.get("available"):
        for line in nvidia_smi["stdout"].splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) == 4:
                gpus.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "memory_total": parts[2],
                        "driver_version": parts[3],
                    }
                )
    cuda_available = bool(torch.cuda.is_available())
    torch_devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            torch_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    pci = run_command(["lspci"])
    nvidia_pci = [
        line.strip()
        for line in pci.get("stdout", "").splitlines()
        if "nvidia" in line.lower()
        and ("vga compatible" in line.lower() or "3d controller" in line.lower())
    ]
    driver_version = None
    driver_source = None
    if gpus:
        driver_version = gpus[0].get("driver_version")
        driver_source = "nvidia-smi"
    else:
        proc_version = Path("/proc/driver/nvidia/version")
        if proc_version.is_file():
            try:
                text = proc_version.read_text(encoding="utf-8")
            except OSError:
                text = ""
            match = re.search(r"Kernel Module\s+([0-9.]+)", text)
            if match:
                driver_version = match.group(1)
                driver_source = "/proc/driver/nvidia/version"
    return {
        "nvidia_smi": nvidia_smi,
        "gpus_from_nvidia_smi": gpus,
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_devices": torch_devices,
        "nvidia_pci_functions": nvidia_pci,
        "driver_version": driver_version,
        "driver_version_source": driver_source,
        "note": (
            "GPU model and memory are reported as unknown when nvidia-smi/NVML is "
            "unavailable; PCI presence alone is not treated as a usable device."
        ),
    }


def collect_packages() -> dict[str, str | None]:
    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "pytest",
        "PyYAML",
        "pandas",
        "transformers",
        "triton",
        "nvidia-cuda-runtime-cu12",
        "llmresfi",
    ]
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def collect_report() -> dict[str, Any]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    return {
        "tool": "LLMResFI",
        "tool_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "os_release": read_os_release(),
        "kernel": platform.release(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "conda_prefix": conda_prefix,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cpu": collect_cpu(),
        "gpu": collect_gpu(),
        "nvcc": run_command(["nvcc", "--version"]),
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count(),
        },
        "packages": collect_packages(),
    }


def to_markdown(report: dict[str, Any]) -> str:
    cpu = report["cpu"]
    gpu = report["gpu"]
    torch_info = report["torch"]
    lines = [
        "# LLMResFI Environment Audit",
        "",
        f"- Timestamp (UTC): `{report['timestamp_utc']}`",
        f"- Platform: `{report['platform']}`",
        f"- Kernel: `{report['kernel']}`",
        f"- Conda prefix: `{report['conda_prefix']}`",
        f"- Conda environment: `{report['conda_default_env']}`",
        f"- Python executable: `{report['python_executable']}`",
        f"- Python version: `{report['python_version']}`",
        f"- CPU model: {cpu['model_name']}",
        f"- CPU topology: {cpu['sockets']} socket(s), {cpu['cores_per_socket']} cores/socket, "
        f"{cpu['threads_per_core']} thread(s)/core, {cpu['logical_cpus']} logical CPUs",
        f"- PyTorch: `{torch_info['version']}`; CUDA runtime: `{torch_info['cuda_version']}`; "
        f"cuDNN: `{torch_info['cudnn_version']}`",
        f"- `torch.cuda.is_available()`: `{torch_info['cuda_available']}`; "
        f"device count: `{torch_info['device_count']}`",
    ]
    nvcc = report.get("nvcc", {})
    if nvcc.get("available"):
        output_lines = nvcc.get("stdout", "").splitlines()
        release_line = next(
            (line for line in output_lines if "release" in line.lower()),
            output_lines[0] if output_lines else "unknown",
        )
        lines.append(f"- System NVCC: `{release_line}`")
    else:
        lines.append("- System NVCC: `unknown` (nvcc unavailable or failed)")
    if gpu["gpus_from_nvidia_smi"]:
        for item in gpu["gpus_from_nvidia_smi"]:
            lines.append(
                f"- GPU {item['index']}: {item['name']}, {item['memory_total']}, "
                f"driver {item['driver_version']}"
            )
    else:
        lines.append(
            "- GPU model / VRAM: **unknown**; `nvidia-smi` did not successfully query "
            "an available device."
        )
        if gpu.get("driver_version"):
            lines.append(
                f"- NVIDIA driver version: `{gpu['driver_version']}` "
                f"(source: `{gpu.get('driver_version_source')}`)"
            )
        else:
            lines.append("- NVIDIA driver version: **unknown**")
        if gpu["nvidia_smi"].get("stderr"):
            lines.append(f"- `nvidia-smi` error: `{gpu['nvidia_smi']['stderr']}`")
        elif gpu["nvidia_smi"].get("stdout"):
            lines.append(f"- `nvidia-smi` error: `{gpu['nvidia_smi']['stdout']}`")
    if gpu["nvidia_pci_functions"]:
        lines.append("- NVIDIA PCI functions observed: " + "; ".join(gpu["nvidia_pci_functions"]))
    lines.extend(["", "## Installed Packages", "", "| Package | Version |", "|---|---|"])
    for name, version in report["packages"].items():
        lines.append(f"| `{name}` | `{version if version is not None else 'not installed'}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="Write full JSON report")
    parser.add_argument("--markdown", type=Path, default=None, help="Write Markdown report")
    args = parser.parse_args()
    report = collect_report()
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(to_markdown(report), encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

