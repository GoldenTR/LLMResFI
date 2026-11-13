"""Minimal CPU/GPU bit-flip example for one FP16 tensor element."""

from __future__ import annotations

import argparse

import torch

from llmresfi import inject_bit_flip, pattern_to_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--flat-index", type=int, default=1)
    parser.add_argument("--bit-position", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False")
    tensor = torch.tensor([1.0, -2.5, 0.0], dtype=torch.float16, device=args.device)
    result = inject_bit_flip(
        tensor,
        flat_index=args.flat_index,
        bit_position=args.bit_position,
        inplace=False,
        tensor_name="example.vector",
    )
    print(f"device={result.device} dtype={result.dtype} shape={result.tensor_shape}")
    print(f"flat_index={result.flat_index} bit={result.bit_position} type={result.bit_type.value}")
    print(
        f"pattern {pattern_to_binary(result.original_bit_pattern, result.dtype)} -> "
        f"{pattern_to_binary(result.corrupted_bit_pattern, result.dtype)}"
    )
    print(f"value {result.original_value!r} -> {result.corrupted_value!r}")
    print(f"original tensor remains unchanged: {tensor.tolist()}")
    print(f"returned tensor: {result.tensor.tolist()}")
    print(f"verified single-bit change: {result.only_target_bit_changed}")


if __name__ == "__main__":
    main()

