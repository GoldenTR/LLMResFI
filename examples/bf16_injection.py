"""Representative BF16 sign, exponent, and mantissa bit flips."""

from __future__ import annotations

import argparse

import torch

from llmresfi import inject_bit_flip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False")
    tensor = torch.tensor(
        [1.0, -2.5, 3.0], dtype=torch.bfloat16, device=args.device
    )
    # BF16: mantissa 0-6, exponent 7-14, sign 15.
    for bit_position in (0, 7, 15):
        result = inject_bit_flip(tensor, flat_index=1, bit_position=bit_position)
        print(
            f"BF16 bit {bit_position:2d} ({result.bit_type.value:8s}): "
            f"{result.original_value:.8g} -> {result.corrupted_value:.8g} | "
            f"0x{result.original_bit_pattern:04x} -> 0x{result.corrupted_bit_pattern:04x}"
        )


if __name__ == "__main__":
    main()

