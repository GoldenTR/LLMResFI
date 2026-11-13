"""Representative FP32 sign, exponent, and mantissa bit flips."""

from __future__ import annotations

import torch

from llmresfi import inject_bit_flip


def main() -> None:
    tensor = torch.tensor([1.0, -2.5, 3.0], dtype=torch.float32)
    # FP32: mantissa 0-22, exponent 23-30, sign 31.
    for bit_position in (0, 23, 31):
        result = inject_bit_flip(tensor, flat_index=1, bit_position=bit_position)
        print(
            f"FP32 bit {bit_position:2d} ({result.bit_type.value:8s}): "
            f"{result.original_value:.9g} -> {result.corrupted_value:.9g} | "
            f"0x{result.original_bit_pattern:08x} -> 0x{result.corrupted_bit_pattern:08x}"
        )


if __name__ == "__main__":
    main()

