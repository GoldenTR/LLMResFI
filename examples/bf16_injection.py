"""Representative BF16 sign, exponent, and mantissa bit flips."""

from __future__ import annotations

import torch

from llmresfi import inject_bit_flip


def main() -> None:
    tensor = torch.tensor([1.0, -2.5, 3.0], dtype=torch.bfloat16)
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

