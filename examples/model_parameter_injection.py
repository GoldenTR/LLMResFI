"""Select a named model parameter, inject a fault, and verify forward impact."""

from __future__ import annotations

import argparse

import torch

from llmresfi import inject_bit_flip, select_target_tensor


class TinyTransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.lm_head = torch.nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(token_ids)
        hidden = self.encoder(hidden)
        return self.lm_head(hidden[:, -1, :])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-name", default="encoder.layers.0.linear1.weight"
    )
    parser.add_argument("--bit-position", type=int, default=12)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    torch.manual_seed(42)
    model = TinyTransformerLM().to(device).eval()
    tokens = torch.randint(
        0, 32, (2, 5), generator=torch.Generator().manual_seed(7)
    ).to(device)
    target = select_target_tensor(model, target_name=args.target_name)

    with torch.no_grad():
        before = model(tokens).clone()
    result = inject_bit_flip(
        target.tensor,
        flat_index=0,
        bit_position=args.bit_position,
        inplace=True,
        tensor_name=target.name,
        step=0,
    )
    with torch.no_grad():
        after = model(tokens)

    print(f"device={target.device} target={target.name} shape={target.shape} dtype={target.dtype}")
    print(f"bit={result.bit_position} type={result.bit_type.value} inplace={result.inplace}")
    print(f"max |forward delta| = {(after - before).abs().max().item():.8g}")
    print(f"forward output changed = {not torch.equal(before, after)}")


if __name__ == "__main__":
    main()

