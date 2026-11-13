"""Runnable PyTorch training-loop integration for three precision paths.

Modes
-----
persistent_low_precision
    Model parameters are persistent BF16/FP16 tensors.  The fault is injected
    in place into a leaf parameter before forward, so it affects the current
    and subsequent steps until overwritten by the optimizer.

transient_compute_copy
    FP32 master parameters are retained.  Each step creates a low-precision
    compute-weight copy for the forward pass.  The fault is injected into that
    transient copy without pretending it is a persistent parameter.

amp_fp32_parameter
    Standard PyTorch autocast with FP32 parameters.  LLMResFI injects the FP32
    parameter storage; this is *not* the same as corrupting a persistent
    low-precision compute weight.  Autocast's temporary low-precision operands
    are not exposed by this mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from llmresfi import (
    EventLogger,
    FaultScheduler,
    inject_bit_flip,
    plan_random_injections,
    resolve_float_dtype,
    select_target_tensor,
)


class ToyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 48, d_model: int = 24) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.head = torch.nn.Linear(d_model, vocab_size, bias=False)

    def hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.embedding(tokens))
        return hidden.mean(dim=1)

    def forward(
        self, tokens: torch.Tensor, head_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.hidden_states(tokens)
        if head_weight is None:
            return self.head(hidden)
        return F.linear(hidden.to(head_weight.dtype), head_weight)


def synthetic_batch(step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(10_000 + step)
    tokens = torch.randint(0, 48, (16, 8), generator=generator).to(device)
    labels = torch.randint(0, 48, (16,), generator=generator).to(device)
    return tokens, labels


def make_logger(args: argparse.Namespace) -> EventLogger | None:
    if args.log_dir is None:
        return None
    return EventLogger(
        experiment_id=args.experiment_id,
        output_dir=args.log_dir,
        format="jsonl",
        random_seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["persistent_low_precision", "transient_compute_copy", "amp_fp32_parameter"],
        default="persistent_low_precision",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--injection-steps", type=int, nargs="+", default=[3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    dtype = resolve_float_dtype(args.dtype)
    if args.mode in {"persistent_low_precision", "amp_fp32_parameter"} and dtype == torch.float32:
        pass
    if args.mode == "transient_compute_copy" and args.dtype == "float32":
        raise SystemExit("transient_compute_copy needs a 16-bit --dtype")

    torch.manual_seed(args.seed)
    model = ToyCausalLM().to(device)
    if args.mode in {"persistent_low_precision", "amp_fp32_parameter"}:
        if args.mode == "persistent_low_precision":
            model = model.to(dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = FaultScheduler(
        injection_steps=args.injection_steps,
        max_faults=len(args.injection_steps),
        seed=args.seed,
    )
    logger = make_logger(args)
    print(f"mode={args.mode} device={device} dtype={dtype}")
    print(f"deterministic injection plan: {scheduler.injection_steps}")

    for step in range(args.steps):
        optimizer.zero_grad()
        tokens, labels = synthetic_batch(step, device)
        injection_result = None
        original_copy = None

        if scheduler.should_inject(step):
            if args.mode == "persistent_low_precision":
                target = select_target_tensor(
                    model, target_name="head.weight", seed=args.seed + step
                )
                spec = plan_random_injections(
                    target.tensor,
                    count=1,
                    seed=args.seed + step,
                    bit_type="mantissa",
                )[0]
                original_copy = target.tensor.detach().clone()
                injection_result = inject_bit_flip(
                    target.tensor,
                    spec.flat_index,
                    spec.bit_position,
                    inplace=True,
                    tensor_name=target.name,
                    step=step,
                    random_seed=args.seed + step,
                    parameter_version=step,
                    injection_phase="before_forward",
                )
                logits = model(tokens)
            elif args.mode == "transient_compute_copy":
                compute_weight = model.head.weight.to(dtype=dtype)
                spec = plan_random_injections(
                    compute_weight,
                    count=1,
                    seed=args.seed + step,
                    bit_type="mantissa",
                )[0]
                original_copy = compute_weight.detach().clone()
                injection_result = inject_bit_flip(
                    compute_weight,
                    spec.flat_index,
                    spec.bit_position,
                    inplace=False,
                    tensor_name="head.weight@transient_compute_copy",
                    step=step,
                    random_seed=args.seed + step,
                    parameter_version=step,
                    injection_phase="before_forward_transient_copy",
                )
                logits = model(tokens, head_weight=injection_result.tensor)
            else:
                target = select_target_tensor(model, target_name="head.weight")
                spec = plan_random_injections(
                    target.tensor,
                    count=1,
                    seed=args.seed + step,
                    bit_type="mantissa",
                )[0]
                original_copy = target.tensor.detach().clone()
                injection_result = inject_bit_flip(
                    target.tensor,
                    spec.flat_index,
                    spec.bit_position,
                    inplace=True,
                    tensor_name=target.name,
                    step=step,
                    random_seed=args.seed + step,
                    parameter_version=step,
                    injection_phase="before_forward_amp_fp32_parameter",
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype if dtype != torch.float32 else torch.bfloat16,
                    enabled=dtype != torch.float32,
                ):
                    logits = model(tokens)

            if logger is not None and injection_result is not None:
                logger.log(
                    injection_result,
                    original_tensor=original_copy,
                    corrupted_tensor=injection_result.tensor.detach(),
                )
            print(
                f"step={step} injected flat={injection_result.flat_index} "
                f"bit={injection_result.bit_position} ({injection_result.bit_type.value})"
            )
        else:
            if args.mode == "amp_fp32_parameter":
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype if dtype != torch.float32 else torch.bfloat16,
                    enabled=dtype != torch.float32,
                ):
                    logits = model(tokens)
            else:
                logits = model(tokens)

        loss = F.cross_entropy(logits.float(), labels)
        loss.backward()
        optimizer.step()
        if step % 2 == 0 or step == args.steps - 1:
            print(f"step={step:03d} loss={loss.item():.6f}")

    if logger is not None:
        print(f"fault audit log: {logger.output_dir}")
    print("training completed; fault metadata is for the audit driver, not detection code")


if __name__ == "__main__":
    main()

