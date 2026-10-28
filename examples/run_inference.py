# examples/run_inference.py
"""Single-GPU LLM inference fault injection example.

Loads a Hugging Face causal LM, runs one deterministic (greedy) generation,
injects a single fault per run per ``config.yml``, and logs the result.

Usage:
    python examples/run_inference.py --config config.yml --prompt "..." --new_tokens 16
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmresfi import FIConfig, FaultInjector, ModelInspector, StageTracker


def load_model(cfg: FIConfig):
    kwargs = {"torch_dtype": getattr(torch, cfg.model.dtype)}
    if cfg.model.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if cfg.model.device == "auto":
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg.model.repo, **kwargs)
    if cfg.model.device in ("cuda", "cpu"):
        model = model.to(cfg.model.device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.repo)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token_id is not None else "[PAD]"
        if tokenizer.pad_token == "[PAD]":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("config.yml"))
    ap.add_argument("--prompt", type=str, default="Explain transformers briefly.")
    ap.add_argument("--new_tokens", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = FIConfig.load(args.config)
    random.seed(cfg.execution.random_seed)
    torch.manual_seed(cfg.execution.random_seed)
    rng = random.Random(cfg.execution.random_seed)

    model, tokenizer = load_model(cfg)
    inspector = ModelInspector(model)

    inputs = tokenizer(args.prompt, return_tensors="pt", return_attention_mask=True).to(model.device)

    tracker = StageTracker(cfg.run_mode)
    injector = FaultInjector(cfg, inspector, tracker, rng)
    injector.arm()

    t0 = time.perf_counter()
    with torch.no_grad():
        pre = model(**inputs, use_cache=True, return_dict=True)
    past = pre.past_key_values
    tracker.on_prefill_done()

    with torch.no_grad():
        out1 = model(inputs.input_ids[:, -1:], use_cache=True, past_key_values=past, return_dict=True)
    past = out1.past_key_values
    next_token = torch.argmax(out1.logits[:, -1, :], dim=-1)
    out_ids = torch.cat([inputs.input_ids, next_token.unsqueeze(-1)], dim=-1)
    tracker.on_first_token_emitted()

    tracker.set_random_decode_target(args.new_tokens, rng)
    cur = next_token
    for _ in range(max(0, args.new_tokens)):
        with torch.no_grad():
            out = model(cur.unsqueeze(1), use_cache=True, past_key_values=past, return_dict=True)
        past = out.past_key_values
        cur = torch.argmax(out.logits[:, -1, :], dim=-1)
        out_ids = torch.cat([out_ids, cur.unsqueeze(-1)], dim=-1)
        tracker.inc_decode()

    t1 = time.perf_counter()
    gen_text = tokenizer.decode(out_ids[0, inputs.input_ids.size(1):], skip_special_tokens=True)

    record = {
        "input_text": args.prompt,
        "output_text": gen_text,
        "elapsed_sec": t1 - t0,
        "fault_info": injector.record,
    }
    injector.disarm()

    out_path = args.out or (cfg.log_dir_resolved / "inference_result.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
