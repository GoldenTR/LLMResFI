#!/usr/bin/env python3
"""Aggregate LLMResFI distributed experiment artifacts.

Examples
--------
Analyze every ``summary.json`` below ``runs/``::

    python examples/analyze_runs.py --root runs

Emit machine-readable JSON for plotting or CI::

    python examples/analyze_runs.py --root runs --json --output report.json

Older run directories that only contain ``recovery.jsonl`` still produce a
small event summary, but outcome classification needs ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmresfi.reporting import (
    aggregate_summaries,
    format_legacy_events,
    format_report,
    iter_recovery_events,
    iter_summaries,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs", help="run directory, summary.json, or recovery.jsonl")
    ap.add_argument(
        "--loss_atol",
        type=float,
        default=1e-3,
        help="absolute loss tolerance when comparing a run with its baseline",
    )
    ap.add_argument(
        "--param_atol",
        type=float,
        default=1e-6,
        help="relative parameter-signature tolerance",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    ap.add_argument("--output", default=None, help="optionally write the report to this path")
    args = ap.parse_args()

    root = Path(args.root)
    summaries = iter_summaries(root)
    if summaries:
        payload = aggregate_summaries(
            summaries,
            loss_atol=args.loss_atol,
            param_atol=args.param_atol,
        )
        text = json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_report(payload)
    else:
        events = iter_recovery_events(root)
        if not events:
            print(f"No summary.json or recovery.jsonl found under {root}")
            return 2
        payload = {"mode": "recovery_events", "events": events}
        text = json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_legacy_events(events)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote report to {output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
