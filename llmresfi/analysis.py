# llmresfi/analysis.py
"""Result classification and sensitivity analysis helpers.

Classifies an injection outcome relative to a golden (fault-free) reference
into the standard reliability taxonomy: Masked / DUE / SDC.
"""
from __future__ import annotations

from typing import Iterable, Optional


class Outcome:
    MASKED = "masked"
    DUE = "due"
    SDC = "sdc"


def classify_text(golden: str, faulty: str, crashed: bool = False) -> str:
    """Classify a single run by comparing golden vs faulty output text.

    - crashed (exception / NaN / timeout / assert) -> DUE
    - identical text -> Masked
    - different text -> SDC
    """
    if crashed:
        return Outcome.DUE
    if golden == faulty:
        return Outcome.MASKED
    return Outcome.SDC


def summarize(
    records: Iterable[dict],
    golden_key: str = "golden",
    faulty_key: str = "output",
    crashed_key: str = "crashed",
) -> dict:
    """Aggregate a list of run records into an outcome count summary."""
    counts = {Outcome.MASKED: 0, Outcome.DUE: 0, Outcome.SDC: 0}
    n = 0
    for r in records:
        n += 1
        outcome = classify_text(
            r.get(golden_key, ""),
            r.get(faulty_key, ""),
            bool(r.get(crashed_key, False)),
        )
        counts[outcome] += 1
    return {"total": n, **counts}


def bit_position_sensitivity(records: Iterable[dict], bit_key: str = "bit_idxs") -> dict:
    """Count SDC occurrences per bit index (higher = more sensitive bit)."""
    sensitivity: dict = {}
    for r in records:
        bits = r.get(bit_key)
        if not bits:
            continue
        crashed = bool(r.get("crashed", False))
        is_sdc = classify_text(r.get("golden", ""), r.get("output", ""), crashed) == Outcome.SDC
        if not is_sdc:
            continue
        for b in bits:
            sensitivity[b] = sensitivity.get(b, 0) + 1
    return dict(sorted(sensitivity.items(), key=lambda kv: -kv[1]))
