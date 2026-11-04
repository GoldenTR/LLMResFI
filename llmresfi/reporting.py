# llmresfi/reporting.py
"""Run artifacts, outcome classification, and cross-run analysis.

A run writes ``summary.json`` next to its logs.  The summary contains the
configuration, all-rank injection records, per-rank metrics/loss curves,
recovery events, step times, peak memory, and a parameter signature.  This
module turns a directory of summaries into outcome statistics and overhead
tables.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

SUMMARY_FILENAME = "summary.json"
RECOVERY_LOG = "recovery.jsonl"

_DTYPE_LAYOUT = {
    "float16": (1, 5, 10),
    "fp16": (1, 5, 10),
    "bfloat16": (1, 8, 7),
    "bf16": (1, 8, 7),
    "float32": (1, 8, 23),
    "fp32": (1, 8, 23),
    "float64": (1, 11, 52),
    "fp64": (1, 11, 52),
}

OUTCOME_BASELINE = "baseline"
OUTCOME_NO_FAULT = "no_fault"
OUTCOME_DUE = "due"
OUTCOME_RECOVERED = "recovered"
OUTCOME_MITIGATED = "mitigated"
OUTCOME_DETECTED_UNRECOVERED = "detected_unrecovered"
OUTCOME_SDC = "sdc"
OUTCOME_MASKED = "masked"
OUTCOME_DETECTED = "detected"
OUTCOME_UNDETECTED = "undetected"


# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------
def bit_region(dtype: str, bit_idx: int) -> str:
    """Map an IEEE-754 bit index to ``sign`` / ``exponent`` / ``mantissa``."""
    name = str(dtype).lower()
    if name.startswith("torch."):
        name = name[len("torch."):]
    _sign, exponent, mantissa = _DTYPE_LAYOUT.get(name, (1, 8, 23))
    if bit_idx >= mantissa + exponent:
        return "sign"
    if bit_idx >= mantissa:
        return "exponent"
    return "mantissa"


def record_bit_regions(record: Dict[str, Any]) -> List[str]:
    bits = record.get("bit_idxs")
    if not bits:
        return ["value"]
    dtype = record.get("dtype", "torch.float32")
    regions = {bit_region(dtype, int(b)) for b in bits}
    return sorted(regions)


def param_signature(model, chunk_size: int = 262_144) -> Dict[str, float]:
    """Return a compact, deterministic parameter fingerprint.

    The signature is intentionally stronger than a single parameter sum: it
    combines signed/absolute/L2 moments and a position-weighted checksum.  It is
    still an approximate comparison (summaries are not a replacement for saving
    and diffing the full state dict), but it catches sign changes, magnitude
    drift, and most permutations without writing the model to disk.

    Parameters are processed in float64 chunks so that computing the signature
    does not create a full-model float64 copy.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    total = 0.0
    abs_total = 0.0
    sum_sq = 0.0
    weighted = 0.0
    max_abs = 0.0
    numel = 0
    for p in model.parameters():
        flat = p.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_size):
            end = min(start + chunk_size, flat.numel())
            values = flat[start:end].to(torch.float64)
            total += float(values.sum().item())
            abs_values = values.abs()
            abs_total += float(abs_values.sum().item())
            sum_sq += float(values.square().sum().item())
            if abs_values.numel():
                max_abs = max(max_abs, float(abs_values.max().item()))
            positions = torch.arange(
                start + 1,
                end + 1,
                dtype=torch.float64,
                device=values.device,
            )
            weighted += float((values * positions).sum().item())
            numel += int(values.numel())
    return {
        "param_sum": total,
        "param_abs_sum": abs_total,
        "param_sum_sq": sum_sq,
        "param_weighted_sum": weighted,
        "param_max_abs": max_abs,
        "numel": numel,
    }


def _cfg_get(cfg: Dict[str, Any], *keys, default=None):
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ---------------------------------------------------------------------------
# summary schema
# ---------------------------------------------------------------------------
def build_summary(
    *,
    run_id: str,
    cfg,
    baseline: bool,
    steps: int,
    records: Sequence[dict],
    per_rank_metrics: Sequence[dict],
    recovery_events: Sequence[dict],
    loss_history: Sequence[dict],
    step_times: Sequence[dict],
    peak_memory_gb: Sequence[dict],
    param_signatures: Sequence[dict],
    device: str = "",
    world_size: int = 1,
    wall_time_s: Optional[float] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Assemble the on-disk run summary.

    ``loss_history``, ``step_times``, ``peak_memory_gb``, and
    ``param_signatures`` are lists of per-rank dicts, e.g.
    ``{"rank": 0, "losses": [...]}``.
    """
    cfg_dict = cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else dict(cfg)
    events = sorted(list(recovery_events), key=lambda e: e.get("time", 0.0))
    actions = {e.get("action") for e in events}
    rollback_actions = {"step_rollback", "checkpoint_rollback"}
    recovery_steps = sorted(
        {
            int(e["step"])
            for e in events
            if e.get("action") in rollback_actions and e.get("step") is not None
        }
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "baseline": bool(baseline),
        "device": device,
        "world_size": int(world_size),
        "steps": int(steps),
        "wall_time_s": None if wall_time_s is None else float(wall_time_s),
        "config": cfg_dict,
        "fault_injected": bool(records),
        "detected": bool(events),
        "recovered": bool(actions & rollback_actions),
        "aborted": "abort" in actions,
        "rollback_events": sum(1 for e in events if e.get("action") in rollback_actions),
        "recovery_steps": recovery_steps,
        "records": list(records),
        "per_rank_metrics": list(per_rank_metrics),
        "recovery_events": events,
        "loss_history": list(loss_history),
        "step_times": list(step_times),
        "peak_memory_gb": list(peak_memory_gb),
        "param_signatures": list(param_signatures),
    }
    if extra:
        summary.update(extra)
    return summary


def write_summary(log_dir, summary: dict) -> Path:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SUMMARY_FILENAME
    tmp = directory / (SUMMARY_FILENAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def load_summary(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_summaries(root) -> List[dict]:
    root = Path(root)
    if root.is_file():
        files = [root] if root.name == SUMMARY_FILENAME else []
    else:
        files = sorted(root.rglob(SUMMARY_FILENAME))
    return [load_summary(p) for p in files]


def iter_recovery_events(root) -> List[dict]:
    """Fallback loader for directories that only contain recovery.jsonl."""
    root = Path(root)
    files = [root] if root.is_file() else sorted(root.rglob(RECOVERY_LOG))
    events: List[dict] = []
    for path in files:
        if path.name != RECOVERY_LOG:
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    event = json.loads(line)
                    event.setdefault("_file", str(path))
                    events.append(event)
    return events


# ---------------------------------------------------------------------------
# outcome classification
# ---------------------------------------------------------------------------
def _run_key(summary: dict) -> Tuple:
    cfg = summary.get("config", {})
    return (
        _cfg_get(cfg, "distributed", "backend"),
        _cfg_get(cfg, "model", "repo"),
        _cfg_get(cfg, "model", "device"),
        summary.get("world_size", _cfg_get(cfg, "distributed", "world_size", default=1)),
        _cfg_get(cfg, "execution", "random_seed"),
        summary.get("steps"),
    )


def _metric_map(summary: dict, key: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for item in summary.get(key, []) or []:
        if item.get("rank") is None:
            continue
        value = item.get("loss") if key == "per_rank_metrics" else item.get("gb")
        if value is not None and math.isfinite(float(value)):
            out[int(item["rank"])] = float(value)
    return out


def _signature_map(summary: dict) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for item in summary.get("param_signatures", []) or []:
        if item.get("rank") is None:
            continue
        out[int(item["rank"])] = {
            "param_sum": float(item.get("param_sum", 0.0)),
            "param_abs_sum": float(item.get("param_abs_sum", 0.0)),
            "param_sum_sq": float(item.get("param_sum_sq", 0.0)),
            "param_weighted_sum": float(item.get("param_weighted_sum", 0.0)),
            "param_max_abs": float(item.get("param_max_abs", 0.0)),
            "numel": float(item.get("numel", 0.0)),
        }
    return out


def _loss_curve_map(summary: dict) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for item in summary.get("loss_history", []) or []:
        if item.get("rank") is None:
            continue
        out[int(item["rank"])] = [float(v) for v in item.get("losses", [])]
    return out


def compare_to_baseline(summary: dict, baseline: dict) -> Dict[str, float]:
    run_loss = _metric_map(summary, "per_rank_metrics")
    base_loss = _metric_map(baseline, "per_rank_metrics")
    common_loss = sorted(set(run_loss) & set(base_loss))
    final_loss_diff = (
        max(abs(run_loss[r] - base_loss[r]) for r in common_loss)
        if common_loss
        else float("inf")
    )

    run_curve = _loss_curve_map(summary)
    base_curve = _loss_curve_map(baseline)
    common_curve = sorted(set(run_curve) & set(base_curve))
    curve_max_diff = 0.0
    curve_final_diff = 0.0
    first_divergence_step: Optional[int] = None
    for rank in common_curve:
        run_values = run_curve[rank]
        base_values = base_curve[rank]
        for step, (got, want) in enumerate(zip(run_values, base_values)):
            if not (math.isfinite(got) and math.isfinite(want)):
                if math.isnan(got) and math.isnan(want):
                    diff = 0.0
                else:
                    diff = float("inf")
            else:
                diff = abs(got - want)
            curve_max_diff = max(curve_max_diff, diff)
            if diff > 1e-12 and first_divergence_step is None:
                first_divergence_step = step
        if run_values and base_values:
            got, want = run_values[-1], base_values[-1]
            if math.isfinite(got) and math.isfinite(want):
                curve_final_diff = max(curve_final_diff, abs(got - want))
            elif not (math.isnan(got) and math.isnan(want)):
                curve_final_diff = float("inf")
        if len(run_values) != len(base_values):
            curve_max_diff = float("inf")

    if common_curve:
        loss_diff = max(final_loss_diff, curve_max_diff)
    else:
        loss_diff = final_loss_diff

    run_sig = _signature_map(summary)
    base_sig = _signature_map(baseline)
    common_sig = sorted(set(run_sig) & set(base_sig))
    if common_sig:
        param_abs_diff = max(
            abs(run_sig[r]["param_abs_sum"] - base_sig[r]["param_abs_sum"])
            for r in common_sig
        )
        param_rel_diff = 0.0
        for rank in common_sig:
            for metric, base_value in base_sig[rank].items():
                run_value = run_sig[rank].get(metric, 0.0)
                relation = abs(run_value - base_value) / max(1.0, abs(base_value))
                param_rel_diff = max(param_rel_diff, relation)
    else:
        param_abs_diff = float("inf")
        param_rel_diff = float("inf")
    return {
        "loss_diff": loss_diff,
        "final_loss_diff": final_loss_diff,
        "curve_max_diff": curve_max_diff,
        "curve_final_diff": curve_final_diff,
        "first_divergence_step": first_divergence_step,
        "param_abs_diff": param_abs_diff,
        "param_rel_diff": param_rel_diff,
    }


def classify_summary(
    summary: dict,
    baseline: Optional[dict] = None,
    loss_atol: float = 1e-3,
    param_atol: float = 1e-6,
) -> Dict[str, Any]:
    """Classify one run into the reliability taxonomy.

    With a matching baseline, the outcome is one of:
    ``recovered``, ``mitigated``, ``detected_unrecovered``, ``sdc``,
    ``masked``.  Without a baseline, only ``detected`` / ``undetected`` are
    reported.
    """
    if summary.get("baseline"):
        return {"outcome": OUTCOME_BASELINE, "detected": False, "recovered": False}
    if summary.get("aborted"):
        return {"outcome": OUTCOME_DUE, "detected": True, "recovered": False}
    if not summary.get("fault_injected"):
        return {"outcome": OUTCOME_NO_FAULT, "detected": False, "recovered": False}

    detected = bool(summary.get("detected"))
    recovered = bool(summary.get("recovered"))
    if baseline is None:
        outcome = OUTCOME_DETECTED if detected else OUTCOME_UNDETECTED
        return {"outcome": outcome, "detected": detected, "recovered": recovered}

    diff = compare_to_baseline(summary, baseline)
    diverged = diff["loss_diff"] > loss_atol or diff["param_rel_diff"] > param_atol
    if detected and recovered and not diverged:
        outcome = OUTCOME_RECOVERED
    elif detected and not diverged:
        outcome = OUTCOME_MITIGATED
    elif detected and diverged:
        outcome = OUTCOME_DETECTED_UNRECOVERED
    elif not detected and diverged:
        outcome = OUTCOME_SDC
    else:
        outcome = OUTCOME_MASKED
    return {
        "outcome": outcome,
        "detected": detected,
        "recovered": recovered,
        "diverged": diverged,
        **diff,
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _add(group: Dict[str, Dict[str, Any]], key, outcome: str) -> None:
    if key is None:
        key = "unknown"
    bucket = group.setdefault(str(key), {"runs": 0, "outcomes": Counter()})
    bucket["runs"] += 1
    bucket["outcomes"][outcome] += 1


def _counter_to_dict(counter: Counter) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def aggregate_summaries(
    summaries: Sequence[dict],
    loss_atol: float = 1e-3,
    param_atol: float = 1e-6,
) -> dict:
    """Aggregate summaries into outcome, sensitivity, and overhead reports."""
    baselines = {_run_key(s): s for s in summaries if s.get("baseline")}
    per_run: List[dict] = []
    outcomes: Counter = Counter()

    by_fault_model: Dict[str, dict] = {}
    by_injection_point: Dict[str, dict] = {}
    by_bit_region: Dict[str, dict] = {}
    by_rank: Dict[str, dict] = {}
    by_bucket: Dict[str, dict] = {}
    by_timing: Dict[str, dict] = {}
    by_backend: Dict[str, dict] = {}
    by_mitigation: Dict[str, dict] = {}

    for summary in summaries:
        baseline = baselines.get(_run_key(summary))
        result = classify_summary(summary, baseline, loss_atol, param_atol)
        outcome = result["outcome"]
        outcomes[outcome] += 1

        cfg = summary.get("config", {})
        fault_model = _cfg_get(cfg, "fault", "fault_model", "type", default="unknown")
        timing = _cfg_get(cfg, "fault", "timing", "type", default="unknown")
        backend = _cfg_get(cfg, "distributed", "backend", default="unknown")
        mitigation_enabled = bool(_cfg_get(cfg, "mitigation", "enabled", default=False))
        mitigation_mode = (
            _cfg_get(cfg, "mitigation", "recovery", "mode", default="none")
            if mitigation_enabled
            else "none"
        )
        injection_point = _cfg_get(
            cfg, "distributed", "injection_point", default="unknown"
        )
        records = summary.get("records", []) or []
        if records:
            injection_point = records[0].get("injection_point", injection_point)

        _add(by_backend, backend, outcome)
        if not summary.get("baseline"):
            _add(by_fault_model, fault_model, outcome)
            _add(by_timing, timing, outcome)
            _add(by_mitigation, mitigation_mode, outcome)

            points = {r.get("injection_point", injection_point) for r in records} or {injection_point}
            for point in points:
                _add(by_injection_point, point, outcome)

        regions = set()
        ranks = set()
        buckets = set()
        for rec in records:
            regions.update(record_bit_regions(rec))
            if rec.get("rank") is not None:
                ranks.add(rec["rank"])
            if rec.get("bucket_index") is not None:
                buckets.add(rec["bucket_index"])
        if not summary.get("baseline"):
            for region in regions or {"none"}:
                _add(by_bit_region, region, outcome)
            for rank in ranks or {"none"}:
                _add(by_rank, rank, outcome)
            for bucket in buckets or {"none"}:
                _add(by_bucket, bucket, outcome)

        per_run.append(
            {
                "run_id": summary.get("run_id", "unknown"),
                "outcome": outcome,
                "detected": result.get("detected", False),
                "recovered": result.get("recovered", False),
                "baseline": bool(summary.get("baseline")),
                "baseline_matched": baseline is not None,
                "fault_model": fault_model,
                "timing": timing,
                "backend": backend,
                "injection_point": injection_point,
                "mitigation": mitigation_mode,
                "bit_regions": sorted(regions),
                "ranks": sorted(ranks),
                "buckets": sorted(buckets),
                "faults": len(records),
                "rollback_events": int(summary.get("rollback_events", 0) or 0),
                "wall_time_s": summary.get("wall_time_s"),
                **{
                    k: v
                    for k, v in result.items()
                    if k
                    in (
                        "loss_diff",
                        "final_loss_diff",
                        "curve_max_diff",
                        "curve_final_diff",
                        "first_divergence_step",
                        "param_abs_diff",
                        "param_rel_diff",
                        "diverged",
                    )
                },
            }
        )

    overhead = _overhead_table(summaries)
    injected_runs = sum(
        1
        for summary in summaries
        if not summary.get("baseline") and summary.get("fault_injected")
    )
    detected_runs = sum(
        1
        for summary in summaries
        if not summary.get("baseline") and summary.get("fault_injected") and summary.get("detected")
    )
    recovered_runs = sum(
        1
        for summary in summaries
        if not summary.get("baseline") and summary.get("fault_injected") and summary.get("recovered")
    )
    return {
        "runs": len(summaries),
        "baselines": sum(1 for s in summaries if s.get("baseline")),
        "faulty_runs": sum(1 for s in summaries if not s.get("baseline")),
        "injected_runs": injected_runs,
        "detection_rate": (detected_runs / injected_runs) if injected_runs else None,
        "recovery_rate": (recovered_runs / injected_runs) if injected_runs else None,
        "baseline_missing": sum(
            1
            for s in summaries
            if not s.get("baseline") and _run_key(s) not in baselines
        ),
        "outcomes": _counter_to_dict(outcomes),
        "by_fault_model": _group_to_dict(by_fault_model),
        "by_injection_point": _group_to_dict(by_injection_point),
        "by_bit_region": _group_to_dict(by_bit_region),
        "by_rank": _group_to_dict(by_rank),
        "by_bucket": _group_to_dict(by_bucket),
        "by_timing": _group_to_dict(by_timing),
        "by_backend": _group_to_dict(by_backend),
        "by_mitigation": _group_to_dict(by_mitigation),
        "overhead": overhead,
        "per_run": per_run,
    }


def _group_to_dict(group: Dict[str, dict]) -> Dict[str, dict]:
    return {
        key: {"runs": val["runs"], "outcomes": _counter_to_dict(val["outcomes"])}
        for key, val in sorted(group.items())
    }


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None, None
    mean = statistics.fmean(clean)
    std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return mean, std


def _overhead_table(summaries: Sequence[dict]) -> List[dict]:
    groups: Dict[Tuple, dict] = defaultdict(
        lambda: {
            "step_times": [],
            "peak_mem": [],
            "wall_times": [],
            "recovery_steps": set(),
            "recoveries": 0,
            "runs": 0,
        }
    )
    for summary in summaries:
        cfg = summary.get("config", {})
        backend = _cfg_get(cfg, "distributed", "backend", default="unknown")
        enabled = bool(_cfg_get(cfg, "mitigation", "enabled", default=False))
        mode = (
            _cfg_get(cfg, "mitigation", "recovery", "mode", default="none")
            if enabled
            else "none"
        )
        key = (backend, mode, bool(summary.get("baseline")))
        entry = groups[key]
        entry["runs"] += 1
        if summary.get("wall_time_s") is not None:
            entry["wall_times"].append(float(summary["wall_time_s"]))
        for step in summary.get("recovery_steps", []) or []:
            entry["recovery_steps"].add(int(step))
        entry["recoveries"] += sum(
            1
            for e in summary.get("recovery_events", []) or []
            if e.get("action") in ("step_rollback", "checkpoint_rollback")
        )
        for item in summary.get("step_times", []) or []:
            times = item.get("times", [])
            if times:
                entry["step_times"].append(statistics.fmean(times) * 1000.0)
        for item in summary.get("peak_memory_gb", []) or []:
            if item.get("gb") is not None:
                entry["peak_mem"].append(float(item["gb"]))

    rows = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1], not item[0][2]),
    )
    for (backend, mode, baseline), entry in ordered_groups:
        mean_ms, std_ms = _mean_std(entry["step_times"])
        mean_mem, _ = _mean_std(entry["peak_mem"])
        mean_wall, _ = _mean_std(entry["wall_times"])
        rows.append(
            {
                "backend": backend,
                "mitigation": mode,
                "baseline": baseline,
                "runs": entry["runs"],
                "mean_step_time_ms": mean_ms,
                "std_step_time_ms": std_ms,
                "mean_peak_memory_gb": mean_mem,
                "mean_wall_time_s": mean_wall,
                "recovery_events": entry["recoveries"],
                "retried_steps": len(entry["recovery_steps"]),
                "step_time_overhead_pct": None,
                "peak_memory_overhead_pct": None,
            }
        )

    baseline_rows: Dict[str, dict] = {}
    baseline_rows_by_mode: Dict[Tuple[str, str], dict] = {}
    for row in rows:
        if row["baseline"] and row["mean_step_time_ms"] is not None:
            baseline_rows.setdefault(row["backend"], row)
            baseline_rows_by_mode.setdefault(
                (row["backend"], row["mitigation"]), row
            )
    for row in rows:
        reference = baseline_rows_by_mode.get(
            (row["backend"], row["mitigation"])
        ) or baseline_rows.get(row["backend"])
        if row["baseline"] or reference is None:
            continue
        if row["mean_step_time_ms"] is not None and reference["mean_step_time_ms"]:
            row["step_time_overhead_pct"] = 100.0 * (
                row["mean_step_time_ms"] / reference["mean_step_time_ms"] - 1.0
            )
        if row["mean_peak_memory_gb"] is not None and reference["mean_peak_memory_gb"]:
            row["peak_memory_overhead_pct"] = 100.0 * (
                row["mean_peak_memory_gb"] / reference["mean_peak_memory_gb"] - 1.0
            )
    return rows


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------
def _format_group(title: str, group: Dict[str, dict]) -> List[str]:
    lines = [f"{title}:"]
    if not group:
        lines.append("  (none)")
        return lines
    for key, value in group.items():
        outcomes = ", ".join(f"{k}={v}" for k, v in value["outcomes"].items())
        lines.append(f"  {key}: runs={value['runs']}  {outcomes}")
    return lines


def format_report(report: dict) -> str:
    lines = [
        f"Runs: {report['runs']} (baselines={report['baselines']}, faulty={report['faulty_runs']})",
        "Outcomes: " + ", ".join(f"{k}={v}" for k, v in report["outcomes"].items()),
    ]
    detection = report.get("detection_rate")
    recovery = report.get("recovery_rate")
    if detection is not None:
        lines.append(
            f"Injected runs: {report['injected_runs']}  "
            f"detection={100.0 * detection:.1f}%  recovery={100.0 * recovery:.1f}%"
        )
    if report.get("baseline_missing"):
        lines.append(
            f"WARNING: {report['baseline_missing']} faulty run(s) have no matching baseline; "
            "they are classified only as detected/undetected."
        )
    lines.append("")
    lines.extend(_format_group("By fault model", report["by_fault_model"]))
    lines.append("")
    lines.extend(_format_group("By injection point", report["by_injection_point"]))
    lines.append("")
    lines.extend(_format_group("By bit region", report["by_bit_region"]))
    lines.append("")
    lines.extend(_format_group("By injection rank", report["by_rank"]))
    lines.append("")
    lines.extend(_format_group("By bucket", report["by_bucket"]))
    lines.append("")
    lines.extend(_format_group("By timing", report["by_timing"]))
    lines.append("")
    lines.extend(_format_group("By mitigation", report["by_mitigation"]))
    lines.append("")
    lines.append("Overhead:")
    if not report["overhead"]:
        lines.append("  (none)")
    else:
        lines.append(
            "  backend   mitigation         baseline  runs  mean_ms   std_ms   peak_GB  "
            "time_ovh%  mem_ovh%  retried  events"
        )
        for row in report["overhead"]:
            mean_ms = "-" if row["mean_step_time_ms"] is None else f"{row['mean_step_time_ms']:.2f}"
            std_ms = "-" if row["std_step_time_ms"] is None else f"{row['std_step_time_ms']:.2f}"
            mem = "-" if row["mean_peak_memory_gb"] is None else f"{row['mean_peak_memory_gb']:.4f}"
            time_ovh = (
                "-"
                if row["step_time_overhead_pct"] is None
                else f"{row['step_time_overhead_pct']:.1f}"
            )
            mem_ovh = (
                "-"
                if row["peak_memory_overhead_pct"] is None
                else f"{row['peak_memory_overhead_pct']:.1f}"
            )
            lines.append(
                f"  {row['backend']:<9} {row['mitigation']:<20} "
                f"{str(row['baseline']):<8} {row['runs']:>4} "
                f"{mean_ms:>8} {std_ms:>8} {mem:>8} {time_ovh:>9} {mem_ovh:>8} "
                f"{row['retried_steps']:>7} {row['recovery_events']:>7}"
            )
    return "\n".join(lines)


def format_legacy_events(events: Sequence[dict]) -> str:
    actions = Counter(e.get("action", "unknown") for e in events)
    signals = Counter(
        s.get("kind", "unknown") for e in events for s in (e.get("signals") or [])
    )
    lines = [
        "No summary.json found; reporting recovery.jsonl events only.",
        f"Events: {len(events)}",
        "Actions: " + ", ".join(f"{k}={v}" for k, v in sorted(actions.items())),
        "Signals: " + ", ".join(f"{k}={v}" for k, v in sorted(signals.items())),
    ]
    return "\n".join(lines)
