from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from llmresfi.reporting import (
    OUTCOME_BASELINE,
    OUTCOME_DETECTED,
    OUTCOME_DETECTED_UNRECOVERED,
    OUTCOME_DUE,
    OUTCOME_MASKED,
    OUTCOME_MITIGATED,
    OUTCOME_RECOVERED,
    OUTCOME_SDC,
    OUTCOME_UNDETECTED,
    aggregate_summaries,
    bit_region,
    build_summary,
    classify_summary,
    compare_to_baseline,
    format_report,
    iter_recovery_events,
    iter_summaries,
    load_summary,
    param_signature,
    write_summary,
)


def _config(*, mode: str = "none", enabled: bool = False) -> dict:
    return {
        "model": {"repo": "mlp", "device": "cuda"},
        "distributed": {
            "backend": "ddp",
            "world_size": 2,
            "injection_point": "before_reduce",
        },
        "fault": {
            "fault_model": {"type": "single"},
            "timing": {"type": "transient"},
        },
        "mitigation": {
            "enabled": enabled,
            "recovery": {"mode": mode},
        },
        "execution": {"random_seed": 42},
    }


def _summary(
    run_id: str,
    *,
    baseline: bool = False,
    losses=(1.0, 0.9),
    signature=(10.0, 10.0),
    detected: bool = False,
    recovered: bool = False,
    aborted: bool = False,
    mode: str = "none",
    step_time: float = 0.01,
    peak_memory: float = 1.0,
) -> dict:
    records = []
    events = []
    if not baseline:
        records = [
            {
                "rank": 0,
                "injection_point": "before_reduce",
                "bucket_index": 0,
                "dtype": "torch.float32",
                "bit_idxs": [0],
            }
        ]
    if detected:
        events = [
            {"rank": rank, "step": 0, "action": "sanitized" if mode == "sanitize" else "step_rollback"}
            for rank in (0, 1)
        ]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "baseline": baseline,
        "world_size": 2,
        "steps": len(losses),
        "wall_time_s": 1.0,
        "config": _config(mode=mode, enabled=mode != "none"),
        "fault_injected": bool(records),
        "detected": detected,
        "recovered": recovered,
        "aborted": aborted,
        "rollback_events": 2 if recovered else 0,
        "recovery_steps": [0] if recovered else [],
        "records": records,
        "per_rank_metrics": [
            {"rank": rank, "loss": losses[-1], "grad_norm": 1.0} for rank in (0, 1)
        ],
        "recovery_events": events,
        "loss_history": [
            {"rank": rank, "losses": list(losses)} for rank in (0, 1)
        ],
        "step_times": [
            {"rank": rank, "times": [step_time] * len(losses)} for rank in (0, 1)
        ],
        "peak_memory_gb": [{"rank": rank, "gb": peak_memory} for rank in (0, 1)],
        "param_signatures": [
            {
                "rank": rank,
                "param_sum": signature[0],
                "param_abs_sum": signature[1],
                "param_sum_sq": signature[1] ** 2,
                "param_weighted_sum": signature[0] * 2,
                "param_max_abs": 1.0,
                "numel": 10,
            }
            for rank in (0, 1)
        ],
    }


def test_bit_region_fp32_and_bf16():
    assert bit_region("torch.float32", 0) == "mantissa"
    assert bit_region("float32", 22) == "mantissa"
    assert bit_region("float32", 23) == "exponent"
    assert bit_region("float32", 31) == "sign"
    assert bit_region("bf16", 0) == "mantissa"
    assert bit_region("bfloat16", 7) == "exponent"
    assert bit_region("torch.bfloat16", 15) == "sign"


def test_param_signature_detects_sign_change_and_reports_moments():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    before = param_signature(model, chunk_size=5)
    with torch.no_grad():
        model.weight.neg_()
    after = param_signature(model, chunk_size=5)

    assert before["numel"] == after["numel"] == 15
    assert before["param_abs_sum"] == after["param_abs_sum"]
    assert before["param_sum"] != after["param_sum"]
    assert before["param_weighted_sum"] != after["param_weighted_sum"]
    assert after["param_sum_sq"] > 0


def test_classification_without_baseline():
    detected = _summary("fault-detected", detected=True)
    undetected = _summary("fault-undetected")
    assert classify_summary(detected)["outcome"] == OUTCOME_DETECTED
    assert classify_summary(undetected)["outcome"] == OUTCOME_UNDETECTED


def test_classification_with_baseline():
    baseline = _summary("base", baseline=True)

    recovered = _summary("recovered", detected=True, recovered=True)
    assert classify_summary(recovered, baseline)["outcome"] == OUTCOME_RECOVERED

    mitigated = _summary("mitigated", detected=True, mode="sanitize")
    assert classify_summary(mitigated, baseline)["outcome"] == OUTCOME_MITIGATED

    detected_unrecovered = _summary(
        "detected-unrecovered",
        losses=(1.0, 1.5),
        signature=(12.0, 12.0),
        detected=True,
        recovered=True,
    )
    assert classify_summary(detected_unrecovered, baseline)["outcome"] == OUTCOME_DETECTED_UNRECOVERED

    sdc = _summary("sdc", losses=(1.0, 1.5), signature=(12.0, 12.0))
    assert classify_summary(sdc, baseline)["outcome"] == OUTCOME_SDC

    masked = _summary("masked")
    assert classify_summary(masked, baseline)["outcome"] == OUTCOME_MASKED

    due = _summary("due", aborted=True)
    assert classify_summary(due, baseline)["outcome"] == OUTCOME_DUE
    assert classify_summary(baseline, baseline)["outcome"] == OUTCOME_BASELINE


def test_compare_to_baseline_reports_curve_and_first_divergence():
    baseline = _summary("base", baseline=True, losses=(1.0, 0.9, 0.8))
    run = _summary("run", losses=(1.0, 0.95, 0.8), signature=(10.0, 10.0))

    diff = compare_to_baseline(run, baseline)
    assert diff["final_loss_diff"] == 0.0
    assert diff["curve_max_diff"] > 0.049
    assert diff["first_divergence_step"] == 1
    assert diff["param_rel_diff"] < 1e-12


def test_aggregate_outcomes_rates_and_overhead():
    baseline = _summary("base", baseline=True, step_time=0.01, peak_memory=1.0)
    recovered = _summary(
        "recovered",
        detected=True,
        recovered=True,
        mode="step_rollback",
        step_time=0.015,
        peak_memory=1.2,
    )
    sdc = _summary("sdc", losses=(1.0, 1.2), signature=(11.0, 11.0))

    report = aggregate_summaries([baseline, recovered, sdc])
    assert report["runs"] == 3
    assert report["injected_runs"] == 2
    assert report["detection_rate"] == 0.5
    assert report["recovery_rate"] == 0.5
    assert report["baseline_missing"] == 0
    assert report["outcomes"][OUTCOME_RECOVERED] == 1
    assert report["outcomes"][OUTCOME_SDC] == 1
    assert report["by_fault_model"]["single"]["runs"] == 2
    assert report["by_bit_region"]["mantissa"]["runs"] == 2
    assert report["by_rank"]["0"]["runs"] == 2

    rollback_row = next(row for row in report["overhead"] if row["mitigation"] == "step_rollback")
    assert rollback_row["mean_step_time_ms"] == 15.0
    assert abs(rollback_row["step_time_overhead_pct"] - 50.0) < 1e-9
    assert abs(rollback_row["peak_memory_overhead_pct"] - 20.0) < 1e-9
    assert rollback_row["retried_steps"] == 1

    rendered = format_report(report)
    assert "Outcomes:" in rendered
    assert "detection=50.0%" in rendered
    assert "time_ovh%" in rendered


def test_build_summary_derives_recovery_fields(tmp_path):
    events = [
        {"rank": 1, "step": 2, "action": "step_rollback", "time": 2.0},
        {"rank": 0, "step": 2, "action": "step_rollback", "time": 1.0},
    ]
    summary = build_summary(
        run_id="run",
        cfg={"execution": {"random_seed": 1}},
        baseline=False,
        steps=3,
        records=[{"rank": 0}],
        per_rank_metrics=[],
        recovery_events=events,
        loss_history=[],
        step_times=[],
        peak_memory_gb=[],
        param_signatures=[],
        world_size=2,
        wall_time_s=12.5,
    )
    assert summary["fault_injected"] is True
    assert summary["detected"] is True
    assert summary["recovered"] is True
    assert summary["rollback_events"] == 2
    assert summary["recovery_steps"] == [2]
    assert summary["recovery_events"][0]["rank"] == 0

    path = write_summary(tmp_path, summary)
    assert load_summary(path)["run_id"] == "run"
    assert iter_summaries(tmp_path)[0]["world_size"] == 2


def test_legacy_recovery_jsonl_fallback(tmp_path):
    path = tmp_path / "run" / "recovery.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"rank": 0, "step": 1, "action": "abort", "signals": []}) + "\n",
        encoding="utf-8",
    )
    events = iter_recovery_events(tmp_path)
    assert len(events) == 1
    assert events[0]["_file"].endswith("recovery.jsonl")


def test_analyzer_cli_json(tmp_path):
    baseline = _summary("base", baseline=True)
    recovered = _summary("recovered", detected=True, recovered=True)
    write_summary(tmp_path / "baseline", baseline)
    write_summary(tmp_path / "fault", recovered)

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "examples/analyze_runs.py",
            "--root",
            str(tmp_path),
            "--json",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["outcomes"] == {"baseline": 1, "recovered": 1}
    assert report["detection_rate"] == 1.0
    assert report["recovery_rate"] == 1.0
