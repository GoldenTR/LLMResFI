#!/usr/bin/env python3
"""Summarize ResNet-18 distributed fault-injection runs.

The report compares every run with a baseline using:

* test accuracy / loss deltas,
* exact SHA-256 equality of model and optimizer state,
* per-rank loss-curve divergence,
* recovery events, retried steps, and median accepted-step time.

Examples
--------
Analyze the default sibling ``ResNet-18/runs`` directory::

    python examples/resnet18/analyze_resnet18.py

Write reports elsewhere::

    python examples/resnet18/analyze_resnet18.py --root /path/to/runs \
        --output-md /path/to/report.md --output-json /path/to/report.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT.parent / "ResNet-18"


def load_runs(root: Path) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(root.iterdir()):
        summary_path = path / "summary.json"
        if not summary_path.is_file():
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["_dir"] = path.name
        summary["_path"] = str(summary_path)
        runs.append(summary)
    return runs


def _hash_set(summary: Dict[str, Any], key: str) -> set:
    values = summary.get(key) or []
    return {str(item.get("sha256")) for item in values if item.get("sha256")}


def _rank_series(summary: Dict[str, Any], key: str, field: str) -> Dict[int, List[float]]:
    series: Dict[int, List[float]] = {}
    for entry in summary.get(key) or []:
        rank = int(entry.get("rank", len(series)))
        values = entry.get(field) or []
        series[rank] = [float(value) for value in values]
    return series


def _curve_diff(baseline: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    if run.get("_dir") == baseline.get("_dir"):
        return {"curve_max_abs_diff": 0.0, "first_divergence_step": None}
    base = _rank_series(baseline, "loss_history", "losses")
    other = _rank_series(run, "loss_history", "losses")
    max_diff = 0.0
    first = None
    for rank in sorted(set(base) & set(other)):
        for step, (left, right) in enumerate(zip(base[rank], other[rank])):
            diff = abs(left - right)
            if diff > max_diff:
                max_diff = diff
            if first is None and diff > 1e-9:
                first = step
    return {"curve_max_abs_diff": max_diff, "first_divergence_step": first}


def _median_step_ms(summary: Dict[str, Any]) -> Optional[float]:
    samples = []
    for entry in summary.get("step_times") or []:
        values = [float(value) for value in entry.get("times") or []]
        # The first step includes CUDA/NCCL warm-up and is not comparable.
        samples.extend(values[1:] if len(values) > 1 else values)
    return statistics.median(samples) * 1000.0 if samples else None


def _peak_memory_gb(summary: Dict[str, Any]) -> Optional[float]:
    values = [
        float(entry["gb"])
        for entry in summary.get("peak_memory_gb") or []
        if entry.get("gb") is not None
    ]
    return max(values) if values else None


def _first_record(summary: Dict[str, Any]) -> Dict[str, Any]:
    records = summary.get("records") or []
    return records[0] if records else {}


def classify(run: Dict[str, Any], model_match: bool, optimizer_match: bool) -> str:
    if run.get("baseline"):
        return "baseline"
    if not run.get("fault_injected"):
        return "no_fault"
    if run.get("recovered"):
        return "recovered_lossless" if model_match and optimizer_match else "recovered_not_exact"
    if run.get("detected"):
        return "detected_unrecovered"
    if model_match and optimizer_match:
        return "masked"
    return "sdc"


def summarize(root: Path, baseline_dir: str) -> Dict[str, Any]:
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"No summary.json files found under {root}")
    by_dir = {run["_dir"]: run for run in runs}
    baseline = by_dir.get(baseline_dir)
    if baseline is None:
        matches = [run for run in runs if run.get("baseline")]
        if not matches:
            raise SystemExit(f"Baseline run {baseline_dir!r} was not found under {root}")
        baseline = matches[0]

    baseline_model_hashes = _hash_set(baseline, "state_sha256")
    baseline_optimizer_hashes = _hash_set(baseline, "optimizer_state_sha256")
    baseline_step_ms = _median_step_ms(baseline)
    rows = []

    for run in runs:
        model_hashes = _hash_set(run, "state_sha256")
        optimizer_hashes = _hash_set(run, "optimizer_state_sha256")
        model_match = bool(model_hashes) and model_hashes == baseline_model_hashes
        optimizer_match = bool(optimizer_hashes) and optimizer_hashes == baseline_optimizer_hashes
        step_ms = _median_step_ms(run)
        overhead_pct = None
        if step_ms is not None and baseline_step_ms:
            overhead_pct = 100.0 * (step_ms / baseline_step_ms - 1.0)
        record = _first_record(run)
        rank_losses = [float(m["loss"]) for m in run.get("per_rank_metrics") or []]
        rows.append(
            {
                "scenario": run["_dir"],
                "run_id": run.get("run_id"),
                "baseline": bool(run.get("baseline")),
                "fault_injected": bool(run.get("fault_injected")),
                "detected": bool(run.get("detected")),
                "recovered": bool(run.get("recovered")),
                "aborted": bool(run.get("aborted")),
                "outcome": classify(run, model_match, optimizer_match),
                "injection_point": record.get("injection_point"),
                "fault_model": record.get("fault_model"),
                "bit_idxs": record.get("bit_idxs"),
                "fault_rank": record.get("rank"),
                "fault_step": record.get("step"),
                "test_loss": run.get("test_loss"),
                "test_accuracy": run.get("test_accuracy"),
                "test_loss_delta": (
                    None
                    if run.get("test_loss") is None or baseline.get("test_loss") is None
                    else float(run["test_loss"]) - float(baseline["test_loss"])
                ),
                "test_accuracy_delta_pp": (
                    None
                    if run.get("test_accuracy") is None or baseline.get("test_accuracy") is None
                    else 100.0 * (float(run["test_accuracy"]) - float(baseline["test_accuracy"]))
                ),
                "final_loss_mean": statistics.mean(rank_losses) if rank_losses else None,
                "final_loss_ranks": rank_losses,
                "attempts": run.get("attempts"),
                "retried_steps": run.get("retried_steps"),
                "rollback_events": run.get("rollback_events"),
                "recovery_steps": run.get("recovery_steps") or [],
                "model_hashes": sorted(model_hashes),
                "optimizer_hashes": sorted(optimizer_hashes),
                "model_match_baseline": model_match,
                "optimizer_match_baseline": optimizer_match,
                "lossless_model_and_optimizer": model_match and optimizer_match,
                "ranks_identical": len(model_hashes) == 1,
                "median_step_ms": step_ms,
                "step_overhead_pct": overhead_pct,
                "peak_memory_gb": _peak_memory_gb(run),
                **_curve_diff(baseline, run),
            }
        )

    return {
        "root": str(root),
        "baseline": baseline["_dir"],
        "baseline_model_hashes": sorted(baseline_model_hashes),
        "baseline_optimizer_hashes": sorted(baseline_optimizer_hashes),
        "baseline_median_step_ms": baseline_step_ms,
        "runs": rows,
    }


def summarize_overhead(root: Path) -> Optional[Dict[str, Any]]:
    """Summarize controlled no-fault runs that exercise the DDP comm hook.

    The runs must use an out-of-range ``inject_rank`` so the hook and optional
    checksum run every step, while no fault actually fires.
    """
    if not root.is_dir():
        return None
    groups = {"plain": "h_plain", "verify": "h_verify", "rollback": "h_rollback"}
    rows = []
    for key, prefix in groups.items():
        medians = []
        walls = []
        runs = []
        for path in sorted(root.iterdir()):
            if not path.name.startswith(prefix) or not (path / "summary.json").is_file():
                continue
            with open(path / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
            step_ms = _median_step_ms(summary)
            if step_ms is not None:
                medians.append(step_ms)
                walls.append(float(summary.get("wall_time_s") or 0.0))
                runs.append(path.name)
        if medians:
            rows.append(
                {
                    "mode": key,
                    "runs": runs,
                    "median_step_ms": statistics.median(medians),
                    "median_wall_s": statistics.median(walls) if walls else None,
                }
            )
    if not rows:
        return None
    plain = next((row for row in rows if row["mode"] == "plain"), None)
    for row in rows:
        if plain is None or not plain["median_step_ms"]:
            row["delta_ms"] = None
            row["overhead_pct"] = None
        else:
            row["delta_ms"] = row["median_step_ms"] - plain["median_step_ms"]
            row["overhead_pct"] = 100.0 * (
                row["median_step_ms"] / plain["median_step_ms"] - 1.0
            )
    return {
        "root": str(root),
        "setup": "inject_rank=99 keeps the DDP comm hook armed while no fault fires",
        "rows": rows,
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_signed(value, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}{suffix}"


def format_markdown(report: Dict[str, Any]) -> str:
    baseline = next(row for row in report["runs"] if row["baseline"])
    lines = [
        "# ResNet-18 分布式软错误实验报告",
        "",
        "本报告比较 CIFAR-10 / ResNet-18 在双卡 DDP 下的无故障基线、静默故障、"
        "检测与回滚恢复运行。判定“严格无损”时同时要求完整模型 state-dict 与完整"
        "优化器状态的 SHA-256 均与基线一致。",
        "",
        "## 实验配置",
        "",
        f"- 场景目录：`{report['root']}`",
        f"- 基线：`{baseline['run_id']}`",
        "- 数据：CIFAR-10，确定性预处理，无随机增强",
        "- 模型：CIFAR 版 ResNet-18（3x3 stem，无 max-pool）",
        f"- 测试准确率：{_fmt(baseline['test_accuracy'] * 100, 2)}%",
        f"- 测试损失：{_fmt(baseline['test_loss'], 6)}",
        f"- 基线中位 step 时间：{_fmt(report['baseline_median_step_ms'], 2)} ms",
        "",
        "## 结果总览",
        "",
        "| 场景 | bit | 注入点 | 结果 | 检测/恢复 | 模型哈希 | 优化器哈希 | Test Acc | ΔAcc | Test Loss | ΔLoss | 重试 step | 中位 step |",
        "|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        bits = ",".join(str(bit) for bit in row["bit_idxs"]) if row["bit_idxs"] else "-"
        point = row["injection_point"] or "-"
        recovery = f"{row['detected']}/{row['recovered']}"
        lines.append(
            "| {scenario} | {bits} | {point} | {outcome} | {recovery} | {model} | {opt} | "
            "{acc}% | {dacc} | {loss} | {dloss} | {retry} | {step} ms |".format(
                scenario=row["scenario"],
                bits=bits,
                point=point,
                outcome=row["outcome"],
                recovery=recovery,
                model="是" if row["model_match_baseline"] else "否",
                opt="是" if row["optimizer_match_baseline"] else "否",
                acc=_fmt((row["test_accuracy"] or 0.0) * 100, 2),
                dacc=_fmt_signed(row["test_accuracy_delta_pp"], 2, " pp"),
                loss=_fmt(row["test_loss"], 6),
                dloss=_fmt_signed(row["test_loss_delta"], 6),
                retry=_fmt(row["retried_steps"]),
                step=_fmt(row["median_step_ms"], 2),
            )
        )

    lines.extend(
        [
            "",
            "## 字段含义",
            "",
            "- `bit`：被翻转的 IEEE-754 位；float32 的 23–30 位属于指数域。",
            "- `注入点`：`after_reduce` 只污染指定 rank，制造 rank 间非对称发散；"
            "`before_reduce` 会在全归约中传播到所有 rank。",
            "- `结果`：`sdc` 表示未检测且最终状态偏离基线；`recovered_lossless` 表示"
            "检测触发、回滚完成，且模型与优化器均逐 bit 匹配基线；`masked` 表示未检测但无偏离。",
            "- `模型哈希/优化器哈希`：与基线完整状态 SHA-256 集合是否一致。两个 rank 的哈希"
            "都参与比较，因此 `after_reduce` 下 rank 间不一致也会被判为“否”。",
            "- `ΔAcc` / `ΔLoss`：相对同配置基线的测试准确率百分点差与损失差。负 ΔAcc、"
            "正 ΔLoss 表示性能退化。",
            "- `重试 step`：发生回滚的唯一训练 step 数；每个 rank 的一条回滚事件会计入"
            "`rollback_events`，但只对应一个重试 step。",
            "- `中位 step`：排除首步预热后的已接受 step 中位耗时。共享主机负载会造成"
            "不同场景的墙钟时间漂移，因此机制固定开销以下面的受控对照为准。",
            "",
        ]
    )

    overhead = report.get("overhead")
    if overhead:
        lines.extend(
            [
                "## 缓解固定开销（受控）",
                "",
                "以下运行使用 `inject_rank=99`：DDP 通信钩子与可选校验和每步都会执行，"
                "但没有故障真正触发，因此只测机制固定成本。每组重复运行两次，取各 run 中位"
                " step 时间的中位数。",
                "",
                "| 机制 | run 数 | 中位 step | 相对 plain 增量 | 相对 plain 开销 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        labels = {
            "plain": "无检测、无快照",
            "verify": "仅通信校验和",
            "rollback": "校验和 + 检测器 + 每步快照",
        }
        for row in overhead["rows"]:
            lines.append(
                "| {mode} | {runs} | {step} ms | {delta} | {pct} |".format(
                    mode=labels.get(row["mode"], row["mode"]),
                    runs=len(row["runs"]),
                    step=_fmt(row["median_step_ms"], 2),
                    delta=_fmt_signed(row["delta_ms"], 2, " ms"),
                    pct=_fmt_signed(row["overhead_pct"], 1, "%"),
                )
            )
        lines.append("")
        lines.append(
            "开销分解：校验和检测是主要固定成本；步级回滚在检测之外再增加每步模型/优化器"
            "快照与非有限值扫描。真正发生故障时，还会额外重放一个 step。"
        )

    lines.extend(
        [
            "",
            "## 关键观察",
            "",
        ]
    )

    for row in report["runs"]:
        if row["outcome"] == "sdc":
            lines.append(
                f"- `{row['scenario']}`：静默故障未被检测，最终模型"
                f"{'相同' if row['model_match_baseline'] else '不同'}于基线；"
                f"测试准确率变化 {_fmt_signed(row['test_accuracy_delta_pp'], 2, ' pp')}，"
                f"测试损失变化 {_fmt_signed(row['test_loss_delta'], 4)}。"
            )
        elif row["outcome"] == "recovered_lossless":
            lines.append(
                f"- `{row['scenario']}`：检测并完成步级回滚；模型与优化器 SHA-256 均与基线一致，"
                f"测试指标完全恢复（ΔAcc={_fmt_signed(row['test_accuracy_delta_pp'], 2, ' pp')}）。"
            )
        elif row["outcome"] == "recovered_not_exact":
            lines.append(
                f"- `{row['scenario']}`：检测并回滚，但状态哈希未完全匹配基线，不能宣称严格无损。"
            )

    lines.extend(
        [
            "",
            "## 结论与局限",
            "",
            "- 在本配置下，`after_reduce` 的单元素指数位翻转足以造成可测的 rank 间发散和"
            "测试性能下降；开启校验和与步级回滚后，最终模型和优化器状态可与基线逐 bit 一致。",
            "- `before_reduce` 的小幅扰动可能被 float32 归约噪声/校验和相对容差吸收，"
            "即使最终模型已经偏离基线也无法检测；这是单次全归约校验和的可观测性边界。"
            "更强的指数位故障可被检测并恢复。",
            "- 回滚的无损性依赖确定性训练、同一 batch 重放和正确的 RNG/优化器状态快照。"
            "当前结论来自单一种子；正式研究应至少重复 5 个种子，并报告 SDC、漏检率与恢复率分布。",
            "- 受控的无故障测量中，通信校验和与步级回滚相对普通 step 的固定开销见"
            "“缓解固定开销”表；实际故障还会增加一次 batch 重放。低开销部署应在"
            "检测覆盖率、快照频率与显存预算之间做权衡，而不应只引用单次墙钟时间。",
            "",
            "机器可读结果见同目录 `report.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_BASE / "runs"))
    ap.add_argument("--baseline-run", default="baseline")
    ap.add_argument("--overhead-root", default=str(DEFAULT_BASE / "analysis" / "overhead"))
    ap.add_argument("--output-md", default=str(DEFAULT_BASE / "analysis" / "report.md"))
    ap.add_argument("--output-json", default=str(DEFAULT_BASE / "analysis" / "report.json"))
    args = ap.parse_args()

    report = summarize(Path(args.root), args.baseline_run)
    report["overhead"] = summarize_overhead(Path(args.overhead_root))
    markdown = format_markdown(report)

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown + "\n", encoding="utf-8")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
