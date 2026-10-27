import pytest

from llmresfi.config import FIConfig


def _write(tmp_path, text):
    p = tmp_path / "c.yml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_inference(tmp_path):
    p = _write(
        tmp_path,
        "run_mode: inference\n"
        "model: {repo: x, dtype: float32, device: cpu}\n"
        "fault: {target: weight, stage: {type: prefill}}\n",
    )
    cfg = FIConfig.load(p)
    assert cfg.run_mode == "inference"
    assert cfg.fault.target == "weight"
    assert cfg.execution.random_seed == 42


def test_stage_mismatch_rejected(tmp_path):
    p = _write(
        tmp_path,
        "run_mode: training\n"
        "model: {repo: x}\n"
        "fault: {target: gradient, stage: {type: prefill}}\n",
    )
    with pytest.raises(SystemExit):
        FIConfig.load(p)


def test_unknown_fault_model_rejected(tmp_path):
    p = _write(
        tmp_path,
        "run_mode: inference\n"
        "model: {repo: x}\n"
        "fault: {target: weight, stage: {type: prefill}, fault_model: {type: bogus}}\n",
    )
    with pytest.raises(SystemExit):
        FIConfig.load(p)
