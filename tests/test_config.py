import json
from pathlib import Path

import pytest
import torch
import yaml

from llmresfi.config import LLMResFIConfig, load_config, write_run_metadata
from llmresfi.exceptions import ConfigurationError


def write_config(tmp_path, payload):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def base_payload():
    return {
        "experiment": {"experiment_id": "exp_001", "seed": 42},
        "fault": {
            "enabled": True,
            "target_type": "model_parameter",
            "target_name": "transformer.h.0.attn.c_attn.weight",
            "dtype": "float16",
            "injection_mode": "single_bit",
            "bit_type": "exponent",
            "injection_step": 100,
            "injection_count": 1,
            "selection_mode": "random_element",
        },
        "logging": {
            "enabled": True,
            "format": "jsonl",
            "output_dir": "logs/exp_001",
        },
    }


def test_load_example_config(tmp_path):
    path = write_config(tmp_path, base_payload())
    config = load_config(path)
    assert config.experiment.experiment_id == "exp_001"
    assert config.experiment.seed == 42
    assert config.fault.dtype is torch.float16
    assert config.fault.injection_steps == (100,)
    assert config.fault.injection_count == 1
    assert config.logging.format == "jsonl"
    assert config.source_path == path
    scheduler = config.build_scheduler()
    assert scheduler.injection_steps == (100,)


def test_config_scheduler_is_deterministic(tmp_path):
    payload = base_payload()
    payload["fault"].pop("injection_step")
    payload["fault"]["step_range"] = [10, 50]
    payload["fault"]["random_injection_steps"] = 5
    payload["fault"]["injection_count"] = 3
    path = write_config(tmp_path, payload)
    first = load_config(path).build_scheduler().injection_steps
    second = load_config(path).build_scheduler().injection_steps
    assert first == second
    assert len(first) == 3


def test_missing_experiment_id_is_generated(tmp_path):
    payload = base_payload()
    payload["experiment"].pop("experiment_id")
    config = load_config(write_config(tmp_path, payload))
    assert config.experiment.experiment_id.startswith("exp_")
    assert len(config.experiment.experiment_id) > 10


def test_unknown_config_key_is_rejected(tmp_path):
    payload = base_payload()
    payload["fault"]["typo_key"] = 1
    with pytest.raises(ConfigurationError, match="Unknown key"):
        load_config(write_config(tmp_path, payload))


def test_invalid_fixed_element_configuration(tmp_path):
    payload = base_payload()
    payload["fault"]["selection_mode"] = "fixed_element"
    with pytest.raises(ConfigurationError, match="requires flat_index"):
        load_config(write_config(tmp_path, payload))
    payload["fault"]["flat_index"] = 3
    payload["fault"]["bit_position"] = 2
    config = load_config(write_config(tmp_path, payload))
    assert config.fault.flat_index == 3


def test_mutually_exclusive_scheduling_modes(tmp_path):
    payload = base_payload()
    payload["fault"]["step_range"] = [10, 20]
    with pytest.raises(ConfigurationError, match="cannot be combined"):
        load_config(write_config(tmp_path, payload))

    payload = base_payload()
    payload["fault"].pop("injection_step")
    payload["fault"]["random_injection_steps"] = 3
    with pytest.raises(ConfigurationError, match="requires fault.step_range"):
        load_config(write_config(tmp_path, payload))


def test_fixed_bit_position_is_validated_against_dtype(tmp_path):
    payload = base_payload()
    payload["fault"]["selection_mode"] = "fixed_element"
    payload["fault"]["flat_index"] = 0
    payload["fault"]["bit_position"] = 16
    with pytest.raises(ConfigurationError, match="bit_position"):
        load_config(write_config(tmp_path, payload))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p["fault"].__setitem__("dtype", "float64"),
        lambda p: p["fault"].__setitem__("injection_mode", "row"),
        lambda p: p["fault"].__setitem__("bit_type", "leading"),
        lambda p: p["logging"].__setitem__("format", "xml"),
        lambda p: p["fault"].__setitem__("step_range", [5, 1]),
    ],
)
def test_invalid_config_values(tmp_path, mutator):
    payload = base_payload()
    mutator(payload)
    with pytest.raises(ConfigurationError):
        load_config(write_config(tmp_path, payload))


def test_write_run_metadata_copies_config(tmp_path):
    path = write_config(tmp_path, base_payload())
    config = load_config(path)
    output = tmp_path / "run"
    metadata_path = write_run_metadata(config, output, extra={"stage": "unit-test"})
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "exp_001"
    assert payload["seed"] == 42
    assert payload["tool_version"]
    assert payload["torch_version"] == torch.__version__
    assert "scheduler_plan" in payload
    assert payload["extra"]["stage"] == "unit-test"
    assert (output / "source_config.yaml").read_text(encoding="utf-8") == path.read_text(
        encoding="utf-8"
    )
