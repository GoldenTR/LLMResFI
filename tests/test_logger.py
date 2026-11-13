import csv
import json
from pathlib import Path

import pytest
import torch

from llmresfi.exceptions import LoggingError
from llmresfi.injector import inject_bit_flip
from llmresfi.logger import REQUIRED_EVENT_FIELDS, EventLogger, FaultEvent


def make_result(dtype=torch.float16, value=1.0, flat_index=0, bit_position=10):
    tensor = torch.tensor([value], dtype=dtype)
    return inject_bit_flip(
        tensor,
        flat_index,
        bit_position,
        tensor_name="layer.weight",
        step=7,
        random_seed=123,
        parameter_version=7,
        injection_phase="before_forward",
    )


def test_jsonl_logging_roundtrip_and_required_fields(tmp_path):
    logger = EventLogger("exp_test", tmp_path, format="jsonl", random_seed=42)
    result = make_result()
    logged = logger.log(result)
    assert set(REQUIRED_EVENT_FIELDS) <= set(logged)
    assert logged["experiment_id"] == "exp_test"
    assert logged["tensor_name"] == "layer.weight"
    assert logged["step"] == 7
    assert logged["bit_position"] == 10
    assert logged["bit_type"] == "exponent"
    assert logged["injection_success"] is True
    assert logged["parameter_version"] == 7
    assert logged["injection_phase"] == "before_forward"
    assert logged["tensor_version_before"] is not None
    assert logged["tensor_version_after"] is not None
    assert logged["injection_timestamp_precise"]
    assert logged["original_bit_pattern"] ^ logged["corrupted_bit_pattern"] == (1 << 10)

    lines = (tmp_path / "fault_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed == logged
    assert (tmp_path / "run_metadata.json").is_file()
    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["experiment_id"] == "exp_test"
    assert metadata["torch_version"] == torch.__version__


def test_logger_generates_unique_experiment_id_when_omitted(tmp_path):
    first = EventLogger(None, tmp_path / "a", format="jsonl")
    second = EventLogger(None, tmp_path / "b", format="jsonl")
    assert first.experiment_id.startswith("exp_")
    assert second.experiment_id.startswith("exp_")
    assert first.experiment_id != second.experiment_id


def test_multiple_events_append_without_overwrite(tmp_path):
    logger = EventLogger("exp_multi", tmp_path, format="jsonl")
    for index in range(3):
        logger.log(
            make_result(value=float(index + 1), bit_position=index),
            tensor_name=f"tensor.{index}",
            step=index,
        )
    lines = (tmp_path / "fault_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    ids = [json.loads(line)["fault_id"] for line in lines]
    assert len(set(ids)) == 3
    assert logger.to_dicts()[2]["tensor_name"] == "tensor.2"


def test_csv_fields_and_parsing(tmp_path):
    logger = EventLogger("exp_csv", tmp_path, format="csv")
    logger.log(make_result(dtype=torch.bfloat16, bit_position=7))
    csv_path = tmp_path / "fault_events.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert set(REQUIRED_EVENT_FIELDS) <= set(rows[0])
    assert rows[0]["tensor_dtype"] == "torch.bfloat16"
    assert rows[0]["bit_type"] == "exponent"
    assert rows[0]["injection_success"] == "True"


def test_both_formats_are_written(tmp_path):
    logger = EventLogger("exp_both", tmp_path, format="both")
    logger.log(make_result())
    assert (tmp_path / "fault_events.jsonl").is_file()
    assert (tmp_path / "fault_events.csv").is_file()


def test_failed_event_is_recorded(tmp_path):
    logger = EventLogger("exp_fail", tmp_path, format="both")
    event = logger.log_failure(
        step=3,
        tensor_name="broken.weight",
        failure_reason="out-of-bounds flat index",
        tensor_shape=(2, 2),
        tensor_dtype="torch.float16",
        tensor_device="cpu",
        flat_index=999,
        bit_position=4,
        bit_type="mantissa",
    )
    assert event["injection_success"] is False
    assert event["failure_reason"] == "out-of-bounds flat index"
    assert event["original_value"] is None
    parsed = json.loads((tmp_path / "fault_events.jsonl").read_text(encoding="utf-8"))
    assert parsed["injection_success"] is False


def test_special_float_values_serialize_to_strict_json(tmp_path):
    logger = EventLogger("exp_special", tmp_path, format="jsonl")
    result = make_result(dtype=torch.float16, value=float("inf"), bit_position=10)
    event = logger.log(result)
    assert event["original_value"] == "Infinity"
    payload = (tmp_path / "fault_events.jsonl").read_text(encoding="utf-8")
    json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def test_duplicate_fault_id_is_rejected(tmp_path):
    logger = EventLogger("exp_dup", tmp_path, format="jsonl")
    logger.log(make_result(), fault_id="fixed-id")
    with pytest.raises(LoggingError):
        logger.log(make_result(), fault_id="fixed-id")


def test_tensor_backup_roundtrip(tmp_path):
    logger = EventLogger("exp_backup", tmp_path, format="jsonl", save_tensor_backup=True)
    original = torch.tensor([1.0, 2.0], dtype=torch.float32)
    result = inject_bit_flip(
        original, 0, 0, tensor_name="weight", step=9, random_seed=3
    )
    event = logger.log(result, original_tensor=original, corrupted_tensor=result.tensor)
    backup_path = Path(event["tensor_backup_path"])
    assert backup_path.is_file()
    payload = torch.load(backup_path, weights_only=True)
    assert torch.equal(payload["original_tensor"], original)
    assert not torch.equal(payload["corrupted_tensor"], original)


def test_fault_event_mapping_and_config_copy(tmp_path):
    logger = EventLogger("exp_mapping", tmp_path, format="jsonl")
    event = FaultEvent(
        experiment_id="exp_mapping",
        fault_id="custom-1",
        step=1,
        tensor_name="x",
        tensor_shape=(1,),
        tensor_dtype="torch.float32",
        tensor_device="cpu",
        flat_index=0,
        bit_position=31,
        bit_type="sign",
        original_value=-0.0,
        corrupted_value=0.0,
    )
    assert logger.log(event)["fault_id"] == "custom-1"
    config = tmp_path / "input.yaml"
    config.write_text("experiment: {}\n", encoding="utf-8")
    copied = logger.save_config_copy(config)
    assert copied.read_text(encoding="utf-8") == config.read_text(encoding="utf-8")


def test_logger_validation_errors(tmp_path):
    with pytest.raises(LoggingError):
        EventLogger("", tmp_path)
    with pytest.raises(LoggingError):
        EventLogger("exp", tmp_path, format="xml")
    logger = EventLogger("exp", tmp_path)
    with pytest.raises(LoggingError):
        logger.log({"experiment_id": "different"})
    with pytest.raises(LoggingError):
        logger.log_failure(step=0, tensor_name="x", failure_reason="")

