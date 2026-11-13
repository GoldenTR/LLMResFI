import random

import pytest
import torch

from llmresfi.exceptions import TargetSelectionError, UnsupportedDTypeError
from llmresfi.target_selector import list_target_tensors, select_target_tensor


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Parameter(torch.ones(2, 3))
        self.proj = torch.nn.Linear(3, 4)
        self.other = torch.nn.Parameter(torch.ones(5, dtype=torch.bfloat16))


def test_select_exact_parameter_name():
    model = TinyModel()
    target = select_target_tensor(model, target_name="proj.weight")
    assert target.name == "proj.weight"
    assert target.tensor is model.proj.weight
    assert target.shape == (4, 3)
    assert target.dtype == torch.float32
    assert target.device == model.proj.weight.device
    assert target.numel == 12
    assert target.storage_ptr != 0
    assert "storage_offset" in target.describe()


def test_missing_parameter_has_clear_error():
    with pytest.raises(TargetSelectionError, match="not found"):
        select_target_tensor(TinyModel(), target_name="does.not.exist")


def test_dtype_filter_and_name_pattern():
    model = TinyModel()
    bf16 = list_target_tensors(model, dtype="bfloat16")
    assert [item.name for item in bf16] == ["other"]
    weight_targets = list_target_tensors(model, name_pattern=r"\.weight$")
    assert {item.name for item in weight_targets} == {"proj.weight"}
    with pytest.raises(TargetSelectionError):
        select_target_tensor(model, dtype="float16", name_pattern="proj")


def test_random_selection_is_reproducible_and_local():
    model = TinyModel()
    random.seed(77)
    expected = random.random()
    random.seed(77)
    first = [select_target_tensor(model, dtype="float32", seed=3).name for _ in range(10)]
    actual = random.random()
    second = [select_target_tensor(model, dtype="float32", seed=3).name for _ in range(10)]
    assert first == second
    assert actual == expected
    assert set(first) <= {"embed", "proj.weight", "proj.bias"}


def test_direct_tensor_selection():
    tensor = torch.ones(2, 2, dtype=torch.float16)
    target = select_target_tensor(tensor)
    assert target.name == "<tensor>"
    assert target.tensor is tensor
    labelled = select_target_tensor(tensor, target_name="activation.foo")
    assert labelled.name == "activation.foo"
    with pytest.raises(TargetSelectionError):
        select_target_tensor(tensor, dtype="float32")


def test_invalid_filters():
    model = TinyModel()
    with pytest.raises(TargetSelectionError):
        select_target_tensor(model, name_pattern="[")
    with pytest.raises(TargetSelectionError):
        select_target_tensor(model, seed=True)  # type: ignore[arg-type]
    with pytest.raises(UnsupportedDTypeError):
        select_target_tensor(model, dtype="float64")
    with pytest.raises(TargetSelectionError):
        list_target_tensors(object())  # type: ignore[arg-type]

