import pytest
import torch

from llmresfi.bit_utils import get_float_format, tensor_bit_pattern
from llmresfi.injector import inject_bit_flip


pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_bit_flip_pattern_verification(dtype, cuda_device):
    tensor = torch.tensor([1.0, -2.5, 0.0], dtype=dtype, device=cuda_device)
    fmt = get_float_format(dtype)
    original = tensor_bit_pattern(tensor[1])
    for bit in (0, fmt.exponent_lsb, fmt.sign_bit):
        result = inject_bit_flip(tensor, 1, bit, inplace=False, synchronize=True)
        assert result.device.type == "cuda"
        assert result.corrupted_bit_pattern == original ^ (1 << bit)
        assert result.only_target_bit_changed
        assert tensor_bit_pattern(tensor[1]) == original
        assert torch.equal(result.tensor.cpu(), tensor.cpu()) is False


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_noncontiguous_inplace(dtype, cuda_device):
    base = torch.arange(12, dtype=dtype, device=cuda_device).reshape(3, 4)
    view = base.t()
    before = base.clone()
    inject_bit_flip(view, 5, 0, inplace=True, synchronize=True)
    torch.cuda.synchronize()
    assert int((base != before).sum().item()) == 1

