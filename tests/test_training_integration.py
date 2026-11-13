import torch

from llmresfi.injector import inject_bit_flip


def test_weight_injection_changes_forward_output(device):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3, bias=False).to(device)
    inputs = torch.randn(2, 4, device=device)
    before = model(inputs).detach().clone()
    original_weight = model.weight.detach().clone()
    inject_bit_flip(model.weight, 0, 10, inplace=True)
    after = model(inputs).detach()
    assert not torch.equal(before, after)
    assert torch.equal(model.weight.detach()[1:], original_weight[1:])


def test_injection_then_backward_and_optimizer_step_does_not_break_graph(device):
    torch.manual_seed(1)
    model = torch.nn.Linear(3, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(8, 3, device=device)
    targets = torch.randn(8, 1, device=device)

    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()

    inject_bit_flip(model.weight, 0, 0, inplace=True, step=0)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(model.weight.detach()).all()


def test_non_inplace_result_can_drive_alternative_computation(device):
    torch.manual_seed(2)
    weight = torch.randn(3, 2, device=device)
    inputs = torch.randn(5, 2, device=device)
    result = inject_bit_flip(weight, 0, 0, inplace=False)
    baseline = torch.nn.functional.linear(inputs, weight)
    corrupted = torch.nn.functional.linear(inputs, result.tensor)
    assert torch.equal(torch.nn.functional.linear(inputs, weight), baseline)
    assert not torch.equal(baseline, corrupted)

