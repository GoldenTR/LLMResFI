import torch
import torch.nn as nn

from llmresfi import FIConfig, RecoveryManager, StageTracker
from llmresfi.config import (
    DetectorCfg,
    ExecCfg,
    FaultCfg,
    MitigationCfg,
    ModelCfg,
    RecoveryCfg,
    StageCfg,
)
from llmresfi.mitigation import (
    AnomalyDetector,
    CheckpointManager,
    Signal,
    capture_snapshot,
    sanitize_gradients,
)


def test_detector_flags_non_finite_gradient():
    model = nn.Linear(3, 2)
    model(torch.randn(4, 3)).sum().backward()
    model.weight.grad[0, 0] = float("nan")
    detector = AnomalyDetector(DetectorCfg(grad_norm_ratio=0, loss_ratio=0, warmup_steps=0))
    signals = detector.check_gradients(model, step=0)
    assert [s.kind for s in signals] == ["grad_non_finite"]


def test_detector_flags_loss_spike():
    detector = AnomalyDetector(DetectorCfg(grad_norm_ratio=0, loss_ratio=3.0, warmup_steps=0))
    detector.observe(1.0, step=0)
    signals = detector.check_loss(10.0, step=1)
    assert [s.kind for s in signals] == ["loss_spike"]


def test_sanitize_zeroes_non_finite_gradients():
    model = nn.Linear(2, 1)
    model(torch.randn(1, 2)).sum().backward()
    model.weight.grad[0, 0] = float("inf")
    summary = sanitize_gradients(model)
    assert summary["replaced"] == 1
    assert torch.isfinite(model.weight.grad).all()


def test_snapshot_restores_parameters_and_optimizer_state():
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    snapshot = capture_snapshot(model, optimizer, step=0)
    expected = [p.detach().clone() for p in model.parameters()]
    expected_state = optimizer.state_dict()

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    snapshot.restore(model, optimizer)

    for want, got in zip(expected, model.parameters()):
        assert torch.equal(want, got)
    restored = optimizer.state_dict()
    assert restored["param_groups"] == expected_state["param_groups"]
    assert set(restored["state"].keys()) == set(expected_state["state"].keys())


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path, keep=2, rank=0)
    path = manager.save(5, model, optimizer, extra={"loss": 1.0})
    assert manager.latest() == path

    expected = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)

    payload = manager.load()
    manager.restore(payload, model, optimizer)
    assert payload["step"] == 5
    for want, got in zip(expected, model.parameters()):
        assert torch.equal(want, got)


def _recovery_cfg(tmp_path, mode):
    return FIConfig(
        run_mode="training",
        model=ModelCfg(repo="tiny", dtype="float32", device="cpu"),
        fault=FaultCfg(target="gradient", stage=StageCfg(type="backward")),
        execution=ExecCfg(log_dir=tmp_path),
        mitigation=MitigationCfg(
            enabled=True,
            recovery=RecoveryCfg(mode=mode, max_retries=1),
        ),
    )


def test_recovery_sanitize_path(tmp_path):
    model = nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[float("nan"), 1.0]])
    cfg = _recovery_cfg(tmp_path, "sanitize")
    recovery = RecoveryManager(cfg, model, None, StageTracker("training"))

    action = recovery.decide([Signal("grad_non_finite", {}, 0, 0)])
    assert action == "continue"
    assert torch.isfinite(model.weight.grad).all()
    assert recovery.events[-1]["action"] == "sanitized"


def test_recovery_checkpoint_rollback_path(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    cfg = _recovery_cfg(tmp_path, "checkpoint_rollback")
    recovery = RecoveryManager(cfg, model, optimizer, StageTracker("training"))
    recovery.save_checkpoint(3)
    expected = [p.detach().clone() for p in model.parameters()]

    with torch.no_grad():
        for p in model.parameters():
            p.add_(5.0)

    action = recovery.decide([Signal("grad_non_finite", {}, 0, 0)])
    assert action == "restored_checkpoint"
    assert recovery.last_restored_step == 3
    for want, got in zip(expected, model.parameters()):
        assert torch.equal(want, got)

    action = recovery.decide([Signal("grad_non_finite", {}, 0, 0)])
    assert action == "abort"
    assert recovery.events[-1]["action"] == "abort"


def test_checkpoint_mode_creates_initial_checkpoint(tmp_path):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    cfg = _recovery_cfg(tmp_path, "checkpoint_rollback")
    recovery = RecoveryManager(cfg, model, optimizer, StageTracker("training"))

    recovery.begin_step()
    payload = recovery.checkpoints.load()
    assert payload is not None
    assert payload["step"] == 0
