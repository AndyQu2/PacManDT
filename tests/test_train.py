"""Tests for the ``python -m dt.train`` training program."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from dt.data import (
    Episode,
    compute_returns_to_go,
    dims_from_dataset,
    episodes_to_dataset_dict,
    save_dataset,
)
from dt.dt import load_checkpoint
import dt.train as train_module
from dt.train import (
    _episodes_from_npz,
    _resolve_context_length,
    _resolve_device,
    build_parser,
    configure_cuda,
    main,
    mean_return_of,
    return_scale_of,
    split_train_eval,
    with_undiscounted_returns,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_episode(length: int = 12, obs_shape=(2, 2), layout: str = "small", seed: int = 0) -> Episode:
    rng = np.random.default_rng(seed)
    rewards = np.ones(length, dtype=np.float32)
    return Episode(
        observations=rng.integers(0, 5, size=(length, *obs_shape)).astype(np.int8),
        actions=rng.integers(0, 5, size=length).astype(np.int64),
        rewards=rewards,
        terminated=np.zeros(length, dtype=bool),
        truncated=np.zeros(length, dtype=bool),
        returns_to_go=compute_returns_to_go(rewards, gamma=0.99),
        timesteps=np.arange(length, dtype=np.int64),
        layout=layout,
        score=length,
        total_reward=float(length),
        success=True,
        info={"win": True},
    )


def make_dataset_dict(count: int = 6, length: int = 12):
    episodes = [make_episode(length=length, seed=index) for index in range(count)]
    return episodes_to_dataset_dict(episodes, eval_fraction=0.0, max_length=length)


# ----------------------------------------------------------------------
# Parser / small helpers
# ----------------------------------------------------------------------
def test_parser_defaults_are_sane():
    args = build_parser().parse_args([])
    assert args.data is None
    assert args.context_length is None
    assert args.epochs > 0
    assert args.batch_size > 0
    assert args.render == "none"
    assert args.eval_only is False


def test_parser_accepts_render_human():
    args = build_parser().parse_args(["--render", "human", "--eval-only", "--checkpoint", "x.pt"])
    assert args.render == "human"


def test_resolve_context_length_uses_recommendation():
    args = build_parser().parse_args([])
    assert _resolve_context_length(args, 500) == 30


def test_resolve_context_length_respects_the_flag():
    args = build_parser().parse_args(["--context-length", "8"])
    assert _resolve_context_length(args, 500) == 8


def test_resolve_context_length_rejects_longer_than_episode():
    args = build_parser().parse_args(["--context-length", "64"])
    with pytest.raises(ValueError, match="cannot exceed"):
        _resolve_context_length(args, 32)


def test_return_scale_and_mean_return():
    dataset = make_dataset_dict()
    scale = return_scale_of(dataset["train"])
    assert scale >= 1.0
    mean = mean_return_of(dataset["train"])
    assert mean == pytest.approx(12.0)


def test_split_train_eval_carves_out_a_holdout():
    dataset = make_dataset_dict(count=10)
    train, eval_ds = split_train_eval(dataset, seed=0)
    assert eval_ds is not None
    assert len(train) + len(eval_ds) == 10
    assert len(train) > 0


def test_split_train_eval_keeps_tiny_datasets_whole():
    dataset = make_dataset_dict(count=2)
    train, eval_ds = split_train_eval(dataset, seed=0)
    assert eval_ds is None
    assert len(train) == 2


def test_with_undiscounted_returns_sums_future_rewards():
    dataset = make_dataset_dict(count=3, length=12)
    converted = with_undiscounted_returns(dataset)
    returns = np.asarray(converted["train"]["returns_to_go"], dtype=np.float32)
    rewards = np.asarray(converted["train"]["rewards"], dtype=np.float32)

    # rewards are all ones, so the undiscounted return-to-go is 12, 11, ... 1
    assert returns[0, 0] == pytest.approx(12.0)
    assert returns[0, 1] == pytest.approx(11.0)
    assert returns[0, -1] == pytest.approx(1.0)
    # the plain-discounted version would be smaller at every step
    discounted = np.asarray(dataset["train"]["returns_to_go"], dtype=np.float32)
    assert np.all(returns >= discounted)
    assert returns.shape == rewards.shape


def test_with_undiscounted_returns_respects_episode_length():
    dataset = make_dataset_dict(count=2, length=12)
    trimmed = {}
    for name, split in dataset.items():
        split = split.remove_columns(["returns_to_go", "length"])
        trimmed[name] = split.add_column("length", [4] * len(split))
    converted = with_undiscounted_returns(trimmed)
    returns = np.asarray(converted["train"]["returns_to_go"], dtype=np.float32)
    assert returns[0, 3] == pytest.approx(1.0)  # last real step
    assert returns[0, 4] == pytest.approx(0.0)  # padded steps stay zero


def test_episodes_from_npz_round_trip(tmp_path):
    episodes = [make_episode(seed=index) for index in range(3)]
    path = save_dataset(tmp_path / "episodes.npz", episodes, max_length=12)

    restored = _episodes_from_npz(path)
    assert len(restored) == 3
    for original, copy in zip(episodes, restored):
        assert copy.length == original.length
        np.testing.assert_array_equal(copy.actions, original.actions)
        np.testing.assert_allclose(copy.returns_to_go, original.returns_to_go)
        assert copy.total_reward == pytest.approx(original.total_reward)


# ----------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------
def test_main_trains_from_npz_and_writes_checkpoint(tmp_path):
    episodes = [make_episode(length=16, seed=index) for index in range(4)]
    data = save_dataset(tmp_path / "data.npz", episodes, max_length=16)
    out_dir = tmp_path / "run"

    exit_code = main(
        [
            "--data", str(data),
            "--layout", "small",
            "--epochs", "1",
            "--batch-size", "2",
            "--context-length", "4",
            "--hidden-size", "16",
            "--n-inner", "32",
            "--eval-episodes", "0",
            "--out-dir", str(out_dir),
            "--quiet",
        ]
    )

    assert exit_code == 0
    checkpoint = out_dir / "checkpoint.pt"
    metrics_path = out_dir / "metrics.json"
    assert checkpoint.exists()
    assert metrics_path.exists()

    model, payload = load_checkpoint(checkpoint)
    dims = dims_from_dataset(make_dataset_dict()["train"], layout="small")
    assert model.state_dim == 4  # (2, 2) observations flattened
    assert payload["layout"] == "small"
    assert payload["context_length"] == 4
    assert payload["dims"]["state_dim"] == model.state_dim
    assert payload["dims"]["act_dim"] == 5 == dims.act_dim

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["context_length"] == 4
    assert len(metrics["history"]["train_loss"]) == 1
    assert np.isfinite(metrics["history"]["train_loss"][0])


def test_main_eval_only_requires_checkpoint():
    with pytest.raises(SystemExit, match="--checkpoint"):
        main(["--eval-only"])


def test_main_eval_only_runs_a_rollout(tmp_path):
    episodes = [make_episode(length=16, seed=index) for index in range(4)]
    data = save_dataset(tmp_path / "data.npz", episodes, max_length=16)
    out_dir = tmp_path / "run"
    main(
        [
            "--data", str(data),
            "--layout", "small",
            "--epochs", "1",
            "--batch-size", "2",
            "--context-length", "4",
            "--hidden-size", "16",
            "--n-inner", "32",
            "--eval-episodes", "0",
            "--out-dir", str(out_dir),
            "--quiet",
        ]
    )
    # The synthetic observations are (2, 2), so rollouts need a matching maze;
    # evaluating on the real ``small`` layout (42 inputs) must fail loudly
    # rather than silently produce garbage.
    with pytest.raises(ValueError, match="flattened state"):
        main(
            [
                "--eval-only",
                "--checkpoint", str(out_dir / "checkpoint.pt"),
                "--eval-episodes", "1",
                "--max-steps", "5",
            ]
        )


# ----------------------------------------------------------------------
# CUDA configuration
# ----------------------------------------------------------------------
class FakeStream:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority


# ``torch.backends.*`` flags are dynamic attributes, so ``monkeypatch`` cannot
# restore them; snapshot and put them back by hand.
TORCH_FLAGS = (
    (torch.backends.cuda.matmul, "allow_tf32"),
    (torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"),
    (torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"),
    (torch.backends.cudnn, "allow_tf32"),
    (torch.backends.cudnn, "benchmark"),
    (torch.backends.cudnn, "deterministic"),
)


@pytest.fixture
def torch_flags():
    saved = [(obj, name, getattr(obj, name)) for obj, name in TORCH_FLAGS]
    precision = torch.get_float32_matmul_precision()
    yield
    for obj, name, value in saved:
        setattr(obj, name, value)
    torch.set_float32_matmul_precision(precision)


def test_parser_device_defaults_to_auto_detect():
    args = build_parser().parse_args([])
    assert args.device is None
    assert _resolve_device("cpu").type == "cpu"
    assert _resolve_device("cuda:0").type == "cuda"


def test_configure_cuda_is_a_noop_on_cpu(monkeypatch, torch_flags):
    for obj, name in TORCH_FLAGS:
        setattr(obj, name, False)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: pytest.fail("stream created on cpu"))

    assert configure_cuda(torch.device("cpu")) is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cudnn.benchmark is False


def test_configure_cuda_is_a_noop_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: pytest.fail("stream created without cuda"))

    assert configure_cuda(torch.device("cuda")) is False


def test_configure_cuda_applies_the_settings_on_gpu(monkeypatch, torch_flags):
    applied = []
    stream = FakeStream(priority=-1)
    for obj, name in TORCH_FLAGS:
        setattr(obj, name, False)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", applied.append)
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", applied.append)
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda fraction, *a: applied.append(fraction)
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda priority=0: stream)
    monkeypatch.setattr(torch.cuda, "set_stream", applied.append)

    assert configure_cuda(torch.device("cuda")) is True

    assert "max_split_size_mb:128" in applied
    assert 0.9 in applied
    assert applied[-1] is stream, "the high-priority stream must be the current stream"
    assert torch.get_float32_matmul_precision() == "high"
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is True
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
    assert train_module._CUDA_STREAM is stream


def test_configure_cuda_survives_a_missing_allocator_hook(monkeypatch, torch_flags):
    def boom(*args, **kwargs):
        raise AttributeError("not available")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings", boom)
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", boom)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", boom)
    monkeypatch.setattr(torch.cuda, "Stream", lambda priority=0: FakeStream(priority))
    monkeypatch.setattr(torch.cuda, "set_stream", lambda s: None)

    assert configure_cuda(torch.device("cuda")) is True
