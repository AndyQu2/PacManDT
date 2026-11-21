"""Tests for the Hugging Face ``datasets`` integration (``dt.data.hf_dataset``)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dt.data import Episode, compute_returns_to_go

pytest.importorskip("datasets")

from datasets import DatasetDict  # noqa: E402

from dt.data import (  # noqa: E402
    DEFAULT_HF_DIR,
    HF_DATASET_FORMAT,
    DataCollector,
    DTSequenceDataset,
    dataset_card,
    dataset_statistics_from_dict,
    dataset_to_episodes,
    episode_features,
    episodes_to_dataset,
    episodes_to_dataset_dict,
    flatten_to_transitions,
    split_episodes,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_episode(
    length: int,
    reward: float = 1.0,
    layout: str = "small",
    rows: int = 4,
    cols: int = 3,
    seed: int = 0,
) -> Episode:
    rewards = np.full(length, reward, dtype=np.float32)
    observations = np.arange(length * rows * cols, dtype=np.int64).reshape(
        length, rows, cols
    )
    observations = (observations % 7).astype(np.int8) - 1
    truncated = np.zeros(length, dtype=bool)
    truncated[-1] = True  # every synthetic episode ends on the step limit
    return Episode(
        observations=observations,
        actions=np.arange(length, dtype=np.int64) % 5,
        rewards=rewards,
        terminated=np.zeros(length, dtype=bool),
        truncated=truncated,
        returns_to_go=compute_returns_to_go(rewards),
        timesteps=np.arange(length, dtype=np.int64),
        seed=seed,
        layout=layout,
        score=length * 10,
        total_reward=float(rewards.sum()),
        success=float(rewards.sum()) >= 5.0,
        info={"win": False, "pellets_remaining": 3},
    )


@pytest.fixture()
def episodes():
    return [make_episode(6, seed=1), make_episode(4, reward=2.0, seed=2)]


@pytest.fixture()
def dataset(episodes):
    return episodes_to_dataset(episodes)


# ----------------------------------------------------------------------
# Feature schema
# ----------------------------------------------------------------------
def test_episode_features_declares_padded_observation_shape():
    features = episode_features(obs_shape=(7, 6), max_length=12, obs_dtype="int8")
    observations = features["observations"]
    assert observations.shape == (12, 7, 6)
    assert observations.dtype == "int8"
    assert features["length"].dtype == "int32"
    assert features["layout"].dtype == "string"


def test_episode_features_rejects_one_dimensional_observations():
    with pytest.raises(ValueError):
        episode_features(obs_shape=(), max_length=5)


# ----------------------------------------------------------------------
# Episode <-> Dataset
# ----------------------------------------------------------------------
def test_episodes_to_dataset_has_one_row_per_episode(dataset, episodes):
    assert len(dataset) == len(episodes)
    assert dataset.features["observations"].shape == (6, 4, 3)
    assert dataset.features["observations"].dtype == "int8"


def test_episodes_to_dataset_pads_short_episodes(dataset):
    numpy_view = dataset.with_format("numpy")
    batch = numpy_view[:]
    assert batch["observations"].shape == (2, 6, 4, 3)
    assert batch["actions"].shape == (2, 6)
    np.testing.assert_array_equal(batch["length"], [6, 4])
    # the padded tail of the shorter episode is masked out
    np.testing.assert_array_equal(batch["mask"][1], [True] * 4 + [False] * 2)
    np.testing.assert_array_equal(batch["actions"][1][4:], [0, 0])


def test_episodes_to_dataset_truncates_to_max_length(episodes):
    dataset = episodes_to_dataset(episodes, max_length=3)
    numpy_view = dataset.with_format("numpy")
    batch = numpy_view[:]
    assert batch["observations"].shape == (2, 3, 4, 3)
    np.testing.assert_array_equal(batch["length"], [3, 3])
    np.testing.assert_array_equal(batch["mask"], np.ones((2, 3), dtype=bool))


def test_episodes_to_dataset_pads_up_to_max_length(episodes):
    dataset = episodes_to_dataset(episodes, max_length=9)
    assert dataset.features["observations"].shape == (9, 4, 3)
    batch = dataset.with_format("numpy")[:]
    assert batch["actions"].shape == (2, 9)
    np.testing.assert_array_equal(batch["mask"][1], [True] * 4 + [False] * 5)


def test_episodes_to_dataset_rejects_empty_input():
    with pytest.raises(ValueError):
        episodes_to_dataset([])


def test_episodes_to_dataset_supports_rgb_observations():
    length = 5
    observations = np.zeros((length, 8, 8, 3), dtype=np.uint8)
    observations[:, 0, 0, 0] = np.arange(length)
    rewards = np.ones(length, dtype=np.float32)
    episode = Episode(
        observations=observations,
        actions=np.zeros(length, dtype=np.int64),
        rewards=rewards,
        terminated=np.zeros(length, dtype=bool),
        truncated=np.ones(length, dtype=bool),
        returns_to_go=compute_returns_to_go(rewards),
        timesteps=np.arange(length, dtype=np.int64),
        seed=0,
        layout="small",
        score=0,
        total_reward=float(rewards.sum()),
        success=False,
        info={"win": False, "pellets_remaining": 0},
    )
    dataset = episodes_to_dataset([episode])
    assert dataset.features["observations"].shape == (length, 8, 8, 3)
    restored = dataset_to_episodes(dataset)[0]
    np.testing.assert_array_equal(restored.observations, observations)


def test_dataset_to_episodes_round_trips(dataset, episodes):
    rebuilt = dataset_to_episodes(dataset)
    assert len(rebuilt) == len(episodes)
    for original, restored in zip(episodes, rebuilt):
        assert restored.length == original.length
        np.testing.assert_array_equal(restored.actions, original.actions)
        np.testing.assert_array_equal(restored.observations, original.observations)
        np.testing.assert_allclose(restored.rewards, original.rewards)
        np.testing.assert_allclose(restored.returns_to_go, original.returns_to_go)
        assert restored.seed == original.seed
        assert restored.layout == original.layout
        assert restored.score == original.score
        assert restored.info["pellets_remaining"] == 3


def test_dataset_to_episodes_survives_a_save_load_cycle(dataset, tmp_path):
    path = tmp_path / "ds"
    dataset.save_to_disk(str(path))
    from datasets import load_from_disk

    restored = dataset_to_episodes(load_from_disk(str(path)))
    np.testing.assert_array_equal(restored[0].actions, [0, 1, 2, 3, 4, 0])


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------
def test_split_episodes_holds_out_the_tail(episodes):
    train, eval_ = split_episodes(episodes, 0.5)
    assert len(train) == 1 and len(eval_) == 1
    assert eval_[0].seed == episodes[-1].seed


def test_split_episodes_with_zero_fraction_keeps_everything(episodes):
    train, eval_ = split_episodes(episodes, 0.0)
    assert train == episodes and eval_ == []


def test_split_episodes_ignores_a_fraction_that_rounds_to_zero(episodes):
    train, eval_ = split_episodes(episodes, 0.1)
    assert train == episodes and eval_ == []


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_split_episodes_rejects_out_of_range_fractions(episodes, fraction):
    with pytest.raises(ValueError):
        split_episodes(episodes, fraction)


def test_episodes_to_dataset_dict_builds_train_and_eval(episodes):
    dataset_dict = episodes_to_dataset_dict(episodes, eval_fraction=0.5)
    assert isinstance(dataset_dict, DatasetDict)
    assert set(dataset_dict) == {"train", "eval"}
    assert len(dataset_dict["train"]) == 1
    assert len(dataset_dict["eval"]) == 1


def test_episodes_to_dataset_dict_omits_eval_when_fraction_is_zero(episodes):
    assert set(episodes_to_dataset_dict(episodes, eval_fraction=0.0)) == {"train"}


# ----------------------------------------------------------------------
# Transition view
# ----------------------------------------------------------------------
def test_flatten_to_transitions_has_one_row_per_step(dataset):
    flat = flatten_to_transitions(dataset)
    assert len(flat) == 10
    assert flat.features["observation"].shape == (4, 3)
    numpy_view = flat.with_format("numpy")[:]
    np.testing.assert_array_equal(numpy_view["action"][:6], [0, 1, 2, 3, 4, 0])
    np.testing.assert_array_equal(numpy_view["episode_id"], [0] * 6 + [1] * 4)


def test_flatten_to_transitions_marks_only_the_last_step_done(dataset):
    numpy_view = flatten_to_transitions(dataset).with_format("numpy")[:]
    np.testing.assert_array_equal(np.flatnonzero(numpy_view["done"]), [5, 9])


def test_flatten_to_transitions_keeps_returns_to_go_monotonic(dataset):
    numpy_view = flatten_to_transitions(dataset).with_format("numpy")[:]
    first_episode = numpy_view["returns_to_go"][:6]
    assert np.all(np.diff(first_episode) <= 1e-6)


# ----------------------------------------------------------------------
# DT windowing
# ----------------------------------------------------------------------
def test_dt_sequence_dataset_counts_full_windows(dataset):
    windows = DTSequenceDataset(dataset, context_length=4, stride=2)
    # episode 0: starts 0, 2  -> 2 windows; episode 1 (len 4): 1 window
    assert len(windows) == 3


def test_dt_sequence_dataset_returns_padded_windows(dataset):
    windows = DTSequenceDataset(dataset, context_length=4, stride=2)
    item = windows[0]
    assert item["observations"].shape == (4, 4, 3)
    assert item["observations"].dtype == np.int8
    assert item["actions"].shape == (4,)
    assert item["returns_to_go"].shape == (4,)
    assert item["mask"].all()


def test_dt_sequence_dataset_pads_short_episodes(dataset):
    windows = DTSequenceDataset(dataset, context_length=5, stride=1)
    index = [i for i, (episode, _) in enumerate(windows._index) if episode == 1]
    item = windows[index[0]]
    assert item["observations"].shape == (5, 4, 3)
    np.testing.assert_array_equal(item["mask"], [True] * 4 + [False])
    np.testing.assert_array_equal(item["actions"][4], 0)
    np.testing.assert_array_equal(item["timesteps"], [0, 1, 2, 3, 4])


def test_dt_sequence_dataset_validates_arguments(dataset):
    with pytest.raises(ValueError):
        DTSequenceDataset(dataset, context_length=0)
    with pytest.raises(ValueError):
        DTSequenceDataset(dataset, context_length=4, stride=0)


def test_make_dt_dataloader_batches_windows(dataset):
    pytest.importorskip("torch")
    from dt.data import make_dt_dataloader

    loader = make_dt_dataloader(
        dataset, context_length=4, batch_size=2, shuffle=False, stride=2
    )
    batch = next(iter(loader))
    assert batch["observations"].shape == (2, 4, 4, 3)
    assert batch["actions"].shape == (2, 4)


# ----------------------------------------------------------------------
# Statistics and card
# ----------------------------------------------------------------------
def test_dataset_statistics_from_dict_matches_episode_statistics(episodes, dataset):
    stats = dataset_statistics_from_dict(DatasetDict({"train": dataset}))
    assert stats["episodes"] == len(episodes)
    assert stats["steps"] == sum(len(e.rewards) for e in episodes)
    assert stats["mean_length"] == pytest.approx(np.mean([6, 4]))
    assert stats["max_return"] == pytest.approx(8.0)


def test_dataset_statistics_from_dict_handles_empty_dataset(dataset):
    stats = dataset_statistics_from_dict(DatasetDict({"train": dataset.select([])}))
    assert stats == {"episodes": 0, "steps": 0}


def test_dataset_card_contains_hub_metadata(episodes):
    dataset_dict = episodes_to_dataset_dict(episodes)
    metadata = {
        "layout": "small",
        "policy": "heuristic",
        "gamma": 0.99,
        "max_length": 6,
        "episodes": len(episodes),
        "statistics": dataset_statistics_from_dict(dataset_dict),
    }
    card = dataset_card(metadata, features=dataset_dict["train"].features)
    assert card.startswith("---")
    assert "license: mit" in card
    assert "decision-transformer" in card
    assert "- small" in card
    assert "mean return" in card
    assert "Array3D" in card


# ----------------------------------------------------------------------
# DataCollector
# ----------------------------------------------------------------------
def test_collector_collects_and_builds_a_dataset_dict():
    collector = DataCollector(
        layout="small", max_steps=20, policy="heuristic", progress=False
    )
    dataset_dict = collector.collect(num_episodes=3)
    assert set(dataset_dict) == {"train"}
    assert len(dataset_dict["train"]) == 3
    assert collector.metadata["format"] == HF_DATASET_FORMAT
    assert collector.metadata["statistics"]["episodes"] == 3
    assert len(collector) == 3
    assert collector.dataset is dataset_dict["train"]
    assert "DataCollector(" in repr(collector)


def test_collector_respects_eval_fraction():
    collector = DataCollector(
        layout="small",
        max_steps=10,
        policy="random",
        progress=False,
        eval_fraction=0.5,
        seed=3,
    )
    dataset_dict = collector.collect(num_episodes=4)
    assert len(dataset_dict["train"]) == 2
    assert len(dataset_dict["eval"]) == 2


def test_collector_from_episodes_sets_metadata(episodes):
    collector = DataCollector(layout="small")
    dataset_dict = collector.from_episodes(episodes, eval_fraction=0.5)
    assert len(dataset_dict["train"]) == 1
    assert collector.metadata["splits"] == ["train", "eval"]
    assert collector.metadata["max_length"] == 6
    assert collector.statistics["episodes"] == 2


def test_collector_from_episodes_rejects_empty_input():
    with pytest.raises(ValueError):
        DataCollector().from_episodes([])


def test_collector_filters_low_return_episodes():
    collector = DataCollector(layout="small", max_steps=10, progress=False)
    with pytest.raises(ValueError):
        collector.collect(num_episodes=2, min_return=1e9)


def test_collector_requires_a_dataset_before_saving(tmp_path):
    with pytest.raises(RuntimeError):
        DataCollector().save_to_disk(tmp_path / "nope")


def test_collector_save_and_load_round_trip(tmp_path, episodes):
    collector = DataCollector(layout="small", policy="heuristic")
    collector.from_episodes(episodes, eval_fraction=0.5)
    path = collector.save_to_disk(tmp_path / "hf")

    assert (path / "README.md").exists()
    assert (path / "pacman_dt_metadata.json").exists()
    assert json.loads((path / "pacman_dt_metadata.json").read_text())["format"] == (
        HF_DATASET_FORMAT
    )

    restored = DataCollector.load_from_disk(path)
    assert set(restored.dataset_dict) == {"train", "eval"}
    assert len(restored) == 2
    assert restored.layout == "small"
    assert restored.policy == "heuristic"
    assert restored.dataset_dict["train"].features == collector.dataset.features
    np.testing.assert_array_equal(
        restored.episodes[0].actions, collector.episodes[0].actions
    )


def test_collector_save_without_card(tmp_path, episodes):
    collector = DataCollector(layout="small")
    collector.from_episodes(episodes)
    path = collector.save_to_disk(tmp_path / "hf", card=False)
    assert not (path / "README.md").exists()


def test_collector_default_output_directory_is_relative():
    assert Path(DEFAULT_HF_DIR).parts[0] == "data"


# ----------------------------------------------------------------------
# CLI (--format hf)
# ----------------------------------------------------------------------
def test_data_collection_cli_writes_hf_dataset(tmp_path):
    from dt.data import main

    out = tmp_path / "hf"
    exit_code = main(
        [
            "--format", "hf",
            "--episodes", "4",
            "--max-steps", "30",
            "--layout", "small",
            "--policy", "heuristic",
            "--seed", "5",
            "--eval-fraction", "0.25",
            "--quiet",
            "--hf-out", str(out),
        ]
    )
    assert exit_code == 0

    restored = DataCollector.load_from_disk(out)
    assert set(restored.dataset_dict) == {"train", "eval"}
    assert len(restored) == 4
    assert restored.metadata["layout"] == "small"
    assert restored.metadata["format"] == HF_DATASET_FORMAT
    assert (out / "README.md").read_text(encoding="utf-8").startswith("---")


def test_data_collection_cli_rejects_hub_repo_without_hf_format(tmp_path):
    from dt.data import main

    exit_code = main(
        [
            "--episodes", "1",
            "--layout", "small",
            "--quiet",
            "--hub-repo", "someone/pacman-dt",
            "--out", str(tmp_path / "x.npz"),
        ]
    )
    assert exit_code == 2
