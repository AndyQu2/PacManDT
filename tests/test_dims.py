"""Tests for the ``state_dim`` / ``act_dim`` / ``max_length`` helpers."""

from __future__ import annotations

import numpy as np
import pytest

from dt.data import (
    DEFAULT_CONTEXT_LENGTH,
    MAX_LENGTH_ROUNDING,
    RECOMMENDED_MAX_LENGTH,
    DTDims,
    Episode,
    compute_returns_to_go,
    describe_layouts,
    dims_from_arrays,
    dims_from_dataset,
    dims_from_episodes,
    dims_from_env,
    episode_length_summary,
    get_act_dim,
    get_dims,
    get_obs_shape,
    get_state_dim,
    load_dataset,
    pad_episodes,
    recommend_context_length,
    recommend_max_length,
    save_dataset,
)
from pacman_env import PacmanEnv
from pacman_env.game.action import Action
from pacman_env.game.map import Map
from pacman_env.wrappers import flatten_float32


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_episode(length: int, obs_shape=(2, 2), action: int = 0) -> Episode:
    rewards = np.full(length, 1.0, dtype=np.float32)
    return Episode(
        observations=np.zeros((length, *obs_shape), dtype=np.int8),
        actions=np.full(length, action, dtype=np.int64),
        rewards=rewards,
        terminated=np.zeros(length, dtype=bool),
        truncated=np.zeros(length, dtype=bool),
        returns_to_go=compute_returns_to_go(rewards),
        timesteps=np.arange(length, dtype=np.int64),
        seed=0,
        layout="classic",
        score=length,
        total_reward=float(rewards.sum()),
        success=False,
        info={"win": False, "game_over": True},
    )


# ----------------------------------------------------------------------
# get_state_dim / get_act_dim
# ----------------------------------------------------------------------
@pytest.mark.parametrize("layout", Map.available_layouts())
def test_get_state_dim_is_the_maze_area(layout):
    rows, cols = Map(layout).shape
    assert get_state_dim(layout) == rows * cols
    assert get_obs_shape(layout) == (rows, cols)


def test_get_state_dim_matches_the_env_observation_space():
    env = PacmanEnv(layout="small")
    try:
        assert get_state_dim("small") == int(np.prod(env.observation_space.shape))
        assert get_act_dim() == int(env.action_space.n)
    finally:
        env.close()


def test_get_state_dim_rgb_observation():
    width, height = Map("classic").get_map_sizes()
    assert get_obs_shape("classic", obs_type="rgb") == (height, width, 3)
    assert get_state_dim("classic", obs_type="rgb") == height * width * 3


def test_get_state_dim_frame_stack_multiplies():
    assert get_state_dim("classic", frame_stack=4) == 4 * 22 * 19
    assert get_obs_shape("classic", frame_stack=2) == (2, 22, 19)


def test_get_state_dim_rejects_bad_arguments():
    with pytest.raises(ValueError):
        get_state_dim("classic", obs_type="depth")
    with pytest.raises(ValueError):
        get_state_dim("classic", frame_stack=0)


def test_get_act_dim_counts_the_action_enum():
    assert get_act_dim() == len(Action) == 5
    # The layout is irrelevant but accepted for uniform call sites.
    assert get_act_dim("small") == get_act_dim("classic")


# ----------------------------------------------------------------------
# DTDims
# ----------------------------------------------------------------------
def test_get_dims_flatten_vs_raw():
    flat = get_dims("classic")
    raw = get_dims("classic", flatten=False)

    assert flat.state_dim == raw.state_dim == 418
    assert flat.act_dim == raw.act_dim == 5
    assert flat.obs_shape == (418,)
    assert raw.obs_shape == (22, 19)
    assert flat.flatten and not raw.flatten
    assert flat.layout == "classic"


def test_dtdims_is_frozen_and_serialises():
    dims = get_dims("small")
    with pytest.raises(Exception):
        dims.state_dim = 1  # type: ignore[misc]
    payload = dims.as_dict()
    assert payload["state_dim"] == 42
    assert payload["obs_shape"] == [42]
    assert payload["act_dim"] == 5


def test_dims_from_env_plain_and_flattened():
    env = PacmanEnv(layout="small")
    try:
        plain = dims_from_env(env)
        flat = dims_from_env(flatten_float32(env))
    finally:
        env.close()

    assert plain.state_dim == 42 and plain.obs_shape == (7, 6)
    assert plain.act_dim == 5 and plain.layout == "small"
    assert not plain.flatten

    assert flat.state_dim == 42 and flat.obs_shape == (42,)
    assert flat.obs_dtype == "float32" and flat.flatten


def test_dims_from_arrays_and_episodes():
    episodes = [make_episode(3, obs_shape=(3, 4)), make_episode(5, obs_shape=(3, 4))]
    from_arrays = dims_from_arrays(pad_episodes(episodes))
    from_episodes = dims_from_episodes(episodes)

    for dims in (from_arrays, from_episodes):
        assert dims.state_dim == 12
        assert dims.act_dim == 5
        assert dims.obs_shape == (3, 4)
        assert dims.obs_dtype == "int8"


def test_dims_from_arrays_rejects_bad_input():
    with pytest.raises(KeyError):
        dims_from_arrays({"actions": np.zeros((1, 2))})
    with pytest.raises(ValueError):
        dims_from_arrays({"observations": np.zeros((2, 2))})
    with pytest.raises(ValueError):
        dims_from_episodes([])


def test_dims_from_dataset():
    pytest.importorskip("datasets")
    from dt.data import episodes_to_dataset_dict

    episodes = [make_episode(4, obs_shape=(7, 6)), make_episode(6, obs_shape=(7, 6))]
    dataset_dict = episodes_to_dataset_dict(episodes)

    dims = dims_from_dataset(dataset_dict)
    assert dims.state_dim == 42
    assert dims.act_dim == 5
    assert dims.obs_shape == (7, 6)


# ----------------------------------------------------------------------
# max_length guidance
# ----------------------------------------------------------------------
def test_episode_length_summary_percentiles():
    summary = episode_length_summary([10, 20, 30, 40])
    assert summary["count"] == 4
    assert summary["min"] == 10 and summary["max"] == 40
    assert summary["p50"] == 25
    assert summary["mean"] == 25
    with pytest.raises(ValueError):
        episode_length_summary([])


def test_recommend_max_length_rounds_up_to_the_block_size():
    value = recommend_max_length([10, 20, 30, 40], percentile=100, cap=None)
    assert value == 64
    assert value % MAX_LENGTH_ROUNDING == 0


def test_recommend_max_length_uses_the_measured_table():
    assert RECOMMENDED_MAX_LENGTH["small"] == 256
    assert recommend_max_length(layout="small") == 256
    assert recommend_max_length(layout="medium_small") == 576
    # never above the time limit the data was collected with
    assert recommend_max_length(layout="classic", max_steps=500) == 500


def test_recommend_max_length_defaults_to_max_steps():
    assert recommend_max_length() == 1000
    assert recommend_max_length(max_steps=750) == 750
    assert recommend_max_length(layout="unknown-layout", max_steps=64) == 64


def test_recommend_max_length_cap_and_validation():
    assert recommend_max_length([10] * 10, cap=8) == 8
    with pytest.raises(ValueError):
        recommend_max_length([1, 2], percentile=0)
    with pytest.raises(ValueError):
        recommend_max_length([1, 2], round_to=0)


def test_recommend_context_length():
    assert recommend_context_length() == DEFAULT_CONTEXT_LENGTH == 30
    assert recommend_context_length(1000) == 30
    assert recommend_context_length(10) == 10
    assert recommend_context_length(1) == 2


def test_describe_layouts_covers_every_layout():
    table = describe_layouts()
    assert "state_dim" in table and "max_length" in table
    for layout in Map.available_layouts():
        assert layout in table


# ----------------------------------------------------------------------
# max_length is honoured by padding / saving
# ----------------------------------------------------------------------
def test_pad_episodes_truncates_to_max_length():
    batch = pad_episodes([make_episode(5), make_episode(2)], max_length=3)
    assert batch["observations"].shape == (2, 3, 2, 2)
    assert batch["mask"].tolist() == [[True, True, True], [True, True, False]]
    np.testing.assert_array_equal(batch["lengths"], [3, 2])
    assert batch["returns_to_go"].shape == (2, 3)


def test_pad_episodes_pads_up_to_max_length():
    batch = pad_episodes([make_episode(2)], max_length=6)
    assert batch["observations"].shape == (1, 6, 2, 2)
    assert batch["mask"].sum() == 2
    assert not batch["mask"][0, 2:].any()
    np.testing.assert_array_equal(batch["lengths"], [2])


def test_pad_episodes_rejects_invalid_max_length():
    with pytest.raises(ValueError):
        pad_episodes([make_episode(2)], max_length=0)


def test_save_dataset_honours_max_length(tmp_path):
    path = save_dataset(
        tmp_path / "tiny.npz",
        [make_episode(5), make_episode(2)],
        metadata={"layout": "classic"},
        max_length=3,
    )
    arrays, metadata = load_dataset(path)
    assert metadata["max_length"] == 3
    assert arrays["observations"].shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(arrays["lengths"], [3, 2])


def test_cli_show_dims_exits_without_collecting(capsys):
    from dt.data import main

    assert main(["--show-dims"]) == 0
    out = capsys.readouterr().out
    assert "state_dim" in out
    assert "classic" in out


def test_dtdims_defaults():
    dims = DTDims(state_dim=4, act_dim=5, obs_shape=(4,))
    assert dims.state_shape == (4,)
    assert dims.flatten is True
    assert dims.layout == "unknown"
