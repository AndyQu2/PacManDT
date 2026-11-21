"""Tests for the Decision-Transformer trajectory collection utilities."""

from __future__ import annotations

import numpy as np
import pytest

from dt.data import (
    DATASET_FORMAT,
    Episode,
    HeuristicPolicy,
    bfs_distance_map,
    collect_episode,
    collect_trajectories,
    compute_returns_to_go,
    dataset_statistics,
    epsilon_greedy_policy,
    filter_episodes,
    load_dataset,
    make_policy,
    pad_episodes,
    random_policy,
    save_dataset,
)
from dt.data import UNREACHABLE, bfs_with_parents
from pacman_env import ENV_ID, PacmanEnv
from pacman_env.game.map import Map


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_episode(length: int, reward: float = 1.0, layout: str = "classic") -> Episode:
    rewards = np.full(length, reward, dtype=np.float32)
    return Episode(
        observations=np.zeros((length, 2, 2), dtype=np.int8),
        actions=np.zeros(length, dtype=np.int64),
        rewards=rewards,
        terminated=np.zeros(length, dtype=bool),
        truncated=np.zeros(length, dtype=bool),
        returns_to_go=compute_returns_to_go(rewards),
        timesteps=np.arange(length, dtype=np.int64),
        seed=length,
        layout=layout,
        score=length * 10,
        total_reward=float(rewards.sum()),
        success=float(rewards.sum()) >= 50.0,
        info={"win": False, "game_over": True, "lives": 0, "pellets_remaining": 5},
    )


# ----------------------------------------------------------------------
# Returns-to-go
# ----------------------------------------------------------------------
def test_compute_returns_to_go_matches_hand_computation():
    # gamma = 0.5 -> [1 + 2*0.5 + 3*0.25, 2 + 3*0.5, 3]
    expected = np.array([2.75, 3.5, 3.0], dtype=np.float32)
    np.testing.assert_allclose(
        compute_returns_to_go([1.0, 2.0, 3.0], gamma=0.5), expected
    )


def test_compute_returns_to_go_with_gamma_one_is_suffix_sum():
    rewards = np.array([1.0, -2.0, 5.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(
        compute_returns_to_go(rewards, gamma=1.0), [4.0, 3.0, 5.0, 0.0]
    )


def test_compute_returns_to_go_handles_empty_and_single():
    assert compute_returns_to_go([], gamma=0.9).shape == (0,)
    np.testing.assert_allclose(compute_returns_to_go([4.0], gamma=0.9), [4.0])


def test_compute_returns_to_go_first_entry_equals_discounted_return():
    gamma = 0.9
    np.testing.assert_allclose(
        compute_returns_to_go([0.0, 0.0, 1.0], gamma=gamma)[0], gamma**2, rtol=1e-6
    )


# ----------------------------------------------------------------------
# BFS helpers
# ----------------------------------------------------------------------
def test_bfs_distance_map_on_classic_layout():
    maze = Map(layout_name="classic")
    distances = bfs_distance_map(maze, [(1, 1)])
    assert distances[1, 1] == 0
    # A reachable neighbour of the source is exactly one step away.
    assert distances[1, 2] == 1 or distances[2, 1] == 1
    # Walls are never entered.
    wall = next(
        (r, c)
        for r in range(maze.shape[0])
        for c in range(maze.shape[1])
        if maze.is_wall(r, c)
    )
    assert distances[wall] == UNREACHABLE


def test_bfs_with_parents_can_reconstruct_first_step():
    maze = Map(layout_name="classic")
    start = (1, 1)
    distances, parents = bfs_with_parents(maze, start)
    cols = maze.shape[1]

    target = None
    for row in range(maze.shape[0]):
        for col in range(maze.shape[1]):
            if distances[row, col] == 3:
                target = (row, col)
                break
        if target is not None:
            break
    assert target is not None

    current = target
    for _ in range(distances.size):
        parent = int(parents[current])
        assert parent >= 0
        parent_cell = divmod(parent, cols)
        if parent_cell == start:
            break
        current = parent_cell
    else:  # pragma: no cover - defensive
        pytest.fail("parent chain did not reach the source")


# ----------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------
def test_make_policy_resolves_names():
    assert make_policy("random") is random_policy
    assert isinstance(make_policy("heuristic"), HeuristicPolicy)
    assert callable(make_policy("epsilon_greedy", epsilon=0.2))


def test_make_policy_rejects_unknown_names():
    with pytest.raises(ValueError):
        make_policy("does-not-exist")
    with pytest.raises(TypeError):
        make_policy(123)


def test_policies_only_emit_valid_actions():
    env = PacmanEnv(layout="classic", max_steps=60)
    for name in ("random", "heuristic", "epsilon_greedy"):
        policy = make_policy(name, epsilon=0.5)
        rng = np.random.default_rng(0)
        obs, info = env.reset(seed=0)
        if hasattr(policy, "reset"):
            policy.reset()
        for _ in range(60):
            action = int(policy(obs, info, env, rng))
            assert env.action_space.contains(action)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break


def test_epsilon_greedy_with_epsilon_zero_matches_base():
    def rollout(policy):
        env = PacmanEnv(layout="small", max_steps=40)
        rng = np.random.default_rng(3)
        obs, info = env.reset(seed=3)
        if hasattr(policy, "reset"):
            policy.reset()
        actions = []
        while True:
            action = int(policy(obs, info, env, rng))
            actions.append(action)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        return actions

    base_actions = rollout(HeuristicPolicy())
    wrapped_actions = rollout(epsilon_greedy_policy(HeuristicPolicy(), epsilon=0.0))
    assert base_actions == wrapped_actions


def test_heuristic_outperforms_random():
    random_eps = collect_trajectories(
        layout="classic", num_episodes=2, max_steps=200, policy="random",
        seed=0, progress=False,
    )
    heuristic_eps = collect_trajectories(
        layout="classic", num_episodes=2, max_steps=200, policy="heuristic",
        seed=0, progress=False,
    )
    random_return = float(np.mean([e.total_reward for e in random_eps]))
    heuristic_return = float(np.mean([e.total_reward for e in heuristic_eps]))
    assert heuristic_return > random_return
    assert heuristic_return > 5.0


# ----------------------------------------------------------------------
# Episode collection
# ----------------------------------------------------------------------
def test_collect_episode_is_consistent():
    env = PacmanEnv(layout="small", max_steps=30)
    episode = collect_episode(
        env, make_policy("heuristic"), np.random.default_rng(0), seed=7
    )

    assert episode.length == len(episode.actions) == len(episode.rewards)
    assert episode.length <= 30
    assert episode.observations.shape[0] == episode.length
    assert episode.observations.shape[1:] == env.observation_space.shape
    assert episode.actions.dtype == np.int64
    assert episode.rewards.dtype == np.float32
    assert episode.timesteps[0] == 0
    np.testing.assert_array_equal(episode.timesteps, np.arange(episode.length))
    assert episode.terminated[-1] or episode.truncated[-1]
    assert episode.seed == 7
    assert episode.layout == "small"
    np.testing.assert_allclose(episode.total_reward, episode.rewards.sum())
    np.testing.assert_allclose(
        episode.returns_to_go[0], compute_returns_to_go(episode.rewards)[0]
    )
    assert episode.done[-1]
    assert not episode.done[:-1].any()


def test_collect_episode_is_reproducible():
    env = PacmanEnv(layout="classic", max_steps=80)
    policy = make_policy("heuristic")
    first = collect_episode(env, policy, np.random.default_rng(0), seed=11)
    second = collect_episode(env, policy, np.random.default_rng(0), seed=11)
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.rewards, second.rewards)
    np.testing.assert_array_equal(first.observations, second.observations)


def test_collect_trajectories_uses_distinct_seeds():
    episodes = collect_trajectories(
        layout="small", num_episodes=3, max_steps=25, policy="random", seed=100,
        progress=False,
    )
    assert len(episodes) == 3
    assert [e.seed for e in episodes] == [100, 101, 102]


def test_collect_trajectories_respects_episode_count_and_layout():
    episodes = collect_trajectories(
        layout="medium", num_episodes=2, max_steps=20, policy="random", seed=0,
        progress=False,
    )
    assert len(episodes) == 2
    assert all(e.layout == "medium" for e in episodes)
    assert all(e.length <= 20 for e in episodes)


def test_collect_trajectories_accepts_callable_policy():
    def stay(obs, info, env, rng):
        return 4  # NOOP

    episodes = collect_trajectories(
        layout="small", num_episodes=1, max_steps=15, policy=stay, progress=False
    )
    assert len(episodes) == 1
    assert set(episodes[0].actions.tolist()) == {4}


def test_collect_trajectories_num_episode_alias_warns():
    with pytest.warns(DeprecationWarning):
        episodes = collect_trajectories(
            layout="small", num_episodes=1, num_episode=2, max_steps=10,
            policy="random", progress=False,
        )
    assert len(episodes) == 2


def test_success_threshold_labels_episodes():
    episodes = collect_trajectories(
        layout="small", num_episodes=1, max_steps=20, policy="heuristic",
        success_threshold=-1e9, progress=False,
    )
    assert episodes[0].success is True
    episodes = collect_trajectories(
        layout="small", num_episodes=1, max_steps=20, policy="heuristic",
        success_threshold=1e9, progress=False,
    )
    assert episodes[0].success is False


# ----------------------------------------------------------------------
# Padding
# ----------------------------------------------------------------------
def test_pad_episodes_shapes_and_mask():
    episodes = [make_episode(3), make_episode(5), make_episode(4)]
    batch = pad_episodes(episodes)

    assert batch["observations"].shape == (3, 5, 2, 2)
    assert batch["actions"].shape == (3, 5)
    assert batch["actions"].dtype == np.int64
    assert batch["rewards"].dtype == np.float32
    assert batch["returns_to_go"].shape == (3, 5)
    assert batch["timesteps"].shape == (3, 5)
    assert batch["mask"].dtype == bool
    np.testing.assert_array_equal(batch["mask"].sum(axis=1), [3, 5, 4])
    np.testing.assert_array_equal(batch["lengths"], [3, 5, 4])
    assert batch["mask"][0].tolist() == [True, True, True, False, False]
    assert batch["mask"][1].tolist() == [True] * 5
    # Padding is zero-filled.
    assert np.all(batch["actions"][0, 3:] == 0)
    assert np.all(batch["rewards"][2, 4:] == 0.0)


def test_pad_episodes_collects_metadata_columns():
    episodes = [make_episode(2), make_episode(6)]
    batch = pad_episodes(episodes)
    np.testing.assert_allclose(batch["episode_returns"], [2.0, 6.0])
    np.testing.assert_array_equal(batch["episode_scores"], [20, 60])
    np.testing.assert_array_equal(batch["episode_seeds"], [2, 6])
    np.testing.assert_array_equal(batch["episode_success"], [False, False])


def test_pad_episodes_rejects_empty_input():
    with pytest.raises(ValueError):
        pad_episodes([])


# ----------------------------------------------------------------------
# Filtering & statistics
# ----------------------------------------------------------------------
def test_filter_episodes_by_return_and_cap():
    episodes = [make_episode(length) for length in (1, 5, 10, 20)]
    kept = filter_episodes(episodes, min_return=5.0)
    assert [e.length for e in kept] == [5, 10, 20]

    capped = filter_episodes(episodes, max_episodes=2)
    assert [e.length for e in capped] == [20, 10]

    assert filter_episodes(episodes) == episodes
    assert filter_episodes([], min_return=1.0) == []


def test_dataset_statistics():
    episodes = [make_episode(2), make_episode(6)]
    stats = dataset_statistics(episodes)
    assert stats["episodes"] == 2
    assert stats["steps"] == 8
    assert stats["mean_return"] == pytest.approx(4.0)
    assert stats["min_return"] == pytest.approx(2.0)
    assert stats["max_return"] == pytest.approx(6.0)
    assert stats["mean_length"] == pytest.approx(4.0)
    assert stats["success_rate"] == pytest.approx(0.0)
    assert stats["win_rate"] == pytest.approx(0.0)


def test_dataset_statistics_on_empty_input():
    assert dataset_statistics([])["episodes"] == 0


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def test_save_and_load_roundtrip(tmp_path):
    episodes = [make_episode(3), make_episode(5)]
    metadata = {"layout": "classic", "policy": "heuristic", "gamma": 0.99}
    path = save_dataset(tmp_path / "nested" / "dataset.npz", episodes, metadata=metadata)
    assert path.exists()

    arrays, loaded_metadata = load_dataset(path)
    assert loaded_metadata["format"] == DATASET_FORMAT
    assert loaded_metadata["layout"] == "classic"
    assert loaded_metadata["policy"] == "heuristic"
    assert loaded_metadata["gamma"] == pytest.approx(0.99)
    assert loaded_metadata["episodes"] == 2
    assert loaded_metadata["steps"] == 8

    expected = pad_episodes(episodes)
    for key in ("observations", "actions", "rewards", "returns_to_go", "timesteps", "mask"):
        np.testing.assert_array_equal(arrays[key], expected[key])


def test_save_and_load_uncompressed(tmp_path):
    path = save_dataset(tmp_path / "plain.npz", [make_episode(2)], compress=False)
    arrays, metadata = load_dataset(path)
    assert arrays["actions"].shape == (1, 2)
    assert metadata["episodes"] == 1


def test_save_dataset_requires_episodes(tmp_path):
    with pytest.raises(ValueError):
        save_dataset(tmp_path / "empty.npz", [])


def test_load_dataset_rejects_foreign_files(tmp_path):
    path = tmp_path / "foreign.npz"
    np.savez(path, actions=np.zeros((1, 2), dtype=np.int64))
    with pytest.raises(ValueError):
        load_dataset(path)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def test_data_collection_cli_writes_dataset(tmp_path):
    from dt.data import main

    out = tmp_path / "cli.npz"
    exit_code = main(
        [
            "--episodes", "2",
            "--max-steps", "30",
            "--layout", "small",
            "--policy", "heuristic",
            "--seed", "5",
            "--quiet",
            "--out", str(out),
        ]
    )
    assert exit_code == 0
    assert out.exists()

    arrays, metadata = load_dataset(out)
    assert metadata["layout"] == "small"
    assert metadata["policy"] == "heuristic"
    assert metadata["collected_episodes"] == 2
    assert arrays["observations"].shape[0] == 2


def test_data_collection_cli_returns_error_when_everything_filtered(tmp_path):
    from dt.data import main

    out = tmp_path / "none.npz"
    exit_code = main(
        [
            "--episodes", "1",
            "--max-steps", "10",
            "--layout", "small",
            "--policy", "random",
            "--min-return", "1e9",
            "--quiet",
            "--out", str(out),
        ]
    )
    assert exit_code == 1
    assert not out.exists()


# ----------------------------------------------------------------------
# Registered spec
# ----------------------------------------------------------------------
def test_env_id_is_registered():
    import gymnasium as gym

    env = gym.make(ENV_ID)
    try:
        assert env.spec is not None
        assert env.spec.max_episode_steps == 1000
    finally:
        env.close()
