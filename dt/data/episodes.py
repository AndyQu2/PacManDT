"""Trajectory collection and storage for Decision-Transformer training.

A Decision Transformer is trained on *episodes*, where every step carries the
observation, the action taken, the reward received and the **return-to-go**
(discounted sum of future rewards).  This module produces exactly that and can
persist it in a single compressed ``.npz`` file:

====================  ==============================================
array                 shape
====================  ==============================================
``observations``      ``(episodes, max_len, *obs_shape)``
``actions``           ``(episodes, max_len)``
``rewards``           ``(episodes, max_len)``
``returns_to_go``     ``(episodes, max_len)``
``timesteps``         ``(episodes, max_len)``
``terminated``        ``(episodes, max_len)``
``truncated``         ``(episodes, max_len)``
``mask``              ``(episodes, max_len)`` — ``True`` for real steps
``lengths``           ``(episodes,)``
====================  ==============================================

Episodes are padded to the longest one; ``mask`` tells a model which steps are
valid, so variable-length episodes can be batched with normal tensor ops.

``terminated`` marks a real end of episode (win or game over) while
``truncated`` marks a time-limit cut.  Returns-to-go are always computed over
the steps that were actually recorded.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from pacman_env.env import PacmanEnv

from .policies import Policy, make_policy

#: Reward an episode must reach to be labelled ``success=True``.  A competent
#: heuristic scores roughly 100-130 over a 1000-step ``classic`` episode, so this
#: threshold separates "played well" from "flailed around".
DEFAULT_SUCCESS_THRESHOLD = 50.0
DATASET_FORMAT = "pacman-dt/decision-transformer-v1"


# ----------------------------------------------------------------------
# Episode container
# ----------------------------------------------------------------------
@dataclass
class Episode:
    """One recorded episode, ready for Decision-Transformer training."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    returns_to_go: np.ndarray
    timesteps: np.ndarray
    seed: int = 0
    layout: str = "classic"
    score: int = 0
    total_reward: float = 0.0
    success: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def done(self) -> np.ndarray:
        """Boolean array marking the last step of the episode."""
        done = np.zeros(self.length, dtype=bool)
        if self.length:
            done[-1] = True
        return done

    def __len__(self) -> int:
        return self.length


# ----------------------------------------------------------------------
# Returns-to-go
# ----------------------------------------------------------------------
def compute_returns_to_go(rewards: Sequence[float], gamma: float = 0.99) -> np.ndarray:
    """Discounted sum of future rewards for every timestep.

    ``returns_to_go[t] = sum_{t' >= t} gamma ** (t' - t) * rewards[t']``
    """
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    returns_to_go = np.zeros_like(rewards)
    running = 0.0
    for index in range(rewards.shape[0] - 1, -1, -1):
        running = float(rewards[index]) + float(gamma) * running
        returns_to_go[index] = running
    return returns_to_go


# ----------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------
def collect_episode(
    env: PacmanEnv,
    policy: Policy,
    rng: np.random.Generator,
    seed: Optional[int] = None,
    gamma: float = 0.99,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> Episode:
    """Roll out ``policy`` in ``env`` until the episode ends."""
    obs, info = env.reset(seed=seed)
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()

    observations: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    terminated_flags: List[bool] = []
    truncated_flags: List[bool] = []

    while True:
        action = int(policy(obs, info, env, rng))
        observations.append(np.array(obs, copy=True))
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        terminated_flags.append(bool(terminated))
        truncated_flags.append(bool(truncated))

        if terminated or truncated:
            break

    total_reward = float(np.sum(rewards))
    return Episode(
        observations=np.stack(observations).astype(np.asarray(observations[0]).dtype, copy=False),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=bool),
        truncated=np.asarray(truncated_flags, dtype=bool),
        returns_to_go=compute_returns_to_go(rewards, gamma=gamma),
        timesteps=np.arange(len(actions), dtype=np.int64),
        seed=int(seed) if seed is not None else -1,
        layout=env.layout,
        score=int(info.get("score", 0)),
        total_reward=total_reward,
        success=bool(total_reward >= success_threshold),
        info={
            "win": bool(info.get("win", False)),
            "game_over": bool(info.get("game_over", False)),
            "lives": int(info.get("lives", 0)),
            "pellets_remaining": int(info.get("pellets_remaining", 0)),
        },
    )


def _collect_chunk(task: Tuple[Any, ...]) -> List[Episode]:
    """Worker entry point used by the multiprocessing collector."""
    (layout, lives, count, max_steps, policy, epsilon, seed, gamma, threshold,
     frame_skip, render_mode) = task
    return collect_trajectories(
        layout=layout,
        lives=lives,
        num_episodes=count,
        max_steps=max_steps,
        success_threshold=threshold,
        policy=policy,
        seed=seed,
        gamma=gamma,
        frame_skip=frame_skip,
        epsilon=epsilon,
        render_mode=render_mode,
        progress=False,
        workers=1,
    )


def collect_trajectories(
    layout: str = "classic",
    lives: int = 3,
    num_episodes: int = 10_000,
    max_steps: int = 1000,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    policy: Union[str, Policy] = "heuristic",
    seed: int = 0,
    gamma: float = 0.99,
    frame_skip: int = 1,
    epsilon: float = 0.05,
    render_mode: Optional[str] = None,
    progress: bool = True,
    workers: int = 1,
    num_episode: Optional[int] = None,
) -> List[Episode]:
    """Collect ``num_episodes`` Decision-Transformer episodes.

    Parameters
    ----------
    layout, lives, max_steps, frame_skip:
        Forwarded to :class:`~pacman_env.env.PacmanEnv`.  ``max_steps`` is the
        per-episode time limit (``truncated=True`` when it is reached).
    success_threshold:
        An episode is labelled ``success=True`` when its total reward is at
        least this value.
    policy:
        ``"heuristic"`` (default), ``"random"``, ``"epsilon_greedy"`` or any
        callable ``(obs, info, env, rng) -> action``.
    seed:
        Episode ``i`` uses ``seed + i`` for both the environment and the policy,
        so a run is reproducible regardless of how it is parallelised.
    workers:
        Number of processes used to collect chunks of episodes in parallel.
        Results are identical to ``workers=1``.
    num_episode:
        Deprecated alias for ``num_episodes``.
    """
    if num_episode is not None:
        warnings.warn(
            "num_episode is deprecated, use num_episodes instead",
            DeprecationWarning,
            stacklevel=2,
        )
        num_episodes = int(num_episode)
    if num_episodes <= 0:
        return []
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")

    workers = max(1, int(workers))
    if workers > 1 and num_episodes > 1:
        return _collect_parallel(
            layout=layout, lives=lives, num_episodes=num_episodes, max_steps=max_steps,
            success_threshold=success_threshold, policy=policy, seed=seed, gamma=gamma,
            frame_skip=frame_skip, epsilon=epsilon, render_mode=render_mode,
            progress=progress, workers=workers,
        )

    env = PacmanEnv(
        layout=layout,
        render_mode=render_mode,
        player_lives=lives,
        max_steps=max_steps,
        frame_skip=frame_skip,
    )
    behaviour = make_policy(policy, epsilon=epsilon)

    episodes: List[Episode] = []
    returns: List[float] = []
    successes = 0
    started = time.perf_counter()
    report_every = max(1, num_episodes // 20)

    try:
        for index in range(num_episodes):
            episode_seed = int(seed) + index
            rng = np.random.default_rng(episode_seed)
            episode = collect_episode(
                env, behaviour, rng, seed=episode_seed,
                gamma=gamma, success_threshold=success_threshold,
            )
            episodes.append(episode)
            returns.append(episode.total_reward)
            successes += int(episode.success)

            if progress and ((index + 1) % report_every == 0 or index + 1 == num_episodes):
                elapsed = max(time.perf_counter() - started, 1e-9)
                steps = sum(len(item) for item in episodes)
                print(
                    f"[{100 * (index + 1) / num_episodes:5.1f}%] "
                    f"episodes {index + 1}/{num_episodes} | "
                    f"mean return {np.mean(returns):7.2f} | "
                    f"success {100 * successes / (index + 1):5.1f}% | "
                    f"{steps / elapsed:6.0f} steps/s",
                    flush=True,
                )
    finally:
        env.close()

    return episodes


def _collect_parallel(
    layout: str,
    lives: int,
    num_episodes: int,
    max_steps: int,
    success_threshold: float,
    policy: Union[str, Policy],
    seed: int,
    gamma: float,
    frame_skip: int,
    epsilon: float,
    render_mode: Optional[str],
    progress: bool,
    workers: int,
) -> List[Episode]:
    from concurrent.futures import ProcessPoolExecutor

    workers = min(workers, num_episodes)
    chunk_size = int(np.ceil(num_episodes / workers))
    tasks = []
    for start in range(0, num_episodes, chunk_size):
        count = min(chunk_size, num_episodes - start)
        tasks.append((
            layout, lives, count, max_steps, policy, epsilon, int(seed) + start,
            gamma, success_threshold, frame_skip, render_mode,
        ))

    episodes: List[Episode] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for chunk in executor.map(_collect_chunk, tasks):
            episodes.extend(chunk)
            if progress:
                elapsed = max(time.perf_counter() - started, 1e-9)
                steps = sum(len(item) for item in episodes)
                mean_return = float(np.mean([item.total_reward for item in episodes]))
                print(
                    f"collected {len(episodes)}/{num_episodes} episodes | "
                    f"mean return {mean_return:7.2f} | {steps / elapsed:6.0f} steps/s",
                    flush=True,
                )
    return episodes[:num_episodes]


# ----------------------------------------------------------------------
# Batching / padding
# ----------------------------------------------------------------------
def pad_episodes(
    episodes: Sequence[Episode],
    obs_dtype: Optional[np.dtype] = None,
    max_length: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Stack episodes into padded arrays plus a validity ``mask``.

    ``max_length`` sets the time axis explicitly (longer episodes are
    truncated, shorter ones zero-padded); ``None`` pads to the longest
    collected episode.  ``lengths`` always reports the number of *real* steps,
    so ``arrays["lengths"] <= arrays["mask"].shape[1]``.
    """
    if not episodes:
        raise ValueError("cannot pad an empty list of episodes")
    if max_length is not None:
        max_length = int(max_length)
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer")

    longest = max(episode.length for episode in episodes)
    max_len = longest if max_length is None else max_length
    obs_shape = episodes[0].observations.shape[1:]
    dtype = np.dtype(obs_dtype) if obs_dtype is not None else episodes[0].observations.dtype
    lengths = [min(episode.length, max_len) for episode in episodes]

    arrays: Dict[str, np.ndarray] = {
        "observations": np.zeros((len(episodes), max_len, *obs_shape), dtype=dtype),
        "actions": np.zeros((len(episodes), max_len), dtype=np.int64),
        "rewards": np.zeros((len(episodes), max_len), dtype=np.float32),
        "returns_to_go": np.zeros((len(episodes), max_len), dtype=np.float32),
        "timesteps": np.zeros((len(episodes), max_len), dtype=np.int64),
        "terminated": np.zeros((len(episodes), max_len), dtype=bool),
        "truncated": np.zeros((len(episodes), max_len), dtype=bool),
        "mask": np.zeros((len(episodes), max_len), dtype=bool),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "episode_returns": np.asarray([episode.total_reward for episode in episodes], dtype=np.float32),
        "episode_scores": np.asarray([episode.score for episode in episodes], dtype=np.int64),
        "episode_success": np.asarray([episode.success for episode in episodes], dtype=bool),
        "episode_wins": np.asarray(
            [bool(episode.info.get("win", False)) for episode in episodes], dtype=bool
        ),
        "episode_seeds": np.asarray([episode.seed for episode in episodes], dtype=np.int64),
    }

    for index, episode in enumerate(episodes):
        length = lengths[index]
        arrays["observations"][index, :length] = episode.observations[:length]
        arrays["actions"][index, :length] = episode.actions[:length]
        arrays["rewards"][index, :length] = episode.rewards[:length]
        arrays["returns_to_go"][index, :length] = episode.returns_to_go[:length]
        arrays["timesteps"][index, :length] = episode.timesteps[:length]
        arrays["terminated"][index, :length] = episode.terminated[:length]
        arrays["truncated"][index, :length] = episode.truncated[:length]
        arrays["mask"][index, :length] = True

    return arrays


def iter_episodes(arrays: Dict[str, np.ndarray]) -> Iterable[Dict[str, np.ndarray]]:
    """Yield per-episode (unpadded) views from a padded dataset."""
    lengths = arrays["lengths"]
    for index, length in enumerate(lengths):
        length = int(length)
        yield {
            "observations": arrays["observations"][index, :length],
            "actions": arrays["actions"][index, :length],
            "rewards": arrays["rewards"][index, :length],
            "returns_to_go": arrays["returns_to_go"][index, :length],
            "timesteps": arrays["timesteps"][index, :length],
            "terminated": arrays["terminated"][index, :length],
            "truncated": arrays["truncated"][index, :length],
        }


# ----------------------------------------------------------------------
# Statistics / filtering
# ----------------------------------------------------------------------
def dataset_statistics(episodes: Sequence[Episode]) -> Dict[str, float]:
    """Summary statistics of a collected dataset."""
    if not episodes:
        return {"episodes": 0, "steps": 0}
    returns = np.asarray([episode.total_reward for episode in episodes], dtype=np.float64)
    lengths = np.asarray([episode.length for episode in episodes], dtype=np.float64)
    scores = np.asarray([episode.score for episode in episodes], dtype=np.float64)
    wins = np.asarray([bool(episode.info.get("win", False)) for episode in episodes], dtype=np.float64)
    successes = np.asarray([episode.success for episode in episodes], dtype=np.float64)
    return {
        "episodes": int(len(episodes)),
        "steps": int(lengths.sum()),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_length": float(lengths.mean()),
        "mean_score": float(scores.mean()),
        "success_rate": float(successes.mean()),
        "win_rate": float(wins.mean()),
    }


def filter_episodes(
    episodes: Sequence[Episode],
    min_return: Optional[float] = None,
    max_episodes: Optional[int] = None,
) -> List[Episode]:
    """Keep episodes with ``total_reward >= min_return`` (and cap the count).

    When ``max_episodes`` is smaller than the batch, the *best* episodes are
    kept (highest total reward), which is what you want for imitation data.
    """
    kept = list(episodes)
    if min_return is not None:
        kept = [episode for episode in kept if episode.total_reward >= float(min_return)]
    if max_episodes is not None:
        max_episodes = int(max_episodes)
        if max_episodes < len(kept):
            kept = sorted(kept, key=lambda episode: episode.total_reward, reverse=True)
        kept = kept[:max_episodes]
    return kept


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def save_dataset(
    path: Union[str, Path],
    episodes: Sequence[Episode],
    metadata: Optional[Dict[str, Any]] = None,
    obs_dtype: Optional[np.dtype] = None,
    compress: bool = True,
    max_length: Optional[int] = None,
) -> Path:
    """Write episodes to ``path`` (``.npz`` is appended when missing).

    ``max_length`` sets the padded time axis (longer episodes are truncated);
    ``None`` pads to the longest collected episode.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = pad_episodes(episodes, obs_dtype=obs_dtype, max_length=max_length)
    stats = dataset_statistics(episodes)
    header: Dict[str, Any] = {
        "format": DATASET_FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes": stats["episodes"],
        "steps": stats["steps"],
        "obs_shape": list(arrays["observations"].shape[2:]),
        "obs_dtype": str(arrays["observations"].dtype),
        # Callers should pass the real action-space size; the fallback is only
        # an upper bound derived from the recorded actions.
        "action_dim": int(arrays["actions"].max()) + 1 if arrays["actions"].size else 0,
        "max_length": int(arrays["mask"].shape[1]),
        "statistics": stats,
    }
    header.update(metadata or {})
    arrays["metadata"] = np.asarray(json.dumps(header))

    (np.savez_compressed if compress else np.savez)(path, **arrays)
    return path


def load_dataset(path: Union[str, Path]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load a dataset written by :func:`save_dataset`.

    Returns ``(arrays, metadata)`` where ``arrays`` contains the padded tensors
    and ``metadata`` the JSON header.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if "metadata" not in arrays:
        raise ValueError(
            f"{path} is not a PacManDT dataset (missing 'metadata' entry); "
            f"found {sorted(arrays)}"
        )
    metadata = json.loads(str(arrays.pop("metadata")))
    if metadata.get("format") != DATASET_FORMAT:
        raise ValueError(
            f"{path} uses dataset format {metadata.get('format')!r}, "
            f"expected {DATASET_FORMAT!r}"
        )
    return arrays, metadata


def episodes_from_npz(path: Union[str, Path]) -> List[Episode]:
    """Rebuild :class:`Episode` objects from a ``.npz`` written by :func:`save_dataset`.

    This is the inverse of :func:`save_dataset`, so a saved dataset can be fed
    straight back into :func:`dataset_statistics`, :func:`filter_episodes` or the
    Hugging Face exporter.  Padding is dropped: every episode is cut back to its
    stored ``length``.
    """
    arrays, metadata = load_dataset(path)
    episodes: List[Episode] = []
    for index, chunk in enumerate(iter_episodes(arrays)):
        seed = int(arrays["episode_seeds"][index]) if "episode_seeds" in arrays else 0
        episodes.append(
            Episode(
                observations=np.asarray(chunk["observations"]),
                actions=np.asarray(chunk["actions"], dtype=np.int64),
                rewards=np.asarray(chunk["rewards"], dtype=np.float32),
                terminated=np.asarray(chunk["terminated"], dtype=bool),
                truncated=np.asarray(chunk["truncated"], dtype=bool),
                returns_to_go=np.asarray(chunk["returns_to_go"], dtype=np.float32),
                timesteps=np.asarray(chunk["timesteps"], dtype=np.int64),
                seed=seed,
                layout=str(metadata.get("layout", "unknown")),
                score=int(arrays["episode_scores"][index]) if "episode_scores" in arrays else 0,
                total_reward=(
                    float(arrays["episode_returns"][index])
                    if "episode_returns" in arrays
                    else float(np.sum(chunk["rewards"]))
                ),
                success=bool(arrays["episode_success"][index]) if "episode_success" in arrays else False,
                info={
                    "win": bool(arrays["episode_wins"][index])
                    if "episode_wins" in arrays
                    else False
                },
            )
        )
    if not episodes:
        raise ValueError(f"{path} contains no episodes")
    return episodes


__all__ = [
    "Episode",
    "DATASET_FORMAT",
    "compute_returns_to_go",
    "collect_episode",
    "collect_trajectories",
    "pad_episodes",
    "iter_episodes",
    "dataset_statistics",
    "filter_episodes",
    "save_dataset",
    "load_dataset",
    "episodes_from_npz",
]
