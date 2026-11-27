"""Hugging Face ``datasets`` integration for Decision-Transformer training.

:class:`DataCollector` is the Hugging-Face-native counterpart of the
``dt.data`` CLI (:mod:`dt.data.cli`): it plays the environment with a
behaviour policy and materialises the result as a ``datasets.DatasetDict``
that can be saved to disk, pushed to the Hub and fed straight into a Decision
Transformer.

Layout of the produced dataset (one row per episode)::

    observations    Array3D (max_length, *obs_shape)  int8
    actions         list<int64>                       (max_length,)
    rewards         list<float32>
    returns_to_go   list<float32>
    timesteps       list<int64>
    terminated      list<bool>
    truncated       list<bool>
    mask            list<bool>                        True = real step
    length          int32
    episode_id      int64
    seed            int64
    layout          string
    score           int64
    total_reward    float32
    success         bool
    win             bool
    pellets_remaining int32

Episodes are padded to ``max_length`` so every row has the same shape, which
makes Arrow storage compact and lets ``with_format("numpy")`` return stacked
``(batch, max_length, ...)`` tensors with zero Python-level assembly.

Typical use::

    from dt.data.hf_dataset import DataCollector

    collector = DataCollector(layout="classic", policy="heuristic", workers=8)
    dataset = collector.collect(num_episodes=1000)      # DatasetDict(train=..., eval=...)
    collector.push_to_hub("your-name/pacman-dt-classic", private=True)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa

try:  # pragma: no cover - exercised only when the extra is missing
    from datasets import (
        Array2D,
        Array3D,
        Array4D,
        Array5D,
        Dataset,
        DatasetDict,
        Features,
        Value,
    )
    from datasets import Sequence as HFSequence
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dt.data.hf_dataset needs the optional 'hub' extra. "
        'Install it with: pip install -e ".[hub]"'
    ) from exc

from .episodes import (
    DATASET_FORMAT,
    DEFAULT_SUCCESS_THRESHOLD,
    Episode,
    collect_trajectories,
    dataset_statistics,
    filter_episodes,
    pad_episodes,
)
from .observations import (
    GHOST_OBS_MODE,
    STATE_OBS_MODE,
    GhostObservationConfig,
    observation_from_metadata,
    observation_metadata,
    resolve_observation,
)

HF_DATASET_FORMAT = "pacman-dt/hf-decision-transformer-v1"
DEFAULT_HF_DIR = "data/pacman_dt_hf"

#: One-line documentation per ghost-observation channel (used by the dataset card).
_CHANNEL_DOCS: Dict[str, str] = {
    "wall": "1 where the tile is a wall",
    "pellet": "1 where a pellet is still present",
    "power_pellet": "1 where a power pellet is still present",
    "ghost_normal": "number of normal (dangerous) ghosts on the tile",
    "ghost_vulnerable": "number of frightened (edible) ghosts on the tile",
    "ghost_eaten": "number of eaten ghosts (eyes returning home) on the tile",
    "player": "1 on the player's tile",
    "player_distance": "BFS distance to the player, bucketed 0..max_distance",
    "ghost_distance": "BFS distance to the nearest dangerous ghost",
    "frightened_timer": "remaining frightened time, bucketed 0..max_distance",
    "lives": "remaining lives, bucketed 0..max_distance",
}

_ARRAY_TYPES = {1: None, 2: Array2D, 3: Array3D, 4: Array4D, 5: Array5D}


# ----------------------------------------------------------------------
# Feature schema
# ----------------------------------------------------------------------
def _array_feature(shape: Sequence[int], dtype: str):
    """Pick the ``ArrayND`` feature matching ``shape``."""
    ndim = len(tuple(shape))
    if ndim == 1:  # pragma: no cover - 1-D payloads are stored as sequences
        raise ValueError("1-D arrays are stored as Sequence(Value(...)) instead")
    try:
        return _ARRAY_TYPES[ndim](shape=tuple(int(size) for size in shape), dtype=dtype)
    except KeyError:  # pragma: no cover - defensive
        raise ValueError(f"unsupported observation rank {ndim}") from None


def episode_features(
    obs_shape: Sequence[int],
    max_length: int,
    obs_dtype: str = "int8",
) -> Features:
    """Arrow schema for one episode row."""
    return Features(
        {
            "observations": _array_feature((max_length, *obs_shape), obs_dtype),
            "actions": HFSequence(Value("int64")),
            "rewards": HFSequence(Value("float32")),
            "returns_to_go": HFSequence(Value("float32")),
            "timesteps": HFSequence(Value("int64")),
            "terminated": HFSequence(Value("bool")),
            "truncated": HFSequence(Value("bool")),
            "mask": HFSequence(Value("bool")),
            "length": Value("int32"),
            "episode_id": Value("int64"),
            "seed": Value("int64"),
            "layout": Value("string"),
            "policy": Value("string"),
            "obs_mode": Value("string"),
            "score": Value("int64"),
            "total_reward": Value("float32"),
            "success": Value("bool"),
            "win": Value("bool"),
            "pellets_remaining": Value("int32"),
        }
    )


# ----------------------------------------------------------------------
# Episode <-> Dataset conversion
# ----------------------------------------------------------------------
def _resize_time_axis(array: np.ndarray, length: int) -> np.ndarray:
    """Truncate or zero-pad the time axis (axis 1) of ``array`` to ``length``."""
    current = array.shape[1]
    if current == length:
        return array
    if current > length:
        return array[:, :length]
    pad = length - current
    pad_width = [(0, 0), (0, pad)] + [(0, 0)] * (array.ndim - 2)
    return np.pad(array, pad_width)


def _nested_list_column(values: np.ndarray):
    """Wrap ``values`` (shape ``(n, *element_shape)``) in nested Arrow lists.

    Equivalent to ``pa.array(values.tolist())`` but keeps everything in Arrow
    buffers, so a 1000-episode dataset costs a few hundred MB instead of tens
    of GB of Python objects.
    """
    element_shape = values.shape[1:]
    total = int(values.size)
    column = pa.array(np.ascontiguousarray(values).reshape(-1))
    for size in reversed(element_shape):
        offsets = pa.array(np.arange(0, total + 1, size, dtype=np.int32))
        column = pa.ListArray.from_arrays(offsets, column)
        total //= size
    return column


def _array_column(feature, values: np.ndarray):
    """Build the Arrow column for an ``ArrayND`` feature."""
    pa_type = Features({"column": feature}).arrow_schema.field("column").type
    return pa_type.wrap_array(_nested_list_column(values))


def episodes_to_dataset(
    episodes: Sequence[Episode],
    max_length: Optional[int] = None,
    obs_dtype: Optional[Union[str, np.dtype]] = None,
    dataset_name: str = "default",
) -> Dataset:
    """Convert collected :class:`~dt.data.episodes.Episode` objects to a HF ``Dataset``."""
    if not episodes:
        raise ValueError("cannot build a dataset from an empty list of episodes")

    arrays = pad_episodes(episodes, obs_dtype=obs_dtype)
    padded_length = int(arrays["observations"].shape[1])
    length = padded_length if max_length is None else int(max_length)
    if length <= 0:
        raise ValueError("max_length must be a positive integer")

    observations = _resize_time_axis(arrays["observations"], length)
    lengths = np.minimum(arrays["lengths"], length).astype(np.int32)

    features = episode_features(
        obs_shape=observations.shape[2:],
        max_length=length,
        obs_dtype=str(observations.dtype),
    )
    payload = {
        "observations": _array_column(features["observations"], observations),
        "actions": list(_resize_time_axis(arrays["actions"], length)),
        "rewards": list(_resize_time_axis(arrays["rewards"], length)),
        "returns_to_go": list(_resize_time_axis(arrays["returns_to_go"], length)),
        "timesteps": list(_resize_time_axis(arrays["timesteps"], length)),
        "terminated": list(_resize_time_axis(arrays["terminated"], length)),
        "truncated": list(_resize_time_axis(arrays["truncated"], length)),
        "mask": list(_resize_time_axis(arrays["mask"], length)),
        "length": lengths,
        "episode_id": np.arange(len(episodes), dtype=np.int64),
        "seed": np.asarray([episode.seed for episode in episodes], dtype=np.int64),
        "layout": [episode.layout for episode in episodes],
        "policy": [str(getattr(episode, "policy", "unknown")) for episode in episodes],
        "obs_mode": [
            str(getattr(episode, "obs_mode", STATE_OBS_MODE)) for episode in episodes
        ],
        "score": np.asarray([episode.score for episode in episodes], dtype=np.int64),
        "total_reward": np.asarray(
            [episode.total_reward for episode in episodes], dtype=np.float32
        ),
        "success": np.asarray([episode.success for episode in episodes], dtype=bool),
        "win": np.asarray(
            [bool(episode.info.get("win", False)) for episode in episodes], dtype=bool
        ),
        "pellets_remaining": np.asarray(
            [int(episode.info.get("pellets_remaining", 0)) for episode in episodes],
            dtype=np.int32,
        ),
    }
    return Dataset.from_dict(payload, features=features, split=dataset_name)


def dataset_to_episodes(dataset: Dataset) -> List[Episode]:
    """Rebuild :class:`Episode` objects from a HF ``Dataset`` (inverse of above)."""
    numpy_view = dataset.with_format("numpy")
    episodes: List[Episode] = []
    for index in range(len(dataset)):
        row = numpy_view[index]
        length = int(row["length"])
        observations = np.asarray(row["observations"])[:length]
        rewards = np.asarray(row["rewards"], dtype=np.float32)[:length]
        episodes.append(
            Episode(
                observations=observations,
                actions=np.asarray(row["actions"], dtype=np.int64)[:length],
                rewards=rewards,
                terminated=np.asarray(row["terminated"], dtype=bool)[:length],
                truncated=np.asarray(row["truncated"], dtype=bool)[:length],
                returns_to_go=np.asarray(row["returns_to_go"], dtype=np.float32)[:length],
                timesteps=np.asarray(row["timesteps"], dtype=np.int64)[:length],
                seed=int(row["seed"]),
                layout=str(row["layout"]),
                policy=str(row.get("policy", "unknown") or "unknown"),
                obs_mode=str(row.get("obs_mode", STATE_OBS_MODE) or STATE_OBS_MODE),
                score=int(row["score"]),
                total_reward=float(row["total_reward"]),
                success=bool(row["success"]),
                info={
                    "win": bool(row["win"]),
                    "pellets_remaining": int(row["pellets_remaining"]),
                },
            )
        )
    return episodes


def split_episodes(
    episodes: Sequence[Episode],
    eval_fraction: float = 0.0,
) -> Tuple[List[Episode], List[Episode]]:
    """Split off the trailing ``eval_fraction`` of ``episodes`` as a hold-out set."""
    episodes = list(episodes)
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    num_eval = int(round(len(episodes) * eval_fraction))
    if num_eval <= 0 or num_eval >= len(episodes):
        return episodes, []
    return episodes[:-num_eval], episodes[-num_eval:]


def episodes_to_dataset_dict(
    episodes: Sequence[Episode],
    eval_fraction: float = 0.0,
    max_length: Optional[int] = None,
    obs_dtype: Optional[Union[str, np.dtype]] = None,
    *,
    train_episodes: Optional[Sequence[Episode]] = None,
    eval_episodes: Optional[Sequence[Episode]] = None,
) -> DatasetDict:
    """Build a ``DatasetDict`` with ``train`` (and optionally ``eval``) splits.

    Pass ``train_episodes``/``eval_episodes`` to use a split that was decided
    elsewhere - for instance the stable hash split of
    :func:`dt.data.protocol.split_by_hash`, which keeps an evaluation episode on
    the evaluation side even after more data is collected.  Without them the
    trailing ``eval_fraction`` of ``episodes`` is held out.
    """
    if train_episodes is None and eval_episodes is None:
        train_episodes, eval_episodes = split_episodes(episodes, eval_fraction)
    else:
        train_episodes = list(train_episodes or [])
        eval_episodes = list(eval_episodes or [])
    if max_length is None:
        # one shared padded length so train and eval have identical shapes
        max_length = max((int(len(episode.rewards)) for episode in episodes), default=0)
    splits: Dict[str, Dataset] = {}
    if train_episodes:
        splits["train"] = episodes_to_dataset(
            train_episodes, max_length=max_length, obs_dtype=obs_dtype, dataset_name="train"
        )
    if eval_episodes:
        splits["eval"] = episodes_to_dataset(
            eval_episodes, max_length=max_length, obs_dtype=obs_dtype, dataset_name="eval"
        )
    if not splits:  # pragma: no cover - guarded by episodes_to_dataset
        raise ValueError("cannot build a DatasetDict from an empty list of episodes")
    return DatasetDict(splits)


def flatten_to_transitions(dataset: Dataset) -> Dataset:
    """Explode an episode dataset into one row per transition.

    Handy for filtering on ``returns_to_go`` or for building a plain
    ``(state, action, reward)`` offline-RL buffer.
    """
    numpy_view = dataset.with_format("numpy")
    rows: Dict[str, List[Any]] = {
        key: []
        for key in (
            "observation",
            "action",
            "reward",
            "returns_to_go",
            "timestep",
            "done",
            "episode_id",
            "seed",
        )
    }
    obs_shape: Optional[Tuple[int, ...]] = None
    for index in range(len(dataset)):
        row = numpy_view[index]
        length = int(row["length"])
        if length == 0:  # pragma: no cover - defensive
            continue
        observations = np.asarray(row["observations"])[:length]
        obs_shape = observations.shape[1:]
        rows["observation"].append(observations)
        rows["action"].append(np.asarray(row["actions"], dtype=np.int64)[:length])
        rows["reward"].append(np.asarray(row["rewards"], dtype=np.float32)[:length])
        rows["returns_to_go"].append(
            np.asarray(row["returns_to_go"], dtype=np.float32)[:length]
        )
        rows["timestep"].append(np.arange(length, dtype=np.int64))
        done = np.zeros(length, dtype=bool)
        done[-1] = bool(row["terminated"][length - 1] or row["truncated"][length - 1])
        rows["done"].append(done)
        rows["episode_id"].append(np.full(length, int(row["episode_id"]), dtype=np.int64))
        rows["seed"].append(np.full(length, int(row["seed"]), dtype=np.int64))

    if obs_shape is None:  # pragma: no cover - defensive
        raise ValueError("dataset contains no transitions")

    flat = {
        "observation": np.concatenate(rows["observation"], axis=0),
        "action": np.concatenate(rows["action"]),
        "reward": np.concatenate(rows["reward"]),
        "returns_to_go": np.concatenate(rows["returns_to_go"]),
        "timestep": np.concatenate(rows["timestep"]),
        "done": np.concatenate(rows["done"]),
        "episode_id": np.concatenate(rows["episode_id"]),
        "seed": np.concatenate(rows["seed"]),
    }
    features = Features(
        {
            "observation": _array_feature(obs_shape, str(flat["observation"].dtype)),
            "action": Value("int64"),
            "reward": Value("float32"),
            "returns_to_go": Value("float32"),
            "timestep": Value("int64"),
            "done": Value("bool"),
            "episode_id": Value("int64"),
            "seed": Value("int64"),
        }
    )
    payload = dict(flat)
    payload["observation"] = _array_column(
        features["observation"], flat["observation"]
    )
    return Dataset.from_dict(payload, features=features)


# ----------------------------------------------------------------------
# Statistics / dataset card
# ----------------------------------------------------------------------
def dataset_statistics_from_dict(dataset_dict: DatasetDict) -> Dict[str, float]:
    """Aggregate episode statistics straight from the Arrow columns."""
    lengths: List[np.ndarray] = []
    returns: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    success: List[np.ndarray] = []
    win: List[np.ndarray] = []
    for dataset in dataset_dict.values():
        if not len(dataset):
            continue
        lengths.append(np.asarray(dataset["length"], dtype=np.int64))
        returns.append(np.asarray(dataset["total_reward"], dtype=np.float64))
        scores.append(np.asarray(dataset["score"], dtype=np.float64))
        success.append(np.asarray(dataset["success"], dtype=bool))
        win.append(np.asarray(dataset["win"], dtype=bool))
    if not lengths:
        return {"episodes": 0, "steps": 0}
    lengths_arr = np.concatenate(lengths)
    returns_arr = np.concatenate(returns)
    scores_arr = np.concatenate(scores)
    success_arr = np.concatenate(success)
    win_arr = np.concatenate(win)
    episodes = int(lengths_arr.size)
    return {
        "episodes": episodes,
        "steps": int(lengths_arr.sum()),
        "mean_return": float(returns_arr.mean()),
        "std_return": float(returns_arr.std()),
        "min_return": float(returns_arr.min()),
        "max_return": float(returns_arr.max()),
        "mean_length": float(lengths_arr.mean()),
        "mean_score": float(scores_arr.mean()),
        "success_rate": float(success_arr.mean()),
        "win_rate": float(win_arr.mean()),
    }


def dataset_card(
    metadata: Dict[str, Any],
    features: Optional[Features] = None,
    splits: Optional[Sequence[str]] = None,
) -> str:
    """Render the ``README.md`` dataset card for a Hub repository."""
    stats = metadata.get("statistics", {}) or {}
    frontmatter = [
        "---",
        "license: mit",
        "task_categories:",
        "- reinforcement-learning",
        "tags:",
        "- decision-transformer",
        "- offline-rl",
        "- pacman",
        "- gymnasium",
        f"- {metadata.get('layout', 'classic')}",
        "---",
        "",
    ]
    body = [
        "# PacManDT trajectories (Decision Transformer)",
        "",
        "Offline trajectories collected with "
        f"PacManDT's `{metadata.get('policy', 'heuristic')}` behaviour policy on the "
        f"`{metadata.get('layout', 'classic')}` maze.",
        "",
        f"* format: `{HF_DATASET_FORMAT}`",
        f"* episodes: **{stats.get('episodes', metadata.get('episodes', '?'))}**",
        f"* transitions: **{stats.get('steps', metadata.get('steps', '?'))}**",
        f"* observation mode: `{metadata.get('obs_mode', STATE_OBS_MODE)}`",
        f"* gamma: `{metadata.get('gamma', 0.99)}`",
        f"* max episode length: `{metadata.get('max_length', '?')}`",
        f"* splits: {', '.join(splits or ['train'])}",
        "",
        "## Statistics",
        "",
    ]
    if stats:
        body += [
            "| metric | value |",
            "| --- | --- |",
            f"| mean return | {stats.get('mean_return', 0.0):.2f} |",
            f"| std return | {stats.get('std_return', 0.0):.2f} |",
            f"| min / max return | {stats.get('min_return', 0.0):.2f} / {stats.get('max_return', 0.0):.2f} |",
            f"| mean length | {stats.get('mean_length', 0.0):.1f} |",
            f"| mean score | {stats.get('mean_score', 0.0):.1f} |",
            f"| success rate | {100 * stats.get('success_rate', 0.0):.1f}% |",
            f"| win rate | {100 * stats.get('win_rate', 0.0):.1f}% |",
            "",
        ]
    body += [
        "## Schema",
        "",
        "One row per episode, padded to `max_length`. `mask` marks real steps;",
        "`length` is the unpadded length.",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        'ds = load_dataset("<repo-id>", split="train").with_format("numpy")',
        'obs = ds[:]["observations"]        # (N, max_length, rows, cols) int8',
        'actions = ds[:]["actions"]         # (N, max_length) int64',
        'rtg = ds[:]["returns_to_go"]       # (N, max_length) float32',
        'mask = ds[:]["mask"]               # (N, max_length) bool',
        "```",
        "",
    ]
    observation = metadata.get("observation") or {}
    channels = observation.get("channels") or []
    if metadata.get("obs_mode", STATE_OBS_MODE) != STATE_OBS_MODE and channels:
        body += [
            "### Observation channels",
            "",
            f"`obs = ds[:][\"observations\"]` is `(N, max_length, {len(channels)}, rows, cols)` "
            f"{observation.get('dtype', 'int8')} — flatten to "
            f"`state_dim={observation.get('feature_dim', '?')}` for the transformer.",
            "",
            "| # | channel | meaning |",
            "| --- | --- | --- |",
        ]
        body += [
            f"| {index} | `{name}` | {_CHANNEL_DOCS.get(name, '')} |"
            for index, name in enumerate(channels)
        ]
        body.append("")
    if features is not None:
        body += ["### Features", "", "```", str(features), "```", ""]
    body += [
        "## Decision-Transformer loading",
        "",
        "```python",
        "from dt.data.hf_dataset import DTSequenceDataset, make_dt_dataloader",
        "",
        'ds = load_dataset("<repo-id>", split="train")',
        "windows = DTSequenceDataset(ds, context_length=30)",
        "loader = make_dt_dataloader(ds, context_length=30, batch_size=64)",
        "```",
        "",
        "## Collection config",
        "",
        "```json",
    ]
    config = {
        key: value
        for key, value in metadata.items()
        if key not in ("statistics", "features")
    }
    body += [_json_dumps(config), "```", ""]
    return "\n".join(frontmatter + body)


# ----------------------------------------------------------------------
# Torch windowing (optional)
# ----------------------------------------------------------------------
class DTSequenceDataset:
    """Sliding windows of length ``context_length`` over a HF episode dataset.

    Yields numpy arrays shaped ``(context_length, ...)``; windows that run past
    the end of an episode are zero padded and flagged in ``mask``.
    """

    def __init__(
        self,
        dataset: Dataset,
        context_length: int = 30,
        stride: int = 1,
        obs_key: str = "observations",
    ):
        if context_length <= 0:
            raise ValueError("context_length must be a positive integer")
        if stride <= 0:
            raise ValueError("stride must be a positive integer")
        self.dataset = dataset
        self.context_length = int(context_length)
        self.stride = int(stride)
        self.obs_key = obs_key
        self.obs_dtype = self._infer_obs_dtype(dataset, obs_key)

        lengths = np.asarray(dataset["length"], dtype=np.int64)
        index: List[Tuple[int, int]] = []
        for episode_index, length in enumerate(lengths):
            length = int(length)
            if length <= self.context_length:
                index.append((episode_index, 0))
            else:
                for start in range(0, length - self.context_length + 1, self.stride):
                    index.append((episode_index, start))
        self._index = index
        self._view = dataset.with_format("numpy")

    @staticmethod
    def _infer_obs_dtype(dataset: Dataset, obs_key: str) -> Optional[np.dtype]:
        feature = dataset.features.get(obs_key)
        dtype = getattr(feature, "dtype", None)
        return np.dtype(dtype) if dtype else None

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        episode_index, start = self._index[item]
        row = self._view[episode_index]
        length = int(row["length"])
        stop = min(start + self.context_length, length)
        window = stop - start

        observations = np.asarray(row[self.obs_key])[start:stop]
        if self.obs_dtype is not None:
            # ``with_format("numpy")`` widens ArrayND columns to int64.
            observations = observations.astype(self.obs_dtype, copy=False)
        actions = np.asarray(row["actions"], dtype=np.int64)[start:stop]
        returns_to_go = np.asarray(row["returns_to_go"], dtype=np.float32)[start:stop]
        timesteps = np.asarray(row["timesteps"], dtype=np.int64)[start:stop]
        mask = np.asarray(row["mask"], dtype=bool)[start:stop]

        pad = self.context_length - window
        if pad:
            observations = np.concatenate(
                [observations, np.zeros((pad, *observations.shape[1:]), dtype=observations.dtype)]
            )
            actions = np.concatenate([actions, np.zeros(pad, dtype=actions.dtype)])
            returns_to_go = np.concatenate(
                [returns_to_go, np.zeros(pad, dtype=returns_to_go.dtype)]
            )
            timesteps = np.concatenate(
                [timesteps, np.arange(length, length + pad, dtype=timesteps.dtype)]
            )
            mask = np.concatenate([mask, np.zeros(pad, dtype=bool)])

        return {
            self.obs_key: observations,
            "actions": actions,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "mask": mask,
            "episode_id": np.int64(episode_index),
            "start": np.int64(start),
        }


def make_dt_dataloader(
    dataset: Dataset,
    context_length: int = 30,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    stride: int = 1,
    **dataloader_kwargs: Any,
):
    """Wrap a HF dataset in a ``torch.utils.data.DataLoader`` of DT windows."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "make_dt_dataloader needs PyTorch. "
            'Install it with: pip install -e ".[train]"'
        ) from exc

    windows = DTSequenceDataset(dataset, context_length=context_length, stride=stride)
    return DataLoader(
        windows,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        **dataloader_kwargs,
    )


def _infer_obs_from_dataset(
    dataset_dict: DatasetDict,
) -> Tuple[str, Optional[GhostObservationConfig]]:
    """Recover ``(obs_mode, config)`` from a dataset's observation feature shape."""
    try:
        first = next(iter(dataset_dict.values()))
        obs_shape = tuple(int(dim) for dim in first.features["observations"].shape[1:])
    except (StopIteration, KeyError, AttributeError):  # pragma: no cover - defensive
        return STATE_OBS_MODE, None
    if len(obs_shape) == 3:
        return GHOST_OBS_MODE, GhostObservationConfig()
    return STATE_OBS_MODE, None


# ----------------------------------------------------------------------
# The collector
# ----------------------------------------------------------------------
@dataclass
class DataCollector:
    """Collect PacMan episodes and expose them as a Hugging Face dataset.

    The object is both the *configuration* of the behaviour policy and the
    holder of the produced ``DatasetDict``::

        collector = DataCollector(layout="classic", policy="heuristic", workers=8)
        dataset = collector.collect(num_episodes=1000)
        collector.push_to_hub("you/pacman-dt-classic")

    Attributes
    ----------
    layout, lives, max_steps, policy, gamma, frame_skip, epsilon,
    success_threshold, seed, workers, render_mode, progress:
        Collection settings, forwarded to
        :func:`dt.data.episodes.collect_trajectories`.
    obs_dtype:
        dtype used to store observations (``int8`` by default).
    obs_mode:
        Observation space recorded in the dataset: ``"state"`` (the
        environment's own grid) or ``"ghost"`` (the ghost-aware stack from
        :mod:`dt.data.observations`, which adds per-ghost channels, distance
        fields and scalar globals).
    obs_config:
        Optional :class:`~dt.data.observations.GhostObservationConfig` used when
        ``obs_mode="ghost"`` (which optional channels to emit).
    eval_fraction:
        Fraction of episodes moved into an ``eval`` split.
    max_length:
        Pad/truncate every episode to this many steps (``None`` keeps the
        longest collected episode).
    repo_id:
        Default Hub repository for :meth:`push_to_hub`.
    """

    layout: str = "classic"
    lives: int = 3
    max_steps: Optional[int] = 1000
    policy: str = "heuristic"
    gamma: float = 0.99
    frame_skip: int = 1
    epsilon: float = 0.05
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
    seed: int = 0
    workers: int = 1
    render_mode: Optional[str] = None
    progress: bool = True
    obs_dtype: str = "int8"
    obs_mode: str = STATE_OBS_MODE
    obs_config: Optional[GhostObservationConfig] = None
    eval_fraction: float = 0.0
    max_length: Optional[int] = None
    repo_id: Optional[str] = None

    episodes: List[Episode] = field(default_factory=list, repr=False)
    dataset_dict: Optional[DatasetDict] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)

    # -- collection ----------------------------------------------------
    def collect(
        self,
        num_episodes: int = 1000,
        *,
        min_return: Optional[float] = None,
        keep_won_only: bool = False,
        max_kept: Optional[int] = None,
    ) -> DatasetDict:
        """Roll out ``num_episodes`` episodes and build the HF dataset."""
        self.episodes = collect_trajectories(
            layout=self.layout,
            lives=self.lives,
            num_episodes=num_episodes,
            max_steps=self.max_steps,
            success_threshold=self.success_threshold,
            policy=self.policy,
            seed=self.seed,
            gamma=self.gamma,
            frame_skip=self.frame_skip,
            epsilon=self.epsilon,
            render_mode=self.render_mode,
            progress=self.progress,
            workers=self.workers,
            obs_mode=self.obs_mode,
            obs_config=self.obs_config,
        )
        kept = list(self.episodes)
        if keep_won_only:
            kept = [episode for episode in kept if episode.info.get("win", False)]
        kept = filter_episodes(kept, min_return=min_return, max_episodes=max_kept)
        if not kept:
            raise ValueError(
                "no episodes left after filtering; lower min_return or drop keep_won_only"
            )
        return self.from_episodes(kept)

    def from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        eval_fraction: Optional[float] = None,
        train_episodes: Optional[Sequence[Episode]] = None,
        eval_episodes: Optional[Sequence[Episode]] = None,
    ) -> DatasetDict:
        """Build (or rebuild) the HF dataset from already collected episodes.

        Pass ``train_episodes``/``eval_episodes`` to keep a split that was
        decided elsewhere (see :func:`dt.data.protocol.split_by_hash`).
        """
        episodes = list(episodes)
        if not episodes:
            raise ValueError("cannot build a dataset from an empty list of episodes")
        self.episodes = episodes
        fraction = self.eval_fraction if eval_fraction is None else float(eval_fraction)

        self.dataset_dict = episodes_to_dataset_dict(
            episodes,
            eval_fraction=fraction,
            max_length=self.max_length,
            obs_dtype=self.obs_dtype,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
        )
        self.metadata = self._build_metadata(episodes, fraction)
        return self.dataset_dict

    def _build_metadata(self, episodes: Sequence[Episode], eval_fraction: float) -> Dict[str, Any]:
        stats = dataset_statistics(episodes)
        splits = list(self.dataset_dict.keys()) if self.dataset_dict else ["train"]
        max_length = (
            int(self.dataset_dict["train"].features["observations"].shape[0])
            if self.dataset_dict and "train" in self.dataset_dict
            else max((episode.length for episode in episodes), default=0)
        )
        recorded_mode = self.obs_mode
        recorded_config = self.obs_config
        for episode in episodes:
            mode = getattr(episode, "obs_mode", STATE_OBS_MODE)
            if mode and mode != STATE_OBS_MODE:
                recorded_mode = mode
                if not recorded_config:
                    recorded_config = GhostObservationConfig.from_dict(
                        (getattr(episode, "observation_meta", {}) or {}).get("config", {})
                    )
                break
        observation = observation_metadata(
            recorded_mode, layout=self.layout, config=recorded_config
        )
        policies = sorted(
            {
                str(getattr(episode, "policy", "unknown") or "unknown")
                for episode in episodes
            }
        )
        return {
            "format": HF_DATASET_FORMAT,
            "source_format": DATASET_FORMAT,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "layout": self.layout,
            "lives": self.lives,
            "max_steps": self.max_steps,
            "max_length": max_length,
            "policy": self.policy,
            "policies": policies,
            "obs_mode": observation["mode"],
            "observation": observation,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "frame_skip": self.frame_skip,
            "seed": self.seed,
            "success_threshold": self.success_threshold,
            "eval_fraction": eval_fraction,
            "splits": splits,
            "episodes": stats.get("episodes", 0),
            "steps": stats.get("steps", 0),
            "statistics": stats,
        }

    # -- accessors -----------------------------------------------------
    @property
    def dataset(self) -> Dataset:
        """The ``train`` split (raises if nothing has been collected yet)."""
        if self.dataset_dict is None:
            raise RuntimeError("no dataset yet - call collect() or from_episodes() first")
        return self.dataset_dict["train"]

    @property
    def statistics(self) -> Dict[str, float]:
        """Statistics of the collected episodes."""
        if self.episodes:
            return dataset_statistics(self.episodes)
        if self.dataset_dict is not None:
            return dataset_statistics_from_dict(self.dataset_dict)
        return {"episodes": 0, "steps": 0}

    def describe(self) -> str:
        """Human-readable summary of the observation space and splits."""
        resolved = resolve_observation(
            self.obs_mode, layout=self.layout, config=self.obs_config
        )
        lines = [
            f"layout:        {self.layout}",
            f"obs_mode:      {resolved.mode}",
            f"obs_shape:     {resolved.obs_shape} {resolved.dtype}",
            f"state_dim:     {resolved.feature_dim}",
            f"policy:        {self.policy}",
        ]
        if self.dataset_dict is not None:
            lines.append(
                "splits:        "
                + ", ".join(f"{name}={len(split)}" for name, split in self.dataset_dict.items())
            )
        return "\n".join(lines)

    def card(self) -> str:
        """Render the Hub dataset card for the current dataset."""
        features = None
        if self.dataset_dict is not None:
            first = next(iter(self.dataset_dict.values()))
            features = first.features
        return dataset_card(self.metadata, features=features, splits=list(self.dataset_dict or {}))

    # -- persistence ---------------------------------------------------
    def save_to_disk(self, path: Union[str, Path] = DEFAULT_HF_DIR, card: bool = True) -> Path:
        """Write the dataset to ``path`` in ``datasets`` arrow format."""
        dataset_dict = self._require_dataset()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(path))
        if card:
            (path / "README.md").write_text(self.card(), encoding="utf-8")
        (path / "pacman_dt_metadata.json").write_text(
            _json_dumps(self.metadata), encoding="utf-8"
        )
        return path

    @classmethod
    def load_from_disk(cls, path: Union[str, Path] = DEFAULT_HF_DIR) -> "DataCollector":
        """Reload a collector from :meth:`save_to_disk` output."""
        from datasets import load_from_disk

        path = Path(path)
        dataset_dict = load_from_disk(str(path))
        if isinstance(dataset_dict, Dataset):  # single-split directory
            dataset_dict = DatasetDict({"train": dataset_dict})
        collector = cls()
        collector.dataset_dict = dataset_dict
        metadata_path = path / "pacman_dt_metadata.json"
        if metadata_path.exists():
            collector.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            collector.layout = collector.metadata.get("layout", collector.layout)
            collector.policy = collector.metadata.get("policy", collector.policy)
            collector.obs_mode, collector.obs_config = observation_from_metadata(
                collector.metadata
            )
        else:
            collector.obs_mode, collector.obs_config = _infer_obs_from_dataset(dataset_dict)
        collector.episodes = [
            episode
            for split in dataset_dict.values()
            for episode in dataset_to_episodes(split)
        ]
        return collector

    def push_to_hub(
        self,
        repo_id: Optional[str] = None,
        *,
        private: bool = False,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
        card: bool = True,
    ) -> str:
        """Upload the dataset (and its card) to the Hugging Face Hub."""
        dataset_dict = self._require_dataset()
        repo_id = repo_id or self.repo_id
        if not repo_id:
            raise ValueError("push_to_hub needs a repo_id (or set DataCollector.repo_id)")

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )
        dataset_dict.push_to_hub(
            repo_id,
            private=private,
            token=token,
            commit_message=commit_message or f"Collect {len(self.episodes)} PacManDT episodes",
        )
        if card:
            api.upload_file(
                path_or_fileobj=self.card().encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message="Add dataset card",
            )
        return f"https://huggingface.co/datasets/{repo_id}"

    def collect_and_push(self, num_episodes: int = 1000, **kwargs: Any) -> str:
        """``collect(...)`` followed by ``push_to_hub(...)``."""
        self.collect(num_episodes)
        return self.push_to_hub(**kwargs)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        *,
        split: Optional[str] = None,
        token: Optional[str] = None,
        **collector_kwargs: Any,
    ) -> "DataCollector":
        """Load a dataset previously pushed with :meth:`push_to_hub`."""
        from datasets import load_dataset

        loaded = load_dataset(repo_id, token=token)
        if isinstance(loaded, Dataset):
            dataset_dict = DatasetDict({"train": loaded})
        else:
            dataset_dict = loaded
        collector = cls(repo_id=repo_id, **collector_kwargs)
        collector.dataset_dict = dataset_dict
        first_split = split or "train"
        stats = dataset_statistics_from_dict(dataset_dict)
        collector.metadata = {
            "format": HF_DATASET_FORMAT,
            "repo_id": repo_id,
            "splits": list(dataset_dict.keys()),
            "episodes": stats.get("episodes", 0),
            "steps": stats.get("steps", 0),
            "max_length": int(dataset_dict[first_split].features["observations"].shape[0])
            if len(dataset_dict[first_split])
            else 0,
            "statistics": stats,
        }
        collector.obs_mode, collector.obs_config = _infer_obs_from_dataset(dataset_dict)
        collector.metadata["obs_mode"] = collector.obs_mode
        collector.metadata["observation"] = observation_metadata(
            collector.obs_mode, layout=collector.layout, config=collector.obs_config
        )
        return collector

    # -- helpers -------------------------------------------------------
    def _require_dataset(self) -> DatasetDict:
        if self.dataset_dict is None:
            raise RuntimeError("no dataset yet - call collect() or from_episodes() first")
        return self.dataset_dict

    @property
    def num_episodes(self) -> int:
        """Number of episodes held, whether they were collected or loaded."""
        if self.episodes:
            return len(self.episodes)
        if self.dataset_dict is not None:
            return sum(len(split) for split in self.dataset_dict.values())
        return 0

    def __len__(self) -> int:
        return self.num_episodes

    def __repr__(self) -> str:
        if self.dataset_dict is None:
            return (
                f"DataCollector(layout={self.layout!r}, policy={self.policy!r}, "
                f"episodes=<not collected>)"
            )
        splits = {name: len(split) for name, split in self.dataset_dict.items()}
        return (
            f"DataCollector(layout={self.layout!r}, policy={self.policy!r}, "
            f"episodes={self.num_episodes}, splits={splits})"
        )


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


__all__ = [
    "HF_DATASET_FORMAT",
    "DEFAULT_HF_DIR",
    "DataCollector",
    "DTSequenceDataset",
    "dataset_card",
    "dataset_statistics_from_dict",
    "dataset_to_episodes",
    "episode_features",
    "episodes_to_dataset",
    "episodes_to_dataset_dict",
    "flatten_to_transitions",
    "make_dt_dataloader",
    "split_episodes",
]
