"""Observation/action dimensions and episode-length guidance for DT training.

A Decision Transformer needs three numbers before the first forward pass:

``state_dim``
    Number of scalars in one observation (``rows * cols`` for the ``state``
    observation, or ``H * W * 3`` for ``rgb``).
``act_dim``
    Number of discrete actions (``5``: left, right, up, down, no-op).
``max_length``
    Time axis every episode is padded (or truncated) to inside the dataset.

This module derives the first two from the environment, from a ``.npz``
archive or from a Hugging Face dataset, and recommends a ``max_length``
(see :func:`recommend_max_length`).

Quick start::

    from dt.data import get_dims, get_state_dim, get_act_dim

    get_state_dim("classic")                 # 418  (22 * 19)
    get_act_dim()                            # 5
    get_dims("classic", flatten=True)        # DTDims(state_dim=418, act_dim=5, ...)

    # or from data you already collected
    dims = dims_from_arrays(arrays)          # arrays = pad_episodes(...) / load_dataset(...)
    dims = dims_from_dataset(collector.dataset_dict)

Running ``python -m dt.data.dims`` prints the table for every layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from pacman_env.env import RGB_OBS, STATE_OBS
from pacman_env.game.action import Action
from pacman_env.game.map import Map

#: Context length (K) recommended for DT training.  The original paper uses 20
#: for D4RL locomotion and 30 for Atari; PacMan episodes are long enough that
#: 30 gives the model a useful history without inflating the batch.
DEFAULT_CONTEXT_LENGTH = 30

#: ``max_length`` that loses < 5 % of the steps, measured on 30 heuristic
#: episodes per layout with ``max_steps=1000``.  ``classic``/``google``/
#: ``medium`` are truncation-bound, so they need the full time limit; the two
#: small layouts always finish well before it.
RECOMMENDED_MAX_LENGTH: Dict[str, int] = {
    "classic": 1000,
    "google": 1000,
    "medium": 1000,
    "medium_small": 576,
    "small": 256,
}

#: Round ``max_length`` up to a multiple of this so the time axis stays aligned
#: for vectorised/blocked kernels.
MAX_LENGTH_ROUNDING = 32


@dataclass(frozen=True)
class DTDims:
    """``state_dim`` / ``act_dim`` plus the observation layout they came from."""

    state_dim: int
    act_dim: int
    obs_shape: Tuple[int, ...]
    obs_dtype: str = "int8"
    layout: str = "unknown"
    flatten: bool = True
    frame_stack: int = 1
    obs_type: str = STATE_OBS

    @property
    def state_shape(self) -> Tuple[int, ...]:
        """Shape of one stored observation (``(state_dim,)`` when flattened)."""
        return self.obs_shape

    def as_dict(self) -> Dict[str, Any]:
        """Plain ``dict`` — handy for configs, JSON and logging."""
        return {
            "state_dim": self.state_dim,
            "act_dim": self.act_dim,
            "obs_shape": list(self.obs_shape),
            "obs_dtype": self.obs_dtype,
            "layout": self.layout,
            "flatten": self.flatten,
            "frame_stack": self.frame_stack,
            "obs_type": self.obs_type,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DTDims(layout={self.layout!r}, state_dim={self.state_dim}, "
            f"act_dim={self.act_dim}, obs_shape={self.obs_shape}, "
            f"obs_dtype={self.obs_dtype!r})"
        )


# ----------------------------------------------------------------------
# Environment dimensions
# ----------------------------------------------------------------------
def get_obs_shape(
    layout: str = "classic",
    *,
    obs_type: str = STATE_OBS,
    frame_stack: int = 1,
) -> Tuple[int, ...]:
    """Shape of a single observation, exactly as :class:`PacmanEnv` returns it.

    ``obs_type="state"`` gives ``(rows, cols)`` (``int8``); ``obs_type="rgb"``
    gives ``(height, width, 3)`` (``uint8``).  ``frame_stack > 1`` prepends the
    stacking axis, matching the usual ``FrameStack`` wrapper.
    """
    if frame_stack <= 0:
        raise ValueError("frame_stack must be a positive integer")

    maze = Map(layout)
    rows, cols = maze.shape
    if obs_type == STATE_OBS:
        base: Tuple[int, ...] = (rows, cols)
    elif obs_type == RGB_OBS:
        width, height = maze.get_map_sizes()
        base = (height, width, 3)
    else:
        raise ValueError(
            f"Unsupported obs_type {obs_type!r}. Use {STATE_OBS!r} or {RGB_OBS!r}."
        )
    return (frame_stack, *base) if frame_stack > 1 else base


def get_state_dim(
    layout: str = "classic",
    *,
    obs_type: str = STATE_OBS,
    frame_stack: int = 1,
) -> int:
    """Number of scalars in one observation — the DT input width.

    This is what you pass to a Decision Transformer as ``state_dim``::

        get_state_dim("classic")            # 418  (22 * 19)
        get_state_dim("small")              # 42   (7 * 6)
        get_state_dim("classic", frame_stack=4)   # 1672
    """
    return int(np.prod(get_obs_shape(layout, obs_type=obs_type, frame_stack=frame_stack)))


def get_act_dim(layout: Optional[str] = None) -> int:
    """Size of the discrete action space — the DT output width.

    Always ``5`` (left, right, up, down, no-op) regardless of the layout;
    ``layout`` is accepted so call sites can stay uniform.
    """
    return len(Action)


def get_dims(
    layout: str = "classic",
    *,
    flatten: bool = True,
    obs_type: str = STATE_OBS,
    frame_stack: int = 1,
    obs_dtype: str = "int8",
) -> "DTDims":
    """All DT shape information for ``layout`` in one object.

    ``flatten=True`` (the usual choice for a DT, which embeds a flat vector)
    reports ``obs_shape == (state_dim,)``; ``flatten=False`` keeps the raw
    ``(rows, cols)`` shape for CNN encoders.
    """
    raw_shape = get_obs_shape(layout, obs_type=obs_type, frame_stack=frame_stack)
    state_dim = int(np.prod(raw_shape))
    return DTDims(
        state_dim=state_dim,
        act_dim=get_act_dim(layout),
        obs_shape=(state_dim,) if flatten else raw_shape,
        obs_dtype=str(obs_dtype),
        layout=str(layout),
        flatten=bool(flatten),
        frame_stack=int(frame_stack),
        obs_type=str(obs_type),
    )


# ----------------------------------------------------------------------
# Dimensions from live objects / datasets
# ----------------------------------------------------------------------
def _space_shape(space: Any) -> Tuple[int, ...]:
    shape = getattr(space, "shape", None)
    if not shape:
        raise TypeError(f"cannot derive an observation shape from {space!r}")
    return tuple(int(dim) for dim in shape)


def _space_size(space: Any) -> int:
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return int(np.prod(space.nvec))
    return int(np.prod(_space_shape(space)))


def _env_layout(env: Any) -> str:
    for candidate in (env, getattr(env, "unwrapped", env)):
        layout = getattr(candidate, "layout", None)
        if isinstance(layout, str):
            return layout
    return "unknown"


def dims_from_env(env: Any) -> DTDims:
    """Read ``state_dim``/``act_dim`` from a (possibly wrapped) Gymnasium env.

    Works with the plain :class:`PacmanEnv`, with
    :func:`~pacman_env.wrappers.make_flat_env` (flat ``float32`` ``Box``) and
    with vector envs (``single_observation_space`` / ``single_action_space``).
    """
    obs_space = getattr(env, "single_observation_space", None)
    if obs_space is None:
        obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "single_action_space", None)
    if act_space is None:
        act_space = getattr(env, "action_space", None)
    if obs_space is None or act_space is None:
        raise TypeError(f"{type(env).__name__} does not expose Gymnasium spaces")

    obs_shape = _space_shape(obs_space)
    state_dim = int(np.prod(obs_shape))
    return DTDims(
        state_dim=state_dim,
        act_dim=_space_size(act_space),
        obs_shape=obs_shape,
        obs_dtype=str(getattr(obs_space, "dtype", "int8")),
        layout=_env_layout(env),
        flatten=obs_shape == (state_dim,),
    )


def dims_from_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    act_dim: Optional[int] = None,
    layout: Optional[str] = None,
) -> DTDims:
    """Read the observation shape from a padded ``.npz``-style array mapping.

    ``arrays`` is whatever :func:`dt.data.pad_episodes` returned or
    :func:`dt.data.load_dataset` loaded: ``observations`` has shape
    ``(episodes, time, *obs_shape)``.  ``act_dim`` cannot be recovered from the
    data reliably (an unused action leaves no trace), so it defaults to the
    environment action count.
    """
    observations = arrays.get("observations")
    if observations is None:
        raise KeyError("arrays must contain an 'observations' entry")
    observations = np.asarray(observations)
    if observations.ndim < 3:
        raise ValueError(
            "observations must be (episodes, time, ...), got shape "
            f"{observations.shape}"
        )
    obs_shape = tuple(int(dim) for dim in observations.shape[2:])
    state_dim = int(np.prod(obs_shape))
    return DTDims(
        state_dim=state_dim,
        act_dim=get_act_dim() if act_dim is None else int(act_dim),
        obs_shape=obs_shape,
        obs_dtype=str(observations.dtype),
        layout=layout or "unknown",
        flatten=obs_shape == (state_dim,),
    )


def dims_from_episodes(
    episodes: Sequence[Any],
    *,
    act_dim: Optional[int] = None,
    layout: Optional[str] = None,
) -> DTDims:
    """Same as :func:`dims_from_arrays` but for a list of ``Episode`` objects."""
    if not episodes:
        raise ValueError("cannot derive dimensions from an empty list of episodes")
    obs_shape = tuple(int(dim) for dim in np.asarray(episodes[0].observations).shape[1:])
    state_dim = int(np.prod(obs_shape))
    return DTDims(
        state_dim=state_dim,
        act_dim=get_act_dim() if act_dim is None else int(act_dim),
        obs_shape=obs_shape,
        obs_dtype=str(np.asarray(episodes[0].observations).dtype),
        layout=layout or "unknown",
        flatten=obs_shape == (state_dim,),
    )


def dims_from_dataset(
    dataset: Any,
    *,
    act_dim: Optional[int] = None,
    layout: Optional[str] = None,
) -> DTDims:
    """Read the observation shape from a HF ``Dataset`` or ``DatasetDict``.

    Uses the ``observations`` feature (``Array3D(max_length, *obs_shape)``), so
    it works before any row is materialised.
    """
    features = getattr(dataset, "features", None)
    if features is None:
        splits = getattr(dataset, "values", None)
        if splits is None:
            raise TypeError(f"{type(dataset).__name__} is not a Dataset/DatasetDict")
        features = next(iter(splits())).features
    if "observations" not in features:
        raise KeyError("dataset has no 'observations' column")

    obs_feature = features["observations"]
    obs_shape = tuple(int(dim) for dim in obs_feature.shape[1:])
    state_dim = int(np.prod(obs_shape))
    return DTDims(
        state_dim=state_dim,
        act_dim=get_act_dim() if act_dim is None else int(act_dim),
        obs_shape=obs_shape,
        obs_dtype=str(getattr(obs_feature, "dtype", "int8")),
        layout=layout or "unknown",
        flatten=obs_shape == (state_dim,),
    )


# ----------------------------------------------------------------------
# max_length guidance
# ----------------------------------------------------------------------
def episode_length_summary(lengths: Iterable[int]) -> Dict[str, float]:
    """``count``/``mean``/``std``/``min``/percentiles/``max`` of episode lengths."""
    values = np.asarray([int(length) for length in lengths], dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarise an empty list of episode lengths")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def recommend_max_length(
    lengths: Optional[Iterable[int]] = None,
    *,
    layout: Optional[str] = None,
    max_steps: int = 1000,
    percentile: float = 95.0,
    round_to: int = MAX_LENGTH_ROUNDING,
    cap: Optional[int] = None,
) -> int:
    """Recommend the ``max_length`` to pad/truncate episodes to.

    * ``lengths`` given → round the ``percentile``-th episode length up to the
      next multiple of ``round_to`` (default 32).  This keeps almost every
      episode intact while cutting the long tail of padding.
    * ``lengths`` omitted → use the measured per-layout table
      (:data:`RECOMMENDED_MAX_LENGTH`), clamped to ``max_steps``.

    The result is clamped to ``cap`` (or ``max_steps`` when ``cap`` is None).
    """
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if round_to <= 0:
        raise ValueError("round_to must be a positive integer")
    if cap is None and max_steps is not None and max_steps > 0:
        cap = int(max_steps)

    if lengths is not None:
        summary = episode_length_summary(lengths)
        target = summary[f"p{int(percentile)}"] if percentile in (50, 90, 95, 99) else float(
            np.percentile(list(lengths), percentile)
        )
        value = int(np.ceil(target / round_to) * round_to)
        value = max(value, 1)
    elif layout is not None and layout in RECOMMENDED_MAX_LENGTH:
        value = int(RECOMMENDED_MAX_LENGTH[layout])
    else:
        value = int(max_steps)

    if cap is not None:
        value = min(value, int(cap))
    return max(1, value)


def recommend_context_length(
    max_length: Optional[int] = None,
    *,
    default: int = DEFAULT_CONTEXT_LENGTH,
    minimum: int = 2,
) -> int:
    """Recommend the DT training window ``K`` (``context_length``).

    ``K = 30`` matches the Atari experiments of the Decision Transformer paper.
    It is clamped to ``max_length`` so a short dataset cannot produce windows
    longer than an episode.
    """
    if default < minimum:
        raise ValueError("default must be >= minimum")
    if max_length is None:
        return int(default)
    return max(int(minimum), min(int(default), int(max_length)))


def describe_layouts(layouts: Optional[Sequence[str]] = None) -> str:
    """Human-readable table of every layout's dims and recommended ``max_length``."""
    names = list(layouts) if layouts is not None else Map.available_layouts()
    header = f"{'layout':<13}{'obs_shape':<15}{'state_dim':>10}{'act_dim':>9}{'max_length':>12}"
    lines = [header, "-" * len(header)]
    for name in names:
        shape = get_obs_shape(name)
        lines.append(
            f"{name:<13}{str(shape):<15}{get_state_dim(name):>10}{get_act_dim():>9}"
            f"{recommend_max_length(layout=name):>12}"
        )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_CONTEXT_LENGTH",
    "MAX_LENGTH_ROUNDING",
    "RECOMMENDED_MAX_LENGTH",
    "RGB_OBS",
    "STATE_OBS",
    "DTDims",
    "describe_layouts",
    "dims_from_arrays",
    "dims_from_dataset",
    "dims_from_env",
    "dims_from_episodes",
    "episode_length_summary",
    "get_act_dim",
    "get_dims",
    "get_obs_shape",
    "get_state_dim",
    "recommend_context_length",
    "recommend_max_length",
]
