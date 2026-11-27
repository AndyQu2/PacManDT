"""Ghost-aware observation space for Decision-Transformer PacMan.

Why this module exists
----------------------
``pacman_env`` ships a single-channel observation (``obs_type="state"``)::

    -1  wall *or* a normal ghost
     0  empty floor
     1  pellet (power pellets look identical)
     5  frightened *or* eaten ghost

Everything a ghost-aware policy needs is therefore missing or ambiguous:

* the **player is not encoded at all**;
* two ghosts standing on the same tile collapse into one ``-1``;
* a ghost is indistinguishable from a wall, and a frightened ghost from an
  eaten one (eyes travelling home);
* power pellets cannot be told apart from ordinary pellets;
* there is no notion of *distance* between Pac-Man and a ghost, so a flat
  ``state_dim`` embedding has to learn "who is close to whom" from absolute
  pixel coordinates alone - the hardest thing for a Decision Transformer's
  per-cell linear embedding to represent.

This module re-encodes the *same* simulation state into a small stack of
semantic planes **without touching** :mod:`pacman_env`: the wrapper reads the
live ``Game`` object the environment already exposes and writes the extra
information into additional channels.  ``state_dim`` grows from ``rows * cols``
to ``channels * rows * cols``, which the Decision Transformer absorbs through
its existing ``state_dim`` argument - the architecture is unchanged.

Channels
--------
Every channel is ``(rows, cols)`` and the stack is ``(channels, rows, cols)``,
so a flattened sample is channel-major: ``channel * rows * cols + row * cols + col``.

===========  ==========================================================
index name   meaning
===========  ==========================================================
0  wall               1 = impassable (wall, maze border, ghost door)
1  pellet             1 = pellet still on the tile
2  power_pellet       1 = power pellet still on the tile
3  ghost_normal       number of *normal* (dangerous) ghosts on the tile
4  ghost_vulnerable   number of frightened (edible) ghosts on the tile
5  ghost_eaten        number of eaten ghosts (eyes returning home)
6  player             1 = Pac-Man's tile
7  player_distance    BFS steps from Pac-Man, 0-100 (100 = unreachable)
8  ghost_distance     BFS steps to the nearest *normal* ghost, 0-100
9  frightened_timer   remaining frightened time, 0-100 (0 = none)
10 lives              lives left, 0-100
===========  ==========================================================

Channels ``0-6`` are the *core* set.  ``7-8`` are the distance fields and
``9-10`` the global scalars; both groups can be switched off
(:class:`GhostObservationConfig`) to trade information for a smaller
``state_dim`` and smaller datasets.

Usage
-----
Wrap an environment - at collection, training-rollout and evaluation time::

    from pacman_env import PacmanEnv
    from dt.data import make_ghost_env

    env = make_ghost_env(PacmanEnv(layout="classic"))
    obs, info = env.reset(seed=0)
    obs.shape                     # (11, 22, 19) int8
    obs[CH_GHOST_NORMAL].sum()    # 4 normal ghosts on the board

The wrapper is a plain :class:`gymnasium.ObservationWrapper`, so it composes
with the rest of the Gymnasium stack and leaves ``info`` (and therefore the
behaviour policies) untouched.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pacman_env.game.constants import (
    GHOST_DOOR,
    OUTSIDE,
    PELLET,
    POWER_PELLET,
    VULNERABLE_DURATION,
    WALL_END,
    WALL_START,
)
from pacman_env.game.ghost_state import GhostState
from pacman_env.game.map import Map

#: Observation mode of the untouched environment (single channel, legacy).
STATE_OBS_MODE = "state"
#: Observation mode produced by :class:`GhostObservationWrapper`.
GHOST_OBS_MODE = "ghost"
OBS_MODES: Tuple[str, ...] = (STATE_OBS_MODE, GHOST_OBS_MODE)

#: Values of the quantised channels are integers in ``[0, VALUE_SCALE]``; an
#: ``int8`` observation therefore stays compact while remaining readable.
VALUE_SCALE = 100

#: Channel names of the default (:data:`GHOST_OBS_MODE`) stack, in order.
CHANNELS: Tuple[str, ...] = (
    "wall",
    "pellet",
    "power_pellet",
    "ghost_normal",
    "ghost_vulnerable",
    "ghost_eaten",
    "player",
    "player_distance",
    "ghost_distance",
    "frightened_timer",
    "lives",
)

#: Channel indices for the default configuration (see :data:`CHANNELS`).
CH_WALL = 0
CH_PELLET = 1
CH_POWER_PELLET = 2
CH_GHOST_NORMAL = 3
CH_GHOST_VULNERABLE = 4
CH_GHOST_EATEN = 5
CH_PLAYER = 6
CH_PLAYER_DISTANCE = 7
CH_GHOST_DISTANCE = 8
CH_FRIGHTENED_TIMER = 9
CH_LIVES = 10

_CORE_CHANNELS: Tuple[str, ...] = CHANNELS[:7]
_DISTANCE_CHANNELS: Tuple[str, ...] = CHANNELS[7:9]
_GLOBAL_CHANNELS: Tuple[str, ...] = CHANNELS[9:]

_CHANNEL_DOC = {
    "wall": "1 = impassable (wall, border, ghost door)",
    "pellet": "1 = pellet still on the tile",
    "power_pellet": "1 = power pellet still on the tile",
    "ghost_normal": "number of normal (dangerous) ghosts",
    "ghost_vulnerable": "number of frightened (edible) ghosts",
    "ghost_eaten": "number of eaten ghosts (eyes returning home)",
    "player": "1 = Pac-Man's tile",
    "player_distance": "BFS steps from Pac-Man (0-100, 100 = unreachable)",
    "ghost_distance": "BFS steps to the nearest normal ghost (0-100)",
    "frightened_timer": "remaining frightened time (0-100)",
    "lives": "lives left (0-100)",
}


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class GhostObservationConfig:
    """Which channels the ghost-aware observation carries.

    ``distance_fields``
        Add ``player_distance``/``ghost_distance`` - two BFS potential fields
        that make *relative* geometry (who is close to whom) linear for the
        transformer.  They cost two channels and two BFS passes per step.
    ``globals``
        Add the broadcast scalars ``frightened_timer`` and ``lives``.
    ``max_distance``
        BFS distance at which a tile counts as "far away"; distances are
        quantised into ``max_distance + 1`` buckets scaled to ``0-100``.
    ``dtype``
        ``"int8"`` (default, 0-100 integers) or ``"float32"`` (0-1 floats).
    """

    distance_fields: bool = True
    globals: bool = True
    max_distance: int = 40
    dtype: str = "int8"

    def __post_init__(self) -> None:
        if int(self.max_distance) <= 0:
            raise ValueError("max_distance must be a positive integer")
        if str(self.dtype) not in ("int8", "float32"):
            raise ValueError("dtype must be 'int8' or 'float32'")

    # -- channel layout ------------------------------------------------
    @property
    def channel_names(self) -> Tuple[str, ...]:
        """Channel names in the order they appear in an observation."""
        names = list(_CORE_CHANNELS)
        if self.distance_fields:
            names.extend(_DISTANCE_CHANNELS)
        if self.globals:
            names.extend(_GLOBAL_CHANNELS)
        return tuple(names)

    @property
    def num_channels(self) -> int:
        return len(self.channel_names)

    def channel_index(self) -> Dict[str, int]:
        """Mapping ``channel name -> index`` for this configuration."""
        return {name: index for index, name in enumerate(self.channel_names)}

    # -- (de)serialisation ---------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        return {
            "distance_fields": bool(self.distance_fields),
            "globals": bool(self.globals),
            "max_distance": int(self.max_distance),
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "GhostObservationConfig":
        """Rebuild a config from :meth:`as_dict` output (missing keys = default)."""
        data = dict(data or {})
        return cls(
            distance_fields=bool(data.get("distance_fields", True)),
            globals=bool(data.get("globals", True)),
            max_distance=int(data.get("max_distance", 40)),
            dtype=str(data.get("dtype", "int8")),
        )

    def resolve(self, layout: str = "classic") -> "ResolvedObservation":
        """Attach the maze geometry of ``layout`` to this configuration."""
        rows, cols = _layout_shape(layout)
        return ResolvedObservation(
            mode=GHOST_OBS_MODE,
            layout=str(layout),
            rows=rows,
            cols=cols,
            channels=self.channel_names,
            dtype=str(self.dtype),
            max_distance=int(self.max_distance),
            config=self,
        )


@dataclass(frozen=True)
class ResolvedObservation:
    """A concrete observation layout: mode + maze size + channels."""

    mode: str
    layout: str
    rows: int
    cols: int
    channels: Tuple[str, ...]
    dtype: str = "int8"
    max_distance: int = 40
    config: Optional[GhostObservationConfig] = None

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        """Shape of one stored observation (channel-first)."""
        return (len(self.channels), int(self.rows), int(self.cols))

    @property
    def feature_dim(self) -> int:
        """Number of scalars once flattened - the DT's ``state_dim``."""
        return int(np.prod(self.obs_shape))

    def as_dict(self) -> Dict[str, Any]:
        """JSON-friendly description, stored in datasets and checkpoints."""
        payload: Dict[str, Any] = {
            "mode": str(self.mode),
            "layout": str(self.layout),
            "rows": int(self.rows),
            "cols": int(self.cols),
            "channels": list(self.channels),
            "obs_shape": list(self.obs_shape),
            "feature_dim": int(self.feature_dim),
            "dtype": str(self.dtype),
        }
        if self.config is not None:
            payload.update(self.config.as_dict())
        return payload

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ResolvedObservation(mode={self.mode!r}, layout={self.layout!r}, "
            f"obs_shape={self.obs_shape}, state_dim={self.feature_dim})"
        )


def _layout_shape(layout: str) -> Tuple[int, int]:
    rows, cols = Map(layout).shape
    return int(rows), int(cols)


def resolve_observation(
    obs_mode: str = STATE_OBS_MODE,
    *,
    layout: str = "classic",
    config: Optional[GhostObservationConfig] = None,
) -> ResolvedObservation:
    """Resolve ``obs_mode``/``config`` into a concrete observation layout.

    ``obs_mode="state"`` keeps the environment's own single-channel grid, so
    datasets collected before the ghost channels existed stay readable.
    """
    mode = _normalise_mode(obs_mode)
    if mode == STATE_OBS_MODE:
        rows, cols = _layout_shape(layout)
        return ResolvedObservation(
            mode=STATE_OBS_MODE,
            layout=str(layout),
            rows=rows,
            cols=cols,
            channels=("state",),
            dtype="int8",
        )
    return (config or GhostObservationConfig()).resolve(layout)


def _normalise_mode(obs_mode: Optional[str]) -> str:
    if obs_mode is None:
        return STATE_OBS_MODE
    mode = str(obs_mode).lower().replace("-", "_")
    if mode in ("", "none", STATE_OBS_MODE, "env"):
        return STATE_OBS_MODE
    if mode in (GHOST_OBS_MODE, "ghosts", "ghost_aware"):
        return GHOST_OBS_MODE
    raise ValueError(f"Unknown observation mode {obs_mode!r}. Use one of {OBS_MODES}.")


def observation_shape(
    obs_mode: str = STATE_OBS_MODE,
    *,
    layout: str = "classic",
    config: Optional[GhostObservationConfig] = None,
) -> Tuple[int, ...]:
    """Shape of one observation for ``obs_mode`` on ``layout``."""
    return resolve_observation(obs_mode, layout=layout, config=config).obs_shape


def observation_feature_dim(
    obs_mode: str = STATE_OBS_MODE,
    *,
    layout: str = "classic",
    config: Optional[GhostObservationConfig] = None,
) -> int:
    """Flattened observation size (``state_dim``) for ``obs_mode``."""
    return resolve_observation(obs_mode, layout=layout, config=config).feature_dim


def observation_from_metadata(
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Optional[GhostObservationConfig]]:
    """Read ``(obs_mode, config)`` from dataset/checkpoint metadata.

    Understands both ``{"obs_mode": "ghost", "observation": {...}}`` (the
    dataset and checkpoint layout) and a bare ``{"mode": "ghost", ...}`` block.
    Missing information means the legacy single-channel observation.
    """
    metadata = dict(metadata or {})
    block = metadata.get("observation")
    if not isinstance(block, Mapping):
        block = metadata if "mode" in metadata else {}
    mode = metadata.get("obs_mode", block.get("mode", STATE_OBS_MODE))
    mode = _normalise_mode(mode)
    if mode == STATE_OBS_MODE:
        return STATE_OBS_MODE, None
    return mode, GhostObservationConfig.from_dict(block)


def observation_metadata(
    obs_mode: str = STATE_OBS_MODE,
    obs_shape: Optional[Sequence[int]] = None,
    config: Optional[GhostObservationConfig] = None,
    *,
    dtype: Optional[str] = None,
    layout: Optional[str] = None,
) -> Dict[str, Any]:
    """Describe an observation layout, optionally derived from a maze layout.

    Pass ``obs_shape`` when the truth is already on disk (a padded
    ``observations`` array or a dataset feature), or ``layout`` to resolve the
    shape from the maze name.  The result is JSON-friendly and round-trips
    through :func:`observation_from_metadata`.
    """
    mode = _normalise_mode(obs_mode)
    config = (config or GhostObservationConfig()) if mode == GHOST_OBS_MODE else None
    channels = ("state",) if config is None else config.channel_names
    payload: Dict[str, Any] = {
        "mode": mode,
        "channels": list(channels),
        "dtype": str(dtype or (config.dtype if config is not None else "int8")),
    }
    if obs_shape is None and layout is not None:
        obs_shape = resolve_observation(mode, layout=layout, config=config).obs_shape
    if layout is not None:
        payload["layout"] = str(layout)
    if obs_shape is not None:
        shape = tuple(int(dim) for dim in obs_shape)
        payload["obs_shape"] = list(shape)
        payload["feature_dim"] = int(np.prod(shape))
        if len(shape) == 3:
            payload["rows"], payload["cols"] = shape[1], shape[2]
    if config is not None:
        payload.update(config.as_dict())
    return payload


def describe_observation(
    obs_mode: str = GHOST_OBS_MODE,
    *,
    layout: str = "classic",
    config: Optional[GhostObservationConfig] = None,
) -> str:
    """Human-readable table of the channels and the resulting ``state_dim``."""
    resolved = resolve_observation(obs_mode, layout=layout, config=config)
    header = (
        f"{'idx':>4}  {'channel':<18}{'description'}"
    )
    lines = [
        f"observation mode {resolved.mode!r} on layout {resolved.layout!r}",
        f"obs_shape {resolved.obs_shape} {resolved.dtype} -> "
        f"state_dim {resolved.feature_dim}",
        header,
        "-" * len(header),
    ]
    for index, name in enumerate(resolved.channels):
        lines.append(f"{index:>4}  {name:<18}{_CHANNEL_DOC.get(name, '')}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Observation builder
# ----------------------------------------------------------------------
class GhostObservationBuilder:
    """Turns a live :class:`pacman_env.game.game.Game` into a channel stack.

    The maze geometry (walls, corridors) is read once and turned into a
    neighbour table, so the two BFS passes needed per step cost a few hundred
    pointer hops instead of rebuilding the graph.
    """

    def __init__(self, maze: Map, config: Optional[GhostObservationConfig] = None):
        self.config = config or GhostObservationConfig()
        self.rows, self.cols = int(maze.shape[0]), int(maze.shape[1])
        self.channels = self.config.channel_names
        self.channel_index = {name: index for index, name in enumerate(self.channels)}
        self.obs_shape: Tuple[int, int, int] = (len(self.channels), self.rows, self.cols)

        self._wall = _wall_mask(maze)
        self._flat_passable = (~self._wall).reshape(-1)
        self._neighbours = _build_neighbour_table(self._wall)

    # -- encoding ------------------------------------------------------
    def build(self, game: Any) -> np.ndarray:
        """Encode ``game`` as ``(channels, rows, cols)``."""
        index = self.channel_index
        rows, cols = self.rows, self.cols
        obs = np.zeros(self.obs_shape, dtype=np.int8)

        layout = game.maze.map_matrix
        if "wall" in index:
            obs[index["wall"]] = self._wall
        if "pellet" in index:
            obs[index["pellet"]] = layout == PELLET
        if "power_pellet" in index:
            obs[index["power_pellet"]] = layout == POWER_PELLET

        normal_sources = []
        for ghost in game.ghosts:
            row, col = int(ghost.nearest_row), int(ghost.nearest_col)
            if not (0 <= row < rows and 0 <= col < cols):
                continue
            if ghost.state == GhostState.VULNERABLE:
                channel = "ghost_vulnerable"
            elif ghost.state == GhostState.EATEN:
                channel = "ghost_eaten"
            else:
                channel = "ghost_normal"
                normal_sources.append(row * cols + col)
            if channel in index:
                obs[index[channel], row, col] += 1

        player = game.player
        player_row, player_col = int(player.nearest_row), int(player.nearest_col)
        player_inside = 0 <= player_row < rows and 0 <= player_col < cols
        if "player" in index and player_inside:
            obs[index["player"], player_row, player_col] = 1

        if "player_distance" in index:
            if player_inside:
                distances = self._bfs([player_row * cols + player_col])
            else:  # pragma: no cover - the player is always on the board
                distances = np.full((rows, cols), -1, dtype=np.int32)
            obs[index["player_distance"]] = self._quantise(distances)

        if "ghost_distance" in index:
            distances = self._bfs(normal_sources) if normal_sources else None
            obs[index["ghost_distance"]] = (
                self._quantise(distances) if distances is not None else VALUE_SCALE
            )

        if "frightened_timer" in index:
            timer = int(getattr(game, "ghosts_timer", 0))
            frightened = any(
                ghost.state == GhostState.VULNERABLE for ghost in game.ghosts
            )
            remaining = max(0, VULNERABLE_DURATION - timer) if frightened else 0
            obs[index["frightened_timer"]] = (
                remaining * VALUE_SCALE
            ) // max(1, VULNERABLE_DURATION)

        if "lives" in index:
            obs[index["lives"]] = min(VALUE_SCALE, int(player.lives))

        if self.config.dtype == "float32":
            return obs.astype(np.float32) / VALUE_SCALE
        return obs

    # -- BFS helpers ---------------------------------------------------
    def _bfs(self, sources: Iterable[int]) -> np.ndarray:
        """4-connected BFS distance (in tiles) from ``sources``, ``-1`` if unreachable."""
        total = self.rows * self.cols
        distances = np.full(total, -1, dtype=np.int32)
        queue: deque = deque()
        flat_passable = self._flat_passable
        for source in sources:
            if 0 <= source < total and flat_passable[source] and distances[source] < 0:
                distances[source] = 0
                queue.append(source)
        neighbours = self._neighbours
        while queue:
            node = queue.popleft()
            next_distance = distances[node] + 1
            for neighbour in neighbours[node]:
                if distances[neighbour] < 0:
                    distances[neighbour] = next_distance
                    queue.append(neighbour)
        return distances.reshape(self.rows, self.cols)

    def _quantise(self, distances: np.ndarray) -> np.ndarray:
        """Map BFS distances to ``0..VALUE_SCALE`` (unreachable = far)."""
        limit = int(self.config.max_distance)
        clipped = np.where(distances < 0, limit, np.minimum(distances, limit))
        return ((clipped * VALUE_SCALE) // max(1, limit)).astype(np.int8)


def _wall_mask(maze: Map) -> np.ndarray:
    """Impassable tiles according to the layout matrix (ghosts are *not* walls)."""
    layout = maze.map_matrix
    return (
        ((layout >= WALL_START) & (layout <= WALL_END))
        | (layout == OUTSIDE)
        | (layout == GHOST_DOOR)
    )


def _build_neighbour_table(wall: np.ndarray) -> Tuple[Tuple[int, ...], ...]:
    """Adjacency lists (flat indices) of the 4-connected corridor graph."""
    rows, cols = wall.shape
    passable = ~wall
    table: list = [()] * (rows * cols)
    for row in range(rows):
        for col in range(cols):
            if not passable[row, col]:
                continue
            neighbours = []
            if row > 0 and passable[row - 1, col]:
                neighbours.append((row - 1) * cols + col)
            if row + 1 < rows and passable[row + 1, col]:
                neighbours.append((row + 1) * cols + col)
            if col > 0 and passable[row, col - 1]:
                neighbours.append(row * cols + col - 1)
            if col + 1 < cols and passable[row, col + 1]:
                neighbours.append(row * cols + col + 1)
            table[row * cols + col] = tuple(neighbours)
    return tuple(table)


# ----------------------------------------------------------------------
# Gymnasium wrapper
# ----------------------------------------------------------------------
class GhostObservationWrapper(gym.ObservationWrapper):
    """Replaces the single-channel grid with the ghost-aware channel stack.

    The wrapper keeps reading the *environment's* observation for nothing but
    the reset/step plumbing - the channels are built from the live ``Game``
    object, which is the same source the environment's ``info`` dict uses.
    ``info``, rewards, actions and the game itself are untouched.
    """

    def __init__(self, env: gym.Env, config: Optional[GhostObservationConfig] = None):
        super().__init__(env)
        self.observation_config = config or GhostObservationConfig()
        game = env.unwrapped.game  # type: ignore[attr-defined]
        self._builder = GhostObservationBuilder(game.maze, self.observation_config)
        self.obs_mode = GHOST_OBS_MODE
        high = VALUE_SCALE if self.observation_config.dtype == "int8" else 1.0
        self.observation_space = spaces.Box(
            low=0,
            high=high,
            shape=self._builder.obs_shape,
            dtype=np.dtype(self.observation_config.dtype),
        )

    # -- Gymnasium API -------------------------------------------------
    def observation(self, observation: np.ndarray) -> np.ndarray:  # noqa: ARG002
        return self._builder.build(self.env.unwrapped.game)  # type: ignore[attr-defined]

    # -- introspection -------------------------------------------------
    @property
    def layout(self) -> str:
        """Maze layout of the wrapped environment (proxied for the collectors)."""
        return str(getattr(self.env, "layout", "unknown"))

    @property
    def resolved_observation(self) -> ResolvedObservation:
        return self.observation_config.resolve(self.layout)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"GhostObservationWrapper(layout={getattr(self.env, 'layout', '?')!r}, "
            f"obs_shape={self._builder.obs_shape}, "
            f"channels={len(self._builder.channels)})"
        )


def make_ghost_env(
    env: gym.Env, config: Optional[GhostObservationConfig] = None
) -> GhostObservationWrapper:
    """Wrap ``env`` so it returns the ghost-aware observation."""
    return GhostObservationWrapper(env, config)


def wrap_observation(
    env: gym.Env,
    obs_mode: str = STATE_OBS_MODE,
    config: Optional[GhostObservationConfig] = None,
) -> gym.Env:
    """Wrap ``env`` for ``obs_mode`` (``"state"`` returns ``env`` unchanged)."""
    if _normalise_mode(obs_mode) == GHOST_OBS_MODE:
        return make_ghost_env(env, config)
    return env


__all__ = [
    "CHANNELS",
    "CH_FRIGHTENED_TIMER",
    "CH_GHOST_DISTANCE",
    "CH_GHOST_EATEN",
    "CH_GHOST_NORMAL",
    "CH_GHOST_VULNERABLE",
    "CH_LIVES",
    "CH_PELLET",
    "CH_PLAYER",
    "CH_PLAYER_DISTANCE",
    "CH_POWER_PELLET",
    "CH_WALL",
    "GHOST_OBS_MODE",
    "OBS_MODES",
    "STATE_OBS_MODE",
    "VALUE_SCALE",
    "GhostObservationBuilder",
    "GhostObservationConfig",
    "GhostObservationWrapper",
    "ResolvedObservation",
    "describe_observation",
    "make_ghost_env",
    "observation_feature_dim",
    "observation_from_metadata",
    "observation_metadata",
    "observation_shape",
    "resolve_observation",
    "wrap_observation",
]
