"""A Gymnasium-compatible Pac-Man environment.

The environment wraps the Pygame-free simulation in
:mod:`pacman_env.game` and only imports Pygame when a rendering mode (or a
pixel observation) is requested.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game.action import Action
from .game.constants import (
    GHOST_DOOR,
    PELLET,
    POWER_PELLET,
    WALL_END,
    WALL_START,
)
from .game.game import Game
from .game.ghost_state import GhostState

STATE_OBS = "state"
RGB_OBS = "rgb"

ANSI_CHARS = {
    "wall": "#",
    "door": "=",
    "pellet": ".",
    "power_pellet": "o",
    "empty": " ",
    "player": "P",
    "ghost": "G",
    "frightened_ghost": "g",
}


class PacmanEnv(gym.Env):
    """Pac-Man for reinforcement learning.

    Parameters
    ----------
    layout:
        Name of a ``.lay`` file shipped in ``pacman_env/assets/layouts``
        (``"classic"``, ``"google"``, ``"medium"``, ``"medium_small"``,
        ``"small"``).
    render_mode:
        ``None`` (default) runs headless and never imports Pygame.
        ``"human"`` opens a window, ``"rgb_array"`` renders off-screen and
        :meth:`render` returns the frame, ``"ansi"`` returns a text drawing of
        the maze (Pygame is not imported).
    obs_type:
        ``"state"`` (default) returns the ``(rows, cols)`` integer tile matrix.
        ``"rgb"`` returns the rendered frame (implies Pygame is initialised).
    player_lives:
        Number of lives per episode.
    max_steps:
        Optional internal episode length limit; reaching it yields
        ``truncated=True``.  Defaults to ``None`` (no internal limit) because
        :func:`gymnasium.make` already applies a ``TimeLimit`` wrapper through
        the registered ``max_episode_steps``.
    frame_skip:
        Number of simulation ticks executed per :meth:`step` call.
    draw_state:
        Draw the observation matrix next to the maze (human/rgb modes only).

    Actions
    -------
    ``0`` left, ``1`` right, ``2`` up, ``3`` down, ``4`` no-op (brake).
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 30}

    def __init__(
        self,
        layout: str = "classic",
        render_mode: Optional[str] = None,
        obs_type: str = STATE_OBS,
        player_lives: int = 3,
        max_steps: Optional[int] = None,
        frame_skip: int = 1,
        draw_state: bool = False,
        render_fps: int = 30,
    ):
        super().__init__()

        if render_mode not in (None, "human", "rgb_array", "ansi"):
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}. "
                f"Choose from {self.metadata['render_modes']} or None."
            )
        if obs_type not in (STATE_OBS, RGB_OBS):
            raise ValueError(f"Unsupported obs_type {obs_type!r}. Use {STATE_OBS!r} or {RGB_OBS!r}.")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be a positive integer or None")
        if frame_skip <= 0:
            raise ValueError("frame_skip must be a positive integer")

        self.layout = layout
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.player_lives = int(player_lives)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.frame_skip = int(frame_skip)
        self.draw_state = bool(draw_state)

        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = int(render_fps)
        self.render_fps = int(render_fps)

        self.game = Game(layout=layout, player_lives=self.player_lives)
        self.action_space = spaces.Discrete(len(Action))

        rows, cols = self.game.maze.shape
        if obs_type == STATE_OBS:
            self.observation_space = spaces.Box(low=-1, high=5, shape=(rows, cols), dtype=np.int8)
        else:
            width, height = self.game.maze.get_map_sizes()
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(height, width, 3), dtype=np.uint8)

        self._renderer = None
        self._steps = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            # Keep ``action_space.sample()`` reproducible for a given seed.
            self.action_space.seed(seed)

        self.game.rng = self.np_random
        for ghost in self.game.ghosts:
            ghost.rng = self.np_random

        self.game.reset()
        self._steps = 0
        if self._renderer is not None:
            # Keep pixel observations reproducible across episodes.
            self._renderer.reset()

        return self._get_obs(), self._get_info()

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = Action.from_index(int(action))

        total_reward = 0.0
        terminated = False
        for _ in range(self.frame_skip):
            reward, terminated = self.game.step(action)
            total_reward += reward
            if terminated:
                break

        self._steps += 1
        truncated = (
            not terminated
            and self.max_steps is not None
            and self._steps >= self.max_steps
        )

        return self._get_obs(), float(total_reward), bool(terminated), bool(truncated), self._get_info()

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "render() was called but the environment was created with render_mode=None; "
                "pass render_mode='human', 'rgb_array' or 'ansi' to enable rendering."
            )
            return None

        if self.render_mode == "ansi":
            return self._render_ansi()

        renderer = self._ensure_renderer()
        renderer.draw(self.game)
        if self.render_mode == "human":
            renderer.present()
            renderer.pg.event.pump()
            return None
        return renderer.rgb_array()

    def close(self) -> None:
        if self._renderer is not None:
            pygame = self._renderer.pg
            self._renderer.close()
            self._renderer = None
            try:
                pygame.quit()
            except Exception:  # pragma: no cover - defensive
                pass

    # ------------------------------------------------------------------
    # Pickling / copying
    # ------------------------------------------------------------------
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # Pygame surfaces cannot be pickled; the renderer is rebuilt on demand.
        state["_renderer"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_renderer(self):
        if self._renderer is None:
            from .rendering import PacManRenderer

            mode = self.render_mode or "rgb_array"
            self._renderer = PacManRenderer(self.game, render_mode=mode, draw_state=self.draw_state)
        return self._renderer

    def _render_ansi(self) -> str:
        """Text drawing of the current state (no Pygame needed)."""
        maze = self.game.maze
        grid = [[ANSI_CHARS["empty"] for _ in range(maze.shape[1])] for _ in range(maze.shape[0])]

        for row in range(maze.shape[0]):
            for col in range(maze.shape[1]):
                value = int(maze.map_matrix[row, col])
                if WALL_START <= value <= WALL_END:
                    grid[row][col] = ANSI_CHARS["wall"]
                elif value == GHOST_DOOR:
                    grid[row][col] = ANSI_CHARS["door"]
                elif value == PELLET:
                    grid[row][col] = ANSI_CHARS["pellet"]
                elif value == POWER_PELLET:
                    grid[row][col] = ANSI_CHARS["power_pellet"]

        for ghost in self.game.ghosts:
            row, col = ghost.nearest_row, ghost.nearest_col
            if maze.in_bounds(row, col):
                grid[row][col] = (
                    ANSI_CHARS["frightened_ghost"]
                    if ghost.state in (GhostState.VULNERABLE, GhostState.EATEN)
                    else ANSI_CHARS["ghost"]
                )

        col, row = self.game.player.get_position()
        if maze.in_bounds(row, col):
            grid[row][col] = ANSI_CHARS["player"]

        board = "\n".join("".join(line) for line in grid)
        return (
            f"score={self.game.score} lives={self.game.player.lives} "
            f"pellets={self.game.pellets_remaining} steps={self._steps}\n{board}"
        )

    def _get_obs(self) -> np.ndarray:
        if self.obs_type == STATE_OBS:
            return self.game.get_state_matrix().copy()
        renderer = self._ensure_renderer()
        renderer.draw(self.game)
        return renderer.rgb_array()

    def _get_info(self) -> Dict[str, Any]:
        return {
            "layout": self.layout,
            "score": self.game.score,
            "lives": int(self.game.player.lives),
            "pellets_remaining": self.game.pellets_remaining,
            "win": bool(self.game.win),
            "game_over": bool(self.game.game_over),
            "mode": int(self.game.game_mode),
            "steps": self._steps,
            "player_position": self.game.player.get_position(),
            "player_pixel_position": self.game.player.get_pixel_pos(),
            "player_velocity": self.game.player.get_vel(),
            "ghost_positions": [ghost.get_position() for ghost in self.game.ghosts],
            "ghost_states": [int(ghost.state) for ghost in self.game.ghosts],
            "ghosts_vulnerable": self.game.is_at_least_a_ghost_vulnerable(),
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def get_state_matrix(self) -> np.ndarray:
        return self.game.get_state_matrix()

    def get_player_position(self) -> Tuple[int, int]:
        return self.game.player.get_position()

    @property
    def unwrapped_game(self) -> Game:
        return self.game
