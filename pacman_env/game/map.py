"""Tile map loading and observation bookkeeping.

Ported from paolodelia99/py-pacman (MIT).  All Pygame code was removed: this
module only owns the two NumPy matrices (the raw layout and the observation).
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .constants import (
    GHOST_HOME_START,
    GHOST_VULNERABLE_VALUE,
    PATH_FINDER_LOOKUP_TABLE,
    PELLET,
    PLAYER_HOME,
    POWER_PELLET,
    ROOT_DIR,
    STATE_LOOKUP_TABLE,
    TILE_SIZE,
    WALL_END,
    WALL_START,
    GHOST_VALUE,
)
from .ghost_state import GhostState

LAYOUT_DIR = ROOT_DIR / "layouts"


class Map:
    """The maze, stored as an integer matrix.

    ``map_matrix`` keeps the raw layout values (walls, pellets, doors, ...) and
    is mutated when pellets are eaten.  ``state_matrix`` is the agent-facing
    observation: ``-1`` walls/ghosts, ``0`` empty, ``1`` pellets and ``5``
    frightened or eaten ghosts.
    """

    def __init__(self, layout_name: str = "classic"):
        self.layout_name = layout_name
        self._ghost_cells: List[Tuple[int, int]] = []
        self.map_matrix: np.ndarray = np.zeros((0, 0), dtype=np.int8)
        self.state_matrix: np.ndarray = np.zeros((0, 0), dtype=np.int8)
        self.path_matrix: np.ndarray = np.zeros((0, 0), dtype=np.int16)
        self.shape: Tuple[int, int] = (0, 0)
        self.reinit_map()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @property
    def layout_path(self) -> Path:
        return LAYOUT_DIR / f"{self.layout_name}.lay"

    @staticmethod
    def available_layouts() -> List[str]:
        if not LAYOUT_DIR.is_dir():
            return []
        return sorted(path.stem for path in LAYOUT_DIR.glob("*.lay"))

    def reinit_map(self) -> None:
        path = self.layout_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Unknown layout {self.layout_name!r}. Available layouts: "
                f"{', '.join(self.available_layouts()) or '<none found>'}"
            )
        self.map_matrix = np.loadtxt(path).astype(np.int8)
        self.shape = self.map_matrix.shape
        self.state_matrix = self.matrix_from_lookup_table(STATE_LOOKUP_TABLE, np.int8)
        self.path_matrix = self.matrix_from_lookup_table(PATH_FINDER_LOOKUP_TABLE, np.int16)
        self._ghost_cells = []

    def matrix_from_lookup_table(self, lookup_table: Dict[int, int], dtype=np.int8) -> np.ndarray:
        matrix = np.zeros(self.shape, dtype=dtype)
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                matrix[row, col] = lookup_table[int(self.map_matrix[row, col])]
        return matrix

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def is_wall(self, row: int, col: int) -> bool:
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            return True
        return WALL_START <= int(self.map_matrix[row, col]) <= WALL_END

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.shape[0] and 0 <= col < self.shape[1]

    def get_player_home(self) -> Tuple[int, int]:
        home_y, home_x = np.where(self.map_matrix == PLAYER_HOME)
        return int(home_x[0]), int(home_y[0])

    def get_number_of_ghosts(self) -> int:
        return sum(1 for value in range(GHOST_HOME_START, GHOST_HOME_START + 4)
                   if np.any(self.map_matrix == value))

    def get_ghosts_home(self, num_ghosts: int) -> List[Dict[str, int]]:
        homes = []
        for index in range(num_ghosts):
            home_y, home_x = np.where(self.map_matrix == GHOST_HOME_START + index)
            homes.append({"x": int(home_x[0]), "y": int(home_y[0])})
        return homes

    def get_ghost_respawn_home(self) -> Tuple[int, int]:
        ghost_home = GHOST_HOME_START + 2
        if not np.any(self.map_matrix == ghost_home):
            ghost_home = GHOST_HOME_START
        home_y, home_x = np.where(self.map_matrix == ghost_home)
        return int(home_x[0]), int(home_y[0])

    def get_number_of_pellets(self) -> int:
        return int(np.count_nonzero((self.map_matrix == PELLET) | (self.map_matrix == POWER_PELLET)))

    def get_map_sizes(self) -> Tuple[int, int]:
        """Return ``(width, height)`` in pixels, including the HUD row."""
        return self.shape[1] * TILE_SIZE, (self.shape[0] + 1) * TILE_SIZE

    def get_state_matrix(self) -> np.ndarray:
        return self.state_matrix

    def get_tile(self, row: int, col: int) -> int:
        if not self.in_bounds(row, col):
            return 0
        return int(self.map_matrix[row, col])

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def remove_biscuit(self, row: int, col: int) -> None:
        if not self.in_bounds(row, col):
            return
        self.map_matrix[row, col] = 10
        if self.state_matrix[row, col] in (0, 1, 5):
            self.state_matrix[row, col] = 0

    def update_ghosts_position(self, ghosts: Sequence) -> None:
        # Restore whatever was under the ghosts before moving them.
        for row, col in self._ghost_cells:
            if self.in_bounds(row, col):
                self.state_matrix[row, col] = STATE_LOOKUP_TABLE[int(self.map_matrix[row, col])]
        self._ghost_cells = []

        for ghost in ghosts:
            row, col = ghost.nearest_row, ghost.nearest_col
            if not self.in_bounds(row, col):
                continue
            value = GHOST_VULNERABLE_VALUE if ghost.state in (
                GhostState.VULNERABLE, GhostState.EATEN) else GHOST_VALUE
            self.state_matrix[row, col] = value
            self._ghost_cells.append((row, col))
