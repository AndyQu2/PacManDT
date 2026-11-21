"""Ghost entities with A* pursuit (rendering lives in ``pacman_env.rendering``)."""

from typing import List, Optional, Tuple

import numpy as np

from .constants import GHOST_EATEN_SPEED, GHOST_SPEED, GHOST_START_VALUE, TILE_SIZE
from .ghost_state import GhostState
from .map import Map
from .pacman import Pacman
from .path_finder import NoPathFoundException, NotValidPointException, PathFinder


class Ghost:
    def __init__(self, ghost_id: int, ghost_color, path_finder: PathFinder,
                 rng: Optional[np.random.Generator] = None):
        self.id = ghost_id
        self.ghost_color = ghost_color
        self.path_finder = path_finder
        self.rng = rng if rng is not None else np.random.default_rng()

        self.x = 0
        self.y = 0
        self.vel_x = 0
        self.vel_y = 0
        self.speed = GHOST_SPEED
        self.nearest_row = 0
        self.nearest_col = 0

        self.state = GhostState.NORMAL
        self.value = 0

        self.home_x = 0
        self.home_y = 0
        self.respawn_x: Optional[int] = None
        self.respawn_y: Optional[int] = None

        self.current_path: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Placement / lifecycle
    # ------------------------------------------------------------------
    def init_home(self, home_x: int, home_y: int) -> None:
        self.home_x, self.home_y = home_x, home_y
        self.x = home_x * TILE_SIZE
        self.y = home_y * TILE_SIZE
        self.nearest_col, self.nearest_row = home_x, home_y
        self.current_path = None

    def init_respawn_home(self, respawn_x: int, respawn_y: int) -> None:
        self.respawn_x, self.respawn_y = respawn_x, respawn_y

    def init_for_game(self, path_finder: PathFinder, player: Pacman) -> None:
        self.vel_x = 0
        self.vel_y = 0
        self.set_normal()
        self.find_path(path_finder, player, random=True)

    def set_normal(self) -> None:
        self.state = GhostState.NORMAL
        self.value = 0
        self.speed = GHOST_SPEED

    def set_vulnerable(self) -> None:
        self.state = GhostState.VULNERABLE
        self.value = GHOST_START_VALUE
        self.speed = GHOST_SPEED

    def set_spectacles(self, path_finder: PathFinder, player: Pacman) -> None:
        """Ghost was eaten: it becomes a pair of eyes travelling back home."""
        self.state = GhostState.EATEN
        self.value = 0
        self.speed = GHOST_EATEN_SPEED
        self.x = self.nearest_col * TILE_SIZE
        self.y = self.nearest_row * TILE_SIZE
        self.find_path(path_finder, player)

    def duplicate_value(self) -> None:
        self.value *= 2

    def is_vulnerable(self) -> bool:
        return self.state == GhostState.VULNERABLE

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def move(self, player: Pacman) -> None:
        self.x += self.vel_x
        self.y += self.vel_y

        self.nearest_row = int((self.y + TILE_SIZE / 2) / TILE_SIZE)
        self.nearest_col = int((self.x + TILE_SIZE / 2) / TILE_SIZE)

        self.check_ghost_state(self.path_finder, player)

        if (self.x % TILE_SIZE) == 0 and (self.y % TILE_SIZE) == 0:
            if self.current_path:
                self.follow_next_path()
            else:
                self.find_path(self.path_finder, player)

    def find_path(self, path_finder: PathFinder, player: Optional[Pacman], random: bool = False) -> None:
        if random or self.state == GhostState.EATEN and self.respawn_x is None:
            target = path_finder.get_random_allow_position(self.rng)
        elif self.state == GhostState.EATEN:
            target = (self.respawn_x, self.respawn_y)
        elif player is not None:
            target = (player.nearest_col, player.nearest_row)
        else:
            target = path_finder.get_random_allow_position(self.rng)

        try:
            self.current_path = path_finder.get_min_path(
                self.nearest_col, self.nearest_row, target[0], target[1])
        except (NoPathFoundException, NotValidPointException):
            self.current_path = None

        self.follow_next_path()

    def follow_next_path(self) -> None:
        if not self.current_path:
            return
        direction = self.current_path[0]
        if direction == "L":
            self.vel_x, self.vel_y = -self.speed, 0
        elif direction == "R":
            self.vel_x, self.vel_y = self.speed, 0
        elif direction == "U":
            self.vel_x, self.vel_y = 0, -self.speed
        elif direction == "D":
            self.vel_x, self.vel_y = 0, self.speed
        self.current_path = self.current_path[1:]

    def check_ghost_state(self, path_finder: PathFinder, player: Pacman) -> None:
        if self.state != GhostState.EATEN:
            return
        if self.respawn_x is None or self.respawn_y is None:
            return
        if self.nearest_row == self.respawn_y and self.nearest_col == self.respawn_x:
            self.set_normal()
            self.find_path(path_finder, player)

    def check_ghost_position(self, maze: Map) -> None:
        if maze.in_bounds(self.nearest_row, self.nearest_col):
            return
        self.init_home(self.home_x, self.home_y)
        self.find_path(self.path_finder, None, random=True)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_pixel_position(self) -> Tuple[int, int]:
        return self.x, self.y

    def get_position(self) -> Tuple[int, int]:
        return self.nearest_col, self.nearest_row

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Ghost(id={self.id}, col={self.nearest_col}, row={self.nearest_row}, "
                f"state={self.state.name}, value={self.value})")
