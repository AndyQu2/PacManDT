"""Game rules: scoring, collisions, frightened timer and episode bookkeeping.

The original ``Game`` class from paolodelia99/py-pacman mixed simulation with
Pygame drawing and a blocking game loop.  This version is pure simulation; the
renderer and the Gymnasium wrapper drive it.
"""

from typing import List, Optional, Tuple

import numpy as np

from .constants import (
    DOOR_H,
    DOOR_V,
    GHOST_COLORS,
    PELLET,
    POWER_PELLET,
    TILE_SIZE,
    VULNERABLE_DURATION,
)
from .game_mode import GameMode
from .ghost import Ghost
from .ghost_state import GhostState
from .map import Map
from .pacman import Pacman
from .path_finder import PathFinder

HIT_THRESHOLD = (3 * TILE_SIZE) // 4


def check_if_hit(x1: int, y1: int, x2: int, y2: int, threshold: int = HIT_THRESHOLD) -> bool:
    return abs(x1 - x2) < threshold and abs(y1 - y2) < threshold


class Game:
    def __init__(self, layout: str = "classic", player_lives: int = 3,
                 rng: Optional[np.random.Generator] = None):
        self.layout = layout
        self.player_lives = player_lives
        self.rng = rng if rng is not None else np.random.default_rng()

        self.maze = Map(layout)
        self.path_finder = PathFinder(self.maze.path_matrix)
        self.player = Pacman(lives=player_lives)
        self.ghosts: List[Ghost] = [
            Ghost(index, GHOST_COLORS.get(index, (255, 0, 0, 255)), self.path_finder, self.rng)
            for index in range(self.maze.get_number_of_ghosts())
        ]

        self.score = 0
        self.total_rewards = 0
        self.game_mode = GameMode.NORMAL
        self.mode_timer = 0
        self.ghosts_timer = 0
        self.steps = 0
        self.win = False
        self.game_over = False
        self.reset()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.maze.reinit_map()
        self.path_finder.state_map = self.maze.path_matrix

        self.score = 0
        self.total_rewards = 0
        self.mode_timer = 0
        self.ghosts_timer = 0
        self.steps = 0
        self.win = False
        self.game_over = False

        self.player.lives = self.player_lives
        self.init_players_in_map()
        self.make_ghosts_normal()
        self.game_mode = GameMode.NORMAL
        self.maze.update_ghosts_position(self.ghosts)

    def init_players_in_map(self) -> None:
        home_x, home_y = self.maze.get_player_home()
        self.player.init_home(home_x, home_y)
        self.player.set_vel_to_zero()

        ghosts_home = self.maze.get_ghosts_home(len(self.ghosts))
        respawn_x, respawn_y = self.maze.get_ghost_respawn_home()
        for ghost, home in zip(self.ghosts, ghosts_home):
            ghost.init_home(home["x"], home["y"])
            ghost.init_respawn_home(respawn_x, respawn_y)
            ghost.init_for_game(self.path_finder, self.player)

    def respawn(self) -> None:
        self.init_players_in_map()
        self.make_ghosts_normal()
        self.game_mode = GameMode.NORMAL
        self.maze.update_ghosts_position(self.ghosts)

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def step(self, action) -> Tuple[float, bool]:
        """Advance the simulation one tick. Returns ``(reward, terminated)``."""
        if self.is_done():
            return 0.0, True

        previous_reward = self.total_rewards
        self.player.change_player_vel(action, self)
        self.player.move(self)

        if self.game_mode == GameMode.HIT_GHOST:
            self.player.lives -= 1
            if self.player.lives <= 0:
                self.game_mode = GameMode.GAME_OVER
                self.game_over = True
            else:
                self.respawn()
        elif not self.win:
            self.move_ghosts()
            self.update_ghosts_position_in_map()
            self.check_ghosts_state()

        self.mode_timer += 1
        self.steps += 1
        return float(self.total_rewards - previous_reward), self.is_done()

    def is_done(self) -> bool:
        return self.game_over or self.win

    def move_ghosts(self) -> None:
        for ghost in self.ghosts:
            ghost.move(self.player)

    def update_ghosts_position_in_map(self) -> None:
        self.maze.update_ghosts_position(self.ghosts)

    # ------------------------------------------------------------------
    # Frightened timer
    # ------------------------------------------------------------------
    def check_ghosts_state(self) -> None:
        if self.is_at_least_a_ghost_vulnerable():
            if self.ghosts_timer >= VULNERABLE_DURATION:
                self.make_ghosts_normal()
            else:
                self.ghosts_timer += 1

        for ghost in self.ghosts:
            ghost.check_ghost_position(self.maze)

    def make_ghosts_vulnerable(self) -> None:
        self.ghosts_timer = 0
        for ghost in self.ghosts:
            ghost.set_vulnerable()

    def make_ghosts_normal(self) -> None:
        self.ghosts_timer = 0
        for ghost in self.ghosts:
            ghost.set_normal()

    def duplicate_vulnerable_ghosts_value(self) -> None:
        for ghost in self.ghosts:
            if ghost.state == GhostState.VULNERABLE:
                ghost.duplicate_value()

    def is_at_least_a_ghost_vulnerable(self) -> bool:
        return any(ghost.state == GhostState.VULNERABLE for ghost in self.ghosts)

    def is_at_least_a_ghost_spectacles(self) -> bool:
        return any(ghost.state == GhostState.EATEN for ghost in self.ghosts)

    # ------------------------------------------------------------------
    # Collisions and scoring
    # ------------------------------------------------------------------
    def check_if_hit_something(self) -> None:
        for row in range(self.player.nearest_row - 1, self.player.nearest_row + 2):
            for col in range(self.player.nearest_col - 1, self.player.nearest_col + 2):
                if not self.maze.in_bounds(row, col):
                    continue
                if abs(self.player.x - col * TILE_SIZE) >= TILE_SIZE:
                    continue
                if abs(self.player.y - row * TILE_SIZE) >= TILE_SIZE:
                    continue

                tile = int(self.maze.map_matrix[row, col])
                if tile == PELLET:
                    self.maze.remove_biscuit(row, col)
                    self.add_score(10)
                    self.add_reward(1)
                    if self.maze.get_number_of_pellets() == 0:
                        self.win = True
                        self.game_mode = GameMode.WAIT_AFTER_FINISHING_LEVEL
                elif tile == POWER_PELLET:
                    self.maze.remove_biscuit(row, col)
                    self.add_score(100)
                    self.add_reward(2)
                    self.make_ghosts_vulnerable()
                elif tile == DOOR_H:
                    self._wrap_horizontally(row, col)
                elif tile == DOOR_V:
                    self._wrap_vertically(row, col)

    def _wrap_horizontally(self, row: int, col: int) -> None:
        for other_col in range(self.maze.shape[1]):
            if other_col != col and self.maze.map_matrix[row, other_col] == DOOR_H:
                self.player.x = other_col * TILE_SIZE
                self.player.x += TILE_SIZE if self.player.vel_x > 0 else -TILE_SIZE

    def _wrap_vertically(self, row: int, col: int) -> None:
        for other_row in range(self.maze.shape[0]):
            if other_row != row and self.maze.map_matrix[other_row, col] == DOOR_V:
                self.player.y = other_row * TILE_SIZE
                self.player.y += TILE_SIZE if self.player.vel_y > 0 else -TILE_SIZE

    def check_collision_with_ghosts(self) -> None:
        for ghost in self.ghosts:
            if not check_if_hit(self.player.x, self.player.y, ghost.x, ghost.y):
                continue
            if ghost.state == GhostState.NORMAL:
                self.game_mode = GameMode.HIT_GHOST
                self.add_reward(-5)
            elif ghost.state == GhostState.VULNERABLE:
                self.add_score(ghost.value)
                self.add_reward(5)
                self.duplicate_vulnerable_ghosts_value()
                ghost.set_spectacles(self.path_finder, self.player)
                if not self.is_at_least_a_ghost_vulnerable():
                    self.ghosts_timer = 0

    def check_if_player_hit_wall(self, x: int, y: int) -> bool:
        for row in range(self.player.nearest_row - 1, self.player.nearest_row + 2):
            for col in range(self.player.nearest_col - 1, self.player.nearest_col + 2):
                if not self.maze.in_bounds(row, col):
                    continue
                if abs(x - col * TILE_SIZE) >= TILE_SIZE:
                    continue
                if abs(y - row * TILE_SIZE) >= TILE_SIZE:
                    continue
                if self.maze.is_wall(row, col):
                    return True
        return False

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def add_score(self, score_to_add: int) -> None:
        self.score += score_to_add

    def add_reward(self, reward_to_add: float) -> None:
        self.total_rewards += reward_to_add

    @property
    def pellets_remaining(self) -> int:
        return self.maze.get_number_of_pellets()

    @property
    def lives(self) -> int:
        return self.player.lives

    def get_state_matrix(self) -> np.ndarray:
        return self.maze.get_state_matrix()
