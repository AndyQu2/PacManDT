"""Pac-Man entity (movement only, rendering lives in ``pacman_env.rendering``)."""

from typing import Tuple, Union

from .action import Action
from .constants import PACMAN_SPEED, TILE_SIZE


class Pacman:
    def __init__(self, lives: int = 3):
        self._lives = lives
        self.x = 0
        self.y = 0
        self.vel_x = 0
        self.vel_y = 0
        self.speed = PACMAN_SPEED
        self._current_action = Action.LEFT

        self.nearest_row = 0
        self.nearest_col = 0
        self.home_x = 0
        self.home_y = 0

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------
    def init_home(self, home_x: int, home_y: int) -> None:
        self.home_x, self.home_y = home_x, home_y
        self.x = self.home_x * TILE_SIZE
        self.y = self.home_y * TILE_SIZE
        self.nearest_row = self.home_y
        self.nearest_col = self.home_x

    def regenerate(self) -> None:
        self.init_home(self.home_x, self.home_y)
        self.set_vel_to_zero()

    def set_vel_to_zero(self) -> None:
        self.vel_x = 0
        self.vel_y = 0

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------
    def move(self, game) -> None:
        self.nearest_row = int((self.y + TILE_SIZE / 2) / TILE_SIZE)
        self.nearest_col = int((self.x + TILE_SIZE / 2) / TILE_SIZE)
        next_x, next_y = self.x + self.vel_x, self.y + self.vel_y

        if game.check_if_player_hit_wall(next_x, next_y):
            self.set_vel_to_zero()
            return

        self.x += self.vel_x
        self.y += self.vel_y
        game.check_if_hit_something()
        game.check_collision_with_ghosts()

    def change_player_vel(self, action: Union[Action, int], game) -> None:
        action = action if isinstance(action, Action) else Action.from_index(int(action))
        self._current_action = action

        if action == Action.LEFT:
            if not (self.vel_x == -self.speed and self.vel_y == 0) \
                    and not game.check_if_player_hit_wall(self.x - self.speed, self.y):
                self.vel_x, self.vel_y = -self.speed, 0
        elif action == Action.RIGHT:
            if not (self.vel_x == self.speed and self.vel_y == 0) \
                    and not game.check_if_player_hit_wall(self.x + self.speed, self.y):
                self.vel_x, self.vel_y = self.speed, 0
        elif action == Action.UP:
            if not (self.vel_y == -self.speed and self.vel_x == 0) \
                    and not game.check_if_player_hit_wall(self.x, self.y - self.speed):
                self.vel_x, self.vel_y = 0, -self.speed
        elif action == Action.DOWN:
            if not (self.vel_y == self.speed and self.vel_x == 0) \
                    and not game.check_if_player_hit_wall(self.x, self.y + self.speed):
                self.vel_x, self.vel_y = 0, self.speed
        else:  # Action.NOOP -- brake
            self.set_vel_to_zero()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_position(self) -> Tuple[int, int]:
        return self.nearest_col, self.nearest_row

    def get_pixel_pos(self) -> Tuple[int, int]:
        return self.x, self.y

    def get_vel(self) -> Tuple[int, int]:
        return self.vel_x, self.vel_y

    @property
    def lives(self) -> int:
        return self._lives

    @lives.setter
    def lives(self, lives: int) -> None:
        self._lives = lives

    @property
    def current_action(self) -> Action:
        return self._current_action

    @current_action.setter
    def current_action(self, action: Action) -> None:
        self._current_action = action

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Pacman(col={self.nearest_col}, row={self.nearest_row}, lives={self.lives})"
