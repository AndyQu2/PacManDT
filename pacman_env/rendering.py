"""Pygame renderer for the Pac-Man simulation.

Pygame is imported lazily so that headless (``render_mode=None``) training never
pays for it.  The renderer keeps its own animation state and never mutates the
simulation.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np

from .game.constants import (
    EDGE_LIGHT_COLOR,
    EDGE_SHADOW_COLOR,
    FILL_COLOR,
    GHOST_COLORS,
    PELLET_COLOR,
    ROOT_DIR,
    STATE_COLOR_LOOKUP_TABLE,
    TILE_LOOKUP_TABLE,
    TILE_SIZE,
    VULNERABLE_DURATION,
    VULNERABLE_GHOST_COLOR,
    WHITE_GHOST_COLOR,
)
from .game.game import Game
from .game.ghost_state import GhostState

TILES_DIR = ROOT_DIR / "tiles"
SPRITE_DIR = ROOT_DIR / "sprite"
TEXT_DIR = ROOT_DIR / "text"

BLANK_TILES = (11, 12, 33, 34, 35, 36, 40)


class PacManRenderer:
    """Draws a :class:`~pacman_env.game.game.Game` with Pygame."""

    def __init__(self, game: Game, render_mode: str = "human", draw_state: bool = False,
                 caption: str = "PacManDT"):
        import pygame as pg

        self.pg = pg
        self.game = game
        self.render_mode = render_mode
        self.draw_state = draw_state
        self.caption = caption

        self.map_width, self.map_height = game.maze.get_map_sizes()
        self.width = self.map_width + ((self.map_width + TILE_SIZE) if draw_state else 0)
        self.height = self.map_height

        pg.display.init()
        flags = 0 if render_mode == "human" else pg.HIDDEN
        self.surface = pg.display.set_mode((self.width, self.height), flags)
        pg.display.set_caption(caption)

        self._pacman_frame = 1
        self._ghost_frame = 1
        self._animation_counter = 0

        self._load_assets()

    # ------------------------------------------------------------------
    # Asset loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load(path) -> "object":
        import pygame as pg

        return pg.image.load(os.fspath(path)).convert_alpha()

    def _load_assets(self) -> None:
        pg = self.pg

        self.tiles: Dict[int, "pg.Surface"] = {}
        for value, filename in TILE_LOOKUP_TABLE.items():
            if value in BLANK_TILES:
                continue
            surface = self._load(TILES_DIR / filename)
            self._recolor_tile(surface)
            self.tiles[value] = surface
        self.blank_tile = self._load(TILES_DIR / "blank.gif")
        self.door_tile = self._load(TILES_DIR / "ghost-door.gif")
        self.glasses = self._load(TILES_DIR / "glasses.gif")

        self.pacman_frames = {"s": [self._load(SPRITE_DIR / "pacman.gif")]}
        for orientation in ("l", "r", "u", "d"):
            self.pacman_frames[orientation] = [
                self._load(SPRITE_DIR / f"pacman-{orientation} {index}.gif") for index in range(1, 9)
            ]

        base_ghosts = [self._load(SPRITE_DIR / f"ghost {index}.gif") for index in range(1, 7)]
        colors = list(GHOST_COLORS.values()) + [VULNERABLE_GHOST_COLOR, WHITE_GHOST_COLOR]
        self.ghost_frames: Dict[Tuple[int, int, int, int], list] = {}
        for color in colors:
            self.ghost_frames[color] = [self._recolor_ghost(surface, color) for surface in base_ghosts]

        self.digits = {index: self._load(TEXT_DIR / f"{index}.gif") for index in range(10)}
        self.life = self._load(TEXT_DIR / "life.gif")
        self.game_over_img = self._load(TEXT_DIR / "gameover.gif")
        self.ready_img = self._load(TEXT_DIR / "ready.gif")

    @staticmethod
    def _recolor_tile(tile) -> None:
        for y in range(tile.get_height()):
            for x in range(tile.get_width()):
                color = tile.get_at((x, y))
                if color == EDGE_LIGHT_COLOR:
                    tile.set_at((x, y), (0, 0, 255, 255))
                elif color == FILL_COLOR:
                    tile.set_at((x, y), (0, 0, 0, 255))
                elif color == EDGE_SHADOW_COLOR:
                    tile.set_at((x, y), (0, 0, 255, 255))
                elif color == PELLET_COLOR:
                    tile.set_at((x, y), (255, 255, 255, 255))

    @staticmethod
    def _recolor_ghost(surface, color):
        ghost = surface.copy()
        for y in range(ghost.get_height()):
            for x in range(ghost.get_width()):
                if ghost.get_at((x, y)) == (255, 0, 0, 255):
                    ghost.set_at((x, y), color)
        return ghost

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self, game: Optional[Game] = None) -> "object":
        pg = self.pg
        game = game or self.game

        self.surface.fill((0, 0, 0))
        self._draw_maze(game)
        self._draw_pacman(game)
        for ghost in game.ghosts:
            self._draw_ghost(ghost, game)
        self._draw_hud(game)
        self._draw_overlay(game)
        if self.draw_state:
            self._draw_state_matrix(game)
        self._advance_animation(game)
        return self.surface

    def _tile_surface(self, value: int):
        if value in BLANK_TILES:
            return self.blank_tile
        return self.tiles.get(value, self.blank_tile)

    def _draw_maze(self, game: Game) -> None:
        maze = game.maze
        for row in range(maze.shape[0]):
            for col in range(maze.shape[1]):
                surface = self._tile_surface(int(maze.map_matrix[row, col]))
                self.surface.blit(surface, (col * TILE_SIZE, row * TILE_SIZE))

    def _draw_pacman(self, game: Game) -> None:
        player = game.player
        vel_x, vel_y = player.get_vel()
        if vel_x > 0:
            frames = self.pacman_frames["r"]
        elif vel_x < 0:
            frames = self.pacman_frames["l"]
        elif vel_y > 0:
            frames = self.pacman_frames["d"]
        elif vel_y < 0:
            frames = self.pacman_frames["u"]
        else:
            frames = self.pacman_frames["s"]
        frame = frames[(self._pacman_frame - 1) % len(frames)]
        self.surface.blit(frame, (player.x, player.y))

    def _draw_ghost(self, ghost, game: Game) -> None:
        if ghost.state == GhostState.EATEN:
            self.surface.blit(self.glasses, (ghost.x, ghost.y))
            return

        if ghost.state == GhostState.VULNERABLE:
            flashing = (VULNERABLE_DURATION - game.ghosts_timer) < 100
            if flashing and (game.ghosts_timer // 10) % 2 == 0:
                color = WHITE_GHOST_COLOR
            else:
                color = VULNERABLE_GHOST_COLOR
        else:
            color = ghost.ghost_color

        frames = self.ghost_frames.get(color) or self.ghost_frames[GHOST_COLORS[ghost.id]]
        frame = frames[(self._ghost_frame - 1) % len(frames)]
        self.surface.blit(frame, (ghost.x, ghost.y))

    def _draw_hud(self, game: Game) -> None:
        top = game.maze.shape[0] * TILE_SIZE
        for index, char in enumerate(str(game.score)):
            self.surface.blit(self.digits[int(char)], (index * 13, top))

        start_x = (game.maze.shape[1] // 2 - 1) * TILE_SIZE
        for index in range(max(game.player.lives, 0)):
            self.surface.blit(self.life, (start_x + index * 10, top))

    def _draw_overlay(self, game: Game) -> None:
        if game.game_over:
            image = self.game_over_img
            self.surface.blit(image, ((self.map_width - image.get_width()) // 2,
                                      (self.map_height - image.get_height()) // 2))
        elif game.win:
            image = self.ready_img
            self.surface.blit(image, ((self.map_width - image.get_width()) // 2,
                                      (self.map_height - image.get_height()) // 2))

    def _draw_state_matrix(self, game: Game) -> None:
        pg = self.pg
        offset_x = self.map_width + TILE_SIZE
        for row in range(game.maze.shape[0]):
            for col in range(game.maze.shape[1]):
                value = int(game.maze.state_matrix[row, col])
                color = STATE_COLOR_LOOKUP_TABLE.get(value, (255, 255, 255))
                pg.draw.rect(self.surface, color,
                             pg.Rect(offset_x + col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))

    def _advance_animation(self, game: Game) -> None:
        self._animation_counter += 1
        if self._animation_counter % 2:
            return
        vel_x, vel_y = game.player.get_vel()
        if vel_x or vel_y:
            self._pacman_frame = self._pacman_frame % 8 + 1
        self._ghost_frame = self._ghost_frame % 6 + 1

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the animation state so renders are reproducible after ``reset()``."""
        self._pacman_frame = 1
        self._ghost_frame = 1
        self._animation_counter = 0

    def present(self) -> None:
        self.pg.display.flip()

    def rgb_array(self) -> np.ndarray:
        array = self.pg.surfarray.array3d(self.surface)
        return np.transpose(array, (1, 0, 2))

    def close(self) -> None:
        try:
            self.pg.display.quit()
        except Exception:  # pragma: no cover - defensive
            pass
