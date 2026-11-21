"""Game mode enumeration (kept compatible with the upstream project)."""

from enum import IntEnum


class GameMode(IntEnum):
    READY = 0
    NORMAL = 1
    HIT_GHOST = 2
    GAME_OVER = 3
    WAIT_TO_START = 4
    WAIT_AFTER_EATING_GHOST = 5
    WAIT_AFTER_FINISHING_LEVEL = 6
    FLASH_MAZE = 7
    EXTRA_PACMAN = 8
    CHANGE_GHOSTS = 9
    BLACK_SCREEN = 10  # used for "level completed"


# Backwards-compatible aliases (the upstream code used lowercase members).
GameMode.ready = GameMode.READY
GameMode.normal = GameMode.NORMAL
GameMode.hit_ghost = GameMode.HIT_GHOST
GameMode.game_over = GameMode.GAME_OVER
GameMode.wait_to_start = GameMode.WAIT_TO_START
GameMode.wait_after_eating_ghost = GameMode.WAIT_AFTER_EATING_GHOST
GameMode.wait_after_finishing_level = GameMode.WAIT_AFTER_FINISHING_LEVEL
GameMode.flash_maze = GameMode.FLASH_MAZE
GameMode.extra_pacman = GameMode.EXTRA_PACMAN
GameMode.change_ghosts = GameMode.CHANGE_GHOSTS
GameMode.black_screen = GameMode.BLACK_SCREEN

MOVE_MODES = (
    GameMode.NORMAL,
    GameMode.CHANGE_GHOSTS,
    GameMode.WAIT_AFTER_EATING_GHOST,
)
