"""Constants for the Pac-Man simulation.

Adapted from https://github.com/paolodelia99/py-pacman (MIT License,
Copyright (c) 2020 Paolo D'Elia).  Rendering-related constants are kept
here as well so the renderer and the simulation agree on the map format.
"""

from pathlib import Path

# ``assets`` lives one directory above this module.
ROOT_DIR = Path(__file__).resolve().parent.parent / "assets"

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
TILE_SIZE = 24

# --------------------------------------------------------------------------
# Map encoding (see res/layouts/*.lay)
# --------------------------------------------------------------------------
BLANK = 10
DOOR_H = 11
DOOR_V = 12
GHOST_DOOR = 13
PELLET = 14
POWER_PELLET = 15
WALL_START = 16
WALL_END = 32
GHOST_HOME_START = 33
GHOST_HOME_END = 36
PLAYER_HOME = 40
OUTSIDE = 50

# --------------------------------------------------------------------------
# Observation encoding
# --------------------------------------------------------------------------
BLANK_SPACE_VALUE = 0
WALL_VALUE = -1
GHOST_VALUE = -1
PELLET_VALUE = 1
POWER_PELLET_VALUE = 1
GHOST_VULNERABLE_VALUE = 5

STATE_LOOKUP_TABLE = {
    BLANK: BLANK_SPACE_VALUE,
    DOOR_H: BLANK_SPACE_VALUE,
    DOOR_V: BLANK_SPACE_VALUE,
    GHOST_DOOR: WALL_VALUE,
    PELLET: PELLET_VALUE,
    POWER_PELLET: POWER_PELLET_VALUE,
    PLAYER_HOME: BLANK_SPACE_VALUE,
    OUTSIDE: BLANK_SPACE_VALUE,
    **{value: WALL_VALUE for value in range(WALL_START, WALL_END + 1)},
    **{value: GHOST_VALUE for value in range(GHOST_HOME_START, GHOST_HOME_END + 1)},
}

# Walls are *not* passable for the A* path finder; ghost doors are.
PATH_FINDER_LOOKUP_TABLE = {
    BLANK: 0,
    DOOR_H: 0,
    DOOR_V: 0,
    GHOST_DOOR: 0,
    PELLET: 0,
    POWER_PELLET: 0,
    PLAYER_HOME: 0,
    OUTSIDE: 1000,
    **{value: 1000 for value in range(WALL_START, WALL_END + 1)},
    **{value: 0 for value in range(GHOST_HOME_START, GHOST_HOME_END + 1)},
}

TILE_LOOKUP_TABLE = {
    BLANK: "blank.gif",
    DOOR_H: "door-h.gif",
    DOOR_V: "door-v.gif",
    GHOST_DOOR: "ghost-door.gif",
    PELLET: "pellet.gif",
    POWER_PELLET: "pellet-power.gif",
    WALL_START + 0: "wall-corner-ll.gif",
    WALL_START + 1: "wall-corner-lr.gif",
    WALL_START + 2: "wall-corner-ul.gif",
    WALL_START + 3: "wall-corner-ur.gif",
    WALL_START + 4: "wall-end-b.gif",
    WALL_START + 5: "wall-end-l.gif",
    WALL_START + 6: "wall-end-r.gif",
    WALL_START + 7: "wall-end-t.gif",
    WALL_START + 8: "wall-nub.gif",
    WALL_START + 9: "wall-straight-horiz.gif",
    WALL_START + 10: "wall-straight-vert.gif",
    WALL_START + 11: "wall-t-bottom.gif",
    WALL_START + 12: "wall-t-left.gif",
    WALL_START + 13: "wall-t-right.gif",
    WALL_START + 14: "wall-t-top.gif",
    WALL_START + 15: "wall-x.gif",
    WALL_START + 16: "x-paintwall.gif",
    GHOST_HOME_START + 0: "ghost-blinky.gif",
    GHOST_HOME_START + 1: "ghost-inky.gif",
    GHOST_HOME_START + 2: "ghost-pinky.gif",
    GHOST_HOME_START + 3: "ghost-sue.gif",
    OUTSIDE: "blank.gif",
}

INVERT_ORIENTATION_TABLE = {"D": "U", "U": "D", "L": "R", "R": "L"}

# --------------------------------------------------------------------------
# Rendering colours
# --------------------------------------------------------------------------
GHOST_COLORS = {
    0: (255, 0, 0, 255),
    1: (255, 128, 255, 255),
    2: (128, 255, 255, 255),
    3: (255, 128, 0, 255),
}
VULNERABLE_GHOST_COLOR = (50, 50, 255, 255)
WHITE_GHOST_COLOR = (255, 255, 255, 255)

EDGE_LIGHT_COLOR = (255, 206, 255, 255)
FILL_COLOR = (132, 0, 132, 255)
EDGE_SHADOW_COLOR = (255, 0, 255, 255)
PELLET_COLOR = (128, 0, 128, 255)

WHITE_EDGE_LIGHT_COLOR = (255, 255, 254, 255)
WHITE_EDGE_SHADOW_COLOR = (255, 255, 254, 255)
WHITE_FILL_COLOR = (0, 0, 0, 255)

STATE_COLOR_LOOKUP_TABLE = {
    BLANK_SPACE_VALUE: (255, 204, 204),
    PELLET_VALUE: (255, 102, 102),
    POWER_PELLET_VALUE: (255, 0, 0),
    GHOST_VULNERABLE_VALUE: (153, 0, 0),
    WALL_VALUE: (0, 0, 255),
    GHOST_VALUE: (51, 51, 255),
}

SCORE_COLWIDTH = 13

# --------------------------------------------------------------------------
# Simulation tuning
# --------------------------------------------------------------------------
PACMAN_SPEED = 3
GHOST_SPEED = 2
GHOST_EATEN_SPEED = 8
VULNERABLE_DURATION = 360  # simulation ticks
GHOST_START_VALUE = 200
