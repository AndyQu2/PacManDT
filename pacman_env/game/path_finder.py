"""A* path finding for the ghosts.

Adapted from paolodelia99/py-pacman (MIT) and made deterministic by
accepting an explicit NumPy generator for random targets.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .constants import INVERT_ORIENTATION_TABLE
from .matrix_point import MatrixPoint, Point


class NotValidPointException(Exception):
    def __init__(self, x: int, y: int):
        super().__init__(f"The point ({x}, {y}) is outside the map")


class NoPathFoundException(Exception):
    def __init__(self, start: Point, goal: Point):
        super().__init__(f"No path found between {start} and {goal}")
        self.start = start
        self.goal = goal


def get_neighbors(matrix: np.ndarray, row: int, col: int) -> Dict[str, Optional[MatrixPoint]]:
    """Return the passable cross-neighbours of ``(row, col)``."""
    rows, cols = matrix.shape
    neighbours: Dict[str, Optional[MatrixPoint]] = {"L": None, "R": None, "U": None, "D": None}
    candidates = (
        ("U", row - 1, col),
        ("D", row + 1, col),
        ("L", row, col - 1),
        ("R", row, col + 1),
    )
    for orientation, r, c in candidates:
        if 0 <= r < rows and 0 <= c < cols:
            neighbours[orientation] = MatrixPoint(int(matrix[r][c]), c, r)
    return neighbours


class PathFinder:
    def __init__(self, state_map: np.ndarray):
        self.state_map = state_map
        self.map_shape = state_map.shape

    def check_point_validity(self, x: int, y: int) -> None:
        if not (0 <= x < self.map_shape[1] and 0 <= y < self.map_shape[0]):
            raise NotValidPointException(x, y)

    def get_min_path(self, x1: int, y1: int, x2: int, y2: int) -> Optional[List[str]]:
        """Return a list of ``L``/``R``/``U``/``D`` moves from start to goal."""
        self.check_point_validity(x1, y1)
        self.check_point_validity(x2, y2)

        open_set: Set[MatrixPoint] = set()
        closed_set: Set[MatrixPoint] = set()
        start = MatrixPoint(int(self.state_map[y1][x1]), x1, y1)
        goal = MatrixPoint(int(self.state_map[y2][x2]), x2, y2)
        open_set.add(start)

        while open_set:
            current = min(open_set, key=lambda item: item.get_cost(goal))

            if current.same_position(goal):
                return self.return_min_path(current)

            open_set.remove(current)
            closed_set.add(current)

            for point in get_neighbors(self.state_map, current.y, current.x).values():
                if point is None or point in closed_set:
                    continue
                if point in open_set:
                    new_g = current.g + 1
                    if point.g > new_g:
                        point.g = new_g
                        point.set_parent(current)
                else:
                    point.g = current.g + 1
                    point.set_parent(current)
                    open_set.add(point)

        raise NoPathFoundException(start, goal)

    @staticmethod
    def return_min_path(point: MatrixPoint) -> List[str]:
        path: List[str] = []
        while point.parent is not None:
            orientation = point.get_parent_orientation()
            if orientation is None:
                break
            path.append(INVERT_ORIENTATION_TABLE[orientation])
            point = point.parent
        path.reverse()
        return path

    def get_random_allow_position(self, rng: Optional[np.random.Generator] = None) -> Tuple[int, int]:
        """Pick a random non-wall tile (used when a ghost has no target)."""
        generator = rng if rng is not None else np.random.default_rng()
        rows, cols = np.where(self.state_map == 0)
        if len(rows) == 0:
            return 0, 0
        index = int(generator.integers(0, len(rows)))
        return int(cols[index]), int(rows[index])
