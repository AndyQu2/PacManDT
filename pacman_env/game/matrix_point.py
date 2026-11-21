"""Small point helpers used by the A* path finder.

Adapted from paolodelia99/py-pacman (MIT).
"""

from typing import Dict, Optional


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_distance(self, point2: "Point") -> int:
        return manhattan_distance(self.x, self.y, point2.x, point2.y)

    def same_position(self, point2: "Point") -> bool:
        return self.x == point2.x and self.y == point2.y

    def get_cross_neighbors(self) -> Dict[str, "Point"]:
        return {
            "U": Point(self.x, self.y - 1),
            "D": Point(self.x, self.y + 1),
            "L": Point(self.x - 1, self.y),
            "R": Point(self.x + 1, self.y),
        }

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"Point x: {self.x}, y: {self.y}"


class MatrixPoint(Point):
    def __init__(self, value: int, x: int, y: int):
        super().__init__(x, y)
        self.value = value
        self.parent: Optional["MatrixPoint"] = None
        self.g = 0

    def get_cost(self, point2: "MatrixPoint") -> int:
        return int(self.value) + self.g + self.get_distance(point2)

    def set_parent(self, point2: "MatrixPoint") -> None:
        if self.get_distance(point2) == 1:
            self.parent = point2

    def get_parent_orientation(self) -> Optional[str]:
        if self.parent is None:
            return None
        for orientation, point in self.get_cross_neighbors().items():
            if point.same_position(self.parent):
                return orientation
        return None

    def __eq__(self, other):
        return (
            isinstance(other, MatrixPoint)
            and self.x == other.x
            and self.y == other.y
            and self.value == other.value
        )

    def __hash__(self):
        return hash((self.x, self.y, self.value))

    def __str__(self):
        return f"MatrixPoint value: {self.value}, x: {self.x}, y: {self.y}"

    def __copy__(self):
        return MatrixPoint(self.value, self.x, self.y)
