"""Discrete actions understood by the simulation."""

from enum import IntEnum


class Action(IntEnum):
    """Action indices used by :class:`pacman_env.env.PacmanEnv`.

    The first four values intentionally match the original py-pacman
    implementation.  ``NOOP`` was added so agents can wait in place.
    """

    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    NOOP = 4

    @classmethod
    def from_index(cls, index: int) -> "Action":
        try:
            return cls(int(index))
        except ValueError as exc:
            valid = ", ".join(f"{member.value} ({member.name})" for member in cls)
            raise ValueError(f"Invalid action {index!r}. Valid actions: {valid}") from exc
