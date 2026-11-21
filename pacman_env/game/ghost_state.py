"""Ghost state enumeration."""

from enum import IntEnum


class GhostState(IntEnum):
    NORMAL = 1
    VULNERABLE = 2
    EATEN = 3  # upstream called this ``spectacles`` (eyes returning home)

    # Backwards-compatible aliases.
    normal = 1
    vulnerable = 2
    spectacles = 3
