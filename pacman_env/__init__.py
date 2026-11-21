"""PacManDT: a Gymnasium-compatible Pac-Man environment.

The game logic and assets are adapted from the MIT-licensed
`paolodelia99/py-pacman <https://github.com/paolodelia99/py-pacman>`_ project.

This package contains the environment only.  Behaviour policies, trajectory
collection and the Decision-Transformer dataset formats live in
:mod:`data`.
"""

ENV_ID = "PacManDT/PacMan-v0"
MAX_EPISODE_STEPS = 1000
__version__ = "0.1.0"

from .env import PacmanEnv  # noqa: E402
from .game import Action, Game, GameMode, Ghost, GhostState, Map, Pacman  # noqa: E402
from .wrappers import flatten_float32, make_env, make_flat_env  # noqa: E402

__all__ = [
    "PacmanEnv",
    "Action",
    "Game",
    "GameMode",
    "Ghost",
    "GhostState",
    "Map",
    "Pacman",
    "ENV_ID",
    "MAX_EPISODE_STEPS",
    "flatten_float32",
    "make_env",
    "make_flat_env",
    "register_envs",
    "__version__",
]


def register_envs() -> str:
    """Register :class:`PacmanEnv` with Gymnasium (idempotent)."""
    from gymnasium.envs.registration import register, registry

    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="pacman_env.env:PacmanEnv",
            max_episode_steps=MAX_EPISODE_STEPS,
        )
    return ENV_ID


register_envs()
