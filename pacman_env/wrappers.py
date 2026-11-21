"""Helpers that turn :class:`~pacman_env.env.PacmanEnv` into a training-ready env.

``PacmanEnv`` returns a ``(rows, cols)`` ``int8`` observation, which is compact
for tabular methods, CNNs and custom feature extractors.  Most deep RL
libraries (Stable-Baselines3, CleanRL, RLlib, ...) expect a *flat* ``float32``
``Box`` though, so the helpers below apply the standard Gymnasium wrappers in
the right order:

    make_flat_env()  ->  RecordEpisodeStatistics(TransformObservation(FlattenObservation(env)))
"""

from typing import Any

import gymnasium as gym
import numpy as np

from . import ENV_ID
from .env import PacmanEnv


def make_env(**kwargs: Any) -> gym.Env:
    """Create the registered environment with :func:`gymnasium.make`."""
    return gym.make(ENV_ID, **kwargs)


def flatten_float32(env: gym.Env) -> gym.Env:
    """Flatten the observation and cast it to ``float32``.

    Returns ``gym.wrappers.TransformObservation(gym.wrappers.FlattenObservation(env))``.
    """
    env = gym.wrappers.FlattenObservation(env)
    space = gym.spaces.Box(
        low=env.observation_space.low.astype(np.float32),
        high=env.observation_space.high.astype(np.float32),
        dtype=np.float32,
    )
    return gym.wrappers.TransformObservation(
        env, lambda obs: np.asarray(obs, dtype=np.float32), space
    )


def make_flat_env(**kwargs: Any) -> gym.Env:
    """Training-ready environment: flat ``float32`` observations + episode stats.

    ``info["episode"]`` then contains ``r`` (return), ``l`` (length) and
    ``t`` (elapsed time) once an episode finishes.
    """
    env = make_env(**kwargs)
    env = flatten_float32(env)
    return gym.wrappers.RecordEpisodeStatistics(env)


__all__ = ["PacmanEnv", "make_env", "flatten_float32", "make_flat_env"]
