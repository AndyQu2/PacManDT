# PacManDT — Pac-Man for Gymnasium

A [Gymnasium](https://gymnasium.farama.org/) environment for Pac-Man with a **silent
(headless) mode** for reinforcement-learning training and a **human mode** for playing or
recording episodes.

The simulation is an adaptation of the MIT-licensed
[`paolodelia99/py-pacman`](https://github.com/paolodelia99/py-pacman) project. The game
rules, maze layouts and sprite assets are reused; the Pygame-dependent simulation loop has
been refactored so that **importing or stepping the environment never imports Pygame**.

> This document covers the environment package only. Behaviour policies,
> trajectory collection and the Decision-Transformer dataset formats live in the
> sibling [`dt.data`](../README.md#project-layout) package.

## Features

- Standard Gymnasium 1.x API (`reset`/`step` 5-tuple, `observation_space`, `action_space`, seeding).
- Registered with `gymnasium.register` (`PacManDT/PacMan-v0`), so `gym.make`, `gym.make_vec`,
  `gym.spec` and `env_checker` all work out of the box.
- `render_mode=None` → fully headless, zero Pygame imports.
- `render_mode="human"` → playable window with arrow-key / WASD controls.
- `render_mode="rgb_array"` → off-screen rendering, returns `uint8` frames.
- `render_mode="ansi"` → text drawing of the maze (no Pygame).
- `obs_type="rgb"` for pixel observations, `draw_state=True` to visualise the observation matrix.
- Deterministic under `reset(seed=...)`, `frame_skip` for action repeat, `TimeLimit` truncation.
- Picklable / deep-copyable, so `AsyncVectorEnv` (subprocess workers) works.
- Five bundled layouts, 1–4 ghosts, A\* ghost pathfinding, power pellets and frightened mode.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -e .
```

`pygame-ce` is only needed for rendering, but it is installed by default because it provides
the `pygame` module used by the renderer. `pip install -r requirements.txt` installs the
runtime dependencies without the package metadata.

## Quick start

### Silent mode (training)

```python
import gymnasium as gym
import pacman_env

env = gym.make("PacManDT/PacMan-v0", render_mode=None)
obs, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

print(info["score"], info["pellets_remaining"], info["win"])
env.close()
```

### Human mode (playing)

```bash
python example.py                 # arrow keys / WASD, R restart, Esc quit
python example.py --layout medium --lives 5 --state
```

### Recording frames

```bash
python examples/rgb_array_frames.py --frames 10 --out frames
```

### Text rendering

```python
import gymnasium as gym, pacman_env

env = gym.make("PacManDT/PacMan-v0", render_mode="ansi", layout="medium_small")
env.reset(seed=0)
print(env.render())
```

More examples live in [`examples/`](examples): `random_agent.py`, `manual_play.py`,
`rgb_array_frames.py`, `train_qlearning.py`, `train_sb3.py`.

## Reinforcement-learning training

The environment is registered with Gymnasium and is ready for the standard training
tooling.

```python
import gymnasium as gym
import pacman_env

env = gym.make("PacManDT/PacMan-v0", render_mode=None)   # TimeLimit(1000) applied
obs, info = env.reset(seed=0)                            # seeds np_random + action_space
```

### What is supported

| Gymnasium feature | Notes |
| --- | --- |
| `gym.make(id, **kwargs)` | `PassiveEnvChecker`, `OrderEnforcing` and `TimeLimit(max_episode_steps=1000)` are applied automatically |
| `gym.make(id, max_episode_steps=N)` | Override the limit per run; `-1` disables the `TimeLimit` wrapper |
| `gym.make_vec(id, num_envs=N)` | Returns a `SyncVectorEnv`; `AsyncVectorEnv` also works because the env is picklable |
| `gym.spec(id)` / `env.spec` | `max_episode_steps=1000`, `render_modes=["human", "rgb_array", "ansi"]` |
| `reset(seed=...)` | Seeds `np_random`, ghost behaviour **and** `action_space.sample()` |
| `check_env(env)` | Passes `gymnasium.utils.env_checker.check_env` and SB3's `env_checker` |
| Wrappers | `TimeLimit`, `Autoreset`, `RecordEpisodeStatistics`, `FlattenObservation`, `TransformObservation`, `RenderCollection` (`render_mode="rgb_array_list"`) |
| `render_mode="human"` | Native window; no `HumanRendering` wrapper needed |
| Pickling / `deepcopy` | Supported (`Pygame` surfaces are dropped and rebuilt on demand) |

Episodes end with `terminated=True` on win / game over, and `truncated=True` when the
`TimeLimit` (1000 steps by default) is reached. `PacmanEnv(max_steps=...)` adds an
*additional* internal limit; it defaults to `None` so the wrapper stays in charge.

### Training-ready helpers

`pacman_env.wrappers` provides the glue most deep-RL libraries expect:

```python
import pacman_env

env = pacman_env.make_flat_env(layout="classic")   # flat float32 obs + episode stats
obs, info = env.reset(seed=0)
obs.shape   # (418,)  float32
# ... after an episode ends: info["episode"] == {"r": return, "l": length, "t": time}
```

- `pacman_env.make_env(**kwargs)` → `gym.make("PacManDT/PacMan-v0", **kwargs)`
- `pacman_env.flatten_float32(env)` → `FlattenObservation` + `float32` `TransformObservation`
- `pacman_env.make_flat_env(**kwargs)` → the above plus `RecordEpisodeStatistics`

The raw `(rows, cols)` `int8` observation is kept by default because it is compact and
works well with CNNs or custom feature extractors; flatten it only when the policy expects
a 1-D vector.

### Vectorised training

```python
import gymnasium as gym, pacman_env

venv = gym.make_vec("PacManDT/PacMan-v0", num_envs=8, max_episode_steps=1000)
obs, info = venv.reset(seed=0)          # (8, 22, 19) int8
obs, rew, term, trunc, infos = venv.step(venv.action_space.sample())
```

### Stable-Baselines3

```bash
pip install -e ".[train]"
python examples/train_sb3.py --layout small --timesteps 20000
```

```python
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import pacman_env

venv = make_vec_env("PacManDT/PacMan-v0", n_envs=4,
                    env_kwargs={"render_mode": None, "layout": "small"},
                    wrapper_class=pacman_env.flatten_float32)
model = DQN("MlpPolicy", venv, verbose=1).learn(20_000)
```

The same API works with CleanRL, RLlib or a hand-written loop (see
`examples/train_qlearning.py` for a dependency-free tabular baseline).

## Environment API

| Item | Value |
| --- | --- |
| Environment id | `PacManDT/PacMan-v0` (or construct `pacman_env.PacmanEnv` directly) |
| Actions | `Discrete(5)` — `0` left, `1` right, `2` up, `3` down, `4` no-op (brake) |
| Observation (`obs_type="state"`) | `Box(-1, 5, (rows, cols), int8)` — `-1` wall/ghost, `0` empty, `1` pellet, `5` frightened/eaten ghost |
| Observation (`obs_type="rgb"`) | `Box(0, 255, (height, width, 3), uint8)` |
| Reward | pellet `+1`, power pellet `+2`, eating a frightened ghost `+5`, losing a life `-5` |
| `terminated` | all pellets eaten (win) or all lives lost |
| `truncated` | `TimeLimit` reached (`max_episode_steps=1000` via the registered spec) |

Constructor arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `layout` | `"classic"` | Layout name (see below) |
| `render_mode` | `None` | `None`, `"human"`, `"rgb_array"` or `"ansi"` |
| `obs_type` | `"state"` | `"state"` or `"rgb"` |
| `player_lives` | `3` | Lives per episode |
| `max_steps` | `None` | Optional internal limit; `None` leaves truncation to the `TimeLimit` wrapper |
| `frame_skip` | `1` | Simulation ticks per `step()` call |
| `draw_state` | `False` | Draw the observation matrix next to the maze |
| `render_fps` | `30` | Target frame rate for `"human"` mode |

`info` contains `layout`, `score`, `lives`, `pellets_remaining`, `win`, `game_over`, `mode`,
`steps`, `player_position`, `player_pixel_position`, `player_velocity`, `ghost_positions`,
`ghost_states` and `ghosts_vulnerable`.

### Layouts

| Layout | Grid (rows × cols) | Pellets | Ghosts |
| --- | --- | --- | --- |
| `classic` | 22 × 19 | 181 | 4 |
| `google` | 11 × 34 | 166 | 4 |
| `medium` | 7 × 19 | 56 | 2 |
| `medium_small` | 7 × 8 | 25 | 1 |
| `small` | 7 × 6 | 13 | 1 |

## Project layout

```
pacman_env/
  env.py           Gymnasium environment (silent / human / rgb_array / ansi)
  wrappers.py      Training helpers (flat float32 obs, episode statistics)
  rendering.py     Lazy Pygame renderer (imported only when rendering)
  game/            Pygame-free simulation: map, pacman, ghosts, A*, rules
  assets/          Layouts (.lay), tiles, sprites, fonts (MIT, see LICENSE-pypacman.txt)
examples/          Runnable example scripts
tests/             pytest suite
main.py            Keyboard-controlled human-mode entry point
```

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

The suite runs `gymnasium.utils.env_checker.check_env`, checks API shapes, seeded
determinism, reward/termination logic, rendering output (including `ansi`), all layouts, the
registered spec and `TimeLimit`, vector environments (`SyncVectorEnv`, `gym.make_vec`),
wrapper composition (`Autoreset`, `RecordEpisodeStatistics`, `FlattenObservation`,
`TransformObservation`, `RenderCollection`), pickling/deep-copying, and asserts that Pygame
is never imported in silent mode.

## Attribution and license

- Game rules, layouts and assets adapted from
  [`paolodelia99/py-pacman`](https://github.com/paolodelia99/py-pacman) (MIT,
  © 2020 Paolo D'Elia) — see [`assets/LICENSE-pypacman.txt`](assets/LICENSE-pypacman.txt).
- This adaptation is released under the MIT License — see [`LICENSE`](LICENSE).
- Pac-Man is a trademark of Bandai Namco Entertainment; this project is an
  unaffiliated educational reimplementation.
