# PacManDT

A **Gymnasium-compatible Pac-Man environment** for reinforcement learning and
Decision-Transformer research, with a fully headless (silent) mode and a human
play/inference mode.

The game rules, maze layouts and sprite assets are adapted from the MIT-licensed
[`paolodelia99/py-pacman`](https://github.com/paolodelia99/py-pacman) project.
The original Pygame game loop has been rewritten as a pure-Python simulation so
that it can run without a display, and wrapped in a standard
`gymnasium.Env`. See [`pacman_env/assets/LICENSE-pypacman.txt`](pacman_env/assets/LICENSE-pypacman.txt).

## Project layout

The repository is split into two top-level packages so the environment and the
Decision-Transformer tooling stay independent:

| path        | contents |
| ----------- | -------- |
| `pacman_env/` | The Gymnasium environment: `env.py`, `rendering.py`, `wrappers.py`, the Pygame-free simulation in `game/` and the sprite assets in `assets/`. Nothing here depends on DT training. |
| `dt/data`    | Everything about *data*: behaviour policies (`policies.py`), trajectory collection and the `.npz` format (`episodes.py`), the observation stack (`observations.py`), the collection protocol (`protocol.py`), `state_dim`/`act_dim`/`max_length` helpers (`dims.py`), the Hugging Face `DataCollector` (`hf_dataset.py`) and the `python -m dt.data` CLI (`cli.py`). |
| `dt/`         | The model and the trainer: `dt/dt.py` (`HuggingfaceDT`, `DTPolicy`, checkpoints), `dt/train.py` (`python -m dt.train`) and `dt/eval.py` (`python -m dt.eval`). |
| `docs/`       | Long-form notes, including [`docs/ghost-observations.md`](docs/ghost-observations.md): the ghost-observation and collection-protocol report. |
| `tests/`      | pytest suite for both packages. |

---

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Optional extras:

| extra   | what it adds                                             |
| ------- | -------------------------------------------------------- |
| `dev`   | `pytest`                                                  |
| `train` | `torch` + `transformers` + `datasets` for DT training      |
| `rl`    | `stable-baselines3` for classic RL baselines               |
| `hub`   | `datasets` + `huggingface_hub` for the HF collector        |

`pygame-ce` is a hard dependency, but it is imported **lazily**: with
`render_mode=None` (the default) it is never touched, so collection and training
run fine on a headless machine or inside a worker process.

---

## Quick start

```python
import gymnasium as gym
import pacman_env  # registers "PacManDT/PacMan-v0"

# Silent / headless - what you use for training
env = gym.make("PacManDT/PacMan-v0", layout="classic", render_mode=None)
obs, info = env.reset(seed=0)
for _ in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
env.close()
```

Human mode (opens a window; requires a display):

```python
env = gym.make("PacManDT/PacMan-v0", render_mode="human")
```

Other render modes: `"rgb_array"` (returns an `(H, W, 3)` frame) and `"ansi"`
(text board for notebooks / logs).

You can also use the class directly, which avoids the Gymnasium wrappers:

```python
from pacman_env import PacmanEnv

env = PacmanEnv(layout="classic", render_mode=None)
```

---

## Environment API

### Constructor

| argument        | default       | meaning                                                     |
| --------------- | ------------- | ----------------------------------------------------------- |
| `layout`        | `"classic"`   | maze name, see `Map.available_layouts()`                    |
| `render_mode`   | `None`        | `None`, `"human"`, `"rgb_array"`, `"ansi"`                   |
| `obs_type`      | `"state"`     | `"state"` (grid) or `"rgb"` (pixels)                         |
| `player_lives`  | `3`           | lives per episode                                            |
| `max_steps`     | `None`        | internal step cap; `gym.make` adds its own `TimeLimit`       |
| `frame_skip`    | `1`           | simulation frames advanced per `step()`                      |
| `draw_state`    | `False`       | draw the score/lives panel next to the maze in pixel mode    |
| `render_fps`    | `30`          | playback speed for `render_mode="human"`                     |

### Spaces

* **Actions** — `Discrete(5)`:
  `0` LEFT, `1` RIGHT, `2` UP, `3` DOWN, `4` NOOP.
* **Observations**
  * `obs_type="state"` → `Box(-1, 5, (rows, cols), int8)`
    where `-1` is a wall or a ghost, `0` empty floor, `1` pellet,
    `5` frightened/eaten ghost.
  * `obs_type="rgb"` → `Box(0, 255, (H, W, 3), uint8)`.

### Rewards

| event                    | reward |
| ------------------------ | ------ |
| pellet                   | `+1`   |
| power pellet             | `+2`   |
| eat a frightened ghost   | `+5`   |
| lose a life              | `-5`   |

### `info`

`layout`, `score`, `lives`, `pellets_remaining`, `win`, `game_over`, `mode`,
`steps`, `player_position`, `player_pixel_position`, `player_velocity`,
`ghost_positions`, `ghost_states`, `ghosts_vulnerable`.

### Layouts

| layout          | grid   | pellets | ghosts |
| --------------- | ------ | ------- | ------ |
| `classic`       | 22 × 19 | 181    | 4      |
| `google`        | 11 × 34 | 166    | 4      |
| `medium`        | 7 × 19  | 56     | 2      |
| `medium_small`  | 7 × 8   | 25     | 1      |
| `small`         | 7 × 6   | 13     | 1      |

---

## Gymnasium features

`pacman_env` registers `PacManDT/PacMan-v0` with
`max_episode_steps=1000`, so `gym.make` gives you the standard stack:
`PassiveEnvChecker`, `OrderEnforcing`, `TimeLimit`.

* `reset(seed=...)` seeds both the simulation **and** the action space; pixel
  observations are reproducible across resets.
* `render()` follows the Gymnasium contract (`human` / `rgb_array` / `ansi`).
* `__getstate__` / `__setstate__` drop the renderer, so the environment works
  with `gym.make_vec(..., vectorization_mode="async")` and
  `ProcessPoolExecutor`.
* Helper wrappers live in `pacman_env.wrappers`:
  `make_env`, `make_flat_env`, `flatten_float32`.

```python
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from pacman_env import ENV_ID

env = gym.make(ENV_ID, layout="classic")
check_env(env.unwrapped, skip_render_check=True)

vec = gym.make_vec(ENV_ID, num_envs=4, vectorization_mode="sync")
```

---

## Reinforcement-learning training

Install the baselines extra first (`pip install -e ".[rl]"`), then:

```python
import gymnasium as gym
import pacman_env
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

env = gym.make("PacManDT/PacMan-v0", layout="classic", render_mode=None)
check_env(env)
model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-4)
model.learn(total_timesteps=100_000)
model.save("dqn-pacman")
```

For pixel observations use `obs_type="rgb"` plus
`gym.wrappers.GrayscaleObservation` / `FrameStackObservation`, or the
`flatten_float32` wrapper for a flat `float32` vector.

---

## Observation modes: `state` vs `ghost`

`pacman_env` is deliberately untouched: its native observation is a single
`int8` grid (`obs_type="state"`) in which `-1` means *wall **or** a normal
ghost*, `5` means *vulnerable **or** eaten ghost*, and Pac-Man is not encoded at
all. That is a fine input for a CNN, but the Decision Transformer embeds every
**cell** with one linear layer (`state_dim` → `hidden_size`), so the collapsed
grid forces a single weight to mean both "wall" and "ghost".

`dt.data.observations` therefore adds an **observation stack** as a Gymnasium
wrapper. The environment, the maze and the game rules are unchanged; only the
tensor that reaches the transformer changes, and `state_dim` is derived from it
so `HuggingfaceDT`/`DecisionTransformerConfig` need **no modification**.

```python
from dt.data import make_ghost_env, GHOST_OBS_MODE, wrap_observation
from pacman_env import PacmanEnv

env = make_ghost_env(PacmanEnv(layout="classic"))   # PacmanEnv + GhostObservationWrapper
obs, info = env.reset(seed=0)
obs.shape, obs.dtype                                # (11, 22, 19) int8

# or wrap an env you already built (frame stacking etc. keeps working)
env = wrap_observation(PacmanEnv(layout="classic"), obs_mode=GHOST_OBS_MODE)
```

`obs_mode="state"` is the legacy behaviour and returns exactly what `pacman_env`
returns, so old datasets and checkpoints keep loading.

### Channels

| idx | channel            | values | why it is there |
| --- | ------------------ | ------ | --------------- |
| 0 | `wall`              | 0/1 | impassable tiles (border, wall, ghost door) |
| 1 | `pellet`            | 0/1 | pellet still on the tile |
| 2 | `power_pellet`      | 0/1 | power pellet still on the tile |
| 3 | `ghost_normal`      | 0..4 | **count** of dangerous ghosts on the tile |
| 4 | `ghost_vulnerable`  | 0..4 | count of frightened (edible) ghosts |
| 5 | `ghost_eaten`       | 0..4 | count of eaten ghosts (eyes returning home) |
| 6 | `player`            | 0/1 | Pac-Man's tile |
| 7 | `player_distance`   | 0..100 | BFS distance from Pac-Man over walkable tiles |
| 8 | `ghost_distance`    | 0..100 | BFS distance to the nearest **normal** ghost |
| 9 | `frightened_timer`  | 0..100 | remaining frightened time, scaled |
| 10 | `lives`            | 0..100 | lives left, scaled |

Scalars (channels 9–10) are broadcast over the grid so the stack stays a dense
`(C, rows, cols)` array and every mode has the same rank. Distances are capped at
`max_distance` (default `40`) and normalised to `0..100`; `100` marks
*unreachable*, which is exactly the signal a planner needs and a CNN cannot
invent from a static grid.

| layout         | `state` | `ghost` (full) | `ghost` (`--no-distance --no-globals`) |
| -------------- | ------- | -------------- | -------------------------------------- |
| `classic`      | 418     | **4598**       | 2926 |
| `google`       | 374     | **4114**       | 2618 |
| `medium`       | 133     | **1463**       | 931  |
| `medium_small` | 56      | **616**        | 392  |
| `small`        | 42      | **462**        | 294  |

```python
from dt.data import describe_observation, get_dims

print(describe_observation("ghost", layout="classic"))   # channel table above
get_dims("classic", obs_mode="ghost").state_dim          # 4598
get_dims("classic", obs_mode="ghost").channels           # 11 names
```

`GhostObservationConfig` controls the stack: `distance_fields` (channels 7–8),
`globals` (channels 9–10), `max_distance` and `dtype`. Turning the auxiliary
channels off shrinks `classic` from 4598 to 2926 inputs, which is a useful
ablation — the pure "per-tile entity" encoding without the planner hints.

### How the mode travels with the data

Nothing is inferred at training time. The mode is written into every artifact:

* `.npz` → `metadata["obs_mode"]` + `metadata["observation"]`
  (`mode`, `obs_shape`, `feature_dim`, `dtype`, `channels`, `layout`, and the
  config fields `distance_fields`, `globals`, `max_distance`).
* HF dataset → `pacman_dt_metadata.json` and the dataset card channel table;
  `DataCollector.load_from_disk` restores it.
* checkpoint → `payload["obs_mode"]` / `payload["observation"]`, and
  `metrics.json`.

`python -m dt.train` reads the mode from the dataset, `python -m dt.eval` reads
it from the checkpoint, and both reject a mismatched `--layout` instead of
silently feeding the model the wrong tensor. Use `--obs-mode ghost|state` to
force a mode and `--no-distance` / `--no-globals` / `--max-distance N` to shape
the stack.

---

## Collecting trajectories for a Decision Transformer

A Decision Transformer (DT) is trained on episodes in which every step carries
the observation, the action, the reward and the **return-to-go**. The `dt.data`
package ships a behaviour-policy module and a collector that produce exactly that.

### One command

```powershell
python -m dt.data --episodes 1000 --policy heuristic --out data/pacman_dt.npz
```

Progress and a summary are printed as it goes:

```
Collecting 1000 episodes on 'classic' with the heuristic policy ...
[  5.0%] episodes 50/1000 | mean return   44.10 | success  10.0% |   7200 steps/s
...
collected:
  episodes      : 1000
  transitions   : 1000000
  return        : mean 44.12  std 5.30  min 26.00  max 61.00
  score         : mean 1340.2
  episode length: mean 400.0
  success rate  : 12.4%
  win rate      : 0.0%

Saved 1000 episodes to data\pacman_dt.npz
```

> **`python -m dt.collect` is deprecated.** The legacy module still works — it
> translates its old flags and forwards to `python -m dt.data`, printing a
> warning — but `python -m dt.data` is the supported entry point.

### Flags

| flag                  | default              | meaning                                                       |
| --------------------- | -------------------- | ------------------------------------------------------------- |
| `--layout`            | `classic`            | maze layout                                                   |
| `--lives`             | `3`                  | lives per episode                                             |
| `--episodes`          | `1000`               | episodes to collect                                           |
| `--max-steps`         | `1000`               | per-episode step cap (`0` = unlimited)                        |
| `--policy`            | `heuristic`          | `heuristic`, `random` or `epsilon_greedy`                     |
| `--portfolio`         | off                  | mix behaviour policies, e.g. `heuristic=0.6,epsilon_greedy=0.3,random=0.1` (overrides `--policy`) |
| `--obs-mode`          | `state`              | `state` (legacy grid) or `ghost` (11-channel stack)            |
| `--no-distance`       | off                  | `ghost` mode: drop the two BFS distance channels               |
| `--no-globals`        | off                  | `ghost` mode: drop the frightened-timer / lives channels       |
| `--max-distance`      | `40`                 | `ghost` mode: BFS distance cap before normalisation            |
| `--balance-success`   | off                  | cap successful episodes at `--success-ratio` × failures        |
| `--success-ratio`     | `2.0`                | the ratio used by `--balance-success`                          |
| `--num-tiers`         | `4`                  | return quantiles reported (and used for curation)              |
| `--eval-fraction`     | `0.1`                | HF only: share of episodes held out for the `eval` split       |
| `--epsilon`           | `0.05`               | exploration rate for `epsilon_greedy`                         |
| `--gamma`             | `0.99`               | discount used for the returns-to-go                           |
| `--seed`              | `0`                  | base seed; episode *i* uses `seed + i`                        |
| `--frame-skip`        | `1`                  | simulation frames per recorded action                         |
| `--workers`           | `1`                  | parallel collector processes (identical results)              |
| `--success-threshold` | `50.0`               | total reward at which an episode is labelled `success`        |
| `--min-return`        | none                 | drop episodes below this total reward                         |
| `--keep-won-only`     | off                  | keep only episodes that cleared the maze                      |
| `--max-kept`          | none                 | cap the stored episodes (keeps the highest-return ones)       |
| `--max-length`        | longest episode      | pad/truncate every episode to this many steps (both formats)  |
| `--show-dims`         | off                  | print `state_dim`/`act_dim`/`max_length` per layout, then exit |
| `--out`               | `data/pacman_dt.npz` | output file                                                   |
| `--no-compress`       | off                  | write an uncompressed `.npz` (faster, larger)                 |
| `--quiet`             | off                  | hide progress output                                          |

### Behaviour policies

Data quality is the single biggest factor in DT performance, so the default is
`heuristic`, a tile-level BFS planner that eats the nearest pellet, dodges
ghosts, and chases frightened ones. Measured on `classic` over 1000 steps:

| policy      | mean return | mean score | steps/s |
| ----------- | ----------- | ---------- | ------- |
| `random`    | ≈ −6        | ≈ 60       | >20 000 |
| `heuristic` | ≈ 110–125   | ≈ 4 000    | ≈ 7 000 |

`epsilon_greedy` wraps the heuristic with a little random exploration, which
widens the state distribution a DT sees — useful when you want the dataset to
contain recovery behaviour. Custom policies are just callables
`(obs, info, env, rng) -> action`; pass one directly to
`collect_trajectories(policy=my_policy)`.

### Collection protocol

A DT can only be as good as the trajectories it imitates, and a dataset that is
100 % expert has two problems: it never shows the states where an agent must
*recover* (being chased, a life lost, a cornered position), and its
return-to-go distribution is so narrow that the conditioning input carries no
information. `dt.data.protocol` implements the protocol used for all
ghost-observation runs:

1. **Policy portfolio** — every episode is played by one of several behaviour
   policies in fixed, deterministic proportions
   (`heuristic=0.6, epsilon_greedy=0.3, random=0.1` by default). The expert
   supplies competent play, `epsilon_greedy` supplies near-miss and recovery
   states, `random` supplies the low-return anchor that makes the
   return-to-go conditioning meaningful. The schedule is smooth round-robin, so
   `workers=16` sees the same mix as `workers=1`.
2. **Quality tiers** — episodes are bucketed into `--num-tiers` return
   quantiles and the tier table is printed, so you can see whether the dataset
   actually spans a useful range of returns before you spend GPU hours.
3. **Success balancing** — `--balance-success` caps successful episodes at
   `--success-ratio` × the number of failures. Without it a strong portfolio
   can drift to >90 % wins, which makes `target_return` useless.
4. **Stable hash split** — the train/eval split is decided by a hash of the
   episode seed, not by position, so appending episodes to a dataset never moves
   an existing episode from `eval` into `train` (the classic silent-leak bug in
   iteratively collected datasets).

```python
from dt.data import CollectionProtocol, recommended_protocol

protocol = recommended_protocol("classic", episodes=4000, workers=16, seed=0)
print(protocol.describe())          # observation stack + portfolio + curation
result = protocol.run()             # collect -> curate -> split -> report
print(result.report["splits"])      # {'train': 3600, 'eval': 400}
result.dataset_dict()               # HF DatasetDict built from the same split
```

`CollectionProtocol` also exposes the steps individually (`schedule()`,
`collect()`, `curate()`, `split()`, `report()`), and the helpers
`parse_portfolio`, `build_policy_schedule`, `quality_tiers`, `balance_classes`,
`curate_episodes`, `split_by_hash`, `protocol_report` and `format_report` are
public if you want to script your own pipeline. From the command line the same
machinery is one flag:

```powershell
python -m dt.data --episodes 4000 --workers 16 `
    --obs-mode ghost --portfolio heuristic=0.6,epsilon_greedy=0.3,random=0.1 `
    --balance-success --num-tiers 4 --format hf --hf-out data/pacman_dt_hf
```

The printed report is what ends up in the dataset metadata, so a run is
reproducible from the artifact alone:

```text
episodes     : 4000 kept / 4000 collected
transitions  : 4000000
return       : mean 41.87  std 26.14  min -6.00  max 128.00
success rate : 24.6%   win rate 0.4%
split        : train 3600  eval 400
by policy    :
  heuristic         2400 eps  (60.0%)  mean return   61.20  success  40.1%
  epsilon_greedy    1200 eps  (30.0%)  mean return   38.55  success  22.0%
  random             400 eps  (10.0%)  mean return   -5.80  success   0.0%
by tier      :
  tier 0             1000 eps  return   -6.00 ..   12.00  success   0.0%
  tier 1             1000 eps  return   12.00 ..   44.00  success   1.2%
  tier 2             1000 eps  return   44.00 ..   70.00  success  33.5%
  tier 3             1000 eps  return   70.00 ..  128.00  success  63.8%
```


### Dimensions and `max_length`

`dt.data.dims` is the single place that answers "how big is my state and how long
is a sequence". Nothing is hard-coded: the numbers are read from the layout and
from the environment's own spaces.

```python
from dt.data import get_state_dim, get_act_dim, get_dims, describe_layouts

state_dim = get_state_dim("classic")            # 418  = 22 * 19
act_dim = get_act_dim()                         # 5    = LEFT/RIGHT/UP/DOWN/NOOP
dims = get_dims("classic")                      # DTDims(...) dataclass
dims.state_dim, dims.act_dim, dims.obs_shape    # 418, 5, (418,)  <- flattened
get_dims("classic", flatten=False).obs_shape    # (22, 19)        <- raw grid

get_state_dim("classic", frame_stack=4)         # 1672 (4 stacked frames)
get_state_dim("classic", obs_type="rgb")        # 755136 (rendered pixels)
get_state_dim("classic", obs_mode="ghost")      # 4598 (11 x 22 x 19)
get_dims("classic", obs_mode="ghost").channels  # the 11 channel names
print(describe_layouts(obs_mode="ghost"))       # ASCII table, same as --show-dims
```

The helpers also read the numbers off things you already have, so a training
script never needs to duplicate the constants:

```python
from dt.data import dims_from_env, dims_from_episodes, dims_from_dataset, dims_from_arrays

dims_from_env(env)                    # any PacmanEnv or gymnasium wrapper stack
dims_from_episodes(episodes)          # list[Episode] from collect_trajectories
dims_from_arrays(arrays)              # the dict returned by load_dataset
dims_from_dataset(dataset_dict)       # a Hugging Face DatasetDict
```

| layout         | `obs_shape` | `state_dim` | `act_dim` | episodes measured | recommended `max_length` |
| -------------- | ----------- | ----------- | --------- | ----------------- | ------------------------ |
| `classic`      | `(22, 19)`  | 418         | 5         | 30                | **1000**                 |
| `google`       | `(11, 34)`  | 374         | 5         | 30                | **1000**                 |
| `medium`       | `(7, 19)`   | 133         | 5         | 30                | **1000**                 |
| `medium_small` | `(7, 8)`    | 56          | 5         | 30                | **576**                  |
| `small`        | `(7, 6)`    | 42          | 5         | 30                | **256**                  |

`state_dim` is the flattened observation size (`rows * cols` for the default
`obs_type="state"`, times `frame_stack` if you stack frames); `act_dim` is always
`5`. `DTDims` also carries `obs_shape`, `obs_dtype`, `layout`, `frame_stack`,
`obs_type`, `obs_mode` and `channels`, so it can be passed straight into a model
factory. Every `dims_from_*` helper *infers* `obs_mode` from the data it is
given, so ghost datasets report `4598` on `classic` without being told.

The `state_dim` values in the table above are for `obs_mode="state"`; the
`ghost` stack multiplies them by the number of channels (see
[Observation modes](#observation-modes-state-vs-ghost)) — `classic` becomes
**4598**, `small` **462**.

#### Setting `max_length`

`max_length` is the per-episode sequence length after padding/truncation. Every
entry point accepts it, and all of them default to "the longest collected
episode":

```python
# 1. programmatic collection
save_dataset("data/pacman_dt.npz", episodes, max_length=1000)
arrays = pad_episodes(episodes, max_length=1000)["observations"]   # (N, 1000, 22, 19)

# 2. Hugging Face collector
from dt.data import DataCollector
collector = DataCollector(layout="classic", max_length=1000)
collector.collect(num_episodes=1000).save_to_disk("data/pacman_dt_hf")

# 3. command line (both formats)
#    python -m dt.data --episodes 1000 --max-length 1000 --out data/pacman_dt.npz
#    python -m dt.data --format hf --max-length 1000 --hf-out data/pacman_dt_hf
```

Episodes longer than `max_length` are **truncated** and shorter ones are
zero-padded; `mask`/`length` always describe the real steps, so nothing is lost
silently.

#### Why these numbers

Heuristic episodes were measured for 30 seeds per layout with `--max-steps 1000`:

| layout         | mean length | p95  | max  | truncation rate |
| -------------- | ----------- | ---- | ---- | --------------- |
| `classic`      | 1000.0      | 1000 | 1000 | 100 %           |
| `google`       | 1000.0      | 1000 | 1000 | 100 %           |
| `medium`       | 882.5       | 1000 | 1000 | 43 %            |
| `medium_small` | 373.7       | 544  | 551  | 0 %             |
| `small`        | 136.8       | 198  | 209  | 0 %             |

* `classic`, `google` and `medium` are **time-limit bound** — the agent almost
  never clears the maze before the step cap, so `max_length` should equal
  `--max-steps` (`1000`). Raising `--max-steps` without raising `--max-length`
  would throw data away.
* `medium_small` and `small` finish well inside the cap. Use
  `recommend_max_length(lengths)` (or the values above, rounded up to a multiple
  of 32 so the sequence length stays tensor-core friendly) to avoid padding 70 %
  of every row with zeros.

`recommend_max_length` does the arithmetic for you:

```python
from dt.data import recommend_max_length, recommend_context_length

recommend_max_length(layout="small")               # 256
recommend_max_length(lengths=observed_lengths)     # p95, rounded up to 32
recommend_context_length(max_length=1000)          # 30
```

For the transformer itself keep the **context length `K` small (20–64)** — the
Decision Transformer paper uses `K = 30` on Atari and D4RL uses 20. `K` is not
`max_length`: the dataset stores whole episodes, and `DTSequenceDataset` /
`make_dt_dataloader` slide a `context_length`-sized window over them
(`context_length=30` by default). A long `max_length` therefore costs disk and
padding, not attention.

### The `.npz` schema

Every array is `(episodes, max_length, ...)`, padded to the longest episode in
the batch unless `--max-length` / `max_length=` is given, in which case longer
episodes are truncated and shorter ones zero-padded. `mask` tells the model
which steps are real.

| array                 | shape                            | dtype     |
| --------------------- | -------------------------------- | --------- |
| `observations`        | `(N, T, *obs_shape)`             | `int8`    |
| `actions`             | `(N, T)`                         | `int64`   |
| `rewards`             | `(N, T)`                         | `float32` |
| `returns_to_go`       | `(N, T)`                         | `float32` |
| `timesteps`           | `(N, T)`                         | `int64`   |
| `terminated`          | `(N, T)`                         | `bool`    |
| `truncated`           | `(N, T)`                         | `bool`    |
| `mask`                | `(N, T)`                         | `bool`    |
| `lengths`             | `(N,)`                           | `int64`   |
| `episode_returns`     | `(N,)`                           | `float32` |
| `episode_scores`      | `(N,)`                           | `int64`   |
| `episode_success`     | `(N,)`                           | `bool`    |
| `episode_seeds`       | `(N,)`                           | `int64`   |

`metadata` is a JSON string with `format` (`pacman-dt/decision-transformer-v1`),
`created`, `episodes`, `steps`, `obs_shape`, `obs_dtype`, `action_dim`,
`max_length`, `obs_mode`, `observation` (the resolved observation stack:
`mode`, `obs_shape`, `feature_dim`, `dtype`, `channels`, `layout`, and the
config fields `distance_fields`, `globals`, `max_distance`), `policies` (the
behaviour policies that produced the episodes), `portfolio` (the requested
mixture), the CLI arguments, and a `statistics` block. With
`--obs-mode ghost`, `obs_shape` is `(11, rows, cols)` and `state_dim` is 11 × the
maze area.

`terminated` marks a genuine end of episode (win or game over) while `truncated`
marks a time-limit cut; returns-to-go are always computed over the steps that
were actually recorded.

### Loading it

```python
from dt.data.episodes import load_dataset

arrays, metadata = load_dataset("data/pacman_dt.npz")
observations = arrays["observations"]  # (N, T, 22, 19) int8
actions = arrays["actions"]  # (N, T) int64
returns_to_go = arrays["returns_to_go"]  # (N, T) float32
mask = arrays["mask"]  # (N, T) bool

# A DT sample: last `context_len` steps of an episode
context_len = 30
start = max(0, arrays["lengths"][0] - context_len)
sample = {
  "states": observations[0, start:start + context_len],
  "actions": actions[0, start:start + context_len],
  "returns_to_go": returns_to_go[0, start:start + context_len],
  "timesteps": arrays["timesteps"][0, start:start + context_len],
  "mask": mask[0, start:start + context_len],
}
```

### Programmatic collection

```python
from dt.data import collect_trajectories, pad_episodes, save_dataset, dataset_statistics

episodes = collect_trajectories(
  layout="classic",
  num_episodes=500,
  max_steps=1000,
  policy="heuristic",
  seed=0,
  gamma=0.99,
  workers=8,  # identical to workers=1, just faster
  progress=True,
)
print(dataset_statistics(episodes))
save_dataset("data/pacman_dt.npz", episodes, metadata={"layout": "classic"})
```

Individual episodes are `dt.data.Episode` objects with `observations`,
`actions`, `rewards`, `returns_to_go`, `timesteps`, `terminated`, `truncated`,
`seed`, `layout`, `score`, `total_reward`, `success` and `info`.

### Tips

* `--workers N` scales roughly linearly and produces byte-identical data, so
  parallel collection is safe.
* Keep `--max-steps` at or below the length you intend to feed the transformer,
  and set `--max-length` to match (see
  [Dimensions and `max_length`](#dimensions-and-max_length)); a 1000-step
  `classic` episode is ~420 kB of `int8` observations before compression.
* Raise `--success-threshold` (or use `--min-return`) to build a dataset of only
  good episodes, which is usually what makes a DT work.
* Episode *i* uses seed `seed + i`, so a dataset is fully reproducible from the
  metadata stored inside the file.

---

## Hugging Face dataset (`DataCollector`)

`dt.data.hf_dataset.DataCollector` is the Hugging Face native counterpart of the
`python -m dt.data` CLI: it plays the environment with a behaviour policy and
returns a `datasets.DatasetDict` you can save, push to the Hub and stream into a
Decision Transformer.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[hub]"
```

```python
from dt.data.hf_dataset import DataCollector

collector = DataCollector(
  layout="classic",
  policy="heuristic",
  max_steps=1000,
  workers=8,
  eval_fraction=0.1,  # 10% of episodes -> "eval" split
  obs_dtype="int8",
)
dataset_dict = collector.collect(num_episodes=1000)

print(collector)  # DataCollector(layout='classic', ..., splits={'train': 900, 'eval': 100})
print(collector.statistics)  # {'episodes': 1000, 'steps': 812345, ...}

collector.save_to_disk("data/pacman_dt_hf")  # arrow + README.md card + metadata.json
collector.push_to_hub("your-name/pacman-dt-classic", private=True)
```

Reload it later (locally or from the Hub):

```python
from dt.data.hf_dataset import DataCollector

collector = DataCollector.load_from_disk("data/pacman_dt_hf")
collector = DataCollector.from_hub("your-name/pacman-dt-classic")
dataset = collector.dataset_dict["train"]
```

### One row per episode

| column | type | meaning |
| --- | --- | --- |
| `observations` | `Array3D(max_length, rows, cols)` `int8` | padded frame stack |
| `actions` | `list<int64>` | `0` LEFT, `1` RIGHT, `2` UP, `3` DOWN, `4` NOOP |
| `rewards` | `list<float32>` | per-step reward |
| `returns_to_go` | `list<float32>` | discounted return from each step |
| `timesteps` | `list<int64>` | `0 .. length-1` |
| `terminated` / `truncated` | `list<bool>` | Gymnasium episode ends |
| `mask` | `list<bool>` | `True` = real step, `False` = padding |
| `length` | `int32` | unpadded episode length |
| `episode_id`, `seed` | `int64` | provenance / reproducibility |
| `layout` | `string` | maze name |
| `score`, `total_reward` | `int64`, `float32` | game score and summed reward |
| `success`, `win`, `pellets_remaining` | `bool`, `bool`, `int32` | outcome flags |

Every row is padded (or truncated) to the same `max_length`, so
`with_format("numpy")` hands back fully stacked arrays:

```python
ds = collector.dataset_dict["train"].with_format("numpy")
obs = ds[:]["observations"]   # (N, max_length, rows, cols)
actions = ds[:]["actions"]    # (N, max_length)
rtg = ds[:]["returns_to_go"]  # (N, max_length)
```

`flatten_to_transitions(ds)` gives the one-row-per-step alternative, and
`dataset_to_episodes(ds)` converts back to `Episode` objects.

### Feeding a Decision Transformer

```python
from dt.data.hf_dataset import DTSequenceDataset, make_dt_dataloader

windows = DTSequenceDataset(dataset, context_length=30, stride=1)
loader = make_dt_dataloader(dataset, context_length=30, batch_size=64, shuffle=True)

for batch in loader:
  # observations (B, 30, rows, cols) int8, actions (B, 30) int64,
  # returns_to_go (B, 30) float32, timesteps (B, 30), mask (B, 30)
  ...
```

`DTSequenceDataset` yields sliding windows over each episode and zero-pads the
tail of windows that run past the end, flagging the padding in `mask`.

### Command line

`python -m dt.data` writes either format:

```powershell
# numpy archive (default)
.\.venv\Scripts\python.exe -m dt.data --episodes 500 --out data/pacman_dt.npz

# Hugging Face dataset directory
.\.venv\Scripts\python.exe -m dt.data --format hf --episodes 500 `
    --eval-fraction 0.1 --hf-out data/pacman_dt_hf

# ...and upload it
.\.venv\Scripts\python.exe -m dt.data --format hf --episodes 500 `
    --hub-repo your-name/pacman-dt-classic --hub-private
```

Extra flags for `--format hf`: `--hf-out`, `--eval-fraction`, `--max-length`,
`--hub-repo`, `--hub-private`. All the collection/filtering flags
(`--policy`, `--min-return`, `--keep-won-only`, `--max-kept`, `--workers`, ...)
behave exactly as in the `.npz` mode.

---

## Training a Decision Transformer

`dt/train.py` is a complete, readable trainer: it collects (or loads) episodes,
reads `state_dim`/`act_dim` from the data, builds `HuggingfaceDT`, fits it with a
masked MSE loss over `context_length`-step windows, evaluates the greedy
`DTPolicy` in the real environment and writes a checkpoint.

It needs PyTorch and `transformers`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train]"
```

### One command

```powershell
# collect 50 heuristic episodes on the classic maze and train for 5 epochs
.\.venv\Scripts\python.exe -m dt.train --layout classic --episodes 50 --epochs 5

# ...then watch the trained policy play in the pygame window
.\.venv\Scripts\python.exe -m dt.train --eval-only `
    --checkpoint runs/dt-classic/checkpoint.pt --render human

# or train on a dataset you already collected
.\.venv\Scripts\python.exe -m dt.train --data data/pacman_dt_hf --epochs 20
.\.venv\Scripts\python.exe -m dt.train --data data/pacman_dt.npz --layout classic
```

`--data` accepts either a `DataCollector.save_to_disk` directory or a `.npz`
written by `dt.data`; omit it and episodes are collected on the fly. A quick
smoke run on the tiny layout takes a few seconds:

```powershell
.\.venv\Scripts\python.exe -m dt.train --layout small --episodes 16 `
    --max-steps 200 --epochs 2 --eval-episodes 2
```

### Flags

| flag | default | meaning |
| ---- | ------- | ------- |
| `--data` | *collect* | dataset directory or `.npz` file |
| `--layout` | `classic`, or the dataset's | maze layout |
| `--episodes` | `50` | episodes to collect when `--data` is omitted |
| `--max-steps` | `1000` | per-episode step limit |
| `--max-length` | recommended | pad/truncate episodes to this many steps |
| `--obs-mode` | the dataset's, else `state` | `state` or `ghost` observation stack |
| `--no-distance`, `--no-globals`, `--max-distance` | - | shape the `ghost` stack (see [Observation modes](#observation-modes-state-vs-ghost)) |
| `--portfolio`, `--balance-success`, `--num-tiers` | - | collection protocol used when `--data` is omitted |
| `--context-length` | `30` | steps of history the transformer sees (`K`) |
| `--epochs` | `5` | passes over the dataset |
| `--batch-size` | `64` | windows per gradient step |
| `--lr` | `1e-4` | AdamW learning rate |
| `--hidden-size`, `--n-layer`, `--n-head`, `--dropout` | `128`, `3`, `1`, `0.1` | transformer size |
| `--target-return` | mean return of the data | return the policy is conditioned on |
| `--eval-episodes` | `3` | greedy rollouts used to score the model (`0` skips) |
| `--render` | `none` | `human` opens the pygame window during evaluation |
| `--out-dir` | `runs/dt-<layout>` | where `checkpoint.pt` and `metrics.json` go |
| `--eval-only` / `--checkpoint` | - | skip training, just play a saved checkpoint |

Everything else (`--policy`, `--epsilon`, `--workers`, `--eval-fraction`,
`--keep-won-only`, `--device`, `--seed`, `--quiet`, ...) behaves as in
`python -m dt.data`.

> **A note on returns.** `dt.data` stores a `gamma`-discounted return-to-go
> (`--gamma`, default `0.99`), which is useful for statistics but is *not* what a
> Decision Transformer conditions on. The trainer therefore recomputes the
> plain, undiscounted sum of future rewards from the `rewards` column before
> training, so `--target-return 600` really does mean "score 600 points".

### Using the model directly

```python
import numpy as np
from dt.dt import DTPolicy, load_checkpoint
from dt.data import observation_from_metadata, wrap_observation
from pacman_env import PacmanEnv

model, payload = load_checkpoint("runs/dt-classic/checkpoint.pt")
obs_mode, obs_config = observation_from_metadata(payload.get("observation"))
env = wrap_observation(
    PacmanEnv(layout=payload["layout"], render_mode=None),
    obs_mode=obs_mode,          # the stack the weights were trained on
    config=obs_config,
)
policy = DTPolicy(
    model,
    context_length=payload["context_length"],
    target_return=payload["target_return"],   # the return we ask for
    return_scale=payload["return_scale"],     # how returns were normalised
)

obs, info = env.reset(seed=0)
policy.reset()
done = False
while not done:
    action = policy.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    policy.observe_reward(reward)          # keep the return-to-go honest
    done = terminated or truncated
```

The policy conditions the model on the *remaining* return,
`target_return - reward earned so far`, which is exactly the quantity it saw
during training - so a `target_return` of 600 means "play as if you were going
to score 600 more points". Call `policy.observe_reward(reward)` after every
`env.step`; if you skip it the policy simply keeps conditioning on the full
target, which is still valid but less faithful.

`HuggingfaceDT` is a thin `torch.nn.Module` over
`transformers.DecisionTransformerModel`: it takes *integer* actions, one-hot
encodes them, and returns `(loss, predictions)` when you pass `targets`:

```python
from dt.dt import HuggingfaceDT
from transformers import DecisionTransformerConfig

model = HuggingfaceDT(DecisionTransformerConfig(state_dim=418, act_dim=5, action_tanh=False))
loss, predictions = model(states, actions, returns_to_go, timesteps, attention_mask, targets=actions)
```

### Checkpoints

`checkpoint.pt` is a single file holding the weights plus everything needed to
rebuild them:

```python
{
  "format": "pacman-dt/decision-transformer-v1",
  "config": {...},            # DecisionTransformerConfig
  "model_state": {...},       # weights
  "context_length": 30,
  "return_scale": 28.4,       # returns_to_go were divided by this during training
  "target_return": 612.0,     # conditioning target used at inference
  "layout": "classic",
  "obs_mode": "ghost",        # the observation stack the weights expect
  "observation": {...},       # resolved stack: obs_shape, dtype, channels, config fields
  "dims": {"state_dim": 4598, "act_dim": 5, ...},
  "history": {"train_loss": [...], "eval_loss": [...]},
}
```

`load_checkpoint(path)` returns `(model, payload)`, so `--eval-only` needs no
extra flags: the layout, observation mode, context length and return scale all
travel with the weights. `python -m dt.eval` reads the same payload, rebuilds the
matching wrapper and refuses to run if `--layout` disagrees with the checkpoint.

### Evaluating a checkpoint (`python -m dt.eval`)

`dt/eval.py` is the standalone evaluator: it rebuilds the observation stack from
the checkpoint, plays headless episodes, optionally compares against the dataset
the model was trained on, and writes a JSON report.

```powershell
# newest checkpoint in runs/, 10 episodes per target return
python -m dt.eval

# a sweep over conditioning targets, with the full report on disk
python -m dt.eval --checkpoint runs/dt-classic/checkpoint.pt `
    --target-return 100 300 600 --episodes 20 --json

# watch it play
python -m dt.eval --mode interactive --render-fps 30
```

| flag | default | meaning |
| ---- | ------- | ------- |
| `--checkpoint` | newest `runs/*/checkpoint.pt` | checkpoint file or run directory |
| `--layout` | the checkpoint's | maze layout (a mismatch is refused) |
| `--target-return` | the checkpoint's | one or more conditioning targets |
| `--context-length` | the checkpoint's | steps of history (`K`) |
| `--obs-mode` | the checkpoint's | `state` or `ghost`; a mismatch is refused, not silently applied |
| `--mode` | `headless` | `headless` or `interactive` (pygame window) |
| `--episodes` | `10` | games per target return |
| `--max-steps` | `0` | step cap (`0` = let episodes end naturally) |
| `--data` | none | dataset to compare the agent against |
| `--json` | `<run>/eval.json` | write the full report (includes `obs_mode` and `observation`) |

The header always states which stack is in use, so a report is self-describing:

```text
PacManDT evaluation
checkpoint   : runs\dt-classic\checkpoint.pt
layout       : classic
observation  : ghost (11 channels)
model        : state_dim 4598, act_dim 5, ... params
context      : 30 steps
return scale : 28.40
target return: 600.0
device       : cuda
mode         : headless
```

### What to expect


The heuristic policy used for collection walks the maze with BFS, so a few dozen
episodes are enough to see the loss fall. A measured 40-episode, 12-epoch run on
the `small` layout (200-step episodes) looks like this:

```text
[train] epoch 1/12 loss=0.0829 acc=0.728 eval_loss=0.0345
[train] epoch 6/12 loss=0.0076 acc=0.984 eval_loss=0.0035
[train] epoch 12/12 loss=0.0023 acc=0.995 eval_loss=0.0017
[eval] episode 2/3 return=13.0 score=130 steps=115 win=True
[eval] mean_return=8.7 mean_score=120.0 mean_length=143
```

99% next-action accuracy on held-out windows, and the greedy policy wins 2 of 3
mazes (`win=True` means every pellet was eaten). The remaining episode wanders
into a loop and hits the step limit - with a small dataset the transformer
imitates the behaviour policy *locally*, so it can drift once it leaves the
states it was trained on. Scale `--episodes`, `--epochs` and `--hidden-size` up
for stronger play. On `classic` the episodes are 1000 steps, so keep
`--max-length 1000` and
`--context-length 30`: a 30-step window still covers the current maze plus enough
history to know which way Pac-Man is heading.

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

---

## License

MIT. Game logic and assets adapted from
[`paolodelia99/py-pacman`](https://github.com/paolodelia99/py-pacman) (MIT);
see [`pacman_env/assets/LICENSE-pypacman.txt`](pacman_env/assets/LICENSE-pypacman.txt).
