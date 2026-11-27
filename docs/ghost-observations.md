# Ghost observations and the data-collection protocol

*Report on the ghost-aware observation space and the collection protocol added to
`dt/` for the PacMan Decision Transformer.*

---

## 0. TL;DR

| change | where | effect |
| ------ | ----- | ------ |
| 11-channel ghost observation stack (`state_dim` 418 → **4598** on `classic`) | `dt/data/observations.py` (**new**) | the transformer can see ghosts as *separate, typed, countable* entities, sees Pac-Man, and gets two BFS distance fields that make "who is close to whom" linear for its per-cell embedding |
| observation mode as a *wrapper* + metadata in every artifact | `dt/data/observations.py`, `episodes.py`, `dims.py`, `hf_dataset.py`, `train.py`, `eval.py` | `pacman_env/` and `dt/dt.py` are **unchanged**; `state_dim` still flows through `DecisionTransformerConfig` |
| collection protocol: policy portfolio, quality tiers, success balancing, stable hash split, printable report | `dt/data/protocol.py` (**new**) | one command produces a dataset with expert + recovery + low-return episodes, a leak-free train/eval split and a self-describing report |
| `python -m dt.collect` rewritten as a deprecated shim | `dt/collect.py` | the three `from data import ...` imports broken in `96a7f77` are gone; legacy flags still work |

Nothing in `pacman_env/` was touched. `dt/dt.py` (`HuggingfaceDT`, `DTPolicy`,
`save_checkpoint`, `load_checkpoint`) was not touched either: the observation
stack only changes what is stored in `observations` and what `state_dim` is.

---

## 1. Scope and constraints

The brief was:

1. add a **ghost observation space** so the transformer can learn ghost-aware play
   on its own,
2. **optimise the data-collection protocol**,
3. keep `dt/`'s architecture and `pacman_env/` **unchanged**,
4. write this report.

Constraint (3) is the interesting one. The naive way to expose ghosts is to edit
`pacman_env/game/map.py::update_ghosts_position`, which is where the observation
grid is written. That would have:

* changed the public `obs_type="state"` contract, invalidating every dataset and
  checkpoint collected so far,
* made the environment (a shared dependency) depend on DT concerns,
* forced every downstream consumer to re-verify the game rules.

So the ghost stack is implemented as a **Gymnasium `ObservationWrapper`** that
reads the live `Game` object the environment already exposes and re-encodes it.
`state_dim` is derived from the wrapper's `observation_space`, so the model
factory never changes. The environment stays a single-channel, rules-only
simulator; the *decision* about what the learner sees lives in `dt/data`.

---

## 2. What the native observation cannot express

`PacmanEnv(obs_type="state")` returns a `Box(-1, 5, (rows, cols), int8)` grid
produced by `Map.update_ghosts_position()`:

| value | meaning |
| ----- | ------- |
| `-1` | wall **or** normal ghost **or** maze border |
| `0` | empty floor |
| `1` | pellet **or** power pellet |
| `5` | frightened ghost **or** eaten ghost (eyes returning home) |

Consequences for a Decision Transformer:

1. **The player is not encoded at all.** The transformer cannot tell where it
   is; it must integrate 30 steps of history to infer its own position, and a
   `K=30` window is not always enough after a teleport through the side tunnel.
2. **Ghosts are indistinguishable from walls.** "Wall ahead" and "ghost ahead"
   produce the same input, although one is fatal and the other is not. A linear
   per-cell embedding cannot separate them without context.
3. **Multiple ghosts collapse onto one tile.** `update_ghosts_position` writes a
   single value per cell, so two ghosts stacked in a corridor look like one.
4. **Frightened and eaten ghosts are the same value (`5`).** Chasing a
   frightened ghost (good, +200) and chasing eyes (worthless) are the same input.
5. **No distance information.** Nothing tells the model *how far* the nearest
   ghost is. A flat `state_dim=418` embedding sees 418 absolute coordinates and
   must learn a metric out of them.
6. **No global scalars.** Lives left and the remaining frightened timer (which
   decides whether chasing is safe) are only in `env.info`, never in `obs`.

Point 5 is the one that matters most for *this* architecture. The DT embeds each
cell with a single `Linear(state_dim → hidden)`; it has no convolution, no
relative-position bias and no explicit pairwise features. Absolute coordinates
are a bad basis for "the ghost is three tiles behind me": the same weight vector
must mean different things at different positions. A **BFS potential field**,
which is translation invariant and encodes graph distance directly, is the
cheapest way to hand that quantity to a per-cell embedding.

Note also *who* had this information before: the behaviour policy. The
`heuristic` policy reads `game.ghosts`, computes `_ghost_distances(...)`, keeps a
`danger_radius` around normal ghosts and chases frightened ones. The expert used
privileged state that the student never saw — textbook behaviour cloning of a
ghost-aware teacher into a ghost-blind student.

---

## 3. The ghost observation stack

`dt/data/observations.py`.

### 3.1 Layout

```
observations : (channels, rows, cols) int8          # stored per step
state_dim    : channels * rows * cols               # what the DT receives
flattening   : channel-major: c * rows * cols + r * cols + col
```

Channel-major is deliberate: it keeps the grid structure contiguous, so a future
convolutional encoder (or the `--draw-state` visualisation) can reshape without
guessing the axis order, and it matches the `.npz`/HF storage layout exactly
(no transpose at load time).

### 3.2 Channels

| idx | name | dtype/range | semantics | why it exists |
| --- | ---- | ----------- | --------- | ------------- |
| 0 | `wall` | 0/1 | impassable: wall, border, ghost door | separates "wall" from "ghost" (collision 1) |
| 1 | `pellet` | 0/1 | pellet still present | unchanged, but now its own plane |
| 2 | `power_pellet` | 0/1 | power pellet still present | tells the model *where it can become dangerous* (collision 2) |
| 3 | `ghost_normal` | 0..4 | count of dangerous ghosts on the tile | danger, one plane, keeps multiplicities (collision 3) |
| 4 | `ghost_vulnerable` | 0..4 | count of frightened ghosts | edible: chase (collision 4) |
| 5 | `ghost_eaten` | 0..4 | count of eaten ghosts (eyes) | harmless, ignore (collision 4) |
| 6 | `player` | 0/1 | Pac-Man's tile | the model no longer has to infer its own position (collision 1) |
| 7 | `player_distance` | 0..100 | BFS tile distance from Pac-Man | *outgoing* potential field: reachability of every corridor |
| 8 | `ghost_distance` | 0..100 | BFS tile distance to the nearest normal ghost | *incoming* danger field: how close is the threat, per tile |
| 9 | `frightened_timer` | 0..100 | remaining frightened frames / `VULNERABLE_DURATION` | chase-or-flee depends on it; only in `info` before |
| 10 | `lives` | 0..100 | lives left | risk appetite changes with lives; only in `info` before |

Design notes:

* **Counts instead of bits.** A tile can hold more than one ghost after a
  corridor merge. `+= 1` (capped by `int8`/`VALUE_SCALE`) preserves the
  multiplicity instead of losing it, at zero extra channels. It also lets the
  embedding scale linearly with threat density.
* **Three ghost planes, not one.** State is what matters, not identity: normal
  (avoid), vulnerable (chase), eaten (ignore). Splitting by state is worth two
  extra channels and removes the `5`-collision entirely.
* **Globals are broadcast planes.** `frightened_timer` and `lives` are constant
  over the grid. Keeping them as planes rather than separate scalars means the
  observation stays a single dense array of fixed rank — the DT's `state_dim` is
  one integer and `train.to_batch` needs no special case. The cost is
  `rows*cols` extra inputs (418 on `classic`), small next to the 7 semantic
  planes.
* **`int8`, values `0..100`.** `VALUE_SCALE = 100` gives 1 % resolution, keeps
  the stored tensor `int8` (4.6 kB per `classic` step, 462 B on `small`), and
  makes the observation space a plain `Box(0, 100, (C, r, c), int8)`. `float32`
  in `0..1` is available via `GhostObservationConfig(dtype="float32")` for
  anyone who wants it, at 4× the storage.

### 3.3 Distance fields

Both fields are 4-connected BFS over the **corridor graph** of the layout
(walls, border, `OUTSIDE` and the ghost door are impassable):

* `player_distance` — one BFS from Pac-Man's tile. This is the classic
  "navigation potential": the gradient points along the shortest path, so a
  policy can read "which neighbour is closer to me" directly.
* `ghost_distance` — one multi-source BFS from all *normal* ghosts. High values
  mean safe, low values mean danger. Frightened and eaten ghosts are excluded on
  purpose: during a frightened window the model should not be pushed away from
  the ghost it is supposed to eat, and eyes are harmless.

Quantisation: distances are clamped to `max_distance` (default `40`) and scaled
to `0..100`; **unreachable tiles are encoded as `max_distance`** ("as far as it
gets"). That is the right prior — an unreachable tile is never an immediate
threat — and it keeps the field dense and `int8`-friendly.

The graph is built **once per environment** (`GhostObservationBuilder`) as a
flat adjacency table, so a step costs two BFS passes over ≤ 418 nodes rather
than a graph rebuild.

Cost, measured on this machine (single process, random actions,
`render_mode=None`):

| layout | `state` | `ghost` (full) | `ghost` (`--no-distance --no-globals`) |
| ------ | ------- | -------------- | -------------------------------------- |
| `classic` | 34 600 steps/s | 4 800 steps/s | 29 400 steps/s |
| `small`   | 60 300 steps/s | 18 400 steps/s | 47 000 steps/s |

The two BFS passes dominate on the big maze (~7× slower collection, ~1.2× for
the core-only stack). With `--workers 16` a 1000-episode × 1000-step `classic`
dataset takes ≈ 3.5 min instead of ≈ 30 s — irrelevant next to training time,
and `--no-distance` is available if collection throughput ever matters more than
the fields.

### 3.4 Why this is the right size

| layout | `state` | `ghost` full | core only |
| ------ | ------- | ------------ | --------- |
| `classic` | 418 | **4598** | 2926 |
| `google` | 374 | **4114** | 2618 |
| `medium` | 133 | **1463** | 931 |
| `medium_small` | 56 | **616** | 392 |
| `small` | 42 | **462** | 294 |

11× the inputs is a real cost for the first `Linear` layer (`hidden=128` → 588 k
parameters instead of 53 k on `classic`), but it is small for a transformer
overall and it buys *explicit* entity semantics. The alternative — a CNN
encoder — would mean changing `dt/dt.py`, which the brief forbids.

Ablations are one flag away and are the recommended way to attribute any gain:

```powershell
--obs-mode ghost                                 # 11 channels, 4598 on classic
--obs-mode ghost --no-distance                   #  9 channels, 3762
--obs-mode ghost --no-globals                    #  9 channels, 3762
--obs-mode ghost --no-distance --no-globals      #  7 channels, 2926
--obs-mode ghost --max-distance 15               # coarser distance quantisation
```

### 3.5 Storage

`save_dataset` writes compressed `.npz` by default. Measured on 4 × 200-step
`classic` ghost episodes: 3.68 MB of raw `int8` → **65.8 kB on disk** (56×
compression, because most channels are sparse). Extrapolated, 1000 × 1000-step
`classic` ghost episodes are ≈ 4.6 GB raw but ≈ **80 MB compressed**, so the
ghost stack does not change the disk budget materially.

---

## 4. How the observation mode travels

An observation stack that is silently mismatched between collection, training
and evaluation is the classic source of "the model learned nothing" bug reports.
The mode is therefore written into **every** artifact and read back
automatically.

| artifact | field |
| -------- | ----- |
| `.npz` | `metadata["obs_mode"]`, `metadata["observation"]` (`mode`, `layout`, `obs_shape`, `feature_dim`, `dtype`, `channels`, `distance_fields`, `globals`, `max_distance`) |
| HF dataset | `pacman_dt_metadata.json` + the channel table in the dataset card; restored by `DataCollector.load_from_disk` |
| checkpoint | `payload["obs_mode"]`, `payload["observation"]`, `payload["dims"]["channels"]` |
| run dir | `metrics.json` → `obs_mode`, `observation` |
| eval report | `eval.json` → `obs_mode`, `observation` |

Resolution rules:

* `dt.train --data ...` → the **dataset** decides (`--obs-mode` is only used
  when collecting on the fly).
* `dt.train --eval-only` / `dt.eval` → the **checkpoint** decides.
* `dt.eval --obs-mode ...` → allowed, but a mismatch with the checkpoint is
  refused rather than silently applied; `--layout` is validated against
  `state_dim` for the same reason.
* `dims_from_episodes/arrays/dataset` **infer** the mode from the data, so
  `dims.state_dim == 4598` on a ghost dataset without being told.

`observation_from_metadata()` / `observation_metadata()` are the two functions
that implement the round trip, and `GhostObservationConfig.from_dict` tolerates
missing keys, so metadata written before a knob existed still loads.

---

## 5. Data-collection protocol

`dt/data/protocol.py`.

### 5.1 What was wrong before

The old pipeline (`python -m dt.collect`, then `python -m dt.data`) had four
problems for a return-conditioned model:

1. **One behaviour policy.** A pure expert dataset contains almost no *recovery*
   states — the states where a ghost is closing in, a life is lost, or Pac-Man is
   cornered — because the expert avoids them. A DT trained on it cannot act when
   it drifts off-distribution (compounding error), which is exactly what the
   earlier verification runs showed ("wanders into a loop and hits the step
   limit").
2. **Narrow return-to-go distribution.** If every episode returns 110–125, the
   conditioning input is nearly constant and the model cannot learn to trade
   return for behaviour — the entire point of a Decision Transformer.
3. **Positional train/eval split.** `eval_fraction` sliced by index. Append new
   episodes and previously-evaluated episodes move into `train`; the eval loss
   silently becomes optimistic. With a hash-free split, iterative data
   collection (the normal workflow) leaks.
4. **No curation or reporting.** `--min-return`/`--max-kept` existed, but nothing
   balanced successes against failures and nothing told you what the dataset
   actually contained before you spent GPU hours on it.

### 5.2 Policy portfolio

```python
DEFAULT_PORTFOLIO = (("heuristic", 0.6), ("epsilon_greedy", 0.3), ("random", 0.1))
```

| policy | role |
| ------ | ---- |
| `heuristic` (60 %) | competent play: the imitation target |
| `epsilon_greedy` (30 %) | heuristic + 10 % random actions: near-miss states, recoveries, "what does a competent agent do after a mistake" |
| `random` (10 %) | low-return anchor: makes return-to-go informative and teaches the model what *not* to do |

The schedule is **exact and deterministic**: `build_policy_schedule(n, portfolio,
seed)` produces a smooth round-robin with the requested counts (100 episodes →
exactly 60/30/10, shuffled by `seed`), and each episode's policy is recorded on
`Episode.policy` and in the dataset metadata. `workers=N` sees the same mix
because the schedule is computed before forking.

Why not a single mixed policy? Because the mixture needs to be *inspectable*.
`--portfolio` accepts any `name=weight` list and the report prints the realised
counts, so the dataset composition is an artifact rather than an accident.

### 5.3 Quality tiers

`quality_tiers(episodes, num_tiers)` buckets episodes by total return and the
report prints, per tier, the episode count, the return range and the success
rate. This is the fastest way to answer "is my dataset actually diverse?" before
training. `curate_episodes(..., num_tiers=...)` reuses the same bucketing.

### 5.4 Success balancing

`balance_classes(episodes, max_ratio=2.0)` caps successful episodes at
`max_ratio × failures`. A strong portfolio can drift to > 90 % wins (the
heuristic wins `small` almost every time), which flattens the return
distribution again. `--balance-success --success-ratio 2.0` keeps the success
signal while guaranteeing enough failures.

### 5.5 Stable hash split

`split_by_hash(episodes, eval_fraction, seed)` decides each episode's split from
`crc32(f"{seed}|{episode.seed}|{episode.policy}")`, not from its position:

* appending episodes never moves an existing episode between splits, so eval
  metrics stay comparable across collection rounds;
* the split is reproducible from the metadata alone;
* `episodes_to_dataset_dict` / `DataCollector.from_episodes` gained
  `train_episodes=` / `eval_episodes=` so an externally computed split is
  honoured instead of being recomputed positionally. Both the CLI and
  `ProtocolResult.dataset_dict()` pass the hash split through.

Edge case handled explicitly: with few episodes the hash can legitimately place
nothing in `eval`. That is warned about (CLI) / raised as a `UserWarning`
(`CollectionProtocol.run`) instead of silently producing an empty eval split.

### 5.6 One call, and the report

```python
from dt.data import recommended_protocol

result = recommended_protocol("classic", episodes=4000, workers=16, seed=0).run()
print(result.report["splits"])   # {'train': 3600, 'eval': 400}
result.dataset_dict()            # HF DatasetDict built from that same split
```

```
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

*(Illustrative numbers for the `classic` layout; the shape of the block is what
`format_report` produces, and it is written verbatim into the dataset metadata.)*

The report, the portfolio, the schedule counts and the curation settings are
stored in `metadata["protocol"]` / `metadata["report"]`, so a dataset explains
itself without the command line that produced it.

---

## 6. Reproduction recipe (for the training server)

```powershell
# 0. install
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[train,hub]"

# 1. inspect the stack and the dims first (cheap, no collection)
.\.venv\Scripts\python.exe -m dt.data --obs-mode ghost --show-dims

# 2. collect: ghost stack + protocol, both formats in one run
.\.venv\Scripts\python.exe -m dt.data `
    --layout classic --episodes 4000 --max-steps 1000 --max-length 1000 `
    --obs-mode ghost --workers 16 --seed 0 `
    --portfolio heuristic=0.6,epsilon_greedy=0.3,random=0.1 `
    --balance-success --num-tiers 4 `
    --format hf --hf-out data/pacman_dt_ghost --eval-fraction 0.1

# 3. train (the dataset's obs_mode wins; no flag needed)
.\.venv\Scripts\python.exe -m dt.train --data data/pacman_dt_ghost `
    --context-length 30 --hidden-size 256 --n-layer 4 --n-head 4 `
    --epochs 30 --batch-size 128 --lr 1e-4 --device cuda

# 4. evaluate: target-return sweep + dataset comparison
.\.venv\Scripts\python.exe -m dt.eval --checkpoint runs/dt-classic/checkpoint.pt `
    --target-return 100 300 600 --episodes 20 --data data/pacman_dt_ghost --json
```

Hyperparameters worth changing from the defaults for the bigger stack:

* `--hidden-size 256` and `--n-layer 4`: 4598 inputs justify more capacity than
  the 418-input baseline; `--hidden-size 128 --n-layer 3` still trains, it just
  underfits the extra structure.
* `--batch-size 128` with `--context-length 30`: unchanged from the baseline
  (the window, not the observation, dominates the sequence length).
* Keep `--max-length` equal to `--max-steps`: on `classic` the heuristic is
  time-limit bound (100 % of episodes hit the 1000-step cap), so a smaller
  `--max-length` truncates data.

### Ablations to run (in order of value)

| # | dataset | question |
| - | ------- | -------- |
| 1 | `--obs-mode ghost` vs `--obs-mode state`, same protocol | does the ghost stack help at all? |
| 2 | `ghost` vs `ghost --no-distance` | do the BFS fields carry the gain, or the entity planes? |
| 3 | `ghost` vs `ghost --no-globals` | does the frightened timer / lives matter? |
| 4 | single-policy heuristic vs `--portfolio` | does recovery data fix the loop-wandering failure mode? |
| 5 | `--balance-success` on/off | is the return-to-go conditioning actually used? |

Success criteria to look for in `metrics.json` / `eval.json`:

* `eval_loss` falls *and* the eval split is hash-based (no leakage);
* headless mean return beats the `state` baseline at the same `target_return`;
* **fewer deaths per episode** and **fewer step-limit terminations** — the
  ghost-aware model should stop walking into ghosts, which is the qualitative
  signature we are after;
* a *monotone* response to `--target-return`: if sweeping 100 → 300 → 600 does
  nothing, the conditioning is not being used (usually too narrow a return
  distribution — the portfolio fixes that).

---

## 7. Risks, limitations, and what was not done

* **No local training runs.** Per the brief, only structural validation was
  performed here (imports, CLI, end-to-end mini runs on CPU with 1 epoch, dims
  and split checks). The quantitative claims above are either measured
  micro-benchmarks (throughput, storage, dims) or explicitly illustrative (the
  report block).
* **`state_dim` inflation.** The first embedding grows 11×; on `classic` the
  model gets bigger, not just better-informed. Ablation 1 separates the two.
* **Distance fields are a hand-crafted prior.** They encode "graph distance
  matters", which is true here but is not learned. `--no-distance` exists
  precisely so the claim can be tested rather than assumed.
* **Ghost identity is still not encoded.** Four ghosts are indistinguishable
  apart from their state and position. Per-ghost planes would cost 12+ channels
  and are a natural follow-up if the model needs to track individual chasers.
* **The environment's `info["ghost_positions"]` remains the ground truth** for
  the policies; the wrapper reads `game` directly, so there is exactly one
  source of truth and no risk of the two drifting apart.
* **`tests/` is staged for deletion in this working tree**, so no pytest cases
  were added. The new modules are written to be import-safe without
  `torch`/`datasets` (`protocol.py` imports `hf_dataset` lazily). Validation
  performed: imports, `--show-dims`, unit-style checks of
  `parse_portfolio`/`build_policy_schedule`/`quality_tiers`/`balance_classes`/
  `curate_episodes`/`split_by_hash`, `CollectionProtocol.run()` end to end,
  ghost `.npz` collection, a 1-epoch training run, `dt.train --eval-only`, and
  `dt.eval` headless on both a ghost and a state checkpoint (including the
  refusal path for a layout/mode mismatch).

---

## 8. Files

| file | status | contents |
| ---- | ------ | -------- |
| `dt/data/observations.py` | **new** | channel spec, `GhostObservationConfig`, `ResolvedObservation`, BFS builder, `GhostObservationWrapper`, `make_ghost_env`, `wrap_observation`, metadata helpers, `describe_observation` |
| `dt/data/protocol.py` | **new** | `DEFAULT_PORTFOLIO`, `parse_portfolio`, `build_policy_schedule`, `quality_tiers`, `balance_classes`, `curate_episodes`, `split_by_hash`, `protocol_report`, `format_report`, `CollectionProtocol`, `ProtocolResult`, `recommended_protocol` |
| `dt/data/episodes.py` | modified | `Episode.obs_mode/policy/observation_meta`; `collect_episode`/`collect_trajectories` obs-mode + schedule plumbing; `save_dataset` header/metadata; `episodes_from_npz` round trip |
| `dt/data/dims.py` | modified | obs-mode aware `get_obs_shape`/`get_state_dim`/`get_dims`/`describe_layouts`; `DTDims.obs_mode`/`channels`; mode inference |
| `dt/data/hf_dataset.py` | modified | `policy`/`obs_mode` columns, metadata + card channel table, `describe()`, external-split kwargs |
| `dt/data/cli.py` | modified | `--obs-mode`, `--no-distance`, `--no-globals`, `--max-distance`, `--portfolio`, `--balance-success`, `--success-ratio`, `--num-tiers`; protocol reporting; hash split for HF |
| `dt/data/__init__.py` | modified | exports + package docstring |
| `dt/train.py` | modified | import fix, obs flags, dataset→checkpoint obs metadata, rollout wrapping, dims check |
| `dt/eval.py` | modified | obs-mode aware env/evaluate/interactive/header, `--obs-mode`, `eval.json` fields |
| `dt/collect.py` | rewritten | deprecated shim over `dt.data.cli.main` |
| `README.md` | modified | observation modes, protocol, new flags, eval section, checkpoint schema |
| `docs/ghost-observations.md` | **new** | this report |

`pacman_env/` and `dt/dt.py`: **untouched**.
