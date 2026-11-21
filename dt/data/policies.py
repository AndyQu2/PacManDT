"""Behaviour policies used to collect Decision-Transformer trajectories.

Every policy is a callable ``(obs, info, env, rng) -> action`` so it can be
passed straight to :func:`dt.data.episodes.collect_trajectories`.

``random`` gives low-quality data, while ``heuristic`` plays the game
competently (collects pellets, dodges ghosts, chases frightened ghosts), which
is what a Decision Transformer needs in order to learn anything useful.
"""

from collections import deque
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from pacman_env.game.action import Action
from pacman_env.game.constants import PELLET, POWER_PELLET
from pacman_env.game.ghost_state import GhostState

Policy = Callable[[np.ndarray, Dict[str, object], object, np.random.Generator], int]

#: ``(action, delta_row, delta_col)`` for the four movement actions.
MOVES: Tuple[Tuple[Action, int, int], ...] = (
    (Action.LEFT, 0, -1),
    (Action.RIGHT, 0, 1),
    (Action.UP, -1, 0),
    (Action.DOWN, 1, 0),
)

UNREACHABLE = 10_000


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def bfs_distance_map(maze, sources: Iterable[Tuple[int, int]]) -> np.ndarray:
    """Return the 4-connected walking distance from ``sources`` to every tile.

    Walls (as understood by the simulation, i.e. :meth:`Map.is_wall`) are
    impassable; unreachable tiles keep :data:`UNREACHABLE`.
    """
    rows, cols = maze.shape
    distances = np.full((rows, cols), UNREACHABLE, dtype=np.int32)
    queue = deque()

    for row, col in sources:
        if not maze.in_bounds(row, col) or maze.is_wall(row, col):
            continue
        if distances[row, col] == 0:
            continue
        distances[row, col] = 0
        queue.append((row, col))

    while queue:
        row, col = queue.popleft()
        next_distance = distances[row, col] + 1
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < rows and 0 <= n_col < cols):
                continue
            if distances[n_row, n_col] <= next_distance or maze.is_wall(n_row, n_col):
                continue
            distances[n_row, n_col] = next_distance
            queue.append((n_row, n_col))

    return distances


def bfs_with_parents(maze, start: Tuple[int, int], blocked: Optional[np.ndarray] = None):
    """Breadth-first search from ``start`` over walkable, unblocked tiles.

    Returns ``(distances, parents)`` where ``parents`` holds the flat index
    ``row * cols + col`` of each tile's BFS parent (``-1`` when unset).
    """
    rows, cols = maze.shape
    distances = np.full((rows, cols), UNREACHABLE, dtype=np.int32)
    parents = np.full((rows, cols), -1, dtype=np.int32)

    start_row, start_col = start
    if not maze.in_bounds(start_row, start_col):
        return distances, parents

    distances[start_row, start_col] = 0
    queue = deque([(start_row, start_col)])
    while queue:
        row, col = queue.popleft()
        next_distance = distances[row, col] + 1
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < rows and 0 <= n_col < cols):
                continue
            if distances[n_row, n_col] != UNREACHABLE:
                continue
            if maze.is_wall(n_row, n_col):
                continue
            if blocked is not None and blocked[n_row, n_col]:
                continue
            distances[n_row, n_col] = next_distance
            parents[n_row, n_col] = row * cols + col
            queue.append((n_row, n_col))

    return distances, parents


def _first_step(distances, parents, start, target, cols) -> Optional[Tuple[int, int]]:
    """Walk back from ``target`` and return the ``(d_row, d_col)`` first move."""
    if target == start or distances[target] >= UNREACHABLE:
        return None
    current = target
    for _ in range(int(distances.size)):
        parent = int(parents[current])
        if parent < 0:
            return None
        parent_row, parent_col = divmod(parent, cols)
        if (parent_row, parent_col) == start:
            return (current[0] - start[0], current[1] - start[1])
        current = (parent_row, parent_col)
    return None


def _game(env):
    """Unwrap ``env`` and return the :class:`~pacman_env.game.game.Game`."""
    return getattr(env, "unwrapped", env).game


def _is_blocked(game, player, action: Action) -> bool:
    """True when ``action`` would run into a wall on the very next pixel."""
    speed = player.speed
    x, y = player.x, player.y
    if action == Action.LEFT:
        return game.check_if_player_hit_wall(x - speed, y)
    if action == Action.RIGHT:
        return game.check_if_player_hit_wall(x + speed, y)
    if action == Action.UP:
        return game.check_if_player_hit_wall(x, y - speed)
    return game.check_if_player_hit_wall(x, y + speed)


_DELTA_TO_ACTION = {(d_row, d_col): action for action, d_row, d_col in MOVES}


# ----------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------
def random_policy(obs, info, env, rng) -> int:
    """Uniformly sample a valid action."""
    return int(rng.integers(0, int(getattr(env, "action_space").n)))


class HeuristicPolicy:
    """BFS planner that eats pellets, dodges ghosts and hunts frightened ghosts.

    A decision is taken once per tile: the planner searches the maze for the
    cheapest route to a target and then follows it.  The target is a reachable
    frightened ghost when one is close, otherwise the nearest pellet that is at
    least ``min_target_distance`` tiles away -- ties are broken towards the
    least visited tile, which keeps the player exploring instead of
    ping-ponging between two equally good directions.  The chosen action is
    held until the player reaches the next tile.

    Parameters
    ----------
    chase_radius:
        Maximum tile distance at which a frightened ghost is worth chasing.
    danger_radius:
        Tiles within this Chebyshev distance of a normal ghost are planned
        around; the plan is refreshed every step once a ghost gets close.
    min_target_distance:
        Preferred minimum walking distance to the pellet being targeted.
    jitter:
        Random tie-breaking noise, which keeps collected episodes diverse.
    """

    def __init__(
        self,
        chase_radius: int = 10,
        danger_radius: int = 1,
        min_target_distance: int = 3,
        jitter: float = 0.9,
    ):
        self.chase_radius = int(chase_radius)
        self.danger_radius = int(danger_radius)
        self.min_target_distance = max(1, int(min_target_distance))
        self.jitter = float(jitter)
        self._pellet_key: Optional[Tuple[str, int]] = None
        self._pellet_cells: Optional[np.ndarray] = None
        self._visits: Optional[np.ndarray] = None
        self._cell: Optional[Tuple[int, int]] = None
        self._action: Action = Action.NOOP
        self._position: Optional[Tuple[int, int]] = None
        self._target: Optional[Tuple[int, int]] = None

    # -- caching -------------------------------------------------------
    def reset(self) -> None:
        """Drop cached state (called at the start of every episode)."""
        self._pellet_key = None
        self._pellet_cells = None
        self._visits = None
        self._cell = None
        self._action = Action.NOOP
        self._position = None
        self._target = None

    def _pellets(self, maze) -> np.ndarray:
        # Pellets only ever disappear, so the pellet count is a cheap version key.
        key = (maze.layout_name, maze.get_number_of_pellets())
        if key != self._pellet_key:
            self._pellet_cells = np.argwhere(
                (maze.map_matrix == PELLET) | (maze.map_matrix == POWER_PELLET)
            ).astype(np.int32)
            self._pellet_key = key
        return self._pellet_cells

    @staticmethod
    def _ghost_distances(maze, ghosts) -> np.ndarray:
        rows, cols = maze.shape
        if not ghosts:
            return np.full((rows, cols), 99, dtype=np.int16)
        row_index = np.arange(rows)[:, None]
        col_index = np.arange(cols)[None, :]
        stacked = [
            np.maximum(np.abs(row_index - row), np.abs(col_index - col))
            for row, col in ghosts
        ]
        return np.min(stacked, axis=0).astype(np.int16)

    # -- policy --------------------------------------------------------
    def __call__(self, obs, info, env, rng) -> int:
        game = _game(env)
        maze = game.maze
        player = game.player
        row, col = int(player.nearest_row), int(player.nearest_col)

        normal_ghosts = [
            (int(ghost.nearest_row), int(ghost.nearest_col))
            for ghost in game.ghosts
            if ghost.state == GhostState.NORMAL
        ]
        frightened_ghosts = [
            (int(ghost.nearest_row), int(ghost.nearest_col))
            for ghost in game.ghosts
            if ghost.state == GhostState.VULNERABLE
        ]
        ghost_distances = self._ghost_distances(maze, normal_ghosts)

        if self._visits is None or self._visits.shape != maze.shape:
            self._visits = np.zeros(maze.shape, dtype=np.int32)

        position = (int(player.x), int(player.y))
        stalled = position == self._position
        threatened = bool(normal_ghosts) and int(ghost_distances[row, col]) <= self.danger_radius + 1

        if not (stalled or threatened) and self._cell == (row, col):
            self._position = position
            return int(self._action)

        self._visits[row, col] += 1
        action = self._plan(game, (row, col), ghost_distances, frightened_ghosts, rng)
        if stalled:
            # The player is wedged against a wall, so the planned move is illegal.
            action = self._free_action(game, (row, col), ghost_distances, action, rng)

        self._cell = (row, col)
        self._action = action
        self._position = position
        return int(action)

    def _plan(self, game, start, ghost_distances, frightened_ghosts, rng) -> Action:
        maze = game.maze
        blocked = ghost_distances <= self.danger_radius
        blocked[start] = False

        distances, parents = bfs_with_parents(maze, start, blocked)
        target = self._choose_target(maze, start, distances, frightened_ghosts, rng)
        if target is not None:
            step = _first_step(distances, parents, start, target, maze.shape[1])
            if step is not None:
                return _DELTA_TO_ACTION[step]
        return self._flee(maze, start, ghost_distances, rng)

    def _choose_target(self, maze, start, distances, frightened_ghosts, rng):
        reachable = distances < UNREACHABLE

        if frightened_ghosts:
            best = None
            for ghost in frightened_ghosts:
                distance = int(distances[ghost])
                if distance <= 0 or not reachable[ghost]:
                    continue
                if max(abs(ghost[0] - start[0]), abs(ghost[1] - start[1])) > self.chase_radius:
                    continue
                if best is None or distance < best[0]:
                    best = (distance, ghost)
            if best is not None:
                return best[1]

        pellets = self._pellets(maze)
        if pellets.size:
            # Stay committed to the current pellet target so the planner does not
            # dither between two equidistant pellets on opposite sides.
            sticky = self._target
            if sticky is not None:
                row, col = sticky
                if (
                    maze.in_bounds(row, col)
                    and int(maze.map_matrix[row, col]) in (PELLET, POWER_PELLET)
                    and reachable[row, col]
                    and distances[row, col] > 0
                ):
                    return sticky

            rows = pellets[:, 0]
            cols = pellets[:, 1]
            distance = distances[rows, cols].astype(np.float64)
            score = distance * 1000.0 + self._visits[rows, cols].astype(np.float64)
            for mask in (
                (distance >= self.min_target_distance) & reachable[rows, cols],
                (distance > 0) & reachable[rows, cols],
            ):
                candidates = np.flatnonzero(mask)
                if candidates.size:
                    values = score[candidates]
                    if self.jitter:
                        values = values + rng.random(candidates.size) * self.jitter
                    chosen = int(candidates[int(np.argmin(values))])
                    self._target = (int(rows[chosen]), int(cols[chosen]))
                    return self._target
        return None

    def _flee(self, maze, start, ghost_distances, rng) -> Action:
        row, col = start
        best_action, best_score = Action.NOOP, None
        for action, d_row, d_col in MOVES:
            n_row, n_col = row + d_row, col + d_col
            if not maze.in_bounds(n_row, n_col) or maze.is_wall(n_row, n_col):
                continue
            score = -10.0 * float(ghost_distances[n_row, n_col])
            score += float(self._visits[n_row, n_col])
            if self.jitter:
                score += rng.random() * self.jitter
            if best_score is None or score < best_score:
                best_score, best_action = score, action
        return best_action

    def _free_action(self, game, start, ghost_distances, preferred, rng) -> Action:
        """Pick a legal action, preferring ``preferred`` and then fleeing."""
        player = game.player
        if preferred != Action.NOOP and not _is_blocked(game, player, preferred):
            return preferred
        row, col = start
        candidates = []
        for action, d_row, d_col in MOVES:
            n_row, n_col = row + d_row, col + d_col
            if not game.maze.in_bounds(n_row, n_col) or game.maze.is_wall(n_row, n_col):
                continue
            if _is_blocked(game, player, action):
                continue
            candidates.append((float(ghost_distances[n_row, n_col]), float(self._visits[n_row, n_col]), action))
        if not candidates:
            return self._flee(game.maze, start, ghost_distances, rng)
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]


def epsilon_greedy_policy(base: Policy, epsilon: float = 0.1) -> Policy:
    """Wrap ``base`` with ``epsilon``-greedy exploration."""
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        # Keep the RNG stream identical to the wrapped policy.
        return base

    def policy(obs, info, env, rng) -> int:
        if rng.random() < epsilon:
            return random_policy(obs, info, env, rng)
        return int(base(obs, info, env, rng))

    policy.__name__ = f"epsilon_greedy_{getattr(base, '__name__', 'policy')}"
    return policy


def make_policy(spec, epsilon: float = 0.1) -> Policy:
    """Build a policy from a name (or pass a callable through unchanged).

    Accepted names: ``"random"``, ``"heuristic"`` (alias ``"greedy"``) and
    ``"epsilon_greedy"`` (alias ``"mixed"``).
    """
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"policy must be a name or a callable, got {type(spec)!r}")

    name = spec.lower().replace("-", "_")
    if name in ("random", "uniform"):
        return random_policy
    if name in ("heuristic", "greedy", "pellet"):
        return HeuristicPolicy()
    if name in ("epsilon_greedy", "mixed", "explore"):
        return epsilon_greedy_policy(HeuristicPolicy(), epsilon)
    raise ValueError(
        f"Unknown policy {spec!r}. Use 'random', 'heuristic' or 'epsilon_greedy'."
    )


__all__ = [
    "Policy",
    "MOVES",
    "bfs_distance_map",
    "bfs_with_parents",
    "random_policy",
    "HeuristicPolicy",
    "epsilon_greedy_policy",
    "make_policy",
]
