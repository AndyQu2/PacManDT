"""Evaluate a trained Decision Transformer on PacManDT.

One command, two modes::

    python -m dt.eval                                  # headless, newest run
    python -m dt.eval --checkpoint runs/dt-small/checkpoint.pt
    python -m dt.eval --mode interactive               # watch and play

``headless`` (the default)
    Play ``--episodes`` games with no window and print one line per episode plus
    a summary block.  Use this in scripts, over SSH, on a server, or to compare
    checkpoints - it never imports pygame and never opens a display.

``interactive``
    Open the pygame window and drive the evaluation yourself::

        SPACE        pause / resume
        TAB          hand control between the agent and the keyboard
        arrow keys   steer Pac-Man (WASD works too) while you have control
        R            restart the current episode from the same seed
        N            skip to the next episode
        [ / ]        lower / raise the target return by 10%
        Q or ESC     quit and print the report

    Watching the agent is the fastest way to see *why* a checkpoint scores what
    it scores.  Taking over with ``TAB`` is also a sanity check that the
    environment itself still plays properly.

Reading the report
------------------
``return``
    Sum of the per-step rewards (``+1`` pellet, ``+5`` frightened ghost,
    ``-5`` death).  This is what the Decision Transformer is trained on.
``score``
    The game's own score counter, shown for comparison.
``outcome``
    ``died``  = all lives lost, ``cleared`` = maze finished, ``step-limit`` =
    ``--max-steps`` ran out first.  A run where *every* episode ends on the step
    limit means the agent never actually finished a game - raise
    ``--max-steps`` (or pass ``0`` for a natural episode end).
``success rate``
    Share of episodes that cleared the maze *or* whose return reached
    ``--success-threshold``.  On a small maze the threshold is often above the
    best possible return, which is why a win counts as a success by itself.

Conditioning on a return
------------------------
A Decision Transformer plays *towards* a target return.  ``--target-return``
overrides the value stored in the checkpoint and accepts several values at
once, which turns the run into a sweep::

    python -m dt.eval --target-return 50 100 200 400

Same seeds for every target, so the rows are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from dt.data import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SUCCESS_THRESHOLD,
    GhostObservationConfig,
    OBS_MODES,
    dataset_statistics,
    episodes_from_npz,
    get_dims,
    observation_from_metadata,
    observation_metadata,
    wrap_observation,
)
from dt import DTPolicy, HuggingfaceDT, load_checkpoint

CHECKPOINT_NAME = "checkpoint.pt"
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_EPISODES = 10
DEFAULT_RENDER_FPS = 30
DEFAULT_RETURN_SCALE = 1.0
DEFAULT_RETURN_STEP = 0.1
NOOP_ACTION = 4

#: Names the interactive key handler understands (see :func:`apply_key`).
ACTION_QUIT = "quit"
ACTION_PAUSE = "pause"
ACTION_TOGGLE_AUTOPILOT = "toggle_autopilot"
ACTION_RESTART = "restart"
ACTION_NEXT = "next_episode"
ACTION_TARGET_UP = "target_up"
ACTION_TARGET_DOWN = "target_down"


# ----------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------
def resolve_device(name: Optional[str] = None) -> torch.device:
    """``cpu`` / ``cuda`` / ``mps`` when named, otherwise the best available.

    Unlike a hard-coded ``"cuda"`` default this falls back to the CPU, so the
    same command works on a laptop and on a GPU box.
    """
    if name:
        return torch.device(name)
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():  # pragma: no cover - macOS only
        return torch.device("mps")
    return torch.device("cpu")


def find_checkpoint(
    path: Optional[Union[str, Path]] = None, *, runs_dir: Union[str, Path] = DEFAULT_RUNS_DIR
) -> Path:
    """Locate a checkpoint file.

    ``path`` may point at the ``.pt`` file itself or at the directory holding it
    (``runs/dt-small``).  With no ``path`` the newest ``runs/*/checkpoint.pt``
    is used, so ``python -m dt.eval`` usually needs no arguments at all.
    """
    runs_dir = Path(runs_dir)
    if path is not None:
        candidate = Path(path)
        if candidate.is_dir():
            candidate = candidate / CHECKPOINT_NAME
        if not candidate.is_file():
            raise FileNotFoundError(f"no checkpoint at {candidate}")
        return candidate

    candidates = sorted(
        runs_dir.glob(f"*/{CHECKPOINT_NAME}"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint found under {runs_dir}{'/' if str(runs_dir) else ''} - "
            f"train one first with 'python -m dt.train', or pass --checkpoint <path>"
        )
    return candidates[0]


def make_env(
    layout: str,
    *,
    render_mode: Optional[str] = None,
    max_steps: Optional[int] = None,
    draw_state: bool = False,
    render_fps: int = DEFAULT_RENDER_FPS,
    obs_mode: str = "state",
    obs_config: Optional[GhostObservationConfig] = None,
):
    """Create a :class:`pacman_env.PacmanEnv` configured for evaluation.

    ``obs_mode="ghost"`` wraps the environment in
    :class:`dt.data.observations.GhostObservationWrapper` so the policy sees the
    same channels it was trained on.  The environment itself is never modified.
    """
    from pacman_env import PacmanEnv

    env = PacmanEnv(
        layout=layout,
        render_mode=render_mode,
        max_steps=max_steps,
        draw_state=draw_state,
        render_fps=render_fps,
    )
    return wrap_observation(env, obs_mode, obs_config)


def _import_pygame():
    """Import pygame with an actionable error message."""
    try:
        import pygame  # noqa: PLC0415 - only needed in interactive mode
    except ImportError as error:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "interactive mode needs pygame; install it with 'pip install pygame-ce'"
        ) from error
    return pygame


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
@dataclass
class EpisodeResult:
    """One evaluated episode."""

    episode: int
    seed: int
    total_return: float
    score: int
    length: int
    win: bool
    success: bool
    pellets_remaining: int
    terminated: bool
    truncated: bool
    target_return: Optional[float] = None

    @property
    def outcome(self) -> str:
        """``cleared``, ``died`` or ``step-limit``."""
        if self.win:
            return "cleared"
        if self.terminated:
            return "died"
        return "step-limit"

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = self.outcome
        return payload


def summarize(
    results: Sequence[EpisodeResult],
    *,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> Dict[str, Any]:
    """Aggregate per-episode results into the numbers the report prints."""
    if not results:
        raise ValueError("summarize() needs at least one episode")

    returns = np.asarray([item.total_return for item in results], dtype=np.float64)
    scores = np.asarray([item.score for item in results], dtype=np.float64)
    lengths = np.asarray([item.length for item in results], dtype=np.float64)
    wins = [item.win for item in results]
    deaths = [item.terminated and not item.win for item in results]
    timeouts = [item.truncated and not item.win for item in results]
    # Recomputed rather than read from ``item.success`` so the threshold passed
    # here is the one that decides the rate.  A cleared maze always counts:
    # on small layouts the reward total can never reach a classic-sized
    # threshold, and reporting a won game as "not a success" is just confusing.
    successes = [item.win or item.total_return >= success_threshold for item in results]

    return {
        "episodes": len(results),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_score": float(scores.mean()),
        "best_score": int(scores.max()),
        "mean_length": float(lengths.mean()),
        "win_count": int(sum(wins)),
        "death_count": int(sum(deaths)),
        "timeout_count": int(sum(timeouts)),
        "win_rate": float(np.mean(wins)),
        "success_rate": float(np.mean(successes)),
        "success_threshold": float(success_threshold),
        "returns": returns.tolist(),
        "scores": scores.astype(int).tolist(),
        "lengths": lengths.astype(int).tolist(),
    }


def format_episode(result: EpisodeResult, *, total: Optional[int] = None) -> str:
    """The one-line-per-episode log entry."""
    counter = f"{result.episode + 1}/{total}" if total else str(result.episode + 1)
    return (
        f"[eval] episode {counter} return={result.total_return:.1f} "
        f"score={result.score} steps={result.length} win={result.win} "
        f"({result.outcome})"
    )


# ----------------------------------------------------------------------
# Headless evaluation
# ----------------------------------------------------------------------
def evaluate(
    model: HuggingfaceDT,
    *,
    layout: str,
    episodes: int,
    context_length: int,
    target_return: Optional[float],
    return_scale: float = DEFAULT_RETURN_SCALE,
    max_steps: Optional[int] = None,
    seed: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    final_board: bool = False,
    draw_state: bool = False,
    verbose: bool = True,
    obs_mode: str = "state",
    obs_config: Optional[GhostObservationConfig] = None,
) -> List[EpisodeResult]:
    """Play ``episodes`` games headlessly and return one result per game.

    Episode ``i`` uses ``seed + i`` so a run is reproducible and two checkpoints
    evaluated with the same arguments face exactly the same mazes and ghosts.
    ``max_steps=None`` lets every episode end naturally (win or game over).
    """
    if episodes <= 0:
        raise ValueError("episodes must be a positive integer")

    env = make_env(
        layout,
        render_mode="ansi" if final_board else None,
        max_steps=max_steps,
        draw_state=draw_state,
        obs_mode=obs_mode,
        obs_config=obs_config,
    )
    policy = DTPolicy(
        model,
        context_length=context_length,
        target_return=target_return,
        return_scale=return_scale,
        device=device,
    )

    results: List[EpisodeResult] = []
    try:
        for index in range(episodes):
            episode_seed = seed + index
            observation, info = env.reset(seed=episode_seed)
            policy.reset()

            total, steps = 0.0, 0
            terminated = truncated = False
            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = env.step(
                    policy.act(observation)
                )
                policy.observe_reward(float(reward))
                total += float(reward)
                steps += 1

            won = bool(info.get("win", False))
            result = EpisodeResult(
                episode=index,
                seed=episode_seed,
                total_return=total,
                score=int(info.get("score", 0)),
                length=steps,
                win=won,
                success=won or total >= success_threshold,
                pellets_remaining=int(info.get("pellets_remaining", 0)),
                terminated=bool(terminated),
                truncated=bool(truncated),
                target_return=target_return,
            )
            results.append(result)
            if verbose:
                print(format_episode(result, total=episodes))

        if final_board:
            board = env.render()
            if board:
                print("\nFinal board")
                print(board)
    finally:
        env.close()

    return results


# ----------------------------------------------------------------------
# Interactive evaluation
# ----------------------------------------------------------------------
@dataclass
class InteractiveState:
    """Everything the keyboard can change while the window is open."""

    autopilot: bool = True
    paused: bool = False
    quit: bool = False
    restart: bool = False
    next_episode: bool = False
    target_return: Optional[float] = None

    def clear_episode_flags(self) -> None:
        """Drop the per-episode requests before starting a new episode."""
        self.restart = False
        self.next_episode = False


def apply_key(
    state: InteractiveState, name: str, *, step: float = DEFAULT_RETURN_STEP
) -> None:
    """Apply one named key press to ``state``.

    Split out from the pygame loop so it can be tested without a display.
    """
    if name == ACTION_QUIT:
        state.quit = True
    elif name == ACTION_PAUSE:
        state.paused = not state.paused
    elif name == ACTION_TOGGLE_AUTOPILOT:
        state.autopilot = not state.autopilot
        state.paused = False
    elif name == ACTION_RESTART:
        state.restart = True
    elif name == ACTION_NEXT:
        state.next_episode = True
    elif name == ACTION_TARGET_UP:
        if state.target_return is not None:
            state.target_return = round(state.target_return * (1.0 + step), 1)
    elif name == ACTION_TARGET_DOWN:
        if state.target_return is not None:
            state.target_return = max(0.0, round(state.target_return * (1.0 - step), 1))
    else:
        raise ValueError(f"unknown interactive action {name!r}")


def build_keymap(pygame) -> Dict[int, str]:
    """Map pygame key codes onto :func:`apply_key` names."""
    return {
        pygame.K_ESCAPE: ACTION_QUIT,
        pygame.K_q: ACTION_QUIT,
        pygame.K_SPACE: ACTION_PAUSE,
        pygame.K_TAB: ACTION_TOGGLE_AUTOPILOT,
        pygame.K_r: ACTION_RESTART,
        pygame.K_n: ACTION_NEXT,
        pygame.K_RIGHTBRACKET: ACTION_TARGET_UP,
        pygame.K_EQUALS: ACTION_TARGET_UP,
        pygame.K_LEFTBRACKET: ACTION_TARGET_DOWN,
        pygame.K_MINUS: ACTION_TARGET_DOWN,
    }


def build_manual_keymap(pygame) -> Dict[int, int]:
    """Map steering keys onto environment actions (arrows and WASD)."""
    return {
        pygame.K_LEFT: 0,
        pygame.K_a: 0,
        pygame.K_RIGHT: 1,
        pygame.K_d: 1,
        pygame.K_UP: 2,
        pygame.K_w: 2,
        pygame.K_DOWN: 3,
        pygame.K_s: 3,
    }


def manual_action(pressed: Any, keymap: Dict[int, int], *, noop: int = NOOP_ACTION) -> int:
    """First held direction key wins; otherwise stand still.

    ``pressed`` is whatever ``pygame.key.get_pressed()`` returns (any object that
    is indexable by key code), so this is testable with a plain dict.
    """
    for key, action in keymap.items():
        try:
            held = pressed[key]
        except (KeyError, IndexError):
            held = False
        if held:
            return int(action)
    return int(noop)


def _caption(state: InteractiveState, layout: str) -> str:
    who = "AGENT" if state.autopilot else "HUMAN"
    if state.paused:
        who = f"PAUSED ({who})"
    target = "off" if state.target_return is None else f"{state.target_return:.0f}"
    return (
        f"PacManDT eval | {layout} | {who} | target={target} | "
        "SPACE pause  TAB agent/human  R restart  N next  Q quit"
    )


def _tick(clock, fps: int) -> None:
    if fps and fps > 0:
        clock.tick(int(fps))


def interactive(
    model: HuggingfaceDT,
    *,
    layout: str,
    context_length: int,
    target_return: Optional[float],
    return_scale: float = DEFAULT_RETURN_SCALE,
    max_steps: Optional[int] = None,
    seed: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    render_fps: int = DEFAULT_RENDER_FPS,
    draw_state: bool = False,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    autopilot: bool = True,
    verbose: bool = True,
    obs_mode: str = "state",
    obs_config: Optional[GhostObservationConfig] = None,
) -> List[EpisodeResult]:
    """Open a window and evaluate the policy with a human in the loop.

    Runs until the window is closed or ``Q``/``ESC`` is pressed, then returns
    the episodes that were actually completed (restarted and skipped episodes
    are not recorded).
    """
    pygame = _import_pygame()
    env = make_env(
        layout,
        render_mode="human",
        max_steps=max_steps,
        draw_state=draw_state,
        render_fps=render_fps,
        obs_mode=obs_mode,
        obs_config=obs_config,
    )
    policy = DTPolicy(
        model,
        context_length=context_length,
        target_return=target_return,
        return_scale=return_scale,
        device=device,
    )
    keymap = build_keymap(pygame)
    steering = build_manual_keymap(pygame)
    clock = pygame.time.Clock()
    state = InteractiveState(autopilot=autopilot, target_return=target_return)

    results: List[EpisodeResult] = []
    episode = 0
    pygame.display.set_caption(_caption(state, layout))

    try:
        while not state.quit:
            observation, info = env.reset(seed=seed + episode)
            policy.reset(target_return=state.target_return)
            state.clear_episode_flags()
            env.render()

            total, steps = 0.0, 0
            terminated = truncated = False
            while not (
                terminated
                or truncated
                or state.quit
                or state.restart
                or state.next_episode
            ):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        state.quit = True
                    elif event.type == pygame.KEYDOWN:
                        name = keymap.get(event.key)
                        if name is None:
                            continue
                        apply_key(state, name)
                        if name in (ACTION_TARGET_UP, ACTION_TARGET_DOWN):
                            policy.reset(target_return=state.target_return)
                        pygame.display.set_caption(_caption(state, layout))

                if state.quit or state.restart or state.next_episode:
                    break
                if state.paused:
                    env.render()
                    _tick(clock, render_fps)
                    continue

                # The policy always runs: it needs the context window to stay
                # in sync, and a human override is reported back to it.
                action = policy.act(observation)
                if not state.autopilot:
                    action = manual_action(pygame.key.get_pressed(), steering)
                    policy.observe_action(action)

                observation, reward, terminated, truncated, info = env.step(action)
                policy.observe_reward(float(reward))
                total += float(reward)
                steps += 1
                env.render()
                _tick(clock, render_fps)

            if state.quit:
                break

            if terminated or truncated:
                won = bool(info.get("win", False))
                result = EpisodeResult(
                    episode=len(results),
                    seed=seed + episode,
                    total_return=total,
                    score=int(info.get("score", 0)),
                    length=steps,
                    win=won,
                    success=won or total >= success_threshold,
                    pellets_remaining=int(info.get("pellets_remaining", 0)),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    target_return=state.target_return,
                )
                results.append(result)
                if verbose:
                    print(format_episode(result))
                episode += 1
            elif state.next_episode:
                episode += 1
    finally:
        env.close()

    return results


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _row(label: str, value: Any) -> str:
    return f"  {label:<13}: {value}"


def format_target(target_return: Optional[float]) -> str:
    """``off`` for an unconditional policy, otherwise the number."""
    return "off" if target_return is None else f"{target_return:.1f}"


def _observation_label(
    obs_mode: str, config: Optional[GhostObservationConfig]
) -> str:
    """One-line description of the observation stack used for this run."""
    if config is None:
        return "state (single channel grid)"
    extras = []
    if not config.distance_fields:
        extras.append("no distance fields")
    if not config.globals:
        extras.append("no globals")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"ghost ({len(config.channel_names)} channels){suffix}"


def print_header(
    *,
    checkpoint: Path,
    model: HuggingfaceDT,
    layout: str,
    context_length: int,
    return_scale: float,
    targets: Sequence[Optional[float]],
    device: torch.device,
    mode: str,
    obs_mode: str = "state",
    obs_config: Optional[GhostObservationConfig] = None,
) -> None:
    print("PacManDT evaluation")
    print(_row("checkpoint", checkpoint))
    print(_row("layout", layout))
    print(_row("observation", _observation_label(obs_mode, obs_config)))
    print(
        _row(
            "model",
            f"state_dim {model.state_dim}, act_dim {model.act_dim}, "
            f"{model.count_parameters():,} params",
        )
    )
    print(_row("context", f"{context_length} steps"))
    print(_row("return scale", f"{return_scale:.2f}"))
    print(_row("target return", ", ".join(format_target(item) for item in targets)))
    print(_row("device", device))
    print(_row("mode", mode))
    print()


def print_report(summary: Dict[str, Any], *, title: str = "Results") -> None:
    """The summary block printed after the episodes."""
    print(title)
    print(_row("episodes", summary["episodes"]))
    print(
        _row(
            "return",
            f"mean {summary['mean_return']:.1f}  "
            f"std {summary['std_return']:.1f}  "
            f"min {summary['min_return']:.1f}  "
            f"max {summary['max_return']:.1f}",
        )
    )
    print(_row("score", f"mean {summary['mean_score']:.1f}  best {summary['best_score']}"))
    print(_row("length", f"mean {summary['mean_length']:.0f} steps"))
    print(_row("win rate", f"{100 * summary['win_rate']:.1f}%"))
    print(
        _row(
            "success rate",
            f"{100 * summary['success_rate']:.1f}%  "
            f"(win or return >= {summary['success_threshold']:.0f})",
        )
    )
    print(
        _row(
            "outcome",
            f"cleared {summary['win_count']}  "
            f"died {summary['death_count']}  "
            f"step-limit {summary['timeout_count']}",
        )
    )


def print_notes(summary: Dict[str, Any]) -> None:
    """Explain the numbers that most often surprise people."""
    episodes = summary["episodes"]
    if summary["timeout_count"] == episodes:
        print()
        print("note: every episode hit the step limit, so the agent never finished a game.")
        print("      raise --max-steps, or pass --max-steps 0 for a natural episode end.")
    elif summary["timeout_count"]:
        print()
        print(
            f"note: {summary['timeout_count']}/{episodes} episode(s) ended on the step "
            f"limit rather than on a win or a game over."
        )
    if summary["win_count"] == 0 and summary["timeout_count"] < episodes:
        print()
        print("note: the agent never cleared a maze in these episodes.")
    if summary["success_rate"] == 1.0 and summary["win_rate"] < 1.0:
        print()
        print(
            f"note: every episode reached the success threshold "
            f"({summary['success_threshold']:.0f}) without clearing the maze - "
            f"'success' is not 'win'."
        )


def print_sweep(rows: Sequence[Tuple[Optional[float], Dict[str, Any]]]) -> None:
    """Compare several target returns side by side."""
    print("Target-return sweep")
    print(f"  {'target':>8}  {'return':>8}  {'score':>8}  {'length':>8}  {'win%':>6}")
    for target, summary in rows:
        print(
            f"  {format_target(target):>8}  "
            f"{summary['mean_return']:>8.1f}  "
            f"{summary['mean_score']:>8.1f}  "
            f"{summary['mean_length']:>8.0f}  "
            f"{100 * summary['win_rate']:>6.1f}"
        )


def dataset_reference(path: Union[str, Path]) -> Dict[str, Any]:
    """Statistics of a saved dataset, for an agent-vs-demonstrations comparison."""
    path = Path(path)
    if path.is_dir():
        from dt.data import DataCollector

        collector = DataCollector.load_from_disk(path)
        stats = dict(collector.statistics)
        stats.setdefault("layout", collector.layout)
        return stats
    if path.suffix == ".npz":
        episodes = episodes_from_npz(path)
        stats = dict(dataset_statistics(episodes))
        stats["layout"] = episodes[0].layout
        return stats
    raise FileNotFoundError(
        f"--data must be a dataset directory or a .npz file, got {path}"
    )


def print_reference(
    stats: Dict[str, Any], summary: Dict[str, Any], *, layout: Optional[str] = None
) -> None:
    """Print the demonstrations next to the agent."""
    print()
    print("Reference (the demonstrations)")
    print(_row("episodes", stats.get("episodes", 0)))
    print(_row("layout", stats.get("layout", "unknown")))
    print(
        _row(
            "return",
            f"mean {stats.get('mean_return', 0.0):.1f}  "
            f"min {stats.get('min_return', 0.0):.1f}  "
            f"max {stats.get('max_return', 0.0):.1f}",
        )
    )
    print(_row("win rate", f"{100 * stats.get('win_rate', 0.0):.1f}%"))
    reference = float(stats.get("mean_return", 0.0))
    delta = summary["mean_return"] - reference
    percent = 100.0 * delta / reference if reference else 0.0
    print(_row("agent vs data", f"{delta:+.1f} return ({percent:+.1f}%)"))

    recorded = stats.get("layout", "unknown")
    if layout and recorded not in ("unknown", layout):
        print()
        print(
            f"note: the demonstrations were recorded on {recorded!r} but the agent is "
            f"playing {layout!r} - the two returns are not directly comparable."
        )


def print_next_steps(checkpoint: Path) -> None:
    print()
    print("Next steps")
    print(f"  watch it : python -m dt.eval --checkpoint {checkpoint} --mode interactive")
    print(f"  sweep    : python -m dt.eval --checkpoint {checkpoint} --target-return 50 100 200")
    print(f"  train    : python -m dt.train --data <dataset> --epochs 100")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _formatter() -> type:
    """Show ``(default: ...)`` *and* keep the epilog's line breaks."""

    class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    return Formatter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dt.eval",
        description="Evaluate a trained Decision Transformer on PacManDT.",
        formatter_class=_formatter(),
        epilog=(
            "examples:\n"
            "  python -m dt.eval\n"
            "  python -m dt.eval --checkpoint runs/dt-small/checkpoint.pt --episodes 20\n"
            "  python -m dt.eval --mode interactive\n"
            "  python -m dt.eval --target-return 50 100 200 400 --json\n"
            "  python -m dt.eval --data data/pacman_dt_small --episodes 20\n"
        ),
    )

    setup = parser.add_argument_group("what to evaluate")
    setup.add_argument("--checkpoint", default=None,
                       help="checkpoint file or run directory (default: newest runs/*/checkpoint.pt)")
    setup.add_argument("--layout", default=None,
                       help="maze layout (default: the layout stored in the checkpoint)")
    setup.add_argument("--target-return", type=float, nargs="+", default=None,
                       help="return to condition on; several values run a sweep "
                            "(default: the value stored in the checkpoint)")
    setup.add_argument("--context-length", type=int, default=None,
                       help="steps of history fed to the model (default: from the checkpoint)")
    setup.add_argument("--mode", choices=("headless", "interactive"), default="headless",
                       help="headless plays without a window; interactive opens one")
    setup.add_argument("--device", default=None,
                       help="cpu / cuda / mps (default: auto)")
    setup.add_argument("--obs-mode", default=None, choices=list(OBS_MODES),
                       help="override the observation stack (default: the one stored in "
                            "the checkpoint; a mismatch is refused, not silently applied)")

    run = parser.add_argument_group("episodes")
    run.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                     help="games to play per target return (headless mode)")
    run.add_argument("--max-steps", type=int, default=0,
                     help="per-episode step limit (0 = let episodes end naturally)")
    run.add_argument("--seed", type=int, default=0,
                     help="episode i uses seed+i, so runs are reproducible")
    run.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD,
                     help="return at which an episode counts as a success "
                          "(clearing the maze always counts)")
    run.add_argument("--final-board", action="store_true",
                     help="print the board of the last headless episode as text")

    window = parser.add_argument_group("interactive mode")
    window.add_argument("--manual", action="store_true",
                        help="start with the keyboard in control instead of the agent")
    window.add_argument("--render-fps", type=int, default=DEFAULT_RENDER_FPS,
                        help="playback speed for --mode interactive")
    window.add_argument("--draw-state", action="store_true",
                        help="draw the observation matrix next to the maze")

    report = parser.add_argument_group("reporting")
    report.add_argument("--data", default=None,
                        help="dataset to compare the agent against (directory or .npz)")
    report.add_argument("--json", nargs="?", const="auto", default=None,
                        metavar="PATH",
                        help="write the full report as JSON (default: <checkpoint dir>/eval.json)")
    report.add_argument("--quiet", action="store_true",
                        help="hide the per-episode lines")
    return parser


def resolve_json_path(value: Optional[str], checkpoint: Path) -> Optional[Path]:
    """Turn ``--json`` into a concrete path (``auto`` = next to the checkpoint)."""
    if value is None:
        return None
    if value == "auto":
        return checkpoint.parent / "eval.json"
    return Path(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.episodes <= 0 and args.mode == "headless":
        print("--episodes must be a positive integer", file=sys.stderr)
        return 2

    try:
        checkpoint = find_checkpoint(args.checkpoint)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    model, payload = load_checkpoint(checkpoint, device=device)

    layout = args.layout or payload.get("layout") or "classic"
    context_length = int(
        args.context_length or payload.get("context_length") or DEFAULT_CONTEXT_LENGTH
    )
    return_scale = float(payload.get("return_scale") or DEFAULT_RETURN_SCALE)
    max_steps = None if args.max_steps <= 0 else int(args.max_steps)
    targets: List[Optional[float]] = list(args.target_return or [payload.get("target_return")])

    # The observation stack is part of the checkpoint contract: a ghost-trained
    # model must be fed the ghost channels, a legacy one the plain grid.
    obs_mode, obs_config = observation_from_metadata(payload)
    if args.obs_mode and args.obs_mode != obs_mode:
        print(
            f"warning: checkpoint was trained with obs_mode={obs_mode!r}; "
            f"ignoring --obs-mode {args.obs_mode!r}",
            file=sys.stderr,
        )

    # A checkpoint only understands the layout it was trained on; catching the
    # mismatch here beats a confusing shape error inside the policy.
    expected = get_dims(layout, obs_mode=obs_mode, ghost_config=obs_config).state_dim
    if expected != model.state_dim:
        print(
            f"error: the checkpoint expects a {model.state_dim}-dim state but layout "
            f"{layout!r} with obs_mode={obs_mode!r} has {expected}. Pass the layout it "
            f"was trained on with --layout.",
            file=sys.stderr,
        )
        return 2

    print_header(
        checkpoint=checkpoint,
        model=model,
        layout=layout,
        context_length=context_length,
        return_scale=return_scale,
        targets=targets,
        device=device,
        mode=args.mode,
        obs_mode=obs_mode,
        obs_config=obs_config,
    )

    runs: List[Dict[str, Any]] = []
    all_results: List[EpisodeResult] = []

    if args.mode == "interactive":
        results = interactive(
            model,
            layout=layout,
            context_length=context_length,
            target_return=targets[0],
            return_scale=return_scale,
            max_steps=max_steps,
            seed=args.seed,
            device=device,
            render_fps=args.render_fps,
            draw_state=args.draw_state,
            success_threshold=args.success_threshold,
            autopilot=not args.manual,
            verbose=not args.quiet,
            obs_mode=obs_mode,
            obs_config=obs_config,
        )
        all_results.extend(results)
        if not results:
            print("no completed episodes to report", file=sys.stderr)
            return 0
        summary = summarize(results, success_threshold=args.success_threshold)
        print()
        print_report(summary)
        print_notes(summary)
        if args.data:
            print_reference(dataset_reference(args.data), summary, layout=layout)
        runs.append({"target_return": targets[0], "summary": summary})
    else:
        for index, target in enumerate(targets):
            if index:
                print()
            if len(targets) > 1 and not args.quiet:
                print(f"target return {format_target(target)}")
            results = evaluate(
                model,
                layout=layout,
                episodes=args.episodes,
                context_length=context_length,
                target_return=target,
                return_scale=return_scale,
                max_steps=max_steps,
                seed=args.seed,
                device=device,
                success_threshold=args.success_threshold,
                final_board=args.final_board,
                draw_state=args.draw_state,
                verbose=not args.quiet,
                obs_mode=obs_mode,
                obs_config=obs_config,
            )
            all_results.extend(results)
            summary = summarize(results, success_threshold=args.success_threshold)
            runs.append({"target_return": target, "summary": summary})

        print()
        if len(runs) > 1:
            print_sweep([(run["target_return"], run["summary"]) for run in runs])
        else:
            print_report(runs[0]["summary"])
            print_notes(runs[0]["summary"])
            if args.data:
                print_reference(dataset_reference(args.data), runs[0]["summary"], layout=layout)

    json_path = resolve_json_path(args.json, checkpoint)
    if json_path is not None:
        report = {
            "checkpoint": str(checkpoint),
            "layout": layout,
            "state_dim": model.state_dim,
            "act_dim": model.act_dim,
            "obs_mode": obs_mode,
            "observation": observation_metadata(obs_mode, config=obs_config, layout=layout),
            "context_length": context_length,
            "return_scale": return_scale,
            "device": str(device),
            "mode": args.mode,
            "max_steps": max_steps,
            "seed": args.seed,
            "success_threshold": args.success_threshold,
            "runs": runs,
            "episodes": [item.as_dict() for item in all_results],
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print()
        print(f"[save] report written to {json_path}")

    if not args.quiet and args.mode == "headless":
        print_next_steps(checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
