"""Collect PacMan demonstrations and save them to disk.

This is the *data* half of the Decision-Transformer workflow, in one small file::

    python -m dt.collect                                    # 50 games on 'small'
    python -m dt.train --data data/pacman_dt_small           # then train on them

What ends up on disk is everything a Decision Transformer needs::

    observations    (episodes, max_length, *obs_shape)   int8
    actions         (episodes, max_length)               int64
    rewards         (episodes, max_length)               float32
    returns_to_go   (episodes, max_length)               float32
    timesteps       (episodes, max_length)               int64
    mask            (episodes, max_length)               bool   (True = real step)

plus ``length``, ``episode_return``, ``score`` and ``layout`` so the trainer can
read ``state_dim``/``act_dim`` and the maze from the data itself.

The flow is deliberately linear - **collect -> filter -> save -> verify** - and
the program reloads what it just wrote before exiting, so a ``0`` exit status
means the dataset on disk is readable.

Choosing a behaviour policy
---------------------------
``heuristic``       plays well: eats pellets, dodges ghosts, chases frightened
                    ghosts.  This is the data you want for DT training.
``epsilon_greedy``  the same, with ``--epsilon`` random moves mixed in.  The
                    extra noise gives the transformer states it would otherwise
                    never see.
``random``          uniformly random actions - a baseline, not training data.

Quick reference
---------------
``--layout``    maze: ``classic``, ``google``, ``medium``, ``medium_small``,
                ``small`` (default ``small`` so a first run takes seconds)
``--episodes``  how many games to record (default ``50``)
``--out``       where to write; defaults to ``data/pacman_dt_<layout>``
``--format``    ``hf`` (Hugging Face dataset directory, default) or ``npz``
``--max-length`` pad/truncate every episode to this many steps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pacman_env.game.map import Map

from data import (
    DEFAULT_SUCCESS_THRESHOLD,
    DataCollector,
    Episode,
    collect_trajectories,
    dataset_statistics,
    describe_layouts,
    dims_from_arrays,
    dims_from_dataset,
    filter_episodes,
    get_act_dim,
    load_dataset,
    recommend_max_length,
    save_dataset,
)

DEFAULT_LAYOUT = "small"
DEFAULT_EPISODES = 50
DEFAULT_MAX_STEPS = 1000
DEFAULT_FORMAT = "hf"
DATA_DIR = Path("data")


# ----------------------------------------------------------------------
# Where the data goes
# ----------------------------------------------------------------------
def default_output(layout: str, fmt: str) -> Path:
    """``data/pacman_dt_small`` for ``hf``, ``data/pacman_dt_small.npz`` for ``npz``."""
    name = f"pacman_dt_{layout}"
    return DATA_DIR / (name if fmt == "hf" else f"{name}.npz")


def resolve_output(out: Optional[str], layout: str, fmt: str) -> Path:
    """Turn ``--out`` (or its default) into the exact path that will be written."""
    if out is None:
        return default_output(layout, fmt)
    path = Path(out)
    if fmt == "npz" and path.suffix != ".npz":
        path = path.with_suffix(".npz")
    return path


# ----------------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------------
def _row(label: str, value: Any) -> str:
    return f"  {label:<13}: {value}"


def print_plan(
    *,
    layout: str,
    episodes: int,
    policy: str,
    max_steps: Optional[int],
    seed: int,
    workers: int,
    out: Path,
    fmt: str,
) -> None:
    print("PacManDT data collection")
    print(_row("layout", layout))
    print(_row("episodes", episodes))
    print(_row("policy", policy))
    print(_row("step limit", "none" if max_steps is None else max_steps))
    print(_row("seed", seed))
    print(_row("workers", workers))
    print(_row("output", f"{out}  ({fmt})"))
    print()


def print_statistics(episodes: Sequence[Episode], title: str) -> None:
    stats = dataset_statistics(episodes)
    print(title)
    print(_row("episodes", stats["episodes"]))
    print(_row("transitions", stats["steps"]))
    print(
        _row(
            "return",
            f"mean {stats['mean_return']:.2f}  "
            f"min {stats['min_return']:.2f}  max {stats['max_return']:.2f}",
        )
    )
    print(_row("score", f"mean {stats['mean_score']:.1f}"))
    print(_row("length", f"mean {stats['mean_length']:.0f}"))
    print(_row("win rate", f"{100 * stats['win_rate']:.1f}%"))


def print_shapes(
    *,
    state_dim: int,
    act_dim: int,
    obs_shape: Sequence[int],
    max_length: int,
    layout: str,
    max_steps: Optional[int],
) -> None:
    shape = "x".join(str(dim) for dim in obs_shape)
    print("Shapes")
    print(_row("state_dim", f"{state_dim}  (observations {shape}, flattened for the model)"))
    print(_row("act_dim", act_dim))
    print(_row("max_length", max_length))
    hint = recommend_max_length(
        layout=layout, max_steps=max_steps if max_steps else DEFAULT_MAX_STEPS
    )
    if hint < max_length:
        print(_row("hint", f"--max-length {hint} caps long episodes and saves disk space"))


def print_next_steps(out: Path, fmt: str) -> None:
    data = str(out)
    print("\nNext steps")
    print(f"  train     : python -m dt.train --data {data} --epochs 20")
    print(f"  watch it  : python -m dt.train --data {data} --epochs 20 --render human")
    if fmt == "hf":
        print(
            "  inspect   : python -c \"from dt.data import DataCollector; "
            f"print(DataCollector.load_from_disk(r'{data}').dataset_dict)\""
        )
    else:
        print(
            f"  inspect   : python -c \"from dt.data import load_dataset; "
            f"print(load_dataset(r'{data}')[0]['observations'].shape)\""
        )


# ----------------------------------------------------------------------
# Save + verify
# ----------------------------------------------------------------------
def save_hf(
    episodes: Sequence[Episode],
    *,
    out: Path,
    layout: str,
    args: argparse.Namespace,
    max_steps: Optional[int],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Write a Hugging Face ``DatasetDict`` directory and return its summary."""
    collector = DataCollector(
        layout=layout,
        lives=args.lives,
        max_steps=max_steps,
        policy=args.policy,
        gamma=args.gamma,
        frame_skip=args.frame_skip,
        epsilon=args.epsilon,
        success_threshold=args.success_threshold,
        seed=args.seed,
        workers=args.workers,
        progress=False,
        eval_fraction=args.eval_fraction,
        max_length=args.max_length,
    )
    collector.from_episodes(episodes)
    path = collector.save_to_disk(out)

    splits = {name: len(split) for name, split in collector.dataset_dict.items()}
    if verbose:
        print(f"\nSaved {len(episodes)} episodes to {path}")
        for name, split in collector.dataset_dict.items():
            shape = split.features["observations"].shape
            print(_row(name, f"{len(split)} episodes, observations {shape} int8"))
        print(_row("metadata", path / "pacman_dt_metadata.json"))
        print(_row("dataset card", path / "README.md"))
    return {"path": path, "splits": splits, "dataset_dict": collector.dataset_dict}


def save_npz(
    episodes: Sequence[Episode],
    *,
    out: Path,
    layout: str,
    args: argparse.Namespace,
    max_steps: Optional[int],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Write a ``.npz`` archive and return its summary."""
    metadata = {
        "layout": layout,
        "lives": args.lives,
        "max_steps": max_steps,
        "policy": args.policy,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "seed": args.seed,
        "frame_skip": args.frame_skip,
        "success_threshold": args.success_threshold,
        "min_return": args.min_return,
        "keep_won_only": args.keep_won_only,
        "action_dim": get_act_dim(),
    }
    path = save_dataset(
        out,
        episodes,
        metadata=metadata,
        compress=not args.no_compress,
        max_length=args.max_length,
    )

    arrays, _ = load_dataset(path)
    if verbose:
        print(f"\nSaved {len(episodes)} episodes to {path}")
        for key in ("observations", "actions", "rewards", "returns_to_go", "timesteps", "mask"):
            print(_row(key, f"{arrays[key].shape} {arrays[key].dtype}"))
    return {"path": path, "splits": {"train": len(episodes)}, "arrays": arrays}


def verify(
    path: Path, fmt: str, expected_episodes: int, *, verbose: bool = True
) -> Dict[str, Any]:
    """Reload what was just written and check it is complete.

    A dataset that cannot be reloaded is worthless, so this is the last thing
    the program does - and it fails loudly instead of printing a success message.
    """
    if fmt == "hf":
        collector = DataCollector.load_from_disk(path)
        dataset_dict = collector.dataset_dict
        splits = {name: len(split) for name, split in dataset_dict.items()}
        found = sum(splits.values())
        dims = dims_from_dataset(dataset_dict, layout=collector.layout)
        max_length = int(dataset_dict["train"].features["observations"].shape[0])
    else:
        arrays, metadata = load_dataset(path)
        found = int(arrays["observations"].shape[0])
        splits = {"train": found}
        dims = dims_from_arrays(arrays, layout=metadata.get("layout"))
        max_length = int(arrays["mask"].shape[1])

    if found != expected_episodes:
        raise ValueError(
            f"verification failed: wrote {expected_episodes} episodes to {path} "
            f"but read back {found}"
        )

    if verbose:
        print("\nVerified on disk")
        print(_row("episodes", f"{found}  ({', '.join(f'{k} {v}' for k, v in splits.items())})"))
        print(_row("readable", "yes - reloaded with the public loaders"))
    return {
        "episodes": found,
        "splits": splits,
        "dims": dims.as_dict(),
        "max_length": max_length,
    }


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
        prog="python -m dt.collect",
        description="Collect PacMan demonstrations and save them to disk.",
        formatter_class=_formatter(),
        epilog=(
            "examples:\n"
            "  python -m dt.collect\n"
            "  python -m dt.collect --layout classic --episodes 500\n"
            "  python -m dt.collect --episodes 200 --policy epsilon_greedy --epsilon 0.1\n"
            "  python -m dt.collect --format npz --out data/pacman_dt_small.npz\n"
        ),
    )
    parser.add_argument("--layout", default=DEFAULT_LAYOUT, choices=Map.available_layouts(),
                        help="Maze layout to play on.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Number of episodes (games) to record.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Per-episode step limit (0 = no limit).")
    parser.add_argument("--policy", default="heuristic",
                        choices=["heuristic", "epsilon_greedy", "random"],
                        help="Behaviour policy used to generate the data.")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Exploration rate for --policy epsilon_greedy.")
    parser.add_argument("--lives", type=int, default=3, help="Player lives per episode.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount used for the stored returns_to_go column.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Episode i uses seed+i, so runs are reproducible.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel collector processes (results are identical).")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Repeat each action this many simulation frames.")
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD,
                        help="Total reward at which an episode counts as a success.")
    parser.add_argument("--min-return", type=float, default=None,
                        help="Discard episodes whose total reward is below this value.")
    parser.add_argument("--keep-won-only", action="store_true",
                        help="Keep only episodes that cleared the maze.")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Pad/truncate every episode to this many steps "
                             "(default: the longest collected episode).")
    parser.add_argument("--eval-fraction", type=float, default=0.0,
                        help="Fraction of episodes put in an 'eval' split (--format hf).")
    parser.add_argument("--format", default=DEFAULT_FORMAT, choices=["hf", "npz"],
                        help="hf = Hugging Face dataset directory, npz = numpy archive.")
    parser.add_argument("--out", default=None,
                        help="Output path (default: data/pacman_dt_<layout>).")
    parser.add_argument("--no-compress", action="store_true",
                        help="Write an uncompressed .npz (faster, larger).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the output path if it already exists.")
    parser.add_argument("--render", default="none", choices=["none", "human"],
                        help="human opens the pygame window while collecting "
                             "(implies --workers 1).")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                        help="Upload the dataset, e.g. 'you/pacman-dt-small' "
                             "(--format hf only).")
    parser.add_argument("--hub-private", action="store_true",
                        help="Create the Hub repository as private.")
    parser.add_argument("--show-dims", action="store_true",
                        help="Print state_dim/act_dim/max_length for every layout and exit.")
    parser.add_argument("--quiet", action="store_true", help="Hide progress output.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.show_dims:
        print(describe_layouts())
        return 0

    if args.episodes <= 0:
        print("--episodes must be a positive integer", file=sys.stderr)
        return 2
    if args.format != "hf" and args.push_to_hub:
        print("--push-to-hub needs --format hf", file=sys.stderr)
        return 2
    if args.render != "none" and args.workers > 1:
        print("note: --render runs in one process, using --workers 1")
        args.workers = 1

    max_steps = None if args.max_steps <= 0 else int(args.max_steps)
    out = resolve_output(args.out, args.layout, args.format)

    if out.exists() and not args.overwrite:
        print(f"{out} already exists - pass --overwrite to replace it", file=sys.stderr)
        return 2

    if not args.quiet:
        print_plan(
            layout=args.layout,
            episodes=args.episodes,
            policy=args.policy,
            max_steps=max_steps,
            seed=args.seed,
            workers=args.workers,
            out=out,
            fmt=args.format,
        )

    # ---- 1. collect ---------------------------------------------------
    episodes = collect_trajectories(
        layout=args.layout,
        lives=args.lives,
        num_episodes=args.episodes,
        max_steps=max_steps,
        success_threshold=args.success_threshold,
        policy=args.policy,
        seed=args.seed,
        gamma=args.gamma,
        frame_skip=args.frame_skip,
        epsilon=args.epsilon,
        render_mode=None if args.render == "none" else args.render,
        progress=not args.quiet,
        workers=args.workers,
    )

    # ---- 2. filter ----------------------------------------------------
    kept = list(episodes)
    if args.keep_won_only:
        kept = [episode for episode in kept if episode.info.get("win", False)]
    kept = filter_episodes(kept, min_return=args.min_return)
    if not kept:
        hint = "Try lowering --min-return"
        if args.keep_won_only:
            hint = "No episode cleared the maze - raise --max-steps or drop --keep-won-only"
        print(
            f"\nNo episodes left after filtering - nothing written. {hint}.",
            file=sys.stderr,
        )
        return 1

    if not args.quiet:
        print()
        print_statistics(episodes, "Collected")
        if len(kept) != len(episodes):
            print_statistics(kept, "Kept after filtering")

    # ---- 3. save ------------------------------------------------------
    if args.format == "hf":
        saved = save_hf(
            kept, out=out, layout=args.layout, args=args,
            max_steps=max_steps, verbose=not args.quiet,
        )
    else:
        saved = save_npz(
            kept, out=out, layout=args.layout, args=args,
            max_steps=max_steps, verbose=not args.quiet,
        )
    path = saved["path"]

    # ---- 4. verify ----------------------------------------------------
    verified = verify(path, args.format, len(kept), verbose=not args.quiet)
    if not args.quiet:
        print()
        print_shapes(
            state_dim=verified["dims"]["state_dim"],
            act_dim=verified["dims"]["act_dim"],
            obs_shape=verified["dims"]["obs_shape"],
            max_length=verified["max_length"],
            layout=args.layout,
            max_steps=max_steps,
        )

    # ---- 5. optionally upload ----------------------------------------
    if args.push_to_hub:
        collector = DataCollector.load_from_disk(path)
        url = collector.push_to_hub(args.push_to_hub, private=args.hub_private)
        print(f"\nPushed to {url}")

    if not args.quiet:
        print_next_steps(path, args.format)
    else:
        print(path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
