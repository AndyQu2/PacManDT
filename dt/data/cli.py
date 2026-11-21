"""Collect Decision-Transformer trajectories from the PacManDT environment.

Run it directly::

    python -m dt.data --episodes 500 --policy heuristic --out data/pacman_dt.npz

The resulting ``.npz`` contains padded ``observations``, ``actions``,
``rewards``, ``returns_to_go``, ``timesteps`` and a ``mask`` â€” exactly what a
Decision Transformer consumes.  See :mod:`dt.data.episodes` for the format.

Pass ``--format hf`` to write a Hugging Face ``datasets`` directory instead
(and ``--hub-repo you/pacman-dt-classic`` to upload it)::

    python -m dt.data --format hf --episodes 500 --eval-fraction 0.1

The behaviour policy decides how good the data is:

``heuristic``       plays well (eats pellets, dodges ghosts, chases frightened
                    ghosts) â€” this is what you want for DT training
``random``          uniformly random actions â€” useful as a baseline
``epsilon_greedy``  heuristic with a small amount of random exploration
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from pacman_env.game.action import Action
from pacman_env.game.map import Map

from .dims import describe_layouts
from .episodes import (
    DEFAULT_SUCCESS_THRESHOLD,
    Episode,
    collect_trajectories,
    dataset_statistics,
    filter_episodes,
    pad_episodes,
    save_dataset,
)

DEFAULT_OUTPUT = "data/pacman_dt.npz"
# duplicated from dt.data.hf_dataset so this module imports without the
# optional "hub" extra installed
DEFAULT_HF_DIR = "data/pacman_dt_hf"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dt.data",
        description="Collect Decision-Transformer trajectories in PacManDT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layout", default="classic", choices=Map.available_layouts(),
                        help="Maze layout to play on.")
    parser.add_argument("--lives", type=int, default=3, help="Player lives per episode.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to collect.")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Per-episode step limit (0 = no limit). Shorter "
                             "episodes give shorter DT sequences and smaller files.")
    parser.add_argument("--policy", default="heuristic",
                        choices=["heuristic", "random", "epsilon_greedy"],
                        help="Behaviour policy used to generate the data.")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="Exploration rate for --policy epsilon_greedy.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor used for the returns-to-go.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Repeat each action this many simulation frames.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel collector processes.")
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD,
                        help="Total reward at which an episode counts as a success.")
    parser.add_argument("--min-return", type=float, default=None,
                        help="Discard episodes whose total reward is below this value.")
    parser.add_argument("--keep-won-only", action="store_true",
                        help="Keep only episodes that cleared the maze.")
    parser.add_argument("--max-kept", type=int, default=None,
                        help="Cap the number of stored episodes.")
    parser.add_argument("--out", default=DEFAULT_OUTPUT,
                        help="Output .npz path (a directory is created if needed).")
    parser.add_argument("--no-compress", action="store_true",
                        help="Write an uncompressed .npz (faster, larger).")
    parser.add_argument("--format", default="npz", choices=["npz", "hf"],
                        help="npz = numpy archive; hf = Hugging Face dataset "
                             "(needs the 'hub' extra).")
    parser.add_argument("--hf-out", default=DEFAULT_HF_DIR,
                        help="Output directory for --format hf.")
    parser.add_argument("--eval-fraction", type=float, default=0.0,
                        help="Fraction of episodes held out as an 'eval' split "
                             "(--format hf only).")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Pad/truncate every episode to this many steps "
                             "(default: the longest collected episode). "
                             "Applies to both --format npz and --format hf.")
    parser.add_argument("--show-dims", action="store_true",
                        help="Print state_dim / act_dim / recommended "
                             "max_length for every layout and exit.")
    parser.add_argument("--hub-repo", default=None,
                        help="Push the HF dataset to this Hub repo id, e.g. "
                             "'your-name/pacman-dt-classic' (--format hf only).")
    parser.add_argument("--hub-private", action="store_true",
                        help="Create the Hub repository as private.")
    parser.add_argument("--quiet", action="store_true", help="Hide progress output.")
    return parser


def select_episodes(
    episodes: Sequence[Episode],
    min_return: Optional[float] = None,
    keep_won_only: bool = False,
    max_kept: Optional[int] = None,
) -> list[Episode]:
    """Apply the CLI's quality filters to a collected batch of episodes."""
    kept = list(episodes)
    if keep_won_only:
        kept = [episode for episode in kept if episode.info.get("win", False)]
    return filter_episodes(kept, min_return=min_return, max_episodes=max_kept)


def print_statistics(episodes: Sequence[Episode], label: str = "dataset") -> None:
    stats = dataset_statistics(episodes)
    if not stats.get("episodes"):
        print(f"{label}: no episodes")
        return
    print(f"\n{label}:")
    print(f"  episodes      : {stats['episodes']}")
    print(f"  transitions   : {stats['steps']}")
    print(f"  return        : mean {stats['mean_return']:.2f}  "
          f"std {stats['std_return']:.2f}  "
          f"min {stats['min_return']:.2f}  max {stats['max_return']:.2f}")
    print(f"  score         : mean {stats['mean_score']:.1f}")
    print(f"  episode length: mean {stats['mean_length']:.1f}")
    print(f"  success rate  : {100 * stats['success_rate']:.1f}%")
    print(f"  win rate      : {100 * stats['win_rate']:.1f}%")


def _write_hf(args: argparse.Namespace, kept: Sequence[Episode]) -> int:
    """Write the collected episodes as a Hugging Face ``DatasetDict``."""
    try:
        from .hf_dataset import DataCollector
    except ImportError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2

    max_steps = None if args.max_steps <= 0 else int(args.max_steps)
    collector = DataCollector(
        layout=args.layout,
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
        repo_id=args.hub_repo,
    )
    collector.from_episodes(kept)
    path = collector.save_to_disk(args.hf_out)

    print(f"\nSaved {len(kept)} episodes to {path}")
    for name, split in collector.dataset_dict.items():
        shape = split.features["observations"].shape
        print(f"  {name:<13}: {len(split)} episodes, observations {shape} int8")
    print(f"  metadata     : {path / 'pacman_dt_metadata.json'}")
    print(f"  dataset card : {path / 'README.md'}")

    if args.hub_repo:
        url = collector.push_to_hub(private=args.hub_private)
        print(f"\nPushed to {url}")

    print("\nLoad it later with:")
    print("  from dt.data.hf_dataset import DataCollector")
    print(f"  collector = DataCollector.load_from_disk({str(path)!r})")
    print("  dataset = collector.dataset_dict")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.show_dims:
        print(describe_layouts())
        return 0

    max_steps = None if args.max_steps <= 0 else int(args.max_steps)

    if args.format != "hf" and args.hub_repo:
        print("--hub-repo needs --format hf", file=sys.stderr)
        return 2

    print(f"Collecting {args.episodes} episodes on '{args.layout}' "
          f"with the {args.policy} policy ...")
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
        render_mode=None,
        progress=not args.quiet,
        workers=args.workers,
    )
    print_statistics(episodes, "collected")

    kept = select_episodes(
        episodes,
        min_return=args.min_return,
        keep_won_only=args.keep_won_only,
        max_kept=args.max_kept,
    )
    if not kept:
        print("\nNo episodes left after filtering - nothing written. "
              "Try lowering --min-return.", file=sys.stderr)
        return 1
    if len(kept) != len(episodes):
        print_statistics(kept, "kept")

    if args.format == "hf":
        return _write_hf(args, kept)

    metadata = {
        "layout": args.layout,
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
        "collected_episodes": len(episodes),
        "action_dim": len(Action),
    }
    path = save_dataset(args.out, kept, metadata=metadata, compress=not args.no_compress,
                        max_length=args.max_length)

    arrays = pad_episodes(kept, max_length=args.max_length)
    print(f"\nSaved {len(kept)} episodes to {path}")
    print(f"  observations : {arrays['observations'].shape} {arrays['observations'].dtype}")
    print(f"  actions      : {arrays['actions'].shape} {arrays['actions'].dtype}")
    print(f"  returns_to_go: {arrays['returns_to_go'].shape} {arrays['returns_to_go'].dtype}")
    print(f"  mask         : {arrays['mask'].shape} (True = valid step)")
    print("\nLoad it later with:")
    print("  from dt.data.episodes import load_dataset")
    print(f"  arrays, metadata = load_dataset({str(path)!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())