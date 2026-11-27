"""Collect Decision-Transformer trajectories from the PacManDT environment.

Run it directly::

    python -m dt.data --episodes 500 --policy heuristic --out data/pacman_dt.npz

The resulting ``.npz`` contains padded ``observations``, ``actions``,
``rewards``, ``returns_to_go``, ``timesteps`` and a ``mask`` — exactly what a
Decision Transformer consumes.  See :mod:`dt.data.episodes` for the format.

Pass ``--format hf`` to write a Hugging Face ``datasets`` directory instead
(and ``--hub-repo you/pacman-dt-classic`` to upload it)::

    python -m dt.data --format hf --episodes 500 --eval-fraction 0.1

The behaviour policy decides how good the data is:

``heuristic``       plays well (eats pellets, dodges ghosts, chases frightened
                    ghosts) — this is what you want for DT training
``random``          uniformly random actions — useful as a baseline
``epsilon_greedy``  heuristic with a small amount of random exploration

Ghost observations
------------------
``--obs-mode ghost`` records the ghost-aware observation stack from
:mod:`dt.data.observations` instead of the environment's single-channel grid:
per-ghost state channels, BFS distance fields and scalar globals.  It is the
recommended mode for DT training::

    python -m dt.data --obs-mode ghost --episodes 2000

``--no-distance`` drops the two BFS distance planes and ``--no-globals`` drops
the frightened-timer / lives planes (``(7, rows, cols)`` instead of
``(11, rows, cols)``).  Use ``--show-dims --obs-mode ghost`` to print the
resulting ``state_dim`` for every layout.

Collection protocol
-------------------
``--portfolio`` mixes several behaviour policies in one run so the dataset
covers a wide return-to-go spectrum (see :mod:`dt.data.protocol`)::

    python -m dt.data --obs-mode ghost --episodes 2000 \\
        --portfolio heuristic=0.6,epsilon_greedy=0.3,random=0.1 \\
        --balance-success --eval-fraction 0.1 --format hf

Without ``--portfolio`` a single ``--policy`` is rolled out, which is the
legacy behaviour.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence, Tuple

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
from .observations import (
    GHOST_OBS_MODE,
    OBS_MODES,
    STATE_OBS_MODE,
    GhostObservationConfig,
    describe_observation,
    observation_metadata,
    resolve_observation,
)
from .protocol import (
    DEFAULT_PORTFOLIO,
    KNOWN_POLICIES,
    balance_classes,
    build_policy_schedule,
    curate_episodes,
    format_report,
    parse_portfolio,
    protocol_report,
    quality_tiers,
    split_by_hash,
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
                        choices=list(KNOWN_POLICIES),
                        help="Behaviour policy used to generate the data.")
    parser.add_argument("--portfolio", default=None, metavar="NAME=WEIGHT,...",
                        help="Mix several behaviour policies in one run, e.g. "
                             "'heuristic=0.6,epsilon_greedy=0.3,random=0.1'. "
                             "Omit it to roll out a single --policy.")
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
    parser.add_argument("--num-tiers", type=int, default=4,
                        help="Quality tiers used by --max-kept / --balance-success "
                             "so trimming keeps the best episode of every return band.")
    parser.add_argument("--balance-success", action="store_true",
                        help="Down-sample the majority success/failure class "
                             "(keeps the dataset from being all wins or all losses).")
    parser.add_argument("--success-ratio", type=float, default=2.0,
                        help="Max majority:minority ratio for --balance-success.")
    parser.add_argument("--obs-mode", default=STATE_OBS_MODE, choices=list(OBS_MODES),
                        help="Observation space: 'state' is the environment's own "
                             "grid; 'ghost' is the ghost-aware channel stack.")
    parser.add_argument("--no-distance", action="store_true",
                        help="--obs-mode ghost only: drop the BFS distance planes.")
    parser.add_argument("--no-globals", action="store_true",
                        help="--obs-mode ghost only: drop the frightened-timer "
                             "and lives planes.")
    parser.add_argument("--max-distance", type=int, default=None,
                        help="--obs-mode ghost only: BFS distance clamp "
                             "(default 40).")
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


def ghost_config_from_args(args: argparse.Namespace) -> Optional[GhostObservationConfig]:
    """Ghost channel selection from the CLI flags (``None`` for ``--obs-mode state``)."""
    if args.obs_mode != GHOST_OBS_MODE:
        return None
    base = GhostObservationConfig()
    return GhostObservationConfig(
        distance_fields=not args.no_distance,
        globals=not args.no_globals,
        max_distance=int(args.max_distance) if args.max_distance else base.max_distance,
        dtype=base.dtype,
    )


def select_episodes(
    episodes: Sequence[Episode],
    min_return: Optional[float] = None,
    keep_won_only: bool = False,
    max_kept: Optional[int] = None,
    *,
    balance_success: bool = False,
    success_ratio: float = 2.0,
    num_tiers: int = 4,
) -> list[Episode]:
    """Apply the CLI's quality filters to a collected batch of episodes."""
    kept = list(episodes)
    if keep_won_only:
        kept = [episode for episode in kept if episode.info.get("win", False)]
    if balance_success:
        kept = balance_classes(kept, max_ratio=success_ratio)
    if max_kept is not None:
        kept = curate_episodes(kept, max_episodes=max_kept, num_tiers=num_tiers)
    return filter_episodes(kept, min_return=min_return)


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


def _write_hf(
    args: argparse.Namespace,
    kept: Sequence[Episode],
    split: Tuple[Sequence[Episode], Sequence[Episode]],
) -> int:
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
        policy=args.policy if args.portfolio is None else "portfolio",
        gamma=args.gamma,
        frame_skip=args.frame_skip,
        epsilon=args.epsilon,
        success_threshold=args.success_threshold,
        seed=args.seed,
        workers=args.workers,
        progress=False,
        obs_mode=args.obs_mode,
        obs_config=ghost_config_from_args(args),
        eval_fraction=args.eval_fraction,
        max_length=args.max_length,
        repo_id=args.hub_repo,
    )
    train_episodes, eval_episodes = split
    collector.from_episodes(
        kept,
        # stable hash split: appending episodes later never moves an existing
        # episode from eval into train
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
    )
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
    obs_config = ghost_config_from_args(args)
    if args.show_dims:
        print(describe_layouts(obs_mode=args.obs_mode, ghost_config=obs_config))
        return 0

    max_steps = None if args.max_steps <= 0 else int(args.max_steps)

    if args.format != "hf" and args.hub_repo:
        print("--hub-repo needs --format hf", file=sys.stderr)
        return 2

    portfolio = parse_portfolio(args.portfolio) if args.portfolio else None
    schedule = (
        build_policy_schedule(args.episodes, portfolio, seed=args.seed)
        if portfolio
        else None
    )

    print(describe_observation(args.obs_mode, layout=args.layout, config=obs_config))
    if portfolio:
        plan = ", ".join(
            f"{name}={schedule.count(name)}" for name in portfolio
        )
        print(f"Portfolio ({args.episodes} episodes): {plan}")
    print(f"\nCollecting {args.episodes} episodes on '{args.layout}' "
          f"with the {args.policy if schedule is None else 'portfolio'} policy ...")
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
        obs_mode=args.obs_mode,
        obs_config=obs_config,
        policy_schedule=schedule,
    )
    print_statistics(episodes, "collected")

    kept = select_episodes(
        episodes,
        min_return=args.min_return,
        keep_won_only=args.keep_won_only,
        max_kept=args.max_kept,
        balance_success=args.balance_success,
        success_ratio=args.success_ratio,
        num_tiers=args.num_tiers,
    )
    if not kept:
        print("\nNo episodes left after filtering - nothing written. "
              "Try lowering --min-return.", file=sys.stderr)
        return 1
    if len(kept) != len(episodes):
        print_statistics(kept, "kept")

    split = (
        split_by_hash(kept, args.eval_fraction, seed=args.seed)
        if args.format == "hf"
        else None
    )
    if split is not None and args.eval_fraction > 0 and not split[1]:
        print(f"\nwarning: the stable hash split placed no episode in eval "
              f"(--eval-fraction {args.eval_fraction}, {len(kept)} episodes). "
              f"Collect more episodes or raise --eval-fraction.", file=sys.stderr)

    print()
    print(format_report(protocol_report(
        kept,
        collected=episodes,
        num_tiers=args.num_tiers,
        splits=None if split is None else {"train": len(split[0]), "eval": len(split[1])},
    )))

    if args.format == "hf":
        return _write_hf(args, kept, split)

    observation = observation_metadata(args.obs_mode, layout=args.layout, config=obs_config)
    metadata = {
        "layout": args.layout,
        "lives": args.lives,
        "max_steps": max_steps,
        "policy": args.policy if schedule is None else "portfolio",
        "policies": sorted({episode.policy for episode in kept}),
        "portfolio": dict(portfolio) if portfolio else None,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "seed": args.seed,
        "frame_skip": args.frame_skip,
        "success_threshold": args.success_threshold,
        "min_return": args.min_return,
        "keep_won_only": args.keep_won_only,
        "balance_success": args.balance_success,
        "collected_episodes": len(episodes),
        "action_dim": len(Action),
        "obs_mode": observation["mode"],
        "observation": observation,
    }
    path = save_dataset(args.out, kept, metadata=metadata, compress=not args.no_compress,
                        max_length=args.max_length, obs_mode=args.obs_mode,
                        obs_config=obs_config)

    arrays = pad_episodes(kept, max_length=args.max_length)
    print(f"\nSaved {len(kept)} episodes to {path}")
    print(f"  observations : {arrays['observations'].shape} {arrays['observations'].dtype}")
    print(f"  actions      : {arrays['actions'].shape} {arrays['actions'].dtype}")
    print(f"  returns_to_go: {arrays['returns_to_go'].shape} {arrays['returns_to_go'].dtype}")
    print(f"  mask         : {arrays['mask'].shape} (True = valid step)")
    print(f"  state_dim    : {resolve_observation(args.obs_mode, layout=args.layout, config=obs_config).feature_dim}")
    print("\nLoad it later with:")
    print("  from dt.data.episodes import load_dataset")
    print(f"  arrays, metadata = load_dataset({str(path)!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())