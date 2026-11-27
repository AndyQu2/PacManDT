"""Deprecated alias for the data-collection CLI in :mod:`dt.data`.

``python -m dt.collect`` used to be a second, self-contained copy of the
collection pipeline.  Two collectors inevitably drift apart - one grows ghost
observations, the other does not - so this module is now a thin shim: it keeps
the historical flags and defaults (``small`` layout, 50 episodes, Hugging Face
output) and forwards everything to :func:`dt.data.cli.main`.

Use ``python -m dt.data`` instead.  It has every flag this module ever had,
plus the ghost observation stack and the collection protocol::

    python -m dt.data --obs-mode ghost --episodes 2000 \\
        --portfolio heuristic=0.6,epsilon_greedy=0.3,random=0.1 \\
        --balance-success --format hf --hf-out data/pacman_dt_ghost

Command line (unchanged for existing scripts)::

    python -m dt.collect                                     # 50 games on 'small'
    python -m dt.collect --layout classic --episodes 500
    python -m dt.collect --format npz --out data/pacman_dt_small.npz
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from pacman_env.game.map import Map

from dt.data import DEFAULT_SUCCESS_THRESHOLD
from dt.data.cli import main as data_main

DEFAULT_LAYOUT = "small"
DEFAULT_EPISODES = 50
DEFAULT_MAX_STEPS = 1000
DEFAULT_FORMAT = "hf"


def build_parser() -> argparse.ArgumentParser:
    """The historical ``python -m dt.collect`` flags (defaults included)."""
    parser = argparse.ArgumentParser(
        prog="python -m dt.collect",
        description="Deprecated: use 'python -m dt.data'. Collects PacMan "
                    "demonstrations and saves them to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
                        help="Accepted for compatibility; the new CLI always replaces "
                             "the output path.")
    parser.add_argument("--render", default="none", choices=["none", "human"],
                        help="Accepted for compatibility; rendering is not supported "
                             "while collecting (use dt.train --render).")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                        help="Upload the dataset, e.g. 'you/pacman-dt-small' "
                             "(--format hf only).")
    parser.add_argument("--hub-private", action="store_true",
                        help="Create the Hub repository as private.")
    parser.add_argument("--show-dims", action="store_true",
                        help="Print state_dim/act_dim/max_length for every layout and exit.")
    parser.add_argument("--quiet", action="store_true", help="Hide progress output.")
    return parser


def to_data_argv(args: argparse.Namespace) -> List[str]:
    """Translate the legacy namespace into ``python -m dt.data`` arguments.

    Every value is passed explicitly so the historical defaults survive the
    hand-off.
    """
    name = f"pacman_dt_{args.layout}"
    default_out = f"data/{name}" if args.format == "hf" else f"data/{name}.npz"
    out = args.out or default_out

    argv = [
        "--layout", str(args.layout),
        "--lives", str(args.lives),
        "--episodes", str(args.episodes),
        "--max-steps", str(args.max_steps),
        "--policy", str(args.policy),
        "--epsilon", str(args.epsilon),
        "--gamma", str(args.gamma),
        "--seed", str(args.seed),
        "--frame-skip", str(args.frame_skip),
        "--workers", str(args.workers),
        "--success-threshold", str(args.success_threshold),
        "--format", str(args.format),
        "--eval-fraction", str(args.eval_fraction),
    ]
    # The new CLI splits npz output (--out) from HF output (--hf-out).
    argv += ["--hf-out", out] if args.format == "hf" else ["--out", out]

    if args.min_return is not None:
        argv += ["--min-return", str(args.min_return)]
    if args.keep_won_only:
        argv.append("--keep-won-only")
    if args.max_length:
        argv += ["--max-length", str(args.max_length)]
    if args.no_compress:
        argv.append("--no-compress")
    if args.push_to_hub:
        argv += ["--hub-repo", str(args.push_to_hub)]
    if args.hub_private:
        argv.append("--hub-private")
    if args.show_dims:
        argv.append("--show-dims")
    if args.quiet:
        argv.append("--quiet")
    return argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        "note: 'python -m dt.collect' is deprecated and now forwards to "
        "'python -m dt.data' (same protocol, plus --obs-mode ghost, "
        "--portfolio and --balance-success).",
        file=sys.stderr,
    )
    if args.render != "none":
        print("note: --render is no longer supported while collecting; ignoring it.",
              file=sys.stderr)
    return data_main(to_data_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
