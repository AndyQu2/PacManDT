"""Train a Decision Transformer on PacManDT and watch it play.

Quick start
-----------
Collect a small dataset and train for a few epochs::

    python -m dt.train --layout small --episodes 32 --epochs 5

Then watch the trained policy play (``--render human`` opens the pygame window)::

    python -m dt.train --eval-only --checkpoint runs/dt-small/checkpoint.pt --render human

Train on a dataset you already collected::

    python -m dt.data --layout classic --episodes 200 --format hf --hf-out data/classic
    python -m dt.train --data data/classic --epochs 20

What happens
------------
1. **Get data** - load ``--data`` (a ``DataCollector.save_to_disk`` directory or a
   ``.npz`` from ``dt.data``) or collect fresh episodes with the heuristic policy.
2. **Read the shapes** - ``state_dim``/``act_dim`` come from ``dt.data.dims``, never
   hard-coded, so the same script works for every layout.
3. **Build the model** - ``HuggingfaceDT`` around
   ``transformers.DecisionTransformerModel``.
4. **Train** - masked MSE between the predicted action and the recorded action,
   over sliding windows of ``--context-length`` steps.  Returns-to-go are
   recomputed as the plain, *undiscounted* sum of future rewards (what a DT is
   conditioned on), so ``--target-return`` is in raw environment reward units.
5. **Evaluate** - held-out loss plus real rollouts of the greedy ``DTPolicy``.
6. **Save** - ``<out-dir>/checkpoint.pt`` (weights + config + inference settings)
   and ``<out-dir>/metrics.json``.

Key knobs
---------
``--context-length`` (``K``)
    How many past steps the transformer sees.  30 is plenty for PacMan: the agent
    mostly needs the current maze plus a little history.
``--epochs`` / ``--batch-size`` / ``--lr``
    Standard optimiser settings.  Small datasets overfit fast - that is expected.
``--target-return``
    The return the policy is conditioned on at inference.  Defaults to the mean
    episode return of the training data; raise it for a greedier agent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from transformers import DecisionTransformerConfig

from data import (
    DEFAULT_CONTEXT_LENGTH,
    DataCollector,
    Episode,
    dims_from_dataset,
    iter_episodes,
    load_dataset,
    make_dt_dataloader,
    recommend_context_length,
    recommend_max_length,
)
from dt import DTPolicy, HuggingfaceDT, load_checkpoint, save_checkpoint

DEFAULT_OUT_DIR = Path("runs")
DEFAULT_RETURN_SCALE = 1.0


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def _episodes_from_npz(path: Path) -> List[Episode]:
    """Rebuild :class:`Episode` objects from a ``dt.data`` ``.npz`` file."""
    arrays, metadata = load_dataset(path)
    episodes: List[Episode] = []
    for index, chunk in enumerate(iter_episodes(arrays)):
        success = bool(arrays["episode_success"][index]) if "episode_success" in arrays else False
        episodes.append(
            Episode(
                observations=np.asarray(chunk["observations"]),
                actions=np.asarray(chunk["actions"], dtype=np.int64),
                rewards=np.asarray(chunk["rewards"], dtype=np.float32),
                terminated=np.asarray(chunk["terminated"], dtype=bool),
                truncated=np.asarray(chunk["truncated"], dtype=bool),
                returns_to_go=np.asarray(chunk["returns_to_go"], dtype=np.float32),
                timesteps=np.asarray(chunk["timesteps"], dtype=np.int64),
                seed=int(metadata.get("seed", 0)),
                layout=str(metadata.get("layout", "unknown")),
                score=int(arrays["episode_scores"][index]) if "episode_scores" in arrays else 0,
                total_reward=(
                    float(arrays["episode_returns"][index])
                    if "episode_returns" in arrays
                    else float(np.sum(chunk["rewards"]))
                ),
                success=success,
                info={"win": success},
            )
        )
    if not episodes:
        raise ValueError(f"{path} contains no episodes")
    return episodes


def get_dataset(args: argparse.Namespace) -> Tuple[Dict[str, Any], str]:
    """Return ``(dataset_dict, layout)``, collecting episodes when needed."""
    if args.data is None:
        layout = args.layout or "classic"
        max_length = args.max_length or recommend_max_length(
            layout=layout, max_steps=args.max_steps
        )
        print(f"[data] collecting {args.episodes} episode(s) on layout {layout!r}")
        collector = DataCollector(
            layout=layout,
            max_steps=args.max_steps,
            policy=args.policy,
            epsilon=args.epsilon,
            seed=args.seed,
            workers=args.workers,
            eval_fraction=args.eval_fraction,
            max_length=max_length,
            progress=not args.quiet,
        )
        dataset_dict = collector.collect(args.episodes, keep_won_only=args.keep_won_only)
        stats = collector.statistics
        layout = collector.layout
        if args.save_data:
            path = collector.save_to_disk(args.save_data)
            print(f"[data] saved dataset to {path}")
    else:
        path = Path(args.data)
        if path.is_dir():
            print(f"[data] loading Hugging Face dataset from {path}")
            collector = DataCollector.load_from_disk(path)
            dataset_dict = collector.dataset_dict
            stats = collector.statistics
            layout = collector.layout
        elif path.suffix == ".npz":
            print(f"[data] loading .npz dataset from {path}")
            collector = DataCollector(
                layout=args.layout or "classic",
                eval_fraction=args.eval_fraction,
                max_length=args.max_length,
            )
            episodes = _episodes_from_npz(path)
            dataset_dict = collector.from_episodes(episodes)
            stats = collector.statistics
            layout = episodes[0].layout if episodes[0].layout != "unknown" else collector.layout
        else:
            raise FileNotFoundError(f"--data must be a directory or a .npz file, got {path}")

    splits = {name: len(split) for name, split in dataset_dict.items()}
    print(f"[data] splits={splits} episodes={stats.get('episodes', 0)} steps={stats.get('steps', 0)}")
    if "mean_return" in stats:
        print(
            f"[data] return mean={stats['mean_return']:.1f} "
            f"min={stats['min_return']:.1f} max={stats['max_return']:.1f} "
            f"mean_length={stats['mean_length']:.0f} win_rate={stats['win_rate']:.2f}"
        )
    return dataset_dict, layout


def with_undiscounted_returns(dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Replace ``returns_to_go`` with the *undiscounted* return still to come.

    ``dt.data`` stores a ``gamma``-discounted return-to-go (good for statistics),
    but a Decision Transformer conditions on the plain sum of future rewards:
    that is what makes ``--target-return 600`` mean "score 600 points".
    Recomputing it from the ``rewards`` column keeps the two in the same units.
    """
    updated = {}
    for name, split in dataset_dict.items():
        rewards = np.asarray(split["rewards"], dtype=np.float32)
        lengths = np.asarray(split["length"], dtype=np.int64)
        returns = np.zeros_like(rewards)
        for row, length in enumerate(lengths):
            length = int(length)
            returns[row, :length] = np.cumsum(rewards[row, :length][::-1])[::-1]
        if "returns_to_go" in split.column_names:
            split = split.remove_columns(["returns_to_go"])
        updated[name] = split.add_column("returns_to_go", returns.tolist())
    return type(dataset_dict)(updated)


def split_train_eval(
    dataset_dict: Dict[str, Any], seed: int
) -> Tuple[Any, Optional[Any]]:
    """Use the ``eval`` split when present, otherwise carve one out of ``train``."""
    train = dataset_dict["train"]
    if "eval" in dataset_dict:
        return train, dataset_dict["eval"]
    if len(train) >= 4:
        halves = train.train_test_split(test_size=0.1, seed=seed)
        return halves["train"], halves["test"]
    return train, None


def return_scale_of(dataset: Any) -> float:
    """Scale returns-to-go into roughly ``[-1, 1]`` for the transformer."""
    returns = np.asarray(dataset["returns_to_go"], dtype=np.float32)
    if returns.size == 0:
        return DEFAULT_RETURN_SCALE
    return max(float(np.abs(returns).max()), 1.0)


def mean_return_of(dataset: Any) -> float:
    """Mean episode return - the default conditioning target at inference."""
    if "total_reward" in dataset.column_names:
        values = np.asarray(dataset["total_reward"], dtype=np.float64)
    else:
        values = np.asarray(dataset["returns_to_go"], dtype=np.float64)[:, 0]
    return float(values.mean()) if values.size else 0.0


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_model(args: argparse.Namespace, dims: Any, max_ep_len: int) -> HuggingfaceDT:
    """Create the transformer with a config sized for this layout."""
    config = DecisionTransformerConfig(
        state_dim=dims.state_dim,
        act_dim=dims.act_dim,
        hidden_size=args.hidden_size,
        max_ep_len=max(1, int(max_ep_len)),
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner or 4 * args.hidden_size,
        activation_function="gelu",
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        # Actions are predicted by MSE against one-hot vectors, so the output
        # must not be squashed into (-1, 1).
        action_tanh=False,
        # Keep the token ids inside the (single entry) vocabulary so the
        # transformers config validator stays quiet.
        bos_token_id=0,
        eos_token_id=0,
    )
    return HuggingfaceDT(config)


def to_batch(batch: Dict[str, Any], return_scale: float, device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert one ``DTSequenceDataset`` batch into model-ready tensors."""
    observations = batch["observations"]
    if not isinstance(observations, torch.Tensor):
        observations = torch.as_tensor(np.asarray(observations))
    observations = observations.to(device=device, dtype=torch.float32)
    states = observations.flatten(start_dim=2)  # (B, T, state_dim)

    returns_to_go = batch["returns_to_go"]
    if not isinstance(returns_to_go, torch.Tensor):
        returns_to_go = torch.as_tensor(np.asarray(returns_to_go))
    returns_to_go = returns_to_go.to(device=device, dtype=torch.float32)

    mask = batch["mask"]
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(np.asarray(mask))
    mask = mask.to(device=device, dtype=torch.float32)

    return {
        "states": states,
        "actions": torch.as_tensor(np.asarray(batch["actions"])).to(device).long(),
        "returns_to_go": (returns_to_go / return_scale).unsqueeze(-1),
        "timesteps": torch.as_tensor(np.asarray(batch["timesteps"])).to(device).long(),
        "attention_mask": mask,
    }


# ----------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------
def train_model(
    args: argparse.Namespace,
    model: HuggingfaceDT,
    train_ds: Any,
    eval_ds: Optional[Any],
    *,
    context_length: int,
    return_scale: float,
    device: torch.device,
) -> Dict[str, Any]:
    """Fit the model with masked MSE and return the loss history."""
    train_loader = make_dt_dataloader(
        train_ds,
        context_length=context_length,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        stride=args.stride,
        drop_last=len(train_ds) > args.batch_size,
    )
    eval_loader = (
        make_dt_dataloader(
            eval_ds,
            context_length=context_length,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            stride=args.stride,
        )
        if eval_ds is not None
        else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history: Dict[str, List[float]] = {"train_loss": [], "eval_loss": [], "action_accuracy": []}
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, correct, seen = 0.0, 0.0, 0.0
        for step, batch in enumerate(train_loader, start=1):
            tensors = to_batch(batch, return_scale, device)
            loss, predictions = model(
                tensors["states"],
                tensors["actions"],
                tensors["returns_to_go"],
                tensors["timesteps"],
                tensors["attention_mask"],
                targets=tensors["actions"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                chosen = predictions.argmax(dim=-1)
                hits = (chosen == tensors["actions"]).float() * tensors["attention_mask"]
                correct += float(hits.sum().item())
                seen += float(tensors["attention_mask"].sum().item())
            running += float(loss.item())

            if not args.quiet and (step % args.log_every == 0 or step == len(train_loader)):
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        train_loss = running / max(len(train_loader), 1)
        accuracy = correct / max(seen, 1.0)
        eval_loss = evaluate_loss(model, eval_loader, return_scale, device)
        history["train_loss"].append(train_loss)
        history["action_accuracy"].append(accuracy)
        history["eval_loss"].append(None if eval_loss != eval_loss else eval_loss)

        message = f"[train] epoch {epoch}/{args.epochs} loss={train_loss:.4f} acc={accuracy:.3f}"
        if eval_loss == eval_loss:  # not NaN
            message += f" eval_loss={eval_loss:.4f}"
            best_loss = min(best_loss, eval_loss)
        print(message)

    history["best_eval_loss"] = None if best_loss == float("inf") else best_loss
    return history


@torch.no_grad()
def evaluate_loss(
    model: HuggingfaceDT,
    loader: Optional[Any],
    return_scale: float,
    device: torch.device,
) -> float:
    """Mean masked MSE over a dataloader (``nan`` when there is nothing to score)."""
    if loader is None or len(loader) == 0:
        return float("nan")
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        tensors = to_batch(batch, return_scale, device)
        loss, _ = model(
            tensors["states"],
            tensors["actions"],
            tensors["returns_to_go"],
            tensors["timesteps"],
            tensors["attention_mask"],
            targets=tensors["actions"],
        )
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def rollout(
    model: HuggingfaceDT,
    *,
    layout: str,
    episodes: int,
    context_length: int,
    target_return: Optional[float],
    return_scale: float,
    max_steps: Optional[int],
    render_mode: Optional[str],
    render_fps: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Play ``episodes`` games with the greedy ``DTPolicy`` and report returns."""
    from pacman_env import PacmanEnv

    env = PacmanEnv(layout=layout, render_mode=render_mode, max_steps=max_steps)
    policy = DTPolicy(
        model,
        context_length=context_length,
        target_return=target_return,
        return_scale=return_scale,
        device=device,
    )
    returns: List[float] = []
    scores: List[int] = []
    lengths: List[int] = []
    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            policy.reset()
            total, steps, done = 0.0, 0, False
            while not done:
                action = policy.act(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                policy.observe_reward(float(reward))
                total += float(reward)
                steps += 1
                done = terminated or truncated
                if render_mode == "human":
                    env.render()
                    if render_fps > 0:
                        time.sleep(1.0 / render_fps)
            returns.append(total)
            scores.append(int(info.get("score", 0)))
            lengths.append(steps)
            print(
                f"[eval] episode {episode + 1}/{episodes} return={total:.1f} "
                f"score={info.get('score', 0)} steps={steps} "
                f"win={bool(info.get('win', False))}"
            )
    finally:
        env.close()

    summary = {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "mean_score": float(np.mean(scores)),
        "mean_length": float(np.mean(lengths)),
        "returns": returns,
        "scores": scores,
    }
    print(
        f"[eval] mean_return={summary['mean_return']:.1f} "
        f"mean_score={summary['mean_score']:.1f} mean_length={summary['mean_length']:.0f}"
    )
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dt.train",
        description="Train a Decision Transformer on PacManDT and evaluate it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--data", type=str, default=None,
                      help="dataset directory (DataCollector.save_to_disk) or .npz file; "
                           "omit to collect fresh episodes")
    data.add_argument("--save-data", type=str, default=None,
                      help="where to save freshly collected episodes (HF format)")
    data.add_argument("--layout", type=str, default=None,
                      help="maze layout (default: classic, or the layout stored in the dataset)")
    data.add_argument("--episodes", type=int, default=50, help="episodes to collect")
    data.add_argument("--max-steps", type=int, default=1000, help="per-episode step limit")
    data.add_argument("--max-length", type=int, default=None,
                      help="pad/truncate episodes to this length (default: recommended)")
    data.add_argument("--context-length", type=int, default=None,
                      help=f"steps of history the model sees (default: {DEFAULT_CONTEXT_LENGTH})")
    data.add_argument("--stride", type=int, default=1, help="window stride over each episode")
    data.add_argument("--policy", type=str, default="heuristic",
                      help="behaviour policy used for collection")
    data.add_argument("--epsilon", type=float, default=0.05, help="exploration for collection")
    data.add_argument("--workers", type=int, default=8, help="parallel collectors")
    data.add_argument("--keep-won-only", action="store_true", help="keep only winning episodes")
    data.add_argument("--eval-fraction", type=float, default=0.1,
                      help="fraction of episodes held out for evaluation")

    model = parser.add_argument_group("model")
    model.add_argument("--hidden-size", type=int, default=128)
    model.add_argument("--n-layer", type=int, default=5)
    model.add_argument("--n-head", type=int, default=1)
    model.add_argument("--n-inner", type=int, default=None, help="feed-forward width")
    model.add_argument("--dropout", type=float, default=0.1)

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=100)
    optim.add_argument("--batch-size", type=int, default=64)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--weight-decay", type=float, default=1e-4)
    optim.add_argument("--max-grad-norm", type=float, default=0.25)
    optim.add_argument("--num-workers", type=int, default=8, help="dataloader workers")
    optim.add_argument("--seed", type=int, default=0)
    optim.add_argument("--device", type=str, default="cuda", help="cpu / cuda / mps")
    optim.add_argument("--log-every", type=int, default=50)
    optim.add_argument("--quiet", action="store_true")

    run = parser.add_argument_group("output / inference")
    run.add_argument("--out-dir", type=str, default=None,
                     help=f"checkpoint directory (default: {DEFAULT_OUT_DIR}/dt-<layout>)")
    run.add_argument("--eval-only", action="store_true",
                     help="skip training and only evaluate --checkpoint")
    run.add_argument("--checkpoint", type=str, default=None,
                     help="checkpoint to load (required with --eval-only)")
    run.add_argument("--eval-episodes", type=int, default=3,
                     help="rollouts used to score the trained policy (0 to skip)")
    run.add_argument("--target-return", type=float, default=None,
                     help="return the policy is conditioned on (default: mean return of the data)")
    run.add_argument("--render", choices=("none", "human"), default="none",
                     help="render the evaluation rollouts")
    run.add_argument("--render-fps", type=int, default=30, help="playback speed for --render human")
    return parser


def _resolve_context_length(args: argparse.Namespace, max_length: int) -> int:
    if args.context_length is not None:
        if args.context_length > max_length:
            raise ValueError(
                f"--context-length ({args.context_length}) cannot exceed "
                f"the episode length ({max_length})"
            )
        return int(args.context_length)
    return recommend_context_length(max_length, default=DEFAULT_CONTEXT_LENGTH)


def _resolve_device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        return torch.device("cuda")
    return torch.device("cpu")


def _max_length_of(dataset_dict: Dict[str, Any], args: argparse.Namespace) -> int:
    train = dataset_dict["train"]
    if "observations" in train.features:
        return int(train.features["observations"].shape[0])
    return int(args.max_length or args.max_steps)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    device = _resolve_device(args.device)

    # ---- inference only ---------------------------------------------
    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit("--eval-only needs --checkpoint <path>")
        model, payload = load_checkpoint(args.checkpoint, device=device)
        layout = args.layout or payload.get("layout") or "classic"
        target_return = args.target_return
        if target_return is None:
            target_return = payload.get("target_return")
        print(
            f"[load] {args.checkpoint}: state_dim={model.state_dim} act_dim={model.act_dim} "
            f"context_length={payload.get('context_length')} target_return={target_return}"
        )
        rollout(
            model,
            layout=layout,
            episodes=max(1, args.eval_episodes),
            context_length=int(payload.get("context_length", DEFAULT_CONTEXT_LENGTH)),
            target_return=target_return,
            return_scale=float(payload.get("return_scale", DEFAULT_RETURN_SCALE)),
            max_steps=args.max_steps,
            render_mode=None if args.render == "none" else "human",
            render_fps=args.render_fps,
            seed=args.seed,
            device=device,
        )
        return 0

    # ---- data --------------------------------------------------------
    dataset_dict, layout = get_dataset(args)
    dataset_dict = with_undiscounted_returns(dataset_dict)
    train_ds, eval_ds = split_train_eval(dataset_dict, args.seed)
    max_length = _max_length_of(dataset_dict, args)
    context_length = _resolve_context_length(args, max_length)
    dims = dims_from_dataset(dataset_dict, layout=layout)

    return_scale = return_scale_of(train_ds)
    target_return = args.target_return if args.target_return is not None else mean_return_of(train_ds)
    print(f"[setup] layout={layout} state_dim={dims.state_dim} act_dim={dims.act_dim}")
    print(
        f"[setup] max_length={max_length} context_length={context_length} "
        f"return_scale={return_scale:.2f} target_return={target_return:.1f} device={device}"
    )

    # ---- model -------------------------------------------------------
    torch.manual_seed(args.seed)
    model = build_model(args, dims, max_ep_len=max_length).to(device)
    print(f"[model] {model}")

    history = train_model(
        args,
        model,
        train_ds,
        eval_ds,
        context_length=context_length,
        return_scale=return_scale,
        device=device,
    )

    # ---- save --------------------------------------------------------
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR / f"dt-{layout}"
    checkpoint = save_checkpoint(
        out_dir / "checkpoint.pt",
        model,
        context_length=context_length,
        return_scale=return_scale,
        target_return=target_return,
        layout=layout,
        extra={
            "dims": dims.as_dict(),
            "max_length": max_length,
            "history": history,
            "train_args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        },
    )
    print(f"[save] checkpoint written to {checkpoint}")

    # ---- evaluate the policy ----------------------------------------
    eval_summary: Dict[str, Any] = {}
    if args.eval_episodes > 0:
        eval_summary = rollout(
            model,
            layout=layout,
            episodes=args.eval_episodes,
            context_length=context_length,
            target_return=target_return,
            return_scale=return_scale,
            max_steps=args.max_steps,
            render_mode=None if args.render == "none" else "human",
            render_fps=args.render_fps,
            seed=args.seed + 1000,
            device=device,
        )

    metrics = {
        "layout": layout,
        "dims": dims.as_dict(),
        "max_length": max_length,
        "context_length": context_length,
        "return_scale": return_scale,
        "target_return": target_return,
        "history": history,
        "eval": eval_summary,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[save] metrics written to {metrics_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
