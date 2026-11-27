"""Collection protocol: *what* to record, not just *how much*.

A Decision Transformer is trained by behaviour cloning on a fixed dataset, so
the dataset decides the ceiling of what the model can learn.  Naively rolling
out one deterministic heuristic for N episodes produces a narrow band of
returns and states: the transformer never sees a badly played game, so it never
learns what a *high* return-to-go implies, and it never sees the ghost
configurations that only appear when the player takes risks.

This module implements the collection protocol that pairs with the ghost
observation stack (:mod:`dt.data.observations`):

1. **Policy portfolio** - episodes are scheduled across several behaviour
   policies with exact proportions (``heuristic`` for competence,
   ``epsilon_greedy`` for coverage, ``random`` for the low-return tail).  The
   schedule is deterministic and evenly interleaved, so a run is reproducible
   and no policy is clustered into a contiguous block of seeds.
2. **Quality tiers** - episodes are bucketed by return; ``max_episodes``
   trimming keeps the best episode *inside every tier* instead of only the
   global top, which preserves the return-to-go spectrum the DT conditions on.
3. **Success balancing** - optionally caps the majority (successful or failed)
   class so the model does not learn "everything wins".
4. **Stable hash split** - the train/eval split is a hash of the episode seed,
   so re-collecting more episodes never reshuffles the existing assignment.
5. **A report** - per-policy and per-tier statistics, printed before training
   so a bad dataset is visible immediately.

Usage::

    from dt.data import CollectionProtocol

    protocol = CollectionProtocol(layout="classic", episodes=2000, workers=16)
    result = protocol.run()
    result.report          # dict of statistics
    result.dataset_dict()  # HF DatasetDict (train/eval), if the hub extra is installed

Command line (through ``python -m dt.data``)::

    python -m dt.data --obs-mode ghost --episodes 2000 \\
        --portfolio heuristic=0.6,epsilon_greedy=0.3,random=0.1
"""

from __future__ import annotations

import math
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .episodes import (
    DEFAULT_SUCCESS_THRESHOLD,
    Episode,
    collect_trajectories,
    dataset_statistics,
)
from .observations import (
    GHOST_OBS_MODE,
    GhostObservationConfig,
    describe_observation,
    resolve_observation,
)

#: Default mixture of behaviour policies.  Heuristic episodes teach competence,
#: ``epsilon_greedy`` widens the state distribution around the heuristic
#: trajectory, and ``random`` supplies the low-return tail that makes the
#: return-to-go conditioning meaningful.
DEFAULT_PORTFOLIO: Tuple[Tuple[str, float], ...] = (
    ("heuristic", 0.60),
    ("epsilon_greedy", 0.30),
    ("random", 0.10),
)

#: Policies accepted in a portfolio (kept in sync with ``policies.make_policy``).
KNOWN_POLICIES = ("heuristic", "epsilon_greedy", "random")


# ----------------------------------------------------------------------
# Portfolio handling
# ----------------------------------------------------------------------
def parse_portfolio(spec: Union[str, Mapping[str, float], Sequence[Any], None]) -> Dict[str, float]:
    """Normalise a portfolio into ``{policy: weight}`` with weights summing to 1.

    Accepts ``{"heuristic": 0.6, "random": 0.4}``, ``"heuristic=0.6,random=0.4"``
    or ``[("heuristic", 0.6), ("random", 0.4)]``.  ``None`` returns the default
    :data:`DEFAULT_PORTFOLIO`.
    """
    if spec is None:
        return dict(DEFAULT_PORTFOLIO)
    if isinstance(spec, str):
        pairs: List[Tuple[str, float]] = []
        for chunk in spec.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(
                    f"portfolio entry {chunk!r} must look like 'heuristic=0.6'"
                )
            name, weight = chunk.split("=", 1)
            pairs.append((name.strip(), float(weight)))
        portfolio = dict(pairs)
    elif isinstance(spec, Mapping):
        portfolio = {str(name): float(weight) for name, weight in spec.items()}
    else:
        portfolio = {str(name): float(weight) for name, weight in spec}

    portfolio = {name.lower().replace("-", "_"): weight for name, weight in portfolio.items()}
    if not portfolio:
        raise ValueError("portfolio must contain at least one policy")
    for name in portfolio:
        if name not in KNOWN_POLICIES:
            raise ValueError(
                f"unknown policy {name!r} in portfolio; use {KNOWN_POLICIES}"
            )
    negative = [name for name, weight in portfolio.items() if weight < 0]
    if negative:
        raise ValueError(f"portfolio weights must be >= 0 (got {negative})")
    total = float(sum(portfolio.values()))
    if total <= 0:
        raise ValueError("portfolio weights must sum to a positive value")
    return {name: weight / total for name, weight in portfolio.items()}


def _apportion(weights: Mapping[str, float], total: int) -> Dict[str, int]:
    """Split ``total`` items over ``weights`` by largest remainder (exact sum).

    ``weights`` may be proportions (they sum to 1) or raw counts (they sum to
    whatever); they are normalised here so callers can pass either.
    """
    if total <= 0:
        return {name: 0 for name in weights}
    total_weight = float(sum(weights.values()))
    if total_weight <= 0:
        return {name: 0 for name in weights}
    scaled = {name: float(weight) / total_weight * total for name, weight in weights.items()}
    counts = {name: int(math.floor(value)) for name, value in scaled.items()}
    shortfall = total - sum(counts.values())
    if shortfall > 0:
        order = sorted(
            scaled,
            key=lambda name: (scaled[name] - counts[name], -list(weights).index(name)),
            reverse=True,
        )
        for name in order[:shortfall]:
            counts[name] += 1
    return counts


def _smooth_round_robin(counts: Mapping[str, int]) -> List[str]:
    """Interleave ``counts`` as evenly as possible (smooth weighted round robin).

    ``{"heuristic": 6, "random": 4}`` becomes ``h h r h r h r h h r`` instead of
    ``h h h h h h r r r r``, so every contiguous block of seeds (a worker's
    chunk) sees a representative mixture.
    """
    weights = {name: int(count) for name, count in counts.items() if int(count) > 0}
    if not weights:
        return []
    order = list(weights)
    total_weight = sum(weights.values())
    current = {name: 0 for name in order}
    remaining = dict(weights)
    schedule: List[str] = []
    while len(schedule) < total_weight:
        for name in order:
            if remaining[name] > 0:
                current[name] += weights[name]
        available = [name for name in order if remaining[name] > 0]
        pick = max(available, key=lambda name: (current[name], -order.index(name)))
        schedule.append(pick)
        remaining[pick] -= 1
        current[pick] -= total_weight
    return schedule


def build_policy_schedule(
    num_episodes: int,
    portfolio: Union[str, Mapping[str, float], Sequence[Any], None] = None,
    *,
    seed: int = 0,
) -> List[str]:
    """Per-episode behaviour-policy schedule with exact portfolio proportions.

    The schedule is fully deterministic: the same ``(num_episodes, portfolio,
    seed)`` always yields the same list, independent of worker count.  ``seed``
    only rotates the schedule, which changes which *seed* gets which policy
    without changing the proportions.
    """
    num_episodes = int(num_episodes)
    if num_episodes <= 0:
        return []
    weights = parse_portfolio(portfolio)
    schedule = _smooth_round_robin(_apportion(weights, num_episodes))
    offset = int(seed) % len(schedule)
    if offset:
        schedule = schedule[offset:] + schedule[:offset]
    return schedule


# ----------------------------------------------------------------------
# Quality tiers / curation
# ----------------------------------------------------------------------
def quality_tiers(
    episodes: Sequence[Episode], num_tiers: int = 4
) -> List[List[Episode]]:
    """Bucket ``episodes`` by return into ``num_tiers`` contiguous tiers.

    Tier ``0`` holds the worst episodes, the last tier the best ones.  Tiers are
    what keeps the return-to-go spectrum intact when a dataset is trimmed.
    """
    num_tiers = max(1, int(num_tiers))
    ordered = sorted(episodes, key=lambda episode: (episode.total_reward, episode.seed))
    if not ordered:
        return [[] for _ in range(num_tiers)]
    tiers: List[List[Episode]] = []
    for index in range(num_tiers):
        start = (index * len(ordered)) // num_tiers
        stop = ((index + 1) * len(ordered)) // num_tiers
        tiers.append(ordered[start:stop])
    return tiers


def balance_classes(
    episodes: Sequence[Episode], max_ratio: float = 2.0
) -> List[Episode]:
    """Cap the majority outcome class at ``max_ratio`` times the minority.

    ``success`` is ``episode.success`` (return >= threshold).  Without balancing,
    a good policy yields ~100 % successes and the DT never learns what a failing
    trajectory looks like - which is exactly the signal a return-to-go
    conditioned model needs.  The *worst* episodes of the majority class are
    dropped, keeping the useful variety.
    """
    successes = [episode for episode in episodes if episode.success]
    failures = [episode for episode in episodes if not episode.success]
    if not successes or not failures:
        return list(episodes)

    max_ratio = max(1.0, float(max_ratio))
    if len(successes) > len(failures) * max_ratio:
        limit = int(math.floor(len(failures) * max_ratio))
        successes = sorted(successes, key=lambda episode: episode.total_reward, reverse=True)[:limit]
    elif len(failures) > len(successes) * max_ratio:
        limit = int(math.floor(len(successes) * max_ratio))
        failures = sorted(failures, key=lambda episode: episode.total_reward, reverse=True)[:limit]
    keep = {id(episode) for episode in successes} | {id(episode) for episode in failures}
    return [episode for episode in episodes if id(episode) in keep]


def curate_episodes(
    episodes: Sequence[Episode],
    *,
    min_return: Optional[float] = None,
    max_episodes: Optional[int] = None,
    balance_success: bool = False,
    max_ratio: float = 2.0,
    num_tiers: int = 4,
) -> List[Episode]:
    """Filter and trim a collected batch while keeping its return spectrum.

    Order of operations: ``min_return`` filter -> success balancing -> stratified
    trim to ``max_episodes`` (best episodes of every quality tier).  The original
    collection order is preserved in the output.
    """
    kept = list(episodes)
    if min_return is not None:
        kept = [episode for episode in kept if episode.total_reward >= float(min_return)]
    if balance_success:
        kept = balance_classes(kept, max_ratio=max_ratio)
    if max_episodes is not None:
        max_episodes = int(max_episodes)
        if max_episodes <= 0:
            raise ValueError("max_episodes must be a positive integer")
        if max_episodes < len(kept):
            tiers = quality_tiers(kept, num_tiers)
            weights = {f"tier{index}": float(len(tier)) for index, tier in enumerate(tiers)}
            quotas = _apportion(weights, max_episodes)
            selected: List[Episode] = []
            for index, tier in enumerate(tiers):
                quota = quotas.get(f"tier{index}", 0)
                if quota <= 0:
                    continue
                best = sorted(tier, key=lambda episode: episode.total_reward, reverse=True)
                selected.extend(best[:quota])
            # Rounding can leave a slot or two: fill from the global ranking.
            if len(selected) < max_episodes:
                chosen = {id(episode) for episode in selected}
                remainder = sorted(
                    (episode for episode in kept if id(episode) not in chosen),
                    key=lambda episode: episode.total_reward,
                    reverse=True,
                )
                selected.extend(remainder[: max_episodes - len(selected)])
            keep = {id(episode) for episode in selected}
            kept = [episode for episode in kept if id(episode) in keep]
    return kept


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------
def _hash_fraction(*parts: Any) -> float:
    """Deterministic ``[0, 1)`` value derived from ``parts`` (stable across runs)."""
    key = "|".join(str(part) for part in parts).encode("utf-8")
    return (zlib.crc32(key) % 10_000) / 10_000.0


def split_by_hash(
    episodes: Sequence[Episode],
    eval_fraction: float = 0.1,
    *,
    seed: int = 0,
) -> Tuple[List[Episode], List[Episode]]:
    """Split into ``(train, eval)`` using a hash of ``(seed, episode seed, policy)``.

    Unlike a positional split this assignment is *stable*: appending newly
    collected episodes leaves every existing episode on the same side, so an
    evaluation set never leaks into training after a re-collection.
    """
    eval_fraction = float(eval_fraction)
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in [0, 1)")
    train: List[Episode] = []
    evaluation: List[Episode] = []
    for episode in episodes:
        value = _hash_fraction(seed, episode.seed, episode.policy)
        (evaluation if value < eval_fraction else train).append(episode)
    return train, evaluation


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _summary(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "min": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "max": float(array.max()),
    }


def protocol_report(
    episodes: Sequence[Episode],
    *,
    collected: Optional[Sequence[Episode]] = None,
    num_tiers: int = 4,
    splits: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Statistics of a curated dataset: overall, per policy and per tier.

    ``splits`` (e.g. ``{"train": 900, "eval": 100}``) is copied into the report
    so the numbers that were written to disk are the numbers that get printed.
    """
    episodes = list(episodes)
    report: Dict[str, Any] = {
        "episodes": len(episodes),
        "collected_episodes": len(collected) if collected is not None else len(episodes),
        "steps": int(sum(episode.length for episode in episodes)),
        "statistics": dataset_statistics(episodes),
        "return": _summary(episode.total_reward for episode in episodes),
        "length": _summary(episode.length for episode in episodes),
        "by_policy": {},
        "tiers": [],
    }
    if splits is not None:
        report["splits"] = {str(name): int(count) for name, count in splits.items()}

    policies: Dict[str, List[Episode]] = {}
    for episode in episodes:
        policies.setdefault(episode.policy or "unknown", []).append(episode)
    for name, group in sorted(policies.items()):
        report["by_policy"][name] = {
            "episodes": len(group),
            "share": len(group) / max(1, len(episodes)),
            "mean_return": float(np.mean([episode.total_reward for episode in group])),
            "success_rate": float(np.mean([1.0 if episode.success else 0.0 for episode in group])),
            "mean_length": float(np.mean([episode.length for episode in group])),
        }

    for index, tier in enumerate(quality_tiers(episodes, num_tiers)):
        if not tier:
            continue
        report["tiers"].append(
            {
                "tier": index,
                "episodes": len(tier),
                "return_min": float(min(episode.total_reward for episode in tier)),
                "return_max": float(max(episode.total_reward for episode in tier)),
                "success_rate": float(
                    np.mean([1.0 if episode.success else 0.0 for episode in tier])
                ),
            }
        )
    return report


def format_report(report: Mapping[str, Any]) -> str:
    """Render :func:`protocol_report` as a compact text block."""
    stats = report.get("statistics", {})
    lines = [
        f"episodes     : {report.get('episodes', 0)} kept / "
        f"{report.get('collected_episodes', 0)} collected",
        f"transitions  : {report.get('steps', 0)}",
        f"return       : mean {stats.get('mean_return', 0.0):.2f}  "
        f"std {stats.get('std_return', 0.0):.2f}  "
        f"min {stats.get('min_return', 0.0):.2f}  max {stats.get('max_return', 0.0):.2f}",
        f"success rate : {100 * stats.get('success_rate', 0.0):.1f}%   "
        f"win rate {100 * stats.get('win_rate', 0.0):.1f}%",
    ]
    by_policy = report.get("by_policy") or {}
    if report.get("splits"):
        lines.append(
            "split        : "
            + "  ".join(f"{name} {count}" for name, count in report["splits"].items())
        )
    if by_policy:
        lines.append("by policy    :")
        for name, entry in by_policy.items():
            lines.append(
                f"  {name:<15}{entry['episodes']:>6} eps  "
                f"({100 * entry['share']:.1f}%)  "
                f"mean return {entry['mean_return']:7.2f}  "
                f"success {100 * entry['success_rate']:5.1f}%"
            )
    tiers = report.get("tiers") or []
    if tiers:
        lines.append("by tier      :")
        for entry in tiers:
            lines.append(
                f"  tier {entry['tier']:<10}{entry['episodes']:>6} eps  "
                f"return {entry['return_min']:7.2f} .. {entry['return_max']:7.2f}  "
                f"success {100 * entry['success_rate']:5.1f}%"
            )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# The protocol itself
# ----------------------------------------------------------------------
@dataclass
class ProtocolResult:
    """What :meth:`CollectionProtocol.run` produced."""

    protocol: "CollectionProtocol"
    episodes: List[Episode]
    train: List[Episode]
    eval: List[Episode]
    schedule: List[str]
    report: Dict[str, Any]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Dataset metadata describing the protocol (JSON friendly)."""
        protocol = self.protocol
        observation = resolve_observation(
            protocol.obs_mode, layout=protocol.layout, config=protocol.obs_config
        )
        return {
            "obs_mode": observation.mode,
            "observation": observation.as_dict(),
            "protocol": {
                "portfolio": dict(protocol.portfolio),
                "schedule": {name: self.schedule.count(name) for name in protocol.portfolio},
                "epsilon": protocol.epsilon,
                "min_return": protocol.min_return,
                "max_episodes": protocol.max_episodes,
                "balance_success": protocol.balance_success,
                "num_tiers": protocol.num_tiers,
                "eval_fraction": protocol.eval_fraction,
                "seed": protocol.seed,
            },
            "policies": sorted({episode.policy for episode in self.episodes}),
            "report": self.report,
        }

    def dataset_dict(self) -> Any:
        """Build a HF ``DatasetDict`` (``train``/``eval``) from the curated episodes."""
        from .hf_dataset import DataCollector  # optional "hub" extra

        collector = DataCollector(
            layout=self.protocol.layout,
            lives=self.protocol.lives,
            max_steps=self.protocol.max_steps,
            policy="portfolio",
            gamma=self.protocol.gamma,
            frame_skip=self.protocol.frame_skip,
            epsilon=self.protocol.epsilon,
            success_threshold=self.protocol.success_threshold,
            seed=self.protocol.seed,
            workers=self.protocol.workers,
            progress=False,
            eval_fraction=self.protocol.eval_fraction,
            max_length=self.protocol.max_length,
            obs_mode=self.protocol.obs_mode,
            obs_config=self.protocol.obs_config,
        )
        collector.from_episodes(
            self.episodes,
            eval_fraction=self.protocol.eval_fraction,
            # the stable hash split, not the positional one: an eval episode
            # stays in eval even after the dataset grows
            train_episodes=self.train,
            eval_episodes=self.eval,
        )
        return collector.dataset_dict


@dataclass(frozen=True)
class CollectionProtocol:
    """Configuration + execution of the ghost-aware collection protocol."""

    layout: str = "classic"
    episodes: int = 1000
    max_steps: Optional[int] = 1000
    lives: int = 3
    frame_skip: int = 1
    gamma: float = 0.99
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
    obs_mode: str = GHOST_OBS_MODE
    obs_config: Optional[GhostObservationConfig] = None
    portfolio: Mapping[str, float] = field(
        default_factory=lambda: parse_portfolio(None)
    )
    epsilon: float = 0.1
    seed: int = 0
    workers: int = 1
    eval_fraction: float = 0.1
    min_return: Optional[float] = None
    max_episodes: Optional[int] = None
    balance_success: bool = False
    success_ratio: float = 2.0
    num_tiers: int = 4
    max_length: Optional[int] = None
    progress: bool = True

    def __post_init__(self) -> None:
        if int(self.episodes) <= 0:
            raise ValueError("episodes must be a positive integer")
        if not 0.0 <= float(self.eval_fraction) < 1.0:
            raise ValueError("eval_fraction must be in [0, 1)")
        if int(self.num_tiers) < 1:
            raise ValueError("num_tiers must be a positive integer")
        object.__setattr__(self, "portfolio", parse_portfolio(self.portfolio))

    # -- planning ------------------------------------------------------
    def schedule(self) -> List[str]:
        """Behaviour policy per episode (deterministic, exact proportions)."""
        return build_policy_schedule(self.episodes, self.portfolio, seed=self.seed)

    def describe(self) -> str:
        """Multi-line plan: observation stack + portfolio + curation settings."""
        lines = [
            describe_observation(self.obs_mode, layout=self.layout, config=self.obs_config),
            "",
            f"layout {self.layout!r}: {self.episodes} episodes, "
            f"max_steps={self.max_steps}, workers={self.workers}, seed={self.seed}",
            "portfolio:",
        ]
        schedule = self.schedule()
        for name, weight in self.portfolio.items():
            count = schedule.count(name)
            lines.append(
                f"  {name:<15}{100 * weight:5.1f}%  -> {count:>6} episodes"
            )
        lines.append(
            f"curation: min_return={self.min_return} max_episodes={self.max_episodes} "
            f"balance_success={self.balance_success} tiers={self.num_tiers} "
            f"eval_fraction={self.eval_fraction}"
        )
        return "\n".join(lines)

    # -- execution -----------------------------------------------------
    def collect(self) -> List[Episode]:
        """Roll out the scheduled policies and return the raw episodes."""
        return collect_trajectories(
            layout=self.layout,
            lives=self.lives,
            num_episodes=int(self.episodes),
            max_steps=self.max_steps,
            success_threshold=self.success_threshold,
            policy=self.schedule()[0] if len(self.portfolio) == 1 else "heuristic",
            seed=self.seed,
            gamma=self.gamma,
            frame_skip=self.frame_skip,
            epsilon=self.epsilon,
            render_mode=None,
            progress=self.progress,
            workers=self.workers,
            obs_mode=self.obs_mode,
            obs_config=self.obs_config,
            policy_schedule=self.schedule(),
        )

    def curate(self, episodes: Sequence[Episode]) -> List[Episode]:
        """Apply the quality filters of :func:`curate_episodes`."""
        return curate_episodes(
            episodes,
            min_return=self.min_return,
            max_episodes=self.max_episodes,
            balance_success=self.balance_success,
            max_ratio=self.success_ratio,
            num_tiers=self.num_tiers,
        )

    def split(self, episodes: Sequence[Episode]) -> Tuple[List[Episode], List[Episode]]:
        """Stable hash split into ``(train, eval)``."""
        return split_by_hash(episodes, self.eval_fraction, seed=self.seed)

    def run(self) -> ProtocolResult:
        """Collect, curate, split and report in one call."""
        collected = self.collect()
        if not collected:
            raise ValueError("collection produced no episodes")
        kept = self.curate(collected)
        if not kept:
            raise ValueError(
                "no episodes left after curation; lower min_return or drop balance_success"
            )
        train, evaluation = self.split(kept)
        if self.eval_fraction > 0 and not evaluation:
            warnings.warn(
                f"the stable hash split put no episode in eval (eval_fraction="
                f"{self.eval_fraction} with {len(kept)} episodes); collect more "
                f"episodes or raise eval_fraction",
                stacklevel=2,
            )
        report = protocol_report(
            kept,
            collected=collected,
            num_tiers=self.num_tiers,
            splits={"train": len(train), "eval": len(evaluation)},
        )
        return ProtocolResult(
            protocol=self,
            episodes=kept,
            train=train,
            eval=evaluation,
            schedule=self.schedule(),
            report=report,
        )


def recommended_protocol(
    layout: str = "classic",
    episodes: int = 1000,
    *,
    workers: int = 1,
    seed: int = 0,
    obs_mode: str = GHOST_OBS_MODE,
) -> CollectionProtocol:
    """The protocol this project recommends for a fresh dataset."""
    return CollectionProtocol(
        layout=layout,
        episodes=episodes,
        workers=workers,
        seed=seed,
        obs_mode=obs_mode,
    )


__all__ = [
    "DEFAULT_PORTFOLIO",
    "KNOWN_POLICIES",
    "CollectionProtocol",
    "ProtocolResult",
    "balance_classes",
    "build_policy_schedule",
    "curate_episodes",
    "format_report",
    "parse_portfolio",
    "protocol_report",
    "quality_tiers",
    "recommended_protocol",
    "split_by_hash",
]
