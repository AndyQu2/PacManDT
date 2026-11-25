"""Decision-Transformer data tooling for PacManDT.

This package holds everything that is *not* part of the environment itself:
behaviour policies, trajectory collection, the ``.npz`` archive format and the
Hugging Face ``datasets`` integration.

Modules
-------
:mod:`dt.data.policies`
    Behaviour policies (``random``, ``heuristic``, ``epsilon_greedy``) used to
    generate demonstrations.
:mod:`dt.data.episodes`
    :class:`Episode`, parallel collection, padding, statistics and the
    ``.npz`` reader/writer.
:mod:`dt.data.hf_dataset`
    :class:`DataCollector` plus the ``datasets.DatasetDict`` schema,
    :class:`DTSequenceDataset` and :func:`make_dt_dataloader`.  Needs the
    optional ``hub`` extra.
:mod:`dt.data.cli`
    The ``python -m dt.data`` command line.
:mod:`dt.data.dims`
    ``state_dim`` / ``act_dim`` / ``max_length`` helpers
    (``python -m dt.data.dims`` prints them for every layout).

Typical use::

    from dt.data import DataCollector

    collector = DataCollector(layout="classic", policy="heuristic", workers=8)
    dataset = collector.collect(num_episodes=1000)
"""

from .cli import DEFAULT_OUTPUT, build_parser, main
from .dims import (
    DEFAULT_CONTEXT_LENGTH,
    MAX_LENGTH_ROUNDING,
    RECOMMENDED_MAX_LENGTH,
    DTDims,
    describe_layouts,
    dims_from_arrays,
    dims_from_dataset,
    dims_from_env,
    dims_from_episodes,
    episode_length_summary,
    get_act_dim,
    get_dims,
    get_obs_shape,
    get_state_dim,
    recommend_context_length,
    recommend_max_length,
)
from .episodes import (
    DATASET_FORMAT,
    DEFAULT_SUCCESS_THRESHOLD,
    Episode,
    collect_episode,
    collect_trajectories,
    compute_returns_to_go,
    dataset_statistics,
    episodes_from_npz,
    filter_episodes,
    iter_episodes,
    load_dataset,
    pad_episodes,
    save_dataset,
)
from .hf_dataset import DataCollector, DTSequenceDataset, HF_DATASET_FORMAT, DEFAULT_HF_DIR, dataset_card, \
    dataset_statistics_from_dict, dataset_to_episodes, episode_features, episodes_to_dataset, episodes_to_dataset_dict, \
    flatten_to_transitions, make_dt_dataloader, split_episodes
from .policies import (
    UNREACHABLE,
    HeuristicPolicy,
    Policy,
    bfs_distance_map,
    bfs_with_parents,
    epsilon_greedy_policy,
    make_policy,
    random_policy,
)

__all__ = [
    # policies
    "Policy",
    "HeuristicPolicy",
    "random_policy",
    "epsilon_greedy_policy",
    "make_policy",
    "bfs_distance_map",
    "bfs_with_parents",
    "UNREACHABLE",
    # episodes / .npz
    "DATASET_FORMAT",
    "DEFAULT_SUCCESS_THRESHOLD",
    "Episode",
    "collect_episode",
    "collect_trajectories",
    "compute_returns_to_go",
    "dataset_statistics",
    "episodes_from_npz",
    "filter_episodes",
    "iter_episodes",
    "load_dataset",
    "pad_episodes",
    "save_dataset",
    # cli
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    # dims / max_length guidance
    "DEFAULT_CONTEXT_LENGTH",
    "MAX_LENGTH_ROUNDING",
    "RECOMMENDED_MAX_LENGTH",
    "DTDims",
    "describe_layouts",
    "dims_from_arrays",
    "dims_from_dataset",
    "dims_from_env",
    "dims_from_episodes",
    "episode_length_summary",
    "get_act_dim",
    "get_dims",
    "get_obs_shape",
    "get_state_dim",
    "recommend_context_length",
    "recommend_max_length",
    # Hugging Face (lazy, needs the "hub" extra)
    "hf_dataset",
    "DataCollector",
    "DTSequenceDataset",
    "HF_DATASET_FORMAT",
    "DEFAULT_HF_DIR",
    "dataset_card",
    "dataset_statistics_from_dict",
    "dataset_to_episodes",
    "episode_features",
    "episodes_to_dataset",
    "episodes_to_dataset_dict",
    "flatten_to_transitions",
    "make_dt_dataloader",
    "split_episodes",
]

_LAZY_ATTRS = {
    "hf_dataset": "hf_dataset",
    "DataCollector": "hf_dataset",
    "DTSequenceDataset": "hf_dataset",
    "HF_DATASET_FORMAT": "hf_dataset",
    "DEFAULT_HF_DIR": "hf_dataset",
    "dataset_card": "hf_dataset",
    "dataset_statistics_from_dict": "hf_dataset",
    "dataset_to_episodes": "hf_dataset",
    "episode_features": "hf_dataset",
    "episodes_to_dataset": "hf_dataset",
    "episodes_to_dataset_dict": "hf_dataset",
    "flatten_to_transitions": "hf_dataset",
    "make_dt_dataloader": "hf_dataset",
    "split_episodes": "hf_dataset",
}


def __getattr__(name: str):
    """Lazily expose :mod:`dt.data.hf_dataset` (needs the optional ``hub`` extra)."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is not None:
        from importlib import import_module

        module = import_module(f".{module_name}", __name__)
        return module if name == module_name else getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
