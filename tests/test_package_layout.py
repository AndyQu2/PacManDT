"""Guards the split between the environment package and the DT tooling."""

from __future__ import annotations

import importlib.util

import pytest

DT_SYMBOLS = (
    "Episode",
    "collect_episode",
    "collect_trajectories",
    "compute_returns_to_go",
    "dataset_statistics",
    "filter_episodes",
    "load_dataset",
    "pad_episodes",
    "save_dataset",
    "HeuristicPolicy",
    "make_policy",
    "random_policy",
    "DataCollector",
    "DTSequenceDataset",
    "hf_dataset",
    "dataset",
    "policies",
)


def test_environment_package_exposes_only_the_environment():
    import pacman_env

    for name in DT_SYMBOLS:
        assert not hasattr(pacman_env, name), f"pacman_env should not export {name}"
    for name in ("PacmanEnv", "ENV_ID", "register_envs", "make_env", "make_flat_env"):
        assert hasattr(pacman_env, name)


def test_dt_data_package_exposes_the_training_tooling():
    from dt import data

    for name in (
        "Episode",
        "collect_episode",
        "collect_trajectories",
        "compute_returns_to_go",
        "dataset_statistics",
        "filter_episodes",
        "load_dataset",
        "pad_episodes",
        "save_dataset",
        "HeuristicPolicy",
        "make_policy",
        "random_policy",
        "DATASET_FORMAT",
    ):
        assert hasattr(data, name), f"dt.data should export {name}"


def test_dt_data_package_cli_entry_point():
    from dt.data import build_parser, main

    assert callable(main)
    assert build_parser().parse_args([]).format == "npz"


@pytest.mark.skipif(
    importlib.util.find_spec("datasets") is None, reason="needs the 'hub' extra"
)
def test_dt_data_package_lazy_hf_exports():
    import dt
    from dt import data

    assert data.HF_DATASET_FORMAT == "pacman-dt/hf-decision-transformer-v1"
    assert dt.data.hf_dataset.DataCollector is data.DataCollector
    assert data.DEFAULT_HF_DIR == dt.data.hf_dataset.DEFAULT_HF_DIR
