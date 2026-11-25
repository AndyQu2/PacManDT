"""Decision-Transformer tooling for PacManDT.

Two independent halves:

``dt.data``
    Behaviour policies, trajectory collection, the ``.npz`` format and the
    Hugging Face ``DataCollector``.  Importing it does **not** need PyTorch.

``dt.dt`` / ``dt.train``
    The model wrapper (``HuggingfaceDT``), the greedy rollout policy
    (``DTPolicy``), checkpoint helpers and the ``python -m dt.train`` program.

The model symbols are resolved lazily so ``import dt.data`` stays torch-free::

    from dt import HuggingfaceDT   # imports torch/transformers on first use
    from dt import data            # never imports torch
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CHECKPOINT_FORMAT",
    "DTPolicy",
    "HuggingfaceDT",
    "load_checkpoint",
    "masked_mse",
    "one_hot_actions",
    "save_checkpoint",
]


def __getattr__(name: str) -> Any:
    """Resolve the model symbols on first access (keeps ``dt.data`` light)."""
    if name in __all__:
        return getattr(import_module(".dt", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
