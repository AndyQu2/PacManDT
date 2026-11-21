"""Decision-Transformer model, rollout policy and checkpoint helpers.

This module is deliberately small so it can be read top to bottom:

``HuggingfaceDT``
    A ``torch.nn.Module`` wrapping
    :class:`transformers.DecisionTransformerModel`.  It accepts *integer*
    actions, turns them into one-hot vectors, and (optionally) returns the
    masked MSE training loss.

``DTPolicy``
    Stateful greedy inference for one environment.  It keeps the last
    ``context_length`` observations/actions, conditions the model on a target
    return and returns the arg-max action.  This is what
    ``python -m dt.train --eval-only --render human`` uses.

``save_checkpoint`` / ``load_checkpoint``
    One-file checkpoint that stores the model weights *and* everything needed
    to rebuild it (transformers config, ``state_dim``/``act_dim``, the return
    normalisation scale, the layout and the training settings).

Why a wrapper at all?  ``DecisionTransformerModel.forward`` expects
``actions`` already embedded as ``(batch, time, act_dim)`` floats and returns
predictions for *every* timestep in ``output.action_preds``.  The wrapper hides
those details and adds the loss::

    model = HuggingfaceDT(config)
    loss, predictions = model(states, actions, returns_to_go, timesteps,
                              attention_mask, targets=actions)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import DecisionTransformerConfig, DecisionTransformerModel

CHECKPOINT_FORMAT = "pacman-dt/decision-transformer-v1"

__all__ = [
    "CHECKPOINT_FORMAT",
    "DTPolicy",
    "HuggingfaceDT",
    "load_checkpoint",
    "masked_mse",
    "one_hot_actions",
    "save_checkpoint",
]


def one_hot_actions(actions: torch.Tensor, act_dim: int) -> torch.Tensor:
    """``(batch, time)`` integer actions -> ``(batch, time, act_dim)`` floats."""
    return F.one_hot(actions.long(), num_classes=int(act_dim)).to(torch.float32)


def masked_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE averaged over the action dimension and the *valid* timesteps only.

    Padded timesteps carry a zero ``mask`` so they never contribute to the
    gradient.
    """
    per_step = F.mse_loss(predictions, targets, reduction="none").mean(dim=-1)
    if mask is None:
        return per_step.mean()
    weights = mask.to(per_step.dtype)
    return (per_step * weights).sum() / weights.sum().clamp(min=1.0)


class HuggingfaceDT(nn.Module):
    """A ``DecisionTransformerModel`` that speaks integer actions and losses."""

    def __init__(self, config: DecisionTransformerConfig):
        super().__init__()
        self.config = config
        self.dt = DecisionTransformerModel(config)

    # -- shape helpers -------------------------------------------------
    @property
    def state_dim(self) -> int:
        return int(self.config.state_dim)

    @property
    def act_dim(self) -> int:
        return int(self.config.act_dim)

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    def count_parameters(self) -> int:
        """Number of trainable parameters (handy when picking ``hidden_size``)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(state_dim={self.state_dim}, act_dim={self.act_dim}, "
            f"hidden_size={self.hidden_size}, layers={self.config.n_layer}, "
            f"heads={self.config.n_head}, params={self.count_parameters():,})"
        )

    # -- forward -------------------------------------------------------
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ):
        """Run the transformer.

        Parameters
        ----------
        states:
            ``(batch, time, state_dim)`` float tensor.
        actions:
            ``(batch, time)`` integer actions (one-hot encoded here) or an
            already embedded ``(batch, time, act_dim)`` float tensor.
        returns_to_go:
            ``(batch, time, 1)`` float tensor, ideally normalised to ~``[-1, 1]``.
        timesteps:
            ``(batch, time)`` long tensor of environment step indices.
        attention_mask:
            ``(batch, time)`` with ``1`` for real steps and ``0`` for padding.
        targets:
            Optional ``(batch, time)`` integer actions.  When given the return
            value is ``(loss, predictions)``; otherwise just ``predictions``.

        Returns
        -------
        ``predictions`` is ``(batch, time, act_dim)``: the action predicted at
        every timestep from that timestep's state token.
        """
        if actions.dim() == 2:
            actions = one_hot_actions(actions, self.act_dim)
        if targets is not None and targets.dim() == 2:
            targets = one_hot_actions(targets, self.act_dim)

        output = self.dt(
            states=states.float(),
            actions=actions.float(),
            returns_to_go=returns_to_go.float(),
            timesteps=timesteps.long(),
            attention_mask=attention_mask,
        )
        predictions = output.action_preds

        if targets is None:
            return predictions
        return masked_mse(predictions, targets.float(), attention_mask), predictions


class DTPolicy:
    """Greedy Decision-Transformer policy for a single Gymnasium environment.

    The policy is *stateful*: call :meth:`reset` at the start of every episode,
    :meth:`act` once per environment step and :meth:`observe_reward` with the
    reward you got back::

        policy = DTPolicy(model, context_length=30, target_return=120.0)
        obs, info = env.reset()
        policy.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            policy.observe_reward(reward)
            done = terminated or truncated

    The model is conditioned on how much return is still to be earned:
    ``target_return - reward earned so far``, divided by ``return_scale`` - the
    same scale that was applied to the returns-to-go during training - so a
    checkpoint is self-contained.
    """

    def __init__(
        self,
        model: HuggingfaceDT,
        *,
        context_length: int = 30,
        target_return: Optional[float] = None,
        return_scale: float = 1.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if context_length <= 0:
            raise ValueError("context_length must be a positive integer")
        self.model = model
        self.model.eval()
        self.context_length = int(context_length)
        self.target_return = None if target_return is None else float(target_return)
        self.return_scale = float(return_scale) if return_scale else 1.0
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self._eye = np.eye(self.model.act_dim, dtype=np.float32)
        self.reset()

    # -- episode bookkeeping -------------------------------------------
    def reset(self, target_return: Optional[float] = None) -> None:
        """Clear the context buffer and (optionally) set a new target return."""
        if target_return is not None:
            self.target_return = float(target_return)
        self._states: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._timesteps: List[int] = []
        self._step = 0
        self._episode_return = 0.0

    def observe_reward(self, reward: float) -> None:
        """Report the reward earned by the last action.

        Call it right after every ``env.step``.  The policy uses the reward to
        keep the conditioned return honest - the model is fed
        ``target_return - reward earned so far``, which is exactly the quantity
        it saw during training.  Skipping the call simply means the policy keeps
        conditioning on the full target.
        """
        self._episode_return += float(reward)
        if self._rewards:
            self._rewards[-1] = float(reward)

    @property
    def conditioned_return(self) -> float:
        """The target return as the model sees it (after normalisation)."""
        return 0.0 if self.target_return is None else self.target_return / self.return_scale

    @property
    def remaining_return(self) -> float:
        """Normalised return still to be earned - what ``act`` conditions on."""
        if self.target_return is None:
            return 0.0
        return (self.target_return - self._episode_return) / self.return_scale

    def __len__(self) -> int:
        """Number of steps currently in the context window."""
        return len(self._states)

    def _returns_to_go(self, length: int) -> np.ndarray:
        """Return-to-go for every step in the window, normalised like in training.

        During training the model sees a *decaying* return-to-go: the reward of
        every step of the window is subtracted as time passes.  Feeding the same
        shape at inference (instead of one constant value) keeps the model in
        distribution::

            R_i = (rewards from step i onwards) + (target_return - earned so far)
        """
        if self.target_return is None:
            return np.zeros((length, 1), dtype=np.float32)
        rewards = np.asarray(self._rewards[-length:], dtype=np.float32)
        future = np.cumsum(rewards[::-1])[::-1]
        remaining = self.target_return - self._episode_return
        returns = (future + remaining) / self.return_scale
        return returns.reshape(-1, 1).astype(np.float32)

    # -- inference -----------------------------------------------------
    @torch.no_grad()
    def act(self, observation: np.ndarray) -> int:
        """Return the greedy action for ``observation``."""
        state = np.asarray(observation, dtype=np.float32).reshape(-1)
        if state.size != self.model.state_dim:
            raise ValueError(
                f"expected a flattened state of size {self.model.state_dim}, "
                f"got {state.size} (observation shape {np.asarray(observation).shape})"
            )

        # The action of the current step is not known yet: append a zero
        # placeholder.  Causal attention means it cannot influence this step's
        # prediction, and it is overwritten below before the next query.
        self._states.append(state)
        self._actions.append(np.zeros(self.model.act_dim, dtype=np.float32))
        self._rewards.append(0.0)
        max_timestep = max(1, int(self.model.config.max_ep_len)) - 1
        self._timesteps.append(min(self._step, max_timestep))
        self._step += 1

        # Keep only the context window: long episodes must not grow the buffer.
        overflow = len(self._states) - self.context_length
        if overflow > 0:
            del self._states[:overflow]
            del self._actions[:overflow]
            del self._rewards[:overflow]
            del self._timesteps[:overflow]

        start = -min(len(self._states), self.context_length)
        states = np.stack(self._states[start:])
        actions = np.stack(self._actions[start:])
        timesteps = np.asarray(self._timesteps[start:], dtype=np.int64)
        mask = np.ones(len(states), dtype=np.float32)
        returns_to_go = self._returns_to_go(len(states))

        predictions = self.model(
            torch.from_numpy(states).unsqueeze(0).to(self.device),
            torch.from_numpy(actions).unsqueeze(0).to(self.device),
            torch.from_numpy(returns_to_go).unsqueeze(0).to(self.device),
            torch.from_numpy(timesteps).unsqueeze(0).to(self.device),
            torch.from_numpy(mask).unsqueeze(0).to(self.device),
        )
        action = int(predictions[0, -1].argmax().item())
        self._actions[-1] = self._eye[action]
        return action


def save_checkpoint(
    path: Union[str, Path],
    model: HuggingfaceDT,
    *,
    context_length: int = 30,
    return_scale: float = 1.0,
    target_return: Optional[float] = None,
    layout: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write everything needed to reload ``model`` into a single ``.pt`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "config": model.config.to_dict(),
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "context_length": int(context_length),
        "return_scale": float(return_scale),
        "target_return": None if target_return is None else float(target_return),
        "layout": layout,
    }
    payload.update(extra or {})
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[HuggingfaceDT, Dict[str, Any]]:
    """Reload a model written by :func:`save_checkpoint`.

    Returns ``(model, payload)`` where ``payload`` holds ``context_length``,
    ``return_scale``, ``target_return``, ``layout`` and anything passed as
    ``extra`` at save time.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a PacManDT DT checkpoint "
            f"(format {payload.get('format')!r})"
        )
    config = DecisionTransformerConfig(**payload["config"])
    model = HuggingfaceDT(config)
    model.load_state_dict(payload["model_state"])
    if device is not None:
        model.to(device)
    model.eval()
    return model, payload
