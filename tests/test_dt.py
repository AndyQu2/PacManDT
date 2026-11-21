"""Tests for the Decision-Transformer model wrapper, policy and checkpoints."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import DecisionTransformerConfig

from dt.dt import (
    CHECKPOINT_FORMAT,
    DTPolicy,
    HuggingfaceDT,
    load_checkpoint,
    masked_mse,
    one_hot_actions,
    save_checkpoint,
)

STATE_DIM = 8
ACT_DIM = 3
CONTEXT = 6


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_config(**overrides) -> DecisionTransformerConfig:
    options = dict(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        hidden_size=16,
        max_ep_len=32,
        n_layer=1,
        n_head=1,
        n_inner=32,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        action_tanh=False,
        bos_token_id=0,
        eos_token_id=0,
    )
    options.update(overrides)
    return DecisionTransformerConfig(**options)


def make_model(**overrides) -> HuggingfaceDT:
    torch.manual_seed(0)
    model = HuggingfaceDT(make_config(**overrides))
    model.eval()
    return model


def make_batch(batch_size: int = 2, time: int = CONTEXT) -> dict:
    torch.manual_seed(1)
    return {
        "states": torch.randn(batch_size, time, STATE_DIM),
        "actions": torch.randint(0, ACT_DIM, (batch_size, time)),
        "returns_to_go": torch.randn(batch_size, time, 1),
        "timesteps": torch.arange(time).unsqueeze(0).expand(batch_size, time).clone(),
        "attention_mask": torch.ones(batch_size, time),
    }


# ----------------------------------------------------------------------
# one_hot_actions / masked_mse
# ----------------------------------------------------------------------
def test_one_hot_actions_shape_and_values():
    actions = torch.tensor([[0, 2], [1, 1]])
    encoded = one_hot_actions(actions, ACT_DIM)
    assert encoded.shape == (2, 2, ACT_DIM)
    assert encoded.dtype == torch.float32
    assert encoded[0, 0].tolist() == [1.0, 0.0, 0.0]
    assert encoded[0, 1].tolist() == [0.0, 0.0, 1.0]
    assert encoded.sum().item() == 4.0


def test_one_hot_actions_rejects_out_of_range():
    with pytest.raises(RuntimeError):
        one_hot_actions(torch.tensor([[ACT_DIM]]), ACT_DIM)


def test_masked_mse_equals_plain_mse_without_mask():
    predictions = torch.zeros(1, 3, ACT_DIM)
    targets = torch.ones(1, 3, ACT_DIM)
    assert masked_mse(predictions, targets).item() == pytest.approx(1.0)


def test_masked_mse_ignores_padded_steps():
    predictions = torch.zeros(1, 3, ACT_DIM)
    targets = torch.ones(1, 3, ACT_DIM)
    # Two real steps, one padded step with a huge error that must be ignored.
    predictions[0, 2] = 100.0
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    assert masked_mse(predictions, targets, mask).item() == pytest.approx(1.0)


def test_masked_mse_handles_all_zero_mask():
    predictions = torch.zeros(1, 2, ACT_DIM)
    targets = torch.ones(1, 2, ACT_DIM)
    mask = torch.zeros(1, 2)
    assert masked_mse(predictions, targets, mask).item() == pytest.approx(0.0)


# ----------------------------------------------------------------------
# HuggingfaceDT
# ----------------------------------------------------------------------
def test_forward_returns_predictions_for_every_timestep():
    model = make_model()
    batch = make_batch()
    predictions = model(
        batch["states"],
        batch["actions"],
        batch["returns_to_go"],
        batch["timesteps"],
        batch["attention_mask"],
    )
    assert predictions.shape == (2, CONTEXT, ACT_DIM)


def test_forward_with_targets_returns_loss_and_predictions():
    model = make_model()
    batch = make_batch()
    loss, predictions = model(
        batch["states"],
        batch["actions"],
        batch["returns_to_go"],
        batch["timesteps"],
        batch["attention_mask"],
        targets=batch["actions"],
    )
    assert loss.dim() == 0
    assert float(loss.detach()) > 0.0
    assert predictions.shape == (2, CONTEXT, ACT_DIM)


def test_forward_accepts_one_hot_actions():
    model = make_model()
    batch = make_batch()
    embedded = one_hot_actions(batch["actions"], ACT_DIM)
    with torch.no_grad():
        from_int = model(
            batch["states"], batch["actions"], batch["returns_to_go"],
            batch["timesteps"], batch["attention_mask"],
        )
        from_one_hot = model(
            batch["states"], embedded, batch["returns_to_go"],
            batch["timesteps"], batch["attention_mask"],
        )
    torch.testing.assert_close(from_int, from_one_hot)


def test_config_shapes_are_exposed():
    model = make_model()
    assert model.state_dim == STATE_DIM
    assert model.act_dim == ACT_DIM
    assert model.hidden_size == 16
    assert model.count_parameters() > 0
    assert "state_dim=8" in repr(model)


def test_package_lazily_exports_the_model_symbols():
    import dt

    assert dt.HuggingfaceDT is HuggingfaceDT
    assert dt.DTPolicy is DTPolicy
    assert dt.CHECKPOINT_FORMAT == CHECKPOINT_FORMAT
    with pytest.raises(AttributeError):
        dt.does_not_exist


# ----------------------------------------------------------------------
# DTPolicy
# ----------------------------------------------------------------------
def test_policy_returns_valid_actions_and_is_deterministic():
    model = make_model()
    observation = np.zeros((2, 4), dtype=np.int8)  # STATE_DIM == 8

    first = DTPolicy(model, context_length=CONTEXT, target_return=10.0)
    actions_a = []
    for _ in range(4):
        actions_a.append(first.act(observation))
    assert all(0 <= action < ACT_DIM for action in actions_a)

    second = DTPolicy(model, context_length=CONTEXT, target_return=10.0)
    actions_b = [second.act(observation) for _ in range(4)]
    assert actions_a == actions_b


def test_policy_context_window_is_capped():
    model = make_model()
    seen = []
    original = model.forward

    def recording_forward(states, actions, returns_to_go, timesteps, attention_mask=None, targets=None):
        seen.append(states.shape[1])
        return original(states, actions, returns_to_go, timesteps, attention_mask, targets)

    model.forward = recording_forward
    policy = DTPolicy(model, context_length=CONTEXT)
    for _ in range(CONTEXT * 3):
        policy.act(np.zeros(STATE_DIM, dtype=np.float32))

    assert max(seen) == CONTEXT  # never more than the context window
    assert len(policy) == CONTEXT  # the buffer is trimmed, not just sliced
    assert seen[-1] == CONTEXT


def test_policy_reset_clears_history_and_updates_target_return():
    model = make_model()
    policy = DTPolicy(model, context_length=CONTEXT, target_return=5.0, return_scale=5.0)
    assert policy.conditioned_return == pytest.approx(1.0)
    policy.act(np.zeros(STATE_DIM, dtype=np.float32))
    assert len(policy) == 1
    policy.reset(target_return=20.0)
    assert len(policy) == 0
    assert policy.conditioned_return == pytest.approx(4.0)


def test_policy_remaining_return_shrinks_as_rewards_arrive():
    model = make_model()
    policy = DTPolicy(model, context_length=CONTEXT, target_return=10.0, return_scale=5.0)
    assert policy.remaining_return == pytest.approx(2.0)

    policy.observe_reward(2.5)
    assert policy.remaining_return == pytest.approx(1.5)
    policy.observe_reward(2.5)
    assert policy.remaining_return == pytest.approx(1.0)

    policy.reset()
    assert policy.remaining_return == pytest.approx(2.0)  # reward tally cleared


def test_policy_act_conditions_on_remaining_return():
    model = make_model()
    seen = []
    original = model.forward

    def recording_forward(states, actions, returns_to_go, timesteps, attention_mask=None, targets=None):
        seen.append(float(returns_to_go[0, -1, 0]))
        return original(states, actions, returns_to_go, timesteps, attention_mask, targets)

    model.forward = recording_forward
    policy = DTPolicy(model, context_length=CONTEXT, target_return=9.0, return_scale=3.0)
    observation = np.zeros(STATE_DIM, dtype=np.float32)

    policy.act(observation)
    policy.observe_reward(3.0)
    action = policy.act(observation)

    assert seen[0] == pytest.approx(3.0)  # 9 / 3
    assert seen[1] == pytest.approx(2.0)  # (9 - 3) / 3
    assert 0 <= action < ACT_DIM


def test_policy_without_target_return_conditions_on_zero():
    model = make_model()
    policy = DTPolicy(model, context_length=CONTEXT)
    assert policy.conditioned_return == 0.0
    policy.observe_reward(7.0)
    assert policy.remaining_return == 0.0
    assert 0 <= policy.act(np.zeros(STATE_DIM, dtype=np.float32)) < ACT_DIM


def test_policy_feeds_a_decaying_return_to_go_profile():
    model = make_model()
    seen = []
    original = model.forward

    def recording_forward(states, actions, returns_to_go, timesteps, attention_mask=None, targets=None):
        seen.append([round(value, 4) for value in returns_to_go[0, :, 0].tolist()])
        return original(states, actions, returns_to_go, timesteps, attention_mask, targets)

    model.forward = recording_forward
    policy = DTPolicy(model, context_length=CONTEXT, target_return=6.0, return_scale=1.0)
    observation = np.zeros(STATE_DIM, dtype=np.float32)

    for _ in range(3):
        policy.act(observation)
        policy.observe_reward(1.0)

    # training sees a decaying profile, so inference must too
    assert seen[0] == pytest.approx([6.0])
    assert seen[1] == pytest.approx([6.0, 5.0])
    assert seen[2] == pytest.approx([6.0, 5.0, 4.0])
    # the return of the first step of an episode is always the target
    assert all(window[0] == pytest.approx(6.0) for window in seen)


def test_policy_rejects_wrong_observation_size():
    model = make_model()
    policy = DTPolicy(model, context_length=CONTEXT)
    with pytest.raises(ValueError, match="flattened state"):
        policy.act(np.zeros(STATE_DIM + 1, dtype=np.float32))


def test_policy_rejects_non_positive_context_length():
    model = make_model()
    with pytest.raises(ValueError):
        DTPolicy(model, context_length=0)


def test_policy_caps_timesteps_at_max_ep_len():
    model = make_model(max_ep_len=4)
    policy = DTPolicy(model, context_length=2)
    for _ in range(10):
        policy.act(np.zeros(STATE_DIM, dtype=np.float32))
    assert max(policy._timesteps) == 3


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------
def test_checkpoint_round_trip_preserves_logits(tmp_path):
    model = make_model()
    path = save_checkpoint(
        tmp_path / "nested" / "checkpoint.pt",
        model,
        context_length=CONTEXT,
        return_scale=12.0,
        target_return=30.0,
        layout="small",
        extra={"note": "hello"},
    )
    assert path.exists()

    restored, payload = load_checkpoint(path)
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["context_length"] == CONTEXT
    assert payload["return_scale"] == pytest.approx(12.0)
    assert payload["target_return"] == pytest.approx(30.0)
    assert payload["layout"] == "small"
    assert payload["note"] == "hello"
    assert restored.state_dim == STATE_DIM
    assert restored.act_dim == ACT_DIM

    batch = make_batch()
    with torch.no_grad():
        expected = model(
            batch["states"], batch["actions"], batch["returns_to_go"],
            batch["timesteps"], batch["attention_mask"],
        )
        actual = restored(
            batch["states"], batch["actions"], batch["returns_to_go"],
            batch["timesteps"], batch["attention_mask"],
        )
    torch.testing.assert_close(expected, actual)


def test_load_checkpoint_rejects_foreign_file(tmp_path):
    path = tmp_path / "not_a_checkpoint.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(ValueError, match="not a PacManDT DT checkpoint"):
        load_checkpoint(path)


def test_policy_from_checkpoint_matches_original(tmp_path):
    model = make_model()
    path = save_checkpoint(tmp_path / "c.pt", model, context_length=CONTEXT)
    restored, payload = load_checkpoint(path)
    observation = np.arange(STATE_DIM, dtype=np.float32)

    original = DTPolicy(model, context_length=payload["context_length"])
    reloaded = DTPolicy(restored, context_length=payload["context_length"])
    assert [original.act(observation) for _ in range(3)] == [
        reloaded.act(observation) for _ in range(3)
    ]
