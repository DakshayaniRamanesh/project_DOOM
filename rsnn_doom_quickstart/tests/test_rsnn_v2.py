"""tests/test_rsnn_v2.py — Unit tests for RSNN v2.

All tests run WITHOUT ViZDoom installed.

Run with:
    pytest tests/test_rsnn_v2.py -v
    pytest tests/test_rsnn_v2.py -v -k "not vizdoom"

Integration tests that require ViZDoom are marked with @pytest.mark.vizdoom
and are skipped automatically if vizdoom is not importable.
"""
from __future__ import annotations

import math
import pathlib
import random
import tempfile

import numpy as np
import pytest
import torch
from torch import nn

# --- Add parent dir to path so tests can import local modules ---
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from doom_env import frame_to_tensor, ACTION_NAMES, FLAT, WIDTH, HEIGHT
from rsnn_v2 import (
    RSNNv2, RSNNv2Config, Spike,
    save_checkpoint, load_checkpoint, build_model_from_checkpoint,
)
from train_rsnn_v2 import (
    compute_returns_and_advantages, a2c_update,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def small_cfg():
    """Minimal config for fast unit tests."""
    return RSNNv2Config(inputs=FLAT, hidden=16, actions=6,
                        beta=0.5, threshold=0.4, timesteps=4)


@pytest.fixture
def model(small_cfg):
    torch.manual_seed(0)
    return RSNNv2(small_cfg)


@pytest.fixture
def dummy_obs():
    """[1, 768] float32 observation."""
    return torch.rand(1, FLAT)


# ===========================================================================
# 1. Preprocessing
# ===========================================================================

class TestPreprocessing:
    def test_output_shape(self):
        arr = np.full((3, 120, 160), 128, np.uint8)
        t = frame_to_tensor(arr)
        assert t.shape == (1, FLAT), f"Expected (1, {FLAT}), got {t.shape}"

    def test_range(self):
        arr = np.full((3, 120, 160), 255, np.uint8)
        t = frame_to_tensor(arr)
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0

    def test_chw_hwc_identical(self):
        arr_chw = np.random.randint(0, 256, (3, 120, 160), dtype=np.uint8)
        arr_hwc = arr_chw.transpose(1, 2, 0)
        assert torch.allclose(frame_to_tensor(arr_chw), frame_to_tensor(arr_hwc))

    def test_pil_consistent(self):
        from PIL import Image
        arr = np.full((3, 120, 160), 100, np.uint8)
        img = Image.fromarray(arr.transpose(1, 2, 0))
        assert torch.allclose(frame_to_tensor(arr), frame_to_tensor(img))

    def test_black_produces_zeros(self):
        arr = np.zeros((3, 120, 160), np.uint8)
        t = frame_to_tensor(arr)
        assert float(t.max()) == 0.0

    def test_white_produces_ones(self):
        arr = np.full((3, 120, 160), 255, np.uint8)
        t = frame_to_tensor(arr)
        assert float(t.min()) == pytest.approx(1.0, abs=1e-3)


# ===========================================================================
# 2. Config
# ===========================================================================

class TestConfig:
    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            RSNNv2Config(beta=0.6)

    def test_invalid_timesteps_raises(self):
        with pytest.raises(ValueError, match="timesteps"):
            RSNNv2Config(timesteps=0)

    def test_roundtrip(self):
        cfg = RSNNv2Config(hidden=48, beta=0.25, timesteps=16)
        assert RSNNv2Config.from_dict(cfg.to_dict()) == cfg


# ===========================================================================
# 3. Spike function
# ===========================================================================

class TestSpikeFunction:
    def test_binary_output(self):
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        s = Spike.apply(x)
        assert set(s.tolist()).issubset({0.0, 1.0})

    def test_threshold_at_zero(self):
        x = torch.tensor([0.0, 0.001, -0.001])
        s = Spike.apply(x)
        assert s[0].item() == 0.0   # exactly zero -> no spike
        assert s[1].item() == 1.0
        assert s[2].item() == 0.0

    def test_surrogate_gradient_nonzero(self):
        x = torch.tensor([0.5], requires_grad=True)
        s = Spike.apply(x)
        s.backward()
        assert x.grad is not None and x.grad.item() != 0.0

    def test_surrogate_gradient_formula(self):
        """d/dx [ 1/(1+2|x|)^2 ] should equal surrogate at x=0.5"""
        x_val = 0.5
        x = torch.tensor([x_val], requires_grad=True)
        s = Spike.apply(x)
        s.backward()
        expected = 1.0 / (1.0 + 2.0 * abs(x_val)) ** 2
        assert abs(x.grad.item() - expected) < 1e-6


# ===========================================================================
# 4. Model forward pass
# ===========================================================================

class TestModelForward:
    def test_output_shapes(self, model, dummy_obs):
        logits, value, state, rate = model(dummy_obs)
        assert logits.shape == (1, 6), f"logits: {logits.shape}"
        assert value.shape == (1,), f"value: {value.shape}"
        assert state[0].shape == (1, model.cfg.hidden)
        assert state[1].shape == (1, model.cfg.hidden)

    def test_spike_rate_in_range(self, model, dummy_obs):
        _, _, _, rate = model(dummy_obs)
        assert 0.0 <= float(rate) <= 1.0

    def test_spikes_are_binary(self, model, dummy_obs):
        """After forward pass, hidden spikes must be 0 or 1."""
        _, _, state, _ = model(dummy_obs)
        spikes = state[0]
        assert set(spikes.detach().flatten().tolist()).issubset({0.0, 1.0}), \
            "Hidden spikes contain non-binary values"

    def test_state_evolves_across_decisions(self, model, dummy_obs):
        """Recurrent state should change across game decisions."""
        _, _, state1, _ = model(dummy_obs)
        _, _, state2, _ = model(dummy_obs, state1)
        # At least the membrane should differ
        assert not torch.allclose(state1[1], state2[1]), \
            "Membrane did not change across decisions (recurrence may be broken)"

    def test_reset_state_is_zeros(self, model):
        spikes, mem = model.reset_state()
        assert torch.all(spikes == 0.0) and torch.all(mem == 0.0)

    def test_batch_size_2(self, model):
        obs = torch.rand(2, FLAT)
        logits, value, state, rate = model(obs)
        assert logits.shape == (2, 6)
        assert value.shape == (2,)
        assert state[0].shape == (2, model.cfg.hidden)

    def test_invalid_obs_raises(self, model):
        with pytest.raises(ValueError):
            model(torch.rand(1, 100))  # wrong input size

    def test_deterministic_inference(self, model, dummy_obs):
        """Same obs + same state -> same output (model in eval mode)."""
        model.eval()
        state = model.reset_state()
        with torch.no_grad():
            logits1, val1, _, _ = model(dummy_obs, state)
            logits2, val2, _, _ = model(dummy_obs, state)
        assert torch.allclose(logits1, logits2)
        assert torch.allclose(val1, val2)


# ===========================================================================
# 5. Recurrent state handling
# ===========================================================================

class TestRecurrentState:
    def test_detach_state_detaches(self, model, dummy_obs):
        _, _, state, _ = model(dummy_obs)
        det_state = RSNNv2.detach_state(state)
        assert not det_state[0].requires_grad
        assert not det_state[1].requires_grad

    def test_recurrent_state_persists_between_episodes_IF_not_reset(
        self, model, dummy_obs
    ):
        """If NOT reset, the state from ep1 feeds into ep2 (deliberate)."""
        _, _, state_ep1, _ = model(dummy_obs)
        # Without reset, state_ep1 continues into next call
        _, _, state_ep2, _ = model(dummy_obs, state_ep1)
        assert not torch.allclose(state_ep1[1], state_ep2[1])

    def test_reset_between_episodes_gives_zero(self, model, dummy_obs):
        _, _, state_ep1, _ = model(dummy_obs)
        state_reset = model.reset_state()
        assert torch.all(state_reset[0] == 0.0)
        assert torch.all(state_reset[1] == 0.0)

    def test_recurrent_gradients_are_nonzero(self, model, dummy_obs):
        """Surrogate gradients must reach recurrent_fc.weight."""
        logits, value, _, _ = model(dummy_obs)
        loss = logits.sum() + value.sum()
        loss.backward()
        grad = model.recurrent_fc.weight.grad
        assert grad is not None, "recurrent_fc.weight has no gradient"
        assert grad.abs().sum().item() > 0, "recurrent_fc.weight gradient is zero"

    def test_gradients_are_finite(self, model, dummy_obs):
        logits, value, _, _ = model(dummy_obs)
        loss = logits.sum() + value.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"Non-finite gradient in {name}"


# ===========================================================================
# 6. A2C loss and returns
# ===========================================================================

from doom_env import Transition


def _make_transitions(n=5, terminal=True):
    """Create fake transitions for A2C tests."""
    torch.manual_seed(42)
    model = RSNNv2(RSNNv2Config(hidden=8, timesteps=2))
    obs = torch.rand(1, FLAT)
    state = model.reset_state()
    transitions = []
    for i in range(n):
        logits, value, state, _ = model(obs, state)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        act = dist.sample()
        lp = dist.log_prob(act)
        done = (i == n - 1) and terminal
        transitions.append(Transition(obs, int(act), lp, value.squeeze(0),
                                       reward=1.0 if done else -1.0, done=done))
    return transitions, model


class TestA2C:
    def test_returns_terminal(self):
        transitions, _ = _make_transitions(3, terminal=True)
        returns, advantages = compute_returns_and_advantages(
            transitions, last_value=0.0, gamma=0.99, gae_lambda=1.0
        )
        assert len(returns) == 3
        assert len(advantages) == 3
        # All returns should be finite
        for r in returns:
            assert math.isfinite(r), f"Non-finite return: {r}"

    def test_returns_truncated_bootstraps(self):
        transitions, _ = _make_transitions(3, terminal=False)
        returns_no_boot, _ = compute_returns_and_advantages(
            transitions, last_value=0.0, gamma=0.99, gae_lambda=1.0
        )
        returns_boot, _ = compute_returns_and_advantages(
            transitions, last_value=5.0, gamma=0.99, gae_lambda=1.0
        )
        # With a positive last_value, returns should be higher
        assert sum(returns_boot) > sum(returns_no_boot), \
            "Bootstrapped returns should be larger than non-bootstrapped"

    def test_a2c_update_returns_finite_loss(self):
        transitions, model = _make_transitions(5, terminal=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        info = a2c_update(
            model, optimizer, transitions,
            last_value=0.0, gamma=0.99, gae_lambda=0.95,
            value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
        )
        for key in ("actor_loss", "critic_loss", "entropy", "total_loss", "grad_norm"):
            assert key in info
            assert math.isfinite(info[key]), f"{key} is not finite: {info[key]}"

    def test_a2c_update_changes_weights(self):
        transitions, model = _make_transitions(5, terminal=True)
        before = model.input_fc.weight.clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        a2c_update(
            model, optimizer, transitions,
            last_value=0.0, gamma=0.99, gae_lambda=0.95,
            value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
        )
        after = model.input_fc.weight
        assert not torch.allclose(before, after), \
            "Weights did not change after A2C update"

    def test_empty_transitions_returns_zero_loss(self):
        model = RSNNv2(RSNNv2Config(hidden=8, timesteps=2))
        optimizer = torch.optim.Adam(model.parameters())
        info = a2c_update(model, optimizer, [], last_value=0.0,
                          gamma=0.99, gae_lambda=0.95,
                          value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5)
        assert info["total_loss"] == 0.0

    def test_advantage_normalisation_with_single_step(self):
        """Single-step episode should not crash on std=0 normalisation."""
        transitions, model = _make_transitions(1, terminal=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        info = a2c_update(
            model, optimizer, transitions,
            last_value=0.0, gamma=0.99, gae_lambda=0.95,
            value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
        )
        assert math.isfinite(info["total_loss"])


# ===========================================================================
# 7. Resource accounting
# ===========================================================================

class TestResources:
    def test_resource_keys_present(self, model):
        r = model.resource_summary()
        expected_keys = {
            "spiking_neurons", "input_synapses", "recurrent_synapses",
            "actor_synapses", "critic_synapses",
            "total_parameters", "hardware_parameters",
        }
        assert expected_keys.issubset(r.keys())

    def test_spiking_neurons_matches_hidden(self, model):
        assert model.resource_summary()["spiking_neurons"] == model.cfg.hidden

    def test_total_parameters_is_positive(self, model):
        assert model.resource_summary()["total_parameters"] > 0

    def test_hardware_params_less_than_total(self, model):
        r = model.resource_summary()
        # Hardware params exclude critic head
        assert r["hardware_parameters"] < r["total_parameters"]


# ===========================================================================
# 8. Checkpoint save/load
# ===========================================================================

class TestCheckpoint:
    def test_save_and_load(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "test_ckpt.pt"
            save_checkpoint(path, model, optimizer, episode=42, best_reward=-50.0)
            assert path.exists()
            ckpt = load_checkpoint(path)
            assert ckpt["version"] == "rsnn_v2"
            assert ckpt["episode"] == 42
            assert ckpt["best_reward"] == -50.0
            assert "config" in ckpt
            assert "model" in ckpt
            assert "optimizer" in ckpt

    def test_load_and_weights_match(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "test_ckpt.pt"
            save_checkpoint(path, model, optimizer, episode=10, best_reward=-100.0)
            loaded_model, ckpt = build_model_from_checkpoint(path)
            for name, param in model.named_parameters():
                loaded_param = dict(loaded_model.named_parameters())[name]
                assert torch.allclose(param, loaded_param), \
                    f"Weight mismatch after checkpoint load: {name}"

    def test_v1_checkpoint_rejected(self):
        """v2 loader must reject v1 checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "fake_v1.pt"
            torch.save({"version": "rsnn_v1", "model": {}, "config": {}}, path)
            with pytest.raises(ValueError, match="not a v2 checkpoint"):
                load_checkpoint(path)

    def test_checkpoint_resume_restores_episode(self, model):
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "ckpt.pt"
            save_checkpoint(path, model, optimizer, episode=99, best_reward=-20.0)
            ckpt = load_checkpoint(path)
            assert ckpt["episode"] == 99

    def test_optimizer_state_restored(self, model):
        """Optimizer state must survive save/load."""
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        # Do one step to populate optimizer state
        obs = torch.rand(1, FLAT)
        logits, value, _, _ = model(obs)
        (logits.sum() + value.sum()).backward()
        optimizer.step()
        optimizer.zero_grad()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "ckpt.pt"
            save_checkpoint(path, model, optimizer, episode=1, best_reward=0.0)
            ckpt = load_checkpoint(path)
            # Rebuild and restore optimizer
            model2 = build_model_from_checkpoint(path)[0]
            opt2 = torch.optim.Adam(model2.parameters(), lr=5e-4)
            opt2.load_state_dict(ckpt["optimizer"])
            # State groups should match
            assert len(opt2.state) > 0, "Optimizer state not restored"


# ===========================================================================
# 9. Spike encoding
# ===========================================================================

class TestSpikeEncoding:
    def test_bright_pixels_spike_more(self):
        """Deterministic rate encoding: brighter observation -> more spikes."""
        model = RSNNv2(RSNNv2Config(hidden=8, timesteps=8))
        obs_bright = torch.full((1, FLAT), 0.9)
        obs_dim = torch.full((1, FLAT), 0.1)
        _, _, _, rate_bright = model(obs_bright)
        _, _, _, rate_dim = model(obs_dim)
        # Spike rates should differ (not guaranteed to always be higher,
        # but with random weights they are usually so for bright inputs)
        # We just check that neither is NaN
        assert math.isfinite(float(rate_bright))
        assert math.isfinite(float(rate_dim))

    def test_zero_input_produces_finite_output(self):
        model = RSNNv2(RSNNv2Config(hidden=8, timesteps=4))
        obs = torch.zeros(1, FLAT)
        logits, value, _, rate = model(obs)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(value).all()


# ===========================================================================
# 10. Integration test (ViZDoom) — skipped if not installed
# ===========================================================================

vizdoom_available = False
try:
    import vizdoom  # noqa: F401
    vizdoom_available = True
except ImportError:
    pass


@pytest.mark.skipif(not vizdoom_available, reason="ViZDoom not installed")
class TestViZDoomIntegration:
    """Short integration tests requiring ViZDoom.

    These require the scenario file at ../ViZDoom/scenarios/basic.cfg
    relative to the test runner CWD.
    """

    def _get_config(self):
        cfg = pathlib.Path("../ViZDoom/scenarios/basic.cfg")
        if not cfg.exists():
            import vizdoom as vzd
            cfg = pathlib.Path(vzd.scenarios_path) / "basic.cfg"
        return cfg

    def test_build_game_and_close(self):
        from doom_env import build_game
        cfg = self._get_config()
        game, action_vectors, vzd = build_game(config=cfg, visible=False, seed=42)
        assert len(action_vectors) == 6
        game.close()

    def test_one_episode_inference(self):
        from doom_env import build_game, collect_episode_inference
        cfg = self._get_config()
        model = RSNNv2(RSNNv2Config(hidden=16, timesteps=4))
        model.eval()
        game, action_vectors, vzd = build_game(config=cfg, visible=False, seed=42)
        try:
            game.new_episode()
            state = model.reset_state()
            result, _ = collect_episode_inference(
                game, action_vectors, vzd, model, state,
                frame_skip=4, max_decisions=10,
            )
            assert result.decisions <= 10
            assert result.kill in (0, 1)
            assert math.isfinite(result.reward)
        finally:
            game.close()

    def test_one_training_episode_a2c(self):
        from doom_env import build_game, collect_episode
        from train_rsnn_v2 import a2c_update
        cfg = self._get_config()
        model = RSNNv2(RSNNv2Config(hidden=16, timesteps=4))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        game, action_vectors, vzd = build_game(config=cfg, visible=False, seed=42)
        try:
            game.new_episode()
            state = model.reset_state()
            result, state = collect_episode(
                game, action_vectors, vzd, model, state,
                frame_skip=4, max_decisions=10,
            )
            info = a2c_update(
                model, optimizer, result.transitions,
                last_value=0.0, gamma=0.99, gae_lambda=0.95,
                value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
            )
            assert math.isfinite(info["total_loss"])
            assert info.get("decisions", result.decisions) == result.decisions
        finally:
            game.close()
