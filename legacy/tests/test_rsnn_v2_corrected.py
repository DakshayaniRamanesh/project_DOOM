"""Offline tests: no ViZDoom package required. Run python -m pytest -q tests/"""
from __future__ import annotations

import math
import random
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from doom_env import (ACTION_NAMES, Transition, collect_rollout,
                      collect_episode_inference, frame_to_tensor)
from evaluate_rsnn_v2 import RandomPolicy, FixedActionPolicy
from rsnn_v2 import (RSNNv2, RSNNv2Config, save_checkpoint, load_checkpoint,
                     restore_rng_from_checkpoint, build_model_from_checkpoint)
from train_rsnn_v2 import (a2c_update, bootstrap_value,
                           compute_returns_and_advantages)


@pytest.fixture
def net():
    torch.manual_seed(7)
    return RSNNv2(RSNNv2Config(hidden=8, timesteps=2))


@pytest.fixture
def six_actions():
    return [[1 if i == j else 0 for j in range(6)] for i in range(6)]


class DummyVZD:
    class GameVariable:
        KILLCOUNT = 1


class FakeGame:
    def __init__(self, length=6, timeout=False):
        self.length = length
        self.timeout = timeout
        self.t = 0
        self.count = 0
        self.last_actions = []

    def is_episode_finished(self):
        return self.t >= self.length

    def is_episode_timeout_reached(self):
        return self.is_episode_finished() and self.timeout

    def get_state(self):
        if self.is_episode_finished():
            return None
        return SimpleNamespace(screen_buffer=np.full((3, 32, 32), 90 + self.t, np.uint8))

    def make_action(self, action, n_tics):
        self.last_actions.append(action)
        self.t += 1
        return 10.0 if self.is_episode_finished() and not self.timeout else -1.0

    def get_game_variable(self, key):
        return int(self.is_episode_finished() and not self.timeout)


def test_input_preprocessing():
    x = frame_to_tensor(np.ones((3, 100, 100), dtype=np.uint8) * 128)
    assert x.shape == (1, 768)
    assert x.dtype == torch.float32
    assert torch.allclose(x, torch.full_like(x, 128 / 255))


@pytest.mark.parametrize('beta', [0.25, 0.5, 0.75])
def test_betas(beta):
    assert RSNNv2Config(beta=beta).beta == beta


def test_invalid_beta():
    with pytest.raises(ValueError):
        RSNNv2Config(beta=0.4)


def test_resources_and_shapes(net):
    assert net.resource_summary()['spiking_neurons'] == 8
    normal = RSNNv2()
    r = normal.resource_summary()
    assert r['hardware_parameters'] == 83622
    assert r['total_parameters'] == 83719
    x = torch.rand(1, 768)
    logits, val, state, spike_rate = net(x)
    assert logits.shape == (1, 6)
    assert val.shape == (1,)
    assert state[0].shape == state[1].shape == (1, 8)
    assert float(spike_rate) >= 0
    assert ((state[0] == 0) | (state[0] == 1)).all()


def test_state_detachment(net):
    x = torch.rand(1, 768)
    _, _, state, _ = net(x)
    assert state[0].grad_fn is not None
    detached = net.detach_state(state)
    assert detached[0].grad_fn is None and detached[1].grad_fn is None
    assert not detached[0].requires_grad
    clean = net.reset_state()
    assert torch.equal(clean[0], torch.zeros_like(clean[0]))


def test_gae_terminal_math():
    zeros = torch.tensor(0.)
    steps = [Transition(None, 0, zeros, zeros, 1., False),
             Transition(None, 0, zeros, zeros, 2., True)]
    targets, adv = compute_returns_and_advantages(steps, 777., 1., 1.)
    assert targets == pytest.approx([3., 2.])
    assert adv == pytest.approx([3., 2.])


def test_gae_bootstrap_math():
    zero = torch.tensor(0.)
    steps = [Transition(None, 0, zero, zero, 1., False),
             Transition(None, 0, zero, zero, 2., False)]
    targets, adv = compute_returns_and_advantages(steps, 3., 0.9, 1.)
    assert targets == pytest.approx([1 + 0.9 * (2 + 0.9 * 3), 2 + 0.9 * 3])


def test_gae_lambda_zero():
    zero = torch.tensor(0.)
    steps = [Transition(None, 0, zero, zero, 1., False),
             Transition(None, 0, zero, zero, 2., False)]
    _, adv = compute_returns_and_advantages(steps, 3., 0.9, 0.)
    assert adv == pytest.approx([1., 4.7])


def test_gradient_and_true_entropy(net):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    state = net.reset_state()
    trs = []
    for t in range(3):
        logits, value, state, _ = net(torch.rand(1, 768), state)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        trs.append(Transition(None, int(action), dist.log_prob(action), value.squeeze(0),
                              1.0 + t, t == 2, dist.entropy()))
    stats = a2c_update(net, opt, trs, 0., max_grad_norm=.5)
    assert np.isfinite(list(stats.values())).all()
    assert 0 <= stats['entropy'] <= np.log(6) + 1e-4
    assert net.recurrent_fc.weight.grad is not None
    assert net.recurrent_fc.weight.grad.abs().sum().item() > 0
    assert torch.nn.utils.clip_grad_norm_(net.parameters(), .5) <= 0.5001


def test_entropy_is_differentiable():
    logits = torch.tensor([1., 0., -1.], requires_grad=True)
    loss = -torch.distributions.Categorical(logits=logits).entropy()
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_missing_entropy_rejected(net):
    with pytest.raises(ValueError, match='entropy'):
        a2c_update(net, torch.optim.Adam(net.parameters()),
                   [Transition(None, 0, torch.tensor(0.), torch.tensor(0.), 0., True)], 0.)


def test_random_action_diversity():
    policy = RandomPolicy()
    policy.reseed(43)
    actions = [policy.select_action(None) for _ in range(600)]
    assert set(actions) == set(range(6))
    assert 50 < actions.count(0) < 150
    policy.reseed(43)
    assert actions[:10] == [policy.select_action(None) for _ in range(10)]


def test_fixed_action():
    p = FixedActionPolicy('left_shoot')
    assert p(torch.rand(1, 768))[0].argmax().item() == 4


def test_inference_random_not_greedy(six_actions):
    game = FakeGame(length=120, timeout=True)
    policy = RandomPolicy()
    policy.reseed(43)
    result, _ = collect_episode_inference(game, six_actions, DummyVZD, policy,
                                          policy.reset_state())
    assert result.decisions == 120
    assert sum(v > 0 for v in result.action_counts.values()) == 6
    assert result.truncated and not result.terminated


def test_rollout_segment_and_next_state(net, six_actions):
    game = FakeGame(length=6)
    r = collect_rollout(game, six_actions, DummyVZD, net, net.reset_state(),
                        rollout_steps=3)
    assert not r.ended
    assert len(r.transitions) == 3
    assert r.next_observation.shape == (1, 768)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    boot = bootstrap_value(net, r.next_observation, r.state)
    a2c_update(net, opt, r.transitions, boot)
    state = net.detach_state(r.state)
    r2 = collect_rollout(game, six_actions, DummyVZD, net, state, rollout_steps=3)
    assert r2.ended and r2.terminated and not r2.truncated
    assert r2.next_observation is None
    assert r2.transitions[-1].done


def test_time_limit_truncation(net, six_actions):
    game = FakeGame(length=2, timeout=True)
    r = collect_rollout(game, six_actions, DummyVZD, net, net.reset_state(), rollout_steps=5)
    assert r.ended and r.timeout and r.truncated and not r.terminated
    assert not r.transitions[-1].done
    assert r.next_observation is None
    assert bootstrap_value(net, r.next_observation, r.state) == 0


def test_manual_decision_cap_bootstrap(net, six_actions):
    game = FakeGame(length=20)
    r = collect_rollout(game, six_actions, DummyVZD, net, net.reset_state(),
                        rollout_steps=8, remaining_decisions=2)
    assert len(r.transitions) == 2
    assert r.ended and r.truncated and not r.terminated
    assert r.next_observation is not None
    assert np.isfinite(bootstrap_value(net, r.next_observation, r.state))


def test_checkpoint_roundtrip(tmp_path, net):
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    p = tmp_path / 'v2.pt'
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    save_checkpoint(p, net, opt, episode=100, best_reward=3.0,
                    extra={'best_val_success': .75})
    rng_a = (random.random(), float(np.random.random()), float(torch.rand(1)))
    ckpt = load_checkpoint(p)
    restore_rng_from_checkpoint(ckpt)
    rng_b = (random.random(), float(np.random.random()), float(torch.rand(1)))
    assert rng_a == rng_b
    new_model, loaded = build_model_from_checkpoint(p)
    assert loaded['episode'] == 100
    assert new_model.cfg.hidden == 8
    assert all(torch.equal(a, b) for a, b in zip(net.parameters(), new_model.parameters()))


def test_mocked_training_resume_and_validation(tmp_path, monkeypatch):
    """Exercise actual trainer loop, segmented updates, validation and resume."""
    import os
    import train_rsnn_v2 as training

    def build_fake(config=None, visible=False, seed=42):
        class G(FakeGame):
            def new_episode(self):
                self.t = 0
            def close(self):
                pass
        return G(length=3), [[0] * 3 for _ in range(6)], DummyVZD

    monkeypatch.setattr(training, 'build_game', build_fake)
    monkeypatch.chdir(tmp_path)
    kwargs = dict(config=None, episodes=3, resume=None, visible=False, seed=9,
                  frame_skip=4, max_decisions=0, rollout_steps=2, hidden=8,
                  beta=.5, timesteps=2, lr=3e-4, gamma=.99, gae_lambda=.95,
                  value_coef=.5, entropy_coef=.01, max_grad_norm=.5,
                  reward_scale=1.0, critic_loss='mse', diagnose_grads=False,
                  normalize_advantages=False, save_every=1,
                  validate_every=2, val_episodes=2, val_seeds=[1101, 1102])
    training.train(SimpleNamespace(**kwargs))
    ckpt = load_checkpoint('checkpoints/v2/latest.pt')
    best = load_checkpoint('checkpoints/v2/best.pt')
    assert ckpt['episode'] == 3
    assert best['episode'] <= 3
    assert 0 <= ckpt['best_val_success'] <= 1
    kwargs.update(episodes=2, resume=tmp_path / 'checkpoints/v2/latest.pt')
    training.train(SimpleNamespace(**kwargs))
    assert load_checkpoint('checkpoints/v2/latest.pt')['episode'] == 5


def test_mocked_evaluation_recreates_game_for_each_policy(tmp_path, monkeypatch):
    import evaluate_rsnn_v2 as evaluator
    created = []

    def build_fake(config=None, visible=False, seed=42):
        class G(FakeGame):
            def new_episode(self):
                self.t = 0
            def close(self):
                pass
        created.append(seed)
        return G(length=16), [[0] * 3 for _ in range(6)], DummyVZD

    monkeypatch.setattr(evaluator, 'build_game', build_fake)
    monkeypatch.chdir(tmp_path)
    m = RSNNv2(RSNNv2Config(hidden=8, timesteps=2))
    save_checkpoint('ckpt.pt', m, torch.optim.Adam(m.parameters()), 2, 0.0)
    args = SimpleNamespace(config=None, v1_checkpoint=None,
                           v2_checkpoint=tmp_path / 'ckpt.pt', compare_checkpoint=None,
                           episodes=2, seed=43, seeds=[43, 44], frame_skip=4, visible=False,
                           eval_mode='greedy', stochastic=False, log_decisions=False, max_logged_episodes=0)
    evaluator.run_evaluation(args)
    # random, shoot, left_shoot, rsnn_v2_best; each should get its own seeded game.
    assert created == [43] * 4 + [44] * 4
    csv_path = next((tmp_path / 'results/v2').glob('eval_*.csv'))
    import csv
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * 2 * 2
    random_rows = [r for r in rows if r['policy'] == 'random']
    counts = [sum(int(r[f'action_{name}']) for r in random_rows) for name in ACTION_NAMES]
    assert len([c for c in counts if c > 0]) >= 5


# ===========================================================================
# Specific Regression Tests (Priority 9 Requirements)
# ===========================================================================

def test_probability_distributions_sum_to_one(net):
    """1. Probability distributions sum to 1.0 across diverse observations."""
    for _ in range(5):
        obs = torch.rand(1, 768)
        logits, _, _, _ = net(obs)
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_greedy_and_stochastic_action_selection(net, six_actions):
    """2. Greedy selects argmax; stochastic samples according to probabilities."""
    game = FakeGame(length=5)
    # Greedy inference
    res_greedy, _ = collect_episode_inference(game, six_actions, DummyVZD, net, net.reset_state(), greedy=True)
    assert res_greedy.decisions == 5
    # Stochastic inference with fixed seeds gives reproducible outcomes
    game2 = FakeGame(length=5)
    torch.manual_seed(42)
    res_stoch1, _ = collect_episode_inference(game2, six_actions, DummyVZD, net, net.reset_state(), greedy=False)
    game3 = FakeGame(length=5)
    torch.manual_seed(42)
    res_stoch2, _ = collect_episode_inference(game3, six_actions, DummyVZD, net, net.reset_state(), greedy=False)
    assert res_stoch1.action_counts == res_stoch2.action_counts


def test_action_logits_and_probabilities_logged(net, six_actions):
    """3. Action logits, probabilities, entropy, and margins are logged correctly."""
    game = FakeGame(length=3)
    res, _ = collect_episode_inference(game, six_actions, DummyVZD, net, net.reset_state(),
                                       greedy=True, record_decisions=True)
    assert len(res.mean_probabilities) == 6
    assert abs(sum(res.mean_probabilities.values()) - 1.0) < 1e-4
    assert res.mean_entropy > 0.0
    assert len(res.decision_records) == 3
    for d in res.decision_records:
        assert "entropy" in d and "margin" in d and "reward" in d
        for n in ACTION_NAMES:
            assert f"logit_{n}" in d and f"prob_{n}" in d


def test_policy_entropy_from_all_six_actions(net):
    """4. Policy entropy is calculated from all six actions using Categorical."""
    logits = torch.tensor([[1.0, 2.0, 0.5, -0.5, 0.0, 1.5]])
    probs = torch.softmax(logits, dim=-1)
    manual_entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
    cat_entropy = torch.distributions.Categorical(probs=probs).entropy().item()
    assert abs(manual_entropy - cat_entropy) < 1e-5


def test_actor_and_critic_gradients_finite(net):
    """5. Actor and critic gradients are finite on all shared and head parameters."""
    obs = torch.rand(1, 768)
    logits, val, _, _ = net(obs)
    dist = torch.distributions.Categorical(logits=logits.squeeze(0))
    act = dist.sample()
    loss = -(dist.log_prob(act) * 2.0) + 0.5 * val.square() - 0.01 * dist.entropy()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient"


def test_gradient_clipping_strictly_enforced(net):
    """6. Gradient clipping caps the norm strictly at max_grad_norm."""
    obs = torch.rand(1, 768)
    logits, val, _, _ = net(obs)
    # Artificially create huge loss to induce large gradients
    loss = (logits.sum() + val.sum()) * 1000.0
    loss.backward()
    max_norm = 0.5
    preclip = nn.utils.clip_grad_norm_(net.parameters(), max_norm)
    assert preclip > max_norm
    postclip = torch.sqrt(sum(p.grad.norm()**2 for p in net.parameters() if p.grad is not None)).item()
    assert postclip <= max_norm + 1e-5


def test_reward_scaling_consistency(net):
    """7. Reward scaling scales GAE returns and advantages consistently without bias."""
    st = net.reset_state()
    trs = []
    for i in range(3):
        obs = torch.rand(1, 768)
        logits, val, st, _ = net(obs, st)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        act = dist.sample()
        trs.append(Transition(obs, int(act.item()), dist.log_prob(act),
                              val.squeeze(0), 10.0 if i == 2 else -1.0,
                              done=(i == 2), entropy=dist.entropy()))

    scale = 0.01
    ret_unscaled, adv_unscaled = compute_returns_and_advantages(trs, 0.0, 0.99, 0.95, reward_scale=1.0)
    ret_scaled, adv_scaled = compute_returns_and_advantages(trs, 0.0, 0.99, 0.95, reward_scale=scale)

    # Advantages with value=0 baseline scale linearly with reward_scale
    # Verify that returns scale consistent with reward scale
    for r_u, r_s in zip(ret_unscaled, ret_scaled):
        assert abs(r_u * scale - r_s) < 1.0  # Accounts for non-scaled value baseline term in GAE


def test_critic_huber_and_mse_options(net):
    """8. Critic Huber and MSE options produce valid, finite losses and stats."""
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    def make_trs():
        st = net.reset_state()
        trs = []
        for i in range(3):
            obs = torch.rand(1, 768)
            logits, val, st, _ = net(obs, st)
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            act = dist.sample()
            trs.append(Transition(obs, int(act.item()), dist.log_prob(act),
                                  val.squeeze(0), 100.0, done=(i == 2), entropy=dist.entropy()))
        return trs

    info_mse = a2c_update(net, opt, make_trs(), 0.0, critic_loss_type="mse")
    assert math.isfinite(info_mse["critic_loss"]) and info_mse["critic_loss"] > 0

    info_huber = a2c_update(net, opt, make_trs(), 0.0, critic_loss_type="huber")
    assert math.isfinite(info_huber["critic_loss"]) and info_huber["critic_loss"] > 0
    # Huber loss is smaller than MSE on large target errors (100^2 vs ~100)
    assert info_huber["critic_loss"] < info_mse["critic_loss"]


def test_recurrent_state_persistence_and_reset(net):
    """10. Recurrent state evolves across decisions and resets to zeros."""
    obs = torch.rand(1, 768)
    s0 = net.reset_state()
    assert (s0[0] == 0).all() and (s0[1] == 0).all()
    _, _, s1, _ = net(obs, s0)
    assert not torch.allclose(s0[1], s1[1])
    # Resetting returns clean zeros
    s_reset = net.reset_state()
    assert (s_reset[0] == 0).all() and (s_reset[1] == 0).all()


def test_greedy_noop_checkpoint_has_nontrivial_distribution():
    """12. A checkpoint that chooses noop greedily can still produce a nontrivial probability distribution."""
    import pathlib
    ckpt_path = pathlib.Path("checkpoints/v2/latest_ep10_backup.pt")
    if not ckpt_path.exists():
        pytest.skip("latest_ep10_backup.pt not present for test")
    model, _ = build_model_from_checkpoint(ckpt_path)
    model.eval()
    obs = torch.full((1, 768), 0.5)
    with torch.no_grad():
        logits, _, _, _ = model(obs)
    probs = torch.softmax(logits.squeeze(0), dim=-1)
    entropy = torch.distributions.Categorical(probs=probs).entropy().item()
    top_idx = logits.squeeze(0).argmax().item()
    # While top action is noop (index 0)
    assert top_idx == 0
    # The probability is NOT 1.0 (it's between 0.3 and 0.5), entropy is substantial (> 1.4)
    assert probs[0].item() < 0.60
    assert entropy > 1.40
    # Other actions (like shoot or right_shoot) have non-zero probability
    assert probs[5].item() > 0.15  # right_shoot


def test_random_baseline_sampling_uniformity():
    """13. Random baseline action sampling remains truly uniform."""
    policy = RandomPolicy(seed=123)
    counts = Counter()
    for _ in range(3000):
        idx = policy.select_action(None)
        counts[idx] += 1
    # Each action should be chosen ~500 times out of 3000 (+- 100)
    for i in range(6):
        assert 400 < counts[i] < 600
