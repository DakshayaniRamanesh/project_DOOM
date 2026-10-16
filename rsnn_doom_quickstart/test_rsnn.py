"""Tests runnable WITHOUT ViZDoom installed: pytest -q test_rsnn.py"""
import numpy as np
from PIL import Image
import torch
from rsnn_doom import RSNN, frame_to_tensor, episode, policy_update


def test_preprocessing_chw_hwc_and_pil():
    arr = np.full((3, 120, 160), 160, np.uint8)
    assert torch.allclose(frame_to_tensor(arr), frame_to_tensor(arr.transpose(1, 2, 0)))
    assert torch.allclose(frame_to_tensor(arr), frame_to_tensor(Image.fromarray(arr.transpose(1, 2, 0))))
    assert frame_to_tensor(arr).shape == (1, 768)


def test_rsnn_shapes_and_memory():
    torch.manual_seed(42)
    model = RSNN(hidden=96, beta=0.5, timesteps=8)
    x = frame_to_tensor(np.full((3, 120, 160), 200, np.uint8))
    scores1, state1, rate = model(x)
    scores2, state2, _ = model(x, state1)
    assert scores1.shape == scores2.shape == (1, 6)
    assert 0 <= rate <= 1
    assert state1[0].shape == state1[1].shape == (1, 96)
    assert not torch.allclose(state1[1], state2[1])
    assert torch.allclose(model.reset_state()[0], torch.zeros(1, 96))


def test_surrogate_gradient_reaches_recurrent_connections():
    torch.manual_seed(42)
    model = RSNN(hidden=96, beta=0.5, timesteps=8)
    obs = frame_to_tensor(np.full((3, 120, 160), 200, np.uint8))
    scores, _, _ = model(obs)
    torch.nn.functional.cross_entropy(scores, torch.tensor([3])).backward()
    assert model.recurrent_fc.weight.grad is not None
    assert model.recurrent_fc.weight.grad.abs().sum() > 0


class FakeState:
    def __init__(self):
        self.screen_buffer = np.full((3, 120, 160), 200, np.uint8)


class FakeVZD:
    class GameVariable:
        KILLCOUNT = 'kills'


class FakeGame:
    def __init__(self):
        self.steps = 0

    def new_episode(self):
        self.steps = 0

    def is_episode_finished(self):
        return self.steps >= 3

    def get_state(self):
        return FakeState()

    def make_action(self, buttons, frame_skip):
        assert len(buttons) == 3 and frame_skip == 4
        self.steps += 1
        return 10 if self.steps == 3 else -1

    def get_game_variable(self, name):
        return 1 if self.steps == 3 else 0


class Args:
    frame_skip = 4
    max_decisions = 0


def test_mock_doom_inference_and_gradient_update():
    torch.manual_seed(42)
    model = RSNN(hidden=16, beta=0.5, timesteps=3)
    actions = [[0, 0, 0]] * 6
    record, rewards, logs = episode(FakeGame(), actions, FakeVZD(), model, Args(), False)
    assert record['kill'] == 1 and record['decisions'] == 3
    assert abs(record['reward'] - 8) < 1e-6 and not logs
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    record, rewards, logs = episode(FakeGame(), actions, FakeVZD(), model, Args(), True)
    assert len(logs) == 3
    loss = policy_update(model, optim, rewards, logs, 0.99)
    assert np.isfinite(loss)
