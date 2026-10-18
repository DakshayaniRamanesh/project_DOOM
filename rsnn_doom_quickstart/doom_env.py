"""ViZDoom BASIC environment and n-step RSNN rollout helpers.

The environment can end through success/death, engine time limit, or a user
max-decisions cap. Rollout boundaries do not reset the episode or RSNN state.
The environment timeout ends the playable episode: no next observation is
available for bootstrapping, so the last value is zero in that case.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

WIDTH, HEIGHT = 32, 24
FLAT = WIDTH * HEIGHT
ACTION_NAMES = ("noop", "left", "right", "shoot", "left_shoot", "right_shoot")


def frame_to_tensor(frame, width: int = WIDTH, height: int = HEIGHT) -> torch.Tensor:
    """Convert ViZDoom CHW/HWC or a PIL image to [1, H*W] float [0,1]."""
    if isinstance(frame, Image.Image):
        img = frame
    else:
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    small = img.convert("L").resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(arr.reshape(1, -1))


def create_actions(game, vzd):
    """Create six actions using ViZDoom's configured button ordering."""
    buttons = list(game.get_available_buttons())
    needed = {vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK}
    if len(buttons) != 3 or set(buttons) != needed:
        raise RuntimeError(f"Expected BASIC MOVE_LEFT, MOVE_RIGHT, ATTACK; got {buttons}")
    sets = [set(), {vzd.Button.MOVE_LEFT}, {vzd.Button.MOVE_RIGHT},
            {vzd.Button.ATTACK}, {vzd.Button.MOVE_LEFT, vzd.Button.ATTACK},
            {vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK}]
    return [[int(btn in active) for btn in buttons] for active in sets]


def build_game(config: Optional[Path] = None, visible: bool = False, seed: int = 42):
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError("ViZDoom is required for gameplay: pip install vizdoom") from exc
    cfg = (Path(vzd.scenarios_path) / "basic.cfg") if config is None else Path(config)
    cfg = cfg.expanduser().resolve()
    if not cfg.is_file():
        raise FileNotFoundError(f"ViZDoom scenario not found: {cfg}")
    game = vzd.DoomGame()
    game.load_config(str(cfg))
    game.set_window_visible(visible)
    game.set_seed(seed)
    actions = create_actions(game, vzd)
    game.init()
    return game, actions, vzd


@dataclass
class Transition:
    observation: torch.Tensor
    action_idx: int
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool   # True only for a true task terminal; timeouts have done=False
    entropy: Optional[torch.Tensor] = None
    truncated: bool = False


@dataclass
class EpisodeResult:
    reward: float
    kill: int
    decisions: int
    mean_spike_rate: float
    terminated: bool
    action_counts: dict
    transitions: list[Transition] = field(default_factory=list)
    truncated: bool = False


@dataclass
class RolloutResult:
    transitions: list[Transition]
    state: Tuple[torch.Tensor, torch.Tensor]
    next_observation: Optional[torch.Tensor]
    ended: bool
    terminated: bool
    truncated: bool
    timeout: bool
    reward: float
    spike_sum: float
    action_counts: dict


def collect_rollout(game, action_vectors, vzd, model, state, *,
                    frame_skip: int = 4, rollout_steps: int = 8,
                    remaining_decisions: int = 0) -> RolloutResult:
    """Collect <=rollout_steps decisions with gradients across that segment.

    remaining_decisions=0 means unlimited. Graphs are deliberately kept ONLY
    for this segment. Detach the returned state after the optimizer step.
    """
    if rollout_steps < 1 or frame_skip < 1:
        raise ValueError("rollout_steps and frame_skip must be positive")
    if remaining_decisions < 0:
        raise ValueError("remaining_decisions must be >= 0")
    model.train()
    transitions: list[Transition] = []
    counts = {name: 0 for name in ACTION_NAMES}
    total_reward = 0.0
    spike_sum = 0.0

    for _ in range(rollout_steps):
        if game.is_episode_finished():
            break
        game_state = game.get_state()
        if game_state is None:
            break
        obs = frame_to_tensor(game_state.screen_buffer)
        logits, value, state, spike_rate = model(obs, state)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        sampled = dist.sample()
        idx = int(sampled.item())
        reward = float(game.make_action(action_vectors[idx], frame_skip))
        finished = bool(game.is_episode_finished())
        timed_out = bool(game.is_episode_timeout_reached()) if finished else False
        true_terminal = finished and not timed_out
        transitions.append(Transition(obs, idx, dist.log_prob(sampled), value.squeeze(0),
                                      reward, true_terminal, dist.entropy(), timed_out))
        counts[ACTION_NAMES[idx]] += 1
        total_reward += reward
        spike_sum += float(spike_rate.detach())
        if finished:
            break
        if remaining_decisions > 0 and len(transitions) >= remaining_decisions:
            break

    finished = bool(game.is_episode_finished())
    timed_out = bool(game.is_episode_timeout_reached()) if finished else False
    cap_reached = remaining_decisions > 0 and len(transitions) >= remaining_decisions
    ended = finished or cap_reached
    truncated = timed_out or (cap_reached and not finished)
    terminal = finished and not timed_out
    if transitions and cap_reached and not finished:
        transitions[-1].truncated = True

    # A game that has finished (including internal timeout) offers no legal next
    # observation. For the finite-horizon BASIC episode, bootstrap value = 0.
    # At an unfinished rollout boundary/cap, bootstrap from the real next frame.
    gs_next = game.get_state() if not finished else None
    next_obs = frame_to_tensor(gs_next.screen_buffer) if gs_next is not None else None
    return RolloutResult(transitions, state, next_obs, ended, terminal, truncated,
                         timed_out, total_reward, spike_sum, counts)


def collect_episode_inference(game, action_vectors, vzd, model, state,
                              frame_skip: int = 4, max_decisions: int = 0,
                              greedy: bool = True) -> Tuple[EpisodeResult, Tuple]:
    """No-gradient inference. Optional policy.select_action(logits) overrides argmax.

    RandomPolicy implements select_action() to sample uniformly at evaluation;
    learned policies keep greedy argmax unless greedy=False is requested.
    """
    model.eval()
    counts = {n: 0 for n in ACTION_NAMES}
    total_reward = 0.0
    spike_sum = 0.0
    decisions = 0
    with torch.no_grad():
        while not game.is_episode_finished():
            gs = game.get_state()
            if gs is None:
                break
            obs = frame_to_tensor(gs.screen_buffer)
            logits, _, state, spike_rate = model(obs, state)
            if hasattr(model, "select_action"):
                idx = int(model.select_action(logits))
            elif greedy:
                idx = int(logits.squeeze(0).argmax().item())
            else:
                idx = int(torch.distributions.Categorical(logits=logits.squeeze(0)).sample().item())
            if not 0 <= idx < len(ACTION_NAMES):
                raise ValueError(f"Policy returned invalid action index {idx}")
            reward = float(game.make_action(action_vectors[idx], frame_skip))
            total_reward += reward
            spike_sum += float(spike_rate)
            counts[ACTION_NAMES[idx]] += 1
            decisions += 1
            if max_decisions > 0 and decisions >= max_decisions:
                break
    finished = bool(game.is_episode_finished())
    timeout = bool(game.is_episode_timeout_reached()) if finished else False
    kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    result = EpisodeResult(total_reward, int(kills > 0), decisions,
                           spike_sum / decisions if decisions else 0.0,
                           finished and not timeout, counts,
                           truncated=timeout or (max_decisions > 0 and decisions >= max_decisions and not finished))
    return result, state
