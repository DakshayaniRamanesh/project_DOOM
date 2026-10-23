"""Minimal RSNN + ViZDoom BASIC: proof of concept, NOT a trained DOOM agent.

Dependencies: pip install torch pillow numpy vizdoom

Offline (works without ViZDoom installed):
    python rsnn_doom.py --mode smoke --image ../doom_frame.png
ViZDoom inference (untrained unless --checkpoint provided):
    python rsnn_doom.py --mode demo --episodes 3 --visible
Experimental REINFORCE training:
    python rsnn_doom.py --mode train --episodes 500
Evaluate a trained checkpoint:
    python rsnn_doom.py --mode evaluate --episodes 50 --checkpoint checkpoints/rsnn.pt

Notes:
- Uses pure PyTorch (no snnTorch required); explicitly models recurrent LIF spikes.
- Beta=0.5 (approved beta choices: 0.25, 0.5, 0.75).
- Surrogate gradients make spiking recurrent weights trainable.
- Training uses truncated-BPTT REINFORCE, a baseline experiment, not guaranteed convergence.
- Floating-point simulation only; hardware's 16-bit fixed-point constraints are NOT verified.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.distributions import Categorical

WIDTH, HEIGHT = 32, 24
ACTION_NAMES = ("noop", "left", "right", "shoot", "left_shoot", "right_shoot")


class Spike(torch.autograd.Function):
    """Binary spike, with a fast-sigmoid surrogate derivative in backward."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output / (1.0 + 2.0 * x.abs()).square()


def frame_to_tensor(frame, width=WIDTH, height=HEIGHT):
    """Accept ViZDoom CHW / HWC / grayscale, or PIL image. Return [1, H*W] float."""
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
    small = img.convert('L').resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(arr.reshape(1, -1))


class RSNN(nn.Module):
    """Rate-encoded pixels -> recurrent LIF layer -> analog action-score readout.

    This is an RSNN because previous *output spikes* are fed back via a trainable
    recurrent matrix. Membrane potential also persists across game decisions.
    Analog action readout is for software experimentation; FPGA interface TBD.
    """
    def __init__(self, inputs=WIDTH * HEIGHT, hidden=96, actions=len(ACTION_NAMES),
                 beta=0.5, threshold=0.4, timesteps=8):
        super().__init__()
        if beta not in (0.25, 0.5, 0.75):
            raise ValueError('beta must be one of 0.25, 0.5, 0.75')
        if timesteps < 1:
            raise ValueError('timesteps must be positive')
        self.inputs, self.hidden, self.actions = inputs, hidden, actions
        self.beta, self.threshold, self.timesteps = beta, threshold, timesteps
        self.input_fc = nn.Linear(inputs, hidden)
        self.recurrent_fc = nn.Linear(hidden, hidden, bias=False)
        self.output_fc = nn.Linear(hidden, actions)
        nn.init.normal_(self.recurrent_fc.weight, mean=0.0, std=0.06)

    def reset_state(self, batch_size=1, device=None):
        device = device or self.input_fc.weight.device
        z = torch.zeros(batch_size, self.hidden, device=device)
        return (z.clone(), z.clone())  # (previous hidden spikes, membrane)

    def forward(self, observation, state=None):
        """One DOOM decision; T internal neural timesteps. State is returned.

        Do NOT reset between decisions, only at new episodes. Call detach_state()
        between decisions when training to bound backpropagation memory.
        """
        if observation.ndim != 2 or observation.shape[1] != self.inputs:
            raise ValueError(f'expected observations [batch,{self.inputs}]')
        if state is None:
            state = self.reset_state(observation.shape[0], observation.device)
        spikes, mem = state
        pixel_accumulator = torch.zeros_like(observation)
        score_total = torch.zeros(observation.shape[0], self.actions,
                                  device=observation.device)
        total_spikes = torch.zeros((), device=observation.device)

        # Deterministic rate encoding: brighter pixels produce more spikes.
        for _ in range(self.timesteps):
            pixel_accumulator = pixel_accumulator + observation
            input_spikes = (pixel_accumulator >= 1.0).to(observation.dtype)
            pixel_accumulator = pixel_accumulator - input_spikes

            current = self.input_fc(input_spikes) + self.recurrent_fc(spikes)
            mem = self.beta * mem + current - self.threshold * spikes
            spikes = Spike.apply(mem - self.threshold)
            score_total = score_total + self.output_fc(spikes)
            total_spikes = total_spikes + spikes.detach().sum()
        rate = total_spikes / (self.timesteps * spikes.numel())
        return score_total / self.timesteps, (spikes, mem), rate

    @staticmethod
    def detach_state(state):
        return tuple(s.detach() for s in state)


def create_actions(game, vzd):
    buttons = list(game.get_available_buttons())
    needed = {vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK}
    if len(buttons) != 3 or set(buttons) != needed:
        raise RuntimeError(f'Expected BASIC buttons, got {[b.name for b in buttons]}')
    sets = [set(), {vzd.Button.MOVE_LEFT}, {vzd.Button.MOVE_RIGHT},
            {vzd.Button.ATTACK},
            {vzd.Button.MOVE_LEFT, vzd.Button.ATTACK},
            {vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK}]
    return [[int(button in enabled) for button in buttons] for enabled in sets]


def build_game(args):
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError('ViZDoom missing: install it with pip install vizdoom') from exc
    config = Path(args.config) if args.config else Path(vzd.scenarios_path) / 'basic.cfg'
    config = config.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f'Cannot find ViZDoom config: {config}')
    game = vzd.DoomGame()
    game.load_config(str(config))
    game.set_window_visible(args.visible)
    game.set_seed(args.seed)
    action_vectors = create_actions(game, vzd)
    game.init()
    return game, action_vectors, vzd


def episode(game, action_vectors, vzd, model, args, training=False):
    game.new_episode()
    memory = model.reset_state()
    log_probs = []
    rewards = []
    spike_rates = []
    decisions = 0
    while not game.is_episode_finished():
        state = game.get_state()
        if state is None:
            break
        observation = frame_to_tensor(state.screen_buffer)
        if training:
            logits, memory, rate = model(observation, memory)
            distribution = Categorical(logits=logits.squeeze(0))
            action_idx = distribution.sample()
            log_probs.append(distribution.log_prob(action_idx))
            idx = int(action_idx.item())
        else:
            with torch.no_grad():
                logits, memory, rate = model(observation, memory)
            idx = int(torch.argmax(logits, dim=-1).item())
        memory = model.detach_state(memory)  # truncation across DOOM decisions
        reward = float(game.make_action(action_vectors[idx], args.frame_skip))
        rewards.append(reward)
        spike_rates.append(float(rate.detach()))
        decisions += 1
        if args.max_decisions and decisions >= args.max_decisions:
            break
    # BASIC typically terminates on kill / timeout. Include explicit killcount.
    kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
    rec = {
        'reward': round(sum(rewards), 3),
        'kill': int(kills > 0),
        'decisions': decisions,
        'mean_spike_rate': round(sum(spike_rates) / len(spike_rates), 5) if spike_rates else 0.0,
        'finished': int(game.is_episode_finished()),
    }
    return rec, rewards, log_probs


def policy_update(model, optimizer, rewards, log_probs, gamma):
    if not log_probs:
        return 0.0
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = float(r) + gamma * g
        returns.append(g)
    returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
    # Within-episode baseline. This is experimental REINFORCE, not full DQN.
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)
    loss = -(torch.stack(log_probs) * returns).mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def smoke(args):
    model = RSNN(hidden=args.hidden, beta=args.beta, timesteps=args.timesteps)
    if args.image:
        image = Image.open(args.image).convert('RGB')
    else:
        # Artificial gradient image; only a shape/gradient check.
        arr = np.linspace(0, 255, WIDTH * HEIGHT, dtype=np.uint8).reshape(HEIGHT, WIDTH)
        image = Image.fromarray(arr, 'L')
    inp = frame_to_tensor(image)
    outputs, state, rate = model(inp)
    loss = nn.CrossEntropyLoss()(outputs, torch.tensor([3]))
    loss.backward()
    grad_in = float(model.input_fc.weight.grad.abs().sum())
    grad_rec = model.recurrent_fc.weight.grad
    grad_rec_sum = float(grad_rec.abs().sum()) if grad_rec is not None else 0.0
    print('OBSERVATION:', tuple(inp.shape), 'RANGE:', (float(inp.min()), float(inp.max())))
    print('ACTION LOGITS:', outputs.detach().flatten().tolist())
    print('PREDICTED ACTION (UNTRAINED):', ACTION_NAMES[int(outputs.argmax())])
    print('MEAN HIDDEN SPIKE RATE:', float(rate))
    print('BACKPROP GRADIENT SUM:', {'input_fc': round(grad_in, 6),
                                  'recurrent_fc': round(grad_rec_sum, 6)})
    print('SPIKING NEURONS:', model.hidden, '(encoder is external; no hardware mapping verified)')
    print('PARAMETERS:', sum(p.numel() for p in model.parameters()))
    assert grad_in > 0, 'Input weights did not receive gradients'
    assert grad_rec_sum > 0, 'Recurrent weights did not receive gradients'
    print('SMOKE TEST PASSED: RSNN forward, feedback gradients, output, backward.')
    print('No DOOM training or FPGA verification was performed.')


def main():
    p = argparse.ArgumentParser(description='RSNN DOOM software proof of concept')
    p.add_argument('--mode', choices=('smoke', 'demo', 'train', 'evaluate'), default='smoke')
    p.add_argument('--image', type=Path, default=None)
    p.add_argument('--config', type=Path, default=None)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--visible', action='store_true')
    p.add_argument('--frame-skip', type=int, default=4)
    p.add_argument('--max-decisions', type=int, default=0)
    p.add_argument('--timesteps', type=int, default=8)
    p.add_argument('--hidden', type=int, default=96)
    p.add_argument('--beta', type=float, choices=(0.25, 0.5, 0.75), default=0.5)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/rsnn.pt'))
    p.add_argument('--output', type=Path, default=Path('results/rsnn_results.csv'))
    args = p.parse_args()
    if args.episodes < 1 or args.frame_skip < 1 or args.hidden < 1:
        p.error('--episodes, --hidden and --frame-skip must be positive')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.mode == 'smoke':
        smoke(args)
        return
    model = RSNN(hidden=args.hidden, beta=args.beta, timesteps=args.timesteps)
    if args.mode == 'evaluate' and not args.checkpoint.exists():
        p.error(f'evaluate requires a trained checkpoint; missing: {args.checkpoint}')
    if args.checkpoint.exists() and args.mode != 'train':
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        cfg = checkpoint['config']
        for k in ('hidden', 'beta', 'timesteps'):
            if cfg[k] != getattr(args, k):
                p.error(f'Checkpoint {k}={cfg[k]}, but CLI requested {getattr(args, k)}')
        model.load_state_dict(checkpoint['model'])
        print('Loaded checkpoint:', args.checkpoint)
    if args.mode == 'demo' and not args.checkpoint.exists():
        print('WARNING: randomly initialized RSNN; actions are NOT learned.')
    if args.mode == 'train':
        print('Experimental training with REINFORCE; success is NOT guaranteed.')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.mode == 'train' else None
    game, vectors, vzd = build_game(args)
    records = []
    try:
        for ep in range(1, args.episodes + 1):
            model.train(args.mode == 'train')
            record, rewards, log_probs = episode(game, vectors, vzd, model, args,
                                                 training=(args.mode == 'train'))
            if optimizer:
                record['policy_loss'] = round(policy_update(model, optimizer, rewards,
                                                             log_probs, args.gamma), 6)
            record['episode'] = ep
            records.append(record)
            print(f"Episode {ep:04d}: reward={record['reward']:8.2f}, "
                  f"kill={record['kill']}, decisions={record['decisions']}, "
                  f"mean_hidden_spike_rate={record['mean_spike_rate']:.3f}")
            if args.mode == 'train' and ep % 25 == 0:
                args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'model': model.state_dict(), 'config': {
                    'hidden': args.hidden, 'beta': args.beta, 'timesteps': args.timesteps
                }}, args.checkpoint)
    finally:
        game.close()
    if optimizer:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': model.state_dict(), 'config': {
            'hidden': args.hidden, 'beta': args.beta, 'timesteps': args.timesteps
        }}, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Mean reward: {np.mean([r['reward'] for r in records]):.2f}")
    print(f"Success rate: {np.mean([r['kill'] for r in records])*100:.1f}%")
    print(f'CSV saved: {args.output}')
    if optimizer:
        print(f'Last checkpoint: {args.checkpoint} (not validated for hardware)')


if __name__ == '__main__':
    main()
