"""Ablation diagnostics for rsnn_doom.py (does NOT train or modify checkpoint).

Keep this file beside rsnn_doom.py and run, for example:
python check_rsnn_behavior.py --config ../ViZDoom/scenarios/basic.cfg \
  --checkpoint checkpoints/rsnn.pt --episodes 100 --seed 43

Runs the exact same saved checkpoint under four conditions:
  normal       Real observations, unmodified RSNN
  blank_image  Input pixels set to a constant 0.25 every decision
  shuffle      Fixed random spatial permutation of pixels (same permutation throughout)
  no_feedback  Recurrent synaptic weights set to zero (copy in RAM only)

These are deliberately disruptive ablations, not new trained competitors.
Ablation scores alone do not prove vision or recurrence are optimal/necessary.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from rsnn_doom import ACTION_NAMES, RSNN, build_game, frame_to_tensor


def parse_args():
    p = argparse.ArgumentParser(description='Check RSNN visual and recurrent dependency')
    p.add_argument('--config', type=Path, default=None)
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/rsnn.pt'))
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--frame-skip', type=int, default=4)
    p.add_argument('--visible', action='store_true')
    p.add_argument('--output', type=Path, default=Path('results/rsnn_ablation.csv'))
    return p.parse_args()


def evaluate_variant(args, ckpt, variant):
    cfg = ckpt['config']
    model = RSNN(hidden=cfg['hidden'], beta=cfg['beta'], timesteps=cfg['timesteps'])
    model.load_state_dict(ckpt['model'])
    model.eval()
    if variant == 'no_feedback':
        # Only affects this model object, NOT the checkpoint file.
        with torch.no_grad():
            model.recurrent_fc.weight.zero_()

    # One fixed permutation shared across all observations/episodes.
    generator = torch.Generator().manual_seed(123456)
    permutation = torch.randperm(model.inputs, generator=generator)

    game, action_vectors, vzd = build_game(args)
    rows = []
    action_counts = Counter()
    try:
        for ep in range(1, args.episodes + 1):
            game.new_episode()
            recurrent_state = model.reset_state()
            episode_counts = Counter()
            decisions = 0
            with torch.no_grad():
                while not game.is_episode_finished():
                    state = game.get_state()
                    if state is None:
                        break
                    observation = frame_to_tensor(state.screen_buffer)
                    if variant == 'blank_image':
                        observation = torch.full_like(observation, 0.25)
                    elif variant == 'shuffle':
                        observation = observation[:, permutation]
                    logits, recurrent_state, _ = model(observation, recurrent_state)
                    action_id = int(logits.argmax(dim=-1).item())
                    action_name = ACTION_NAMES[action_id]
                    action_counts[action_name] += 1
                    episode_counts[action_name] += 1
                    game.make_action(action_vectors[action_id], args.frame_skip)
                    decisions += 1
            kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT) > 0)
            rows.append({
                'variant': variant,
                'episode': ep,
                'kill': kills,
                'reward': round(float(game.get_total_reward()), 4),
                'decisions': decisions,
                'shoot_actions': episode_counts['shoot'] + episode_counts['left_shoot'] + episode_counts['right_shoot'],
            })
    finally:
        game.close()
    mean_reward = float(np.mean([r['reward'] for r in rows]))
    success = float(np.mean([r['kill'] for r in rows])) * 100
    counts_text = ', '.join(f'{n}:{action_counts[n]}' for n in ACTION_NAMES if action_counts[n])
    print(f'{variant:>13}: kills={sum(r["kill"] for r in rows):3}/{len(rows):3} '
          f'({success:5.1f}%), mean_reward={mean_reward:8.2f}')
    print('              actions:', counts_text)
    return rows


def main():
    args = parse_args()
    if args.episodes < 1 or args.frame_skip < 1:
        raise SystemExit('episodes and frame-skip must be positive')
    if not args.checkpoint.is_file():
        raise SystemExit(f'Checkpoint not found: {args.checkpoint}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    print('Loaded checkpoint:', args.checkpoint)
    print('No weights will be updated or saved. Seed:', args.seed)
    all_rows = []
    for variant in ('normal', 'blank_image', 'shuffle', 'no_feedback'):
        all_rows.extend(evaluate_variant(args, ckpt, variant))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print('Saved:', args.output)
    print('Interpretation: compare action counts and performance; these tests are diagnostics, not proof of what the network learned.')


if __name__ == '__main__':
    main()
