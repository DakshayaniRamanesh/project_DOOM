"""Diagnose a trained RSNN against fixed-action ViZDoom BASIC policies.

Place next to rsnn_doom.py, then run:
    python diagnose_rsnn_policy.py --config ../ViZDoom/scenarios/basic.cfg \
      --checkpoint checkpoints/rsnn.pt --episodes 100 --seed 43

No training occurs; only evaluation. Requires vizdoom, torch, pillow, numpy.
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
    p = argparse.ArgumentParser(description="ViZDoom BASIC fixed-action and RSNN diagnostic")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/rsnn.pt"))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--visible", action="store_true", help="Show the ViZDoom window (slows evaluations)")
    p.add_argument("--output", type=Path, default=Path("results/policy_diagnostics.csv"))
    return p.parse_args()


def evaluate_policy(args, policy_name, model):
    # A new simulator instance per policy ensures the same *initial seed*;
    # state trajectories still differ because each policy acts differently.
    game, actions, vzd = build_game(args)
    selected = Counter()
    rows = []
    policy_rng = random.Random(args.seed)
    try:
        for ep in range(1, args.episodes + 1):
            game.new_episode()
            recurrent_state = model.reset_state() if model is not None else None
            decisions = 0
            while not game.is_episode_finished():
                if policy_name == "rsnn":
                    state = game.get_state()
                    if state is None:
                        break
                    observation = frame_to_tensor(state.screen_buffer)
                    with torch.no_grad():
                        logits, recurrent_state, _ = model(observation, recurrent_state)
                    action_id = int(logits.argmax(dim=-1).item())
                elif policy_name == "random":
                    action_id = policy_rng.randrange(len(ACTION_NAMES))
                else:
                    action_id = ACTION_NAMES.index(policy_name)
                selected[ACTION_NAMES[action_id]] += 1
                game.make_action(actions[action_id], args.frame_skip)
                decisions += 1
            rows.append({
                "policy": policy_name,
                "episode": ep,
                "reward": float(game.get_total_reward()),
                "kill": int(game.get_game_variable(vzd.GameVariable.KILLCOUNT) > 0),
                "decisions": decisions,
                "actions": decisions,
            })
    finally:
        game.close()
    mean_reward = float(np.mean([r["reward"] for r in rows]))
    success_rate = 100.0 * float(np.mean([r["kill"] for r in rows]))
    mean_decisions = float(np.mean([r["decisions"] for r in rows]))
    counts = ", ".join(f"{k}:{selected[k]}" for k in ACTION_NAMES if selected[k])
    print(f"{policy_name:>12}: kills={sum(r['kill'] for r in rows):3d}/{len(rows)}, "
          f"success={success_rate:5.1f}%, mean_reward={mean_reward:8.2f}, "
          f"mean_decisions={mean_decisions:6.1f}")
    print(f"             action counts: {counts}")
    return rows


def main():
    args = parse_args()
    if args.episodes < 1 or args.frame_skip < 1:
        raise SystemExit("--episodes and --frame-skip must be positive")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing trained checkpoint: {args.checkpoint}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    model = RSNN(hidden=cfg["hidden"], beta=cfg["beta"], timesteps=cfg["timesteps"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"RSNN loaded: hidden={cfg['hidden']}, beta={cfg['beta']}, "
          f"timesteps={cfg['timesteps']}")
    print("Comparison policies use the same six-action space, scenario, frame skip and seed.")
    results = []
    for policy_name in ("shoot", "left_shoot", "right_shoot", "random", "rsnn"):
        results.extend(evaluate_policy(args, policy_name, model if policy_name == "rsnn" else None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "episode", "reward", "kill", "decisions", "actions"])
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved to: {args.output}")
    print("Note: a fixed-action policy matching RSNN performance would weaken a claim of adaptive learning.")


if __name__ == "__main__":
    main()
