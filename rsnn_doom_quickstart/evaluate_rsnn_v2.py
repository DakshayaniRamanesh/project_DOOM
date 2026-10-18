"""evaluate_rsnn_v2.py — Common evaluation interface for all policies.

Compares:
  1. Random policy
  2. Always shoot
  3. Always left_shoot  (best fixed-action baseline in v1 results)
  4. RSNN v1  (loaded from v1 checkpoint)
  5. RSNN v2  (loaded from v2 checkpoint)

All policies use identical:
  - ViZDoom scenario (basic.cfg)
  - 6-action space
  - Frame skip
  - Episode count
  - Seed protocol

Usage (PowerShell, from rsnn_doom_quickstart/):
  # Full comparison (requires both v1 and v2 checkpoints):
  python evaluate_rsnn_v2.py --config ../ViZDoom/scenarios/basic.cfg \\
      --v1-checkpoint checkpoints/rsnn.pt \\
      --v2-checkpoint checkpoints/v2/best.pt \\
      --episodes 100 --seed 43

  # Skip v1:
  python evaluate_rsnn_v2.py --config ../ViZDoom/scenarios/basic.cfg \\
      --v2-checkpoint checkpoints/v2/best.pt --episodes 50

  # Visible gameplay:
  python evaluate_rsnn_v2.py --config ../ViZDoom/scenarios/basic.cfg \\
      --v2-checkpoint checkpoints/v2/best.pt --episodes 5 --visible

  # Multiple seeds:
  python evaluate_rsnn_v2.py --config ../ViZDoom/scenarios/basic.cfg \\
      --v2-checkpoint checkpoints/v2/best.pt --episodes 100 \\
      --seeds 43 100 200 300 400

Outputs:
  results/v2/eval_YYYYMMDD_HHMMSS.csv   — per-episode rows for all policies
  results/v2/eval_summary_*.txt         — aggregated statistics printed + saved
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from doom_env import (
    ACTION_NAMES, build_game, frame_to_tensor,
    collect_episode_inference,
)
from rsnn_v2 import RSNNv2, RSNNv2Config, build_model_from_checkpoint


# ---------------------------------------------------------------------------
# V1 model loader (avoids importing rsnn_doom.py, reconstructs inline)
# ---------------------------------------------------------------------------

def _load_v1_model(checkpoint_path: Path):
    """Load a v1 checkpoint. Returns a callable that matches the v2 interface."""
    # Import v1 RSNN for loading only
    try:
        from rsnn_doom import RSNN as RSNNv1
    except ImportError:
        raise RuntimeError(
            "rsnn_doom.py must be present to load v1 checkpoints."
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = RSNNv1(
        hidden=cfg["hidden"], beta=cfg["beta"], timesteps=cfg["timesteps"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


class V1ModelAdapter:
    """Wraps the original RSNN v1 without changing its weights."""

    def __init__(self, model, cfg: dict):
        self._model = model
        self._cfg = cfg

    def reset_state(self, batch_size=1, device=None):
        return self._model.reset_state(batch_size, device)

    def eval(self):
        self._model.eval()
        return self

    def __call__(self, obs, state=None):
        """Returns (logits, dummy_value=0, state, spike_rate)."""
        with torch.no_grad():
            logits, state, rate = self._model(obs, state)
        return logits, torch.zeros(1), state, rate


# ---------------------------------------------------------------------------
# Fixed-action and random policy wrappers
# ---------------------------------------------------------------------------

class FixedActionPolicy:
    """Always chooses the same action index."""

    def __init__(self, action_name: str):
        self.action_idx = ACTION_NAMES.index(action_name)
        self.name = action_name

    def reset_state(self, batch_size=1, device=None):
        return None

    def eval(self):
        return self

    def __call__(self, obs, state=None):
        logits = torch.zeros(1, len(ACTION_NAMES))
        logits[0, self.action_idx] = 1e9  # Force this action
        return logits, torch.zeros(1), state, torch.tensor(0.0)


class RandomPolicy:
    """Truly sample uniformly over all six actions, even in greedy eval mode."""

    def __init__(self):
        self.rng = random.Random(43)

    def reseed(self, seed: int):
        self.rng.seed(seed)

    def reset_state(self, batch_size=1, device=None):
        return None

    def eval(self):
        return self

    def __call__(self, obs, state=None):
        logits = torch.zeros(1, len(ACTION_NAMES))
        return logits, torch.zeros(1), state, torch.tensor(0.0)

    def select_action(self, logits):
        return self.rng.randrange(len(ACTION_NAMES))


# ---------------------------------------------------------------------------
# Evaluation of a single policy over multiple episodes
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy_name: str,
    model,
    game,
    action_vectors,
    vzd,
    n_episodes: int,
    frame_skip: int,
    seed: int,
    greedy: bool = True,
) -> List[dict]:
    """Run n_episodes of a policy and return per-episode records."""
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(model, "reseed"):
        model.reseed(seed)

    records = []
    action_totals = Counter()

    for ep in range(1, n_episodes + 1):
        game.new_episode()
        state = model.reset_state()
        result, _ = collect_episode_inference(
            game, action_vectors, vzd, model, state,
            frame_skip=frame_skip, greedy=greedy,
        )
        action_totals.update(result.action_counts)

        records.append({
            "policy": policy_name,
            "episode": ep,
            "seed": seed,
            "total_reward": round(result.reward, 3),
            "kill": result.kill,
            "decisions": result.decisions,
            "terminated": int(result.terminated),
            "mean_spike_rate": round(result.mean_spike_rate, 5),
            **{f"action_{n}": result.action_counts.get(n, 0) for n in ACTION_NAMES},
        })

    # Summary
    rewards = [r["total_reward"] for r in records]
    kills = [r["kill"] for r in records]
    decs = [r["decisions"] for r in records]
    kill_rate = np.mean(kills) * 100
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_decisions = np.mean(decs)

    counts_str = ", ".join(
        f"{n}:{action_totals[n]}"
        for n in ACTION_NAMES if action_totals[n] > 0
    )
    print(
        f"  {policy_name:<16}: kills={sum(kills):3d}/{n_episodes} "
        f"({kill_rate:5.1f}%) "
        f"reward={mean_reward:8.2f}±{std_reward:.2f} "
        f"mean_dec={mean_decisions:.1f}"
    )
    print(f"  {'':16}  action counts: {counts_str}")

    return records


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def aggregate_summary(records: List[dict], policy_name: str) -> dict:
    subset = [r for r in records if r["policy"] == policy_name]
    if not subset:
        return {}
    rewards = [r["total_reward"] for r in subset]
    kills = [r["kill"] for r in subset]
    decs = [r["decisions"] for r in subset]
    return {
        "policy": policy_name,
        "n_episodes": len(subset),
        "kill_rate_pct": round(np.mean(kills) * 100, 2),
        "mean_reward": round(np.mean(rewards), 3),
        "std_reward": round(np.std(rewards), 3),
        "min_reward": round(min(rewards), 3),
        "max_reward": round(max(rewards), 3),
        "mean_decisions": round(np.mean(decs), 2),
        "std_decisions": round(np.std(decs), 2),
        "total_kills": sum(kills),
    }


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def run_evaluation(args):
    seeds = args.seeds if args.seeds else [args.seed]

    # Build policies
    policies = {}

    # Fixed baselines
    policies["random"] = RandomPolicy()
    policies["always_shoot"] = FixedActionPolicy("shoot")
    policies["always_left_shoot"] = FixedActionPolicy("left_shoot")

    # RSNN v1
    if args.v1_checkpoint and Path(args.v1_checkpoint).exists():
        print(f"Loading v1 checkpoint: {args.v1_checkpoint}")
        v1_model, v1_cfg = _load_v1_model(Path(args.v1_checkpoint))
        policies["rsnn_v1"] = V1ModelAdapter(v1_model, v1_cfg)
    else:
        print("No v1 checkpoint provided or found; skipping rsnn_v1.")

    # RSNN v2
    if args.v2_checkpoint and Path(args.v2_checkpoint).exists():
        print(f"Loading v2 checkpoint: {args.v2_checkpoint}")
        v2_model, v2_ckpt = build_model_from_checkpoint(args.v2_checkpoint)
        v2_model.eval()
        policies["rsnn_v2"] = v2_model
        print(f"  v2 trained for {v2_ckpt.get('episode', '?')} episodes, "
              f"best_reward={v2_ckpt.get('best_reward', '?'):.2f}")
    else:
        print("No v2 checkpoint provided or found; skipping rsnn_v2.")

    if not policies:
        print("No policies to evaluate. Exiting.")
        return

    # Output paths
    results_dir = Path("results/v2")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"eval_{timestamp}.csv"
    summary_path = results_dir / f"eval_summary_{timestamp}.txt"

    fieldnames = [
        "policy", "episode", "seed", "total_reward", "kill", "decisions",
        "terminated", "mean_spike_rate",
        *[f"action_{n}" for n in ACTION_NAMES],
    ]

    all_records = []
    total_seeds = len(seeds)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for seed_idx, seed in enumerate(seeds):
            print(f"\nSeed {seed} ({seed_idx+1}/{total_seeds})")
            print("-" * 60)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            for policy_name, policy in policies.items():
                # New engine process/object for each policy+seed. Re-seeding only
                # Python/NumPy is NOT sufficient to rewind ViZDoom's internal RNG.
                torch.manual_seed(seed)
                game, action_vectors, vzd = build_game(
                    config=args.config,
                    visible=args.visible,
                    seed=seed,
                )
                try:
                    policy.eval()
                    recs = evaluate_policy(
                        policy_name=policy_name,
                        model=policy,
                        game=game,
                        action_vectors=action_vectors,
                        vzd=vzd,
                        n_episodes=args.episodes,
                        frame_skip=args.frame_skip,
                        seed=seed,
                        greedy=True,
                    )
                    for rec in recs:
                        all_records.append(rec)
                        writer.writerow(rec)
                    csv_file.flush()
                finally:
                    game.close()

    # Print aggregate summary
    print("\n" + "=" * 70)
    print("  AGGREGATE EVALUATION SUMMARY")
    print("=" * 70)
    summary_lines = []
    summaries = []
    for policy_name in policies:
        s = aggregate_summary(all_records, policy_name)
        if s:
            summaries.append(s)
            line = (
                f"  {s['policy']:<20}: "
                f"kill_rate={s['kill_rate_pct']:5.1f}% "
                f"mean_reward={s['mean_reward']:8.2f} "
                f"std={s['std_reward']:6.2f} "
                f"(N={s['n_episodes']})"
            )
            print(line)
            summary_lines.append(line)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"RSNN v2 Evaluation Summary — {timestamp}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Episodes per seed: {args.episodes}\n")
        f.write(f"Frame skip: {args.frame_skip}\n")
        f.write("=" * 70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\nCSV saved: {csv_path}")
    print(f"Summary:   {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate RSNN v2 and baselines on ViZDoom BASIC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Path to basic.cfg")
    p.add_argument("--v1-checkpoint", type=Path, default=Path("checkpoints/rsnn.pt"),
                   help="Path to v1 checkpoint (set to '' to skip)")
    p.add_argument("--v2-checkpoint", type=Path, default=Path("checkpoints/v2/best.pt"),
                   help="Path to v2 checkpoint")
    p.add_argument("--episodes", type=int, default=100,
                   help="Episodes per policy per seed")
    p.add_argument("--seed", type=int, default=43,
                   help="Default seed (used if --seeds not given)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Multiple seeds for aggregate evaluation")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--visible", action="store_true",
                   help="Show ViZDoom window (slows evaluation)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.episodes < 1 or args.frame_skip < 1:
        print("--episodes and --frame-skip must be positive")
        sys.exit(1)
    run_evaluation(args)


if __name__ == "__main__":
    main()
