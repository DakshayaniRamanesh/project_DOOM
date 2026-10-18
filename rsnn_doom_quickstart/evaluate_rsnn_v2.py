"""evaluate_rsnn_v2.py — Evaluation interface with dual greedy/stochastic modes and action probability diagnostics.

Compares:
  1. Random policy (samples uniformly across all 6 actions)
  2. Always shoot
  3. Always left_shoot
  4. RSNN v1 (loaded from v1 checkpoint)
  5. RSNN v2 (loaded from v2 checkpoint, evaluated in greedy and/or stochastic mode)

Diagnostic capabilities:
  - Exact network softmax probabilities per action (not inferred from counts)
  - Raw actor logits and logit margins (1st - 2nd highest logit)
  - Policy entropy
  - Action selection fractions vs predicted probabilities
  - Optional per-decision CSV logging (bounded to prevent huge files)
  - Dual-mode evaluation: --eval-mode both compares greedy and stochastic side-by-side
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
    collect_episode_inference, EpisodeResult
)
from rsnn_v2 import RSNNv2, RSNNv2Config, build_model_from_checkpoint


# ---------------------------------------------------------------------------
# V1 model loader
# ---------------------------------------------------------------------------

def _load_v1_model(checkpoint_path: Path):
    """Load a v1 checkpoint. Returns a callable matching the v2 interface."""
    try:
        from rsnn_doom import RSNN as RSNNv1
    except ImportError:
        raise RuntimeError("rsnn_doom.py must be present to load v1 checkpoints.")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = RSNNv1(
        hidden=cfg["hidden"], beta=cfg["beta"], timesteps=cfg["timesteps"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


class V1ModelAdapter:
    """Wraps the original RSNN v1 without modifying its weights."""

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
    """Truly sample uniformly over all six actions, in both greedy and stochastic mode."""

    def __init__(self, seed: int = 43):
        self.rng = random.Random(seed)

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
    record_decisions_limit: int = 0,
) -> tuple[List[dict], List[dict]]:
    """Run n_episodes of a policy and return (episode_records, decision_records)."""
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(model, "reseed"):
        model.reseed(seed)

    records = []
    all_decision_records = []
    action_totals = Counter()
    prob_totals = {n: 0.0 for n in ACTION_NAMES}
    total_decisions = 0
    mode_str = "greedy" if greedy else "stochastic"

    for ep in range(1, n_episodes + 1):
        game.new_episode()
        state = model.reset_state()
        should_record = (ep <= record_decisions_limit)
        result, _ = collect_episode_inference(
            game, action_vectors, vzd, model, state,
            frame_skip=frame_skip, greedy=greedy,
            record_decisions=should_record,
        )
        action_totals.update(result.action_counts)
        for n in ACTION_NAMES:
            prob_totals[n] += result.mean_probabilities.get(n, 0.0) * result.decisions
        total_decisions += result.decisions

        if should_record and result.decision_records:
            for d in result.decision_records:
                d_copy = dict(d)
                d_copy["policy"] = policy_name
                d_copy["eval_mode"] = mode_str
                d_copy["episode"] = ep
                d_copy["seed"] = seed
                all_decision_records.append(d_copy)

        ep_rec = {
            "policy": policy_name,
            "eval_mode": mode_str,
            "episode": ep,
            "seed": seed,
            "total_reward": round(result.reward, 3),
            "kill": result.kill,
            "decisions": result.decisions,
            "terminated": int(result.terminated),
            "mean_spike_rate": round(result.mean_spike_rate, 5),
            "mean_entropy": result.mean_entropy,
            "mean_margin": result.mean_margin,
            **{f"action_{n}": result.action_counts.get(n, 0) for n in ACTION_NAMES},
            **{f"frac_{n}": round((result.action_counts.get(n, 0) / result.decisions) if result.decisions else 0.0, 4)
               for n in ACTION_NAMES},
            **{f"prob_{n}": round(result.mean_probabilities.get(n, 0.0), 4) for n in ACTION_NAMES},
        }
        records.append(ep_rec)

    # Summary
    rewards = [r["total_reward"] for r in records]
    kills = [r["kill"] for r in records]
    decs = [r["decisions"] for r in records]
    kill_rate = np.mean(kills) * 100
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_decisions = np.mean(decs)
    mean_ent = np.mean([r["mean_entropy"] for r in records])
    mean_mar = np.mean([r["mean_margin"] for r in records])

    # True mean action probabilities vs chosen fractions
    mean_probs = {n: (prob_totals[n] / total_decisions) if total_decisions else 0.0 for n in ACTION_NAMES}
    counts_str = ", ".join(f"{n}:{action_totals[n]}" for n in ACTION_NAMES if action_totals[n] > 0)
    probs_str = ", ".join(f"{n}:{mean_probs[n]*100:.1f}%" for n in ACTION_NAMES)

    print(
        f"  {policy_name:<18} [{mode_str:<10}]: kills={sum(kills):3d}/{n_episodes} "
        f"({kill_rate:5.1f}%) "
        f"reward={mean_reward:8.2f}±{std_reward:.2f} "
        f"dec={mean_decisions:.1f} ent={mean_ent:.3f} margin={mean_mar:.3f}"
    )
    print(f"    Selected actions: {counts_str}")
    print(f"    Model prob dist:  {probs_str}")

    return records, all_decision_records


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def aggregate_summary(records: List[dict], policy_name: str, eval_mode: str) -> dict:
    subset = [r for r in records if r["policy"] == policy_name and r["eval_mode"] == eval_mode]
    if not subset:
        return {}
    rewards = [r["total_reward"] for r in subset]
    kills = [r["kill"] for r in subset]
    decs = [r["decisions"] for r in subset]
    ents = [r["mean_entropy"] for r in subset]
    margins = [r["mean_margin"] for r in subset]
    tot_decs = sum(decs)

    mean_probs = {}
    act_fractions = {}
    for n in ACTION_NAMES:
        tot_act = sum(r[f"action_{n}"] for r in subset)
        act_fractions[n] = round((tot_act / tot_decs) * 100 if tot_decs else 0.0, 2)
        mean_probs[n] = round(float(np.mean([r[f"prob_{n}"] for r in subset])) * 100, 2)

    return {
        "policy": policy_name,
        "eval_mode": eval_mode,
        "n_episodes": len(subset),
        "kill_rate_pct": round(float(np.mean(kills)) * 100, 2),
        "mean_reward": round(float(np.mean(rewards)), 3),
        "std_reward": round(float(np.std(rewards)), 3),
        "min_reward": round(float(min(rewards)), 3),
        "max_reward": round(float(max(rewards)), 3),
        "mean_decisions": round(float(np.mean(decs)), 2),
        "mean_entropy": round(float(np.mean(ents)), 4),
        "mean_margin": round(float(np.mean(margins)), 4),
        "total_kills": sum(kills),
        "action_fractions_pct": act_fractions,
        "mean_model_probs_pct": mean_probs,
    }


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def run_evaluation(args):
    seeds = args.seeds if args.seeds else [args.seed]

    # Determine evaluation modes
    if args.stochastic or args.eval_mode == "stochastic":
        eval_modes = ["stochastic"]
    elif args.eval_mode == "both":
        eval_modes = ["greedy", "stochastic"]
    else:
        eval_modes = ["greedy"]

    # Build policies
    policies = {}

    # Fixed baselines
    policies["random"] = RandomPolicy(seed=args.seed)
    policies["always_shoot"] = FixedActionPolicy("shoot")
    policies["always_left_shoot"] = FixedActionPolicy("left_shoot")

    # RSNN v1
    if args.v1_checkpoint and Path(args.v1_checkpoint).exists():
        print(f"Loading v1 checkpoint: {args.v1_checkpoint}")
        v1_model, v1_cfg = _load_v1_model(Path(args.v1_checkpoint))
        policies["rsnn_v1"] = V1ModelAdapter(v1_model, v1_cfg)
    else:
        print("No v1 checkpoint provided or found; skipping rsnn_v1.")

    # RSNN v2 - best checkpoint
    if args.v2_checkpoint and Path(args.v2_checkpoint).exists():
        print(f"Loading v2 checkpoint: {args.v2_checkpoint}")
        v2_model, v2_ckpt = build_model_from_checkpoint(args.v2_checkpoint)
        v2_model.eval()
        policies["rsnn_v2_best"] = v2_model
        print(f"  v2_best trained for episode {v2_ckpt.get('episode', '?')}, "
              f"best_reward={v2_ckpt.get('best_reward', '?')}")
    else:
        print(f"v2 checkpoint not found at: {args.v2_checkpoint}")

    # Additional checkpoint (e.g. latest or ep10 backup for comparative diagnosis)
    if args.compare_checkpoint and Path(args.compare_checkpoint).exists():
        print(f"Loading comparison checkpoint: {args.compare_checkpoint}")
        cmp_model, cmp_ckpt = build_model_from_checkpoint(args.compare_checkpoint)
        cmp_model.eval()
        policies["rsnn_v2_cmp"] = cmp_model
        print(f"  cmp trained for episode {cmp_ckpt.get('episode', '?')}")

    if not policies:
        print("No policies to evaluate. Exiting.")
        return

    # Output paths
    results_dir = Path("results/v2")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"eval_{timestamp}.csv"
    summary_path = results_dir / f"eval_summary_{timestamp}.txt"
    decisions_csv_path = results_dir / f"eval_decisions_{timestamp}.csv" if args.log_decisions else None

    fieldnames = [
        "policy", "eval_mode", "episode", "seed", "total_reward", "kill", "decisions",
        "terminated", "mean_spike_rate", "mean_entropy", "mean_margin",
        *[f"action_{n}" for n in ACTION_NAMES],
        *[f"frac_{n}" for n in ACTION_NAMES],
        *[f"prob_{n}" for n in ACTION_NAMES],
    ]

    all_records = []
    all_decision_records = []
    total_seeds = len(seeds)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for seed_idx, seed in enumerate(seeds):
            print(f"\nSeed {seed} ({seed_idx+1}/{total_seeds})")
            print("-" * 65)

            for policy_name, policy in policies.items():
                for mode in eval_modes:
                    # Random and Fixed policies behave identically in greedy/stochastic,
                    # so avoid redundant duplicate runs if mode=="both"
                    if mode == "stochastic" and policy_name in ("random", "always_shoot", "always_left_shoot") and "greedy" in eval_modes:
                        continue

                    is_greedy = (mode == "greedy")
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    game, action_vectors, vzd = build_game(
                        config=args.config,
                        visible=args.visible,
                        seed=seed,
                    )
                    try:
                        policy.eval()
                        recs, dec_recs = evaluate_policy(
                            policy_name=policy_name,
                            model=policy,
                            game=game,
                            action_vectors=action_vectors,
                            vzd=vzd,
                            n_episodes=args.episodes,
                            frame_skip=args.frame_skip,
                            seed=seed,
                            greedy=is_greedy,
                            record_decisions_limit=args.max_logged_episodes if args.log_decisions else 0,
                        )
                        for rec in recs:
                            all_records.append(rec)
                            writer.writerow(rec)
                        if dec_recs:
                            all_decision_records.extend(dec_recs)
                        csv_file.flush()
                    finally:
                        game.close()

    # Save decisions CSV if requested
    if args.log_decisions and all_decision_records and decisions_csv_path:
        dec_fieldnames = list(all_decision_records[0].keys())
        with decisions_csv_path.open("w", newline="", encoding="utf-8") as df:
            d_writer = csv.DictWriter(df, fieldnames=dec_fieldnames)
            d_writer.writeheader()
            d_writer.writerows(all_decision_records)
        print(f"\nDetailed per-decision log saved: {decisions_csv_path} ({len(all_decision_records)} decisions)")

    # Print aggregate summary
    print("\n" + "=" * 78)
    print("  AGGREGATE EVALUATION SUMMARY (Greedy vs Stochastic Comparison)")
    print("=" * 78)
    summary_lines = []
    
    # Header
    hdr = f"  {'Policy':<16} {'Mode':<11} {'Kill %':<8} {'Reward (mean±std)':<20} {'Dec':<6} {'Ent':<6} {'Margin':<6}"
    print(hdr)
    print("  " + "-" * 74)
    summary_lines.append(hdr)
    summary_lines.append("  " + "-" * 74)

    for policy_name in policies:
        for mode in eval_modes:
            s = aggregate_summary(all_records, policy_name, mode)
            if s:
                rew_str = f"{s['mean_reward']:7.1f}±{s['std_reward']:.1f}"
                line = (
                    f"  {s['policy']:<16} {s['eval_mode']:<11} "
                    f"{s['kill_rate_pct']:5.1f}%  "
                    f"{rew_str:<20} "
                    f"{s['mean_decisions']:5.1f} "
                    f"{s['mean_entropy']:5.3f} "
                    f"{s['mean_margin']:5.3f}"
                )
                print(line)
                summary_lines.append(line)
                # Print action fraction vs model probability comparison
                frac_str = ", ".join(f"{k[:4]}:{v:.0f}%" for k, v in s['action_fractions_pct'].items())
                prob_str = ", ".join(f"{k[:4]}:{v:.0f}%" for k, v in s['mean_model_probs_pct'].items())
                sub_line1 = f"    Actions chosen:  {frac_str}"
                sub_line2 = f"    Network probs:   {prob_str}"
                print(sub_line1)
                print(sub_line2)
                summary_lines.append(sub_line1)
                summary_lines.append(sub_line2)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"RSNN v2 Evaluation Summary — {timestamp}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Episodes per seed: {args.episodes}\n")
        f.write(f"Evaluation modes: {eval_modes}\n")
        f.write(f"Frame skip: {args.frame_skip}\n")
        f.write("=" * 78 + "\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\nPer-episode CSV saved: {csv_path}")
    print(f"Summary saved:         {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate RSNN v2 and baselines on ViZDoom BASIC with action diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=None,
                   help="Path to basic.cfg")
    p.add_argument("--v1-checkpoint", type=Path, default=Path("checkpoints/rsnn.pt"),
                   help="Path to v1 checkpoint (set to '' to skip)")
    p.add_argument("--v2-checkpoint", type=Path, default=Path("checkpoints/v2/best.pt"),
                   help="Path to v2 checkpoint")
    p.add_argument("--compare-checkpoint", type=Path, default=None,
                   help="Optional second v2 checkpoint for side-by-side comparison (e.g. checkpoints/v2/latest.pt)")
    p.add_argument("--eval-mode", choices=("greedy", "stochastic", "both"), default="greedy",
                   help="Evaluation mode: greedy (argmax), stochastic (sample), or both side-by-side")
    p.add_argument("--stochastic", action="store_true",
                   help="Shorthand for --eval-mode stochastic")
    p.add_argument("--log-decisions", action="store_true",
                   help="Log per-decision logits, probabilities, margins and entropy to CSV")
    p.add_argument("--max-logged-episodes", type=int, default=5,
                   help="Max episodes per policy/seed to write to per-decision log (prevents huge CSVs)")
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
