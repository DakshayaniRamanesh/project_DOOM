"""RSNN v2: on-policy single-environment A2C with GAE and truncated BPTT.

Training unit: a bounded segment of <= --rollout-steps DOOM decisions.
Each segment: collect differentiable log_probs/values/true entropies, bootstrap
from next observation if available, update actor and critic, detach LIF state.
The episode is NOT reset at segment boundaries.

ViZDoom is loaded lazily; --mode smoke works without ViZDoom installed.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rsnn_v2 import RSNNv2, RSNNv2Config, save_checkpoint, load_checkpoint, restore_rng_from_checkpoint
from doom_env import ACTION_NAMES, Transition, build_game, collect_rollout, collect_episode_inference, frame_to_tensor


def compute_returns_and_advantages(transitions, last_value: float,
                                   gamma: float, gae_lambda: float):
    """GAE over one rollout using detached state values.

    `done` means a true terminal (no bootstrap). Engine timeouts also use a
    zero bootstrap when no next frame exists; rollout boundaries get the critic's
    actual next-state value. Never bootstrap across a new episode.
    """
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError("gamma and gae_lambda must be between 0 and 1")
    n = len(transitions)
    adv = [0.0] * n
    ret = [0.0] * n
    gae = 0.0
    next_val = float(last_value)
    for t in range(n - 1, -1, -1):
        tr = transitions[t]
        mask = 0.0 if tr.done else 1.0
        this_value = float(tr.value.detach())
        delta = tr.reward + gamma * next_val * mask - this_value
        gae = delta + gamma * gae_lambda * mask * gae
        adv[t] = gae
        ret[t] = gae + this_value
        next_val = this_value
    return ret, adv


def a2c_update(model: RSNNv2, optimizer, transitions, last_value,
               gamma=0.99, gae_lambda=0.95, value_coef=0.5,
               entropy_coef=0.01, max_grad_norm=0.5,
               normalize_advantages=False):
    """One differentiable A2C update across one bounded recurrent rollout."""
    if not transitions:
        raise ValueError("Cannot update using an empty rollout")
    if any(t.entropy is None for t in transitions):
        raise ValueError("Each transition must retain differentiable full policy entropy")
    returns, advantages = compute_returns_and_advantages(
        transitions, last_value, gamma, gae_lambda)
    vals = torch.stack([t.value.reshape(()) for t in transitions])
    log_probs = torch.stack([t.log_prob.reshape(()) for t in transitions])
    entropies = torch.stack([t.entropy.reshape(()) for t in transitions])
    advantage_tensor = vals.new_tensor(advantages)
    if normalize_advantages and advantage_tensor.numel() >= 2:
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
            advantage_tensor.std(unbiased=False) + 1e-8)
    targets = vals.new_tensor(returns)
    actor_loss = -(log_probs * advantage_tensor).mean()
    critic_loss = nn.functional.mse_loss(vals, targets)
    policy_entropy = entropies.mean()
    loss = actor_loss + value_coef * critic_loss - entropy_coef * policy_entropy
    if not torch.isfinite(loss):
        raise FloatingPointError("A2C loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # clip_grad_norm_ returns the norm BEFORE clipping, useful to log as such.
    grad_norm_preclip = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                                 error_if_nonfinite=True)
    optimizer.step()
    return dict(actor_loss=float(actor_loss.detach()),
                critic_loss=float(critic_loss.detach()),
                entropy=float(policy_entropy.detach()),
                total_loss=float(loss.detach()),
                grad_norm_preclip=float(grad_norm_preclip.detach()))


def bootstrap_value(model, next_observation, state):
    """Evaluate the critic on the next real observation (no gradient)."""
    if next_observation is None:
        return 0.0
    with torch.no_grad():
        _, value, _, _ = model(next_observation, RSNNv2.detach_state(state))
        return float(value.squeeze())


def smoke_test(args):
    """Offline end-to-end differentiable rollout with true entropy and clipping."""
    torch.manual_seed(args.seed)
    cfg = RSNNv2Config(hidden=args.hidden, beta=args.beta, timesteps=args.timesteps)
    model = RSNNv2(cfg)
    model.print_resources()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = model.reset_state()
    transitions = []
    for i in range(4):
        observation = torch.rand(1, 768)
        logits, value, state, _ = model(observation, state)
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        transitions.append(Transition(observation, int(action.item()), dist.log_prob(action),
                                      value.squeeze(0), 5.0 if i == 3 else -1.0,
                                      done=(i == 3), entropy=dist.entropy()))
    stats = a2c_update(model, opt, transitions, 0.0, args.gamma, args.gae_lambda,
                       args.value_coef, args.entropy_coef, args.max_grad_norm)
    assert all(math.isfinite(v) for v in stats.values())
    grad = model.recurrent_fc.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    assert not RSNNv2.detach_state(state)[0].requires_grad
    tmp = Path("checkpoints/v2/smoke_test.pt")
    save_checkpoint(tmp, model, opt, episode=4, best_reward=-1.0,
                    extra={"best_val_success": 0.0})
    loaded = load_checkpoint(tmp)
    assert loaded["version"] == "rsnn_v2"
    assert loaded["episode"] == 4
    tmp.unlink()
    print("SMOKE TEST PASSED: surrogate gradients, real entropy, GAE, optimizer, checkpoint")
    print("Loss metrics:", stats)


def validate(model, args):
    """Use designated validation seeds ONLY for checkpoint selection."""
    was_training = model.training
    rewards, kills = [], []
    for seed in args.val_seeds:
        game, vectors, vzd = build_game(config=args.config, visible=False, seed=seed)
        try:
            for _ in range(args.val_episodes):
                game.new_episode()
                result, _ = collect_episode_inference(
                    game, vectors, vzd, model, model.reset_state(),
                    frame_skip=args.frame_skip, greedy=True)
                rewards.append(result.reward)
                kills.append(result.kill)
        finally:
            game.close()
    if was_training:
        model.train()
    return float(np.mean(kills)), float(np.mean(rewards))


def train(args):
    if args.episodes < 1 or args.rollout_steps < 1 or args.frame_skip < 1:
        raise ValueError("Episodes, rollout-steps and frame-skip must be positive")
    if args.validate_every < 1 or args.val_episodes < 1:
        raise ValueError("Validation intervals and episodes must be positive")
    if args.max_decisions < 0:
        raise ValueError("max-decisions must be >= 0")
    if not args.val_seeds:
        raise ValueError("At least one validation seed is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Reconstruct using checkpoint configuration BEFORE constructing optimizer.
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = load_checkpoint(args.resume)
        cfg = RSNNv2Config.from_dict(ckpt["config"])
        if (cfg.hidden, cfg.beta, cfg.timesteps) != (args.hidden, args.beta, args.timesteps):
            raise ValueError("Resume checkpoint model configuration differs from CLI flags. "
                             "Use matching --hidden, --beta and --timesteps values.")
        model = RSNNv2(cfg)
        model.load_state_dict(ckpt["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_episode = int(ckpt["episode"]) + 1
        best_success = float(ckpt.get("best_val_success", -1.0))
        best_reward = float(ckpt.get("best_val_reward", -float("inf")))
        restore_rng_from_checkpoint(ckpt)
        print(f"Resumed at episode {start_episode}; ViZDoom internal RNG cannot be restored exactly")
    else:
        cfg = RSNNv2Config(hidden=args.hidden, beta=args.beta, timesteps=args.timesteps)
        model = RSNNv2(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        start_episode = 1
        best_success, best_reward = -1.0, -float("inf")
    model.print_resources()

    ckpt_dir = Path("checkpoints/v2")
    results_dir = Path("results/v2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"train_corrected_{tag}.csv"
    cfg_file = results_dir / f"config_corrected_{tag}.txt"
    cfg_file.write_text("\n".join([
        "RSNN v2 corrected A2C + GAE",
        f"training_seed={args.seed}",
        f"validation_seeds={args.val_seeds}",
        f"val_episodes={args.val_episodes}",
        f"validate_every={args.validate_every}",
        f"episodes_this_call={args.episodes}",
        f"resume={args.resume}",
        f"rollout_steps={args.rollout_steps}",
        f"lr={args.lr}", f"gamma={args.gamma}", f"gae_lambda={args.gae_lambda}",
        f"value_coef={args.value_coef}", f"entropy_coef={args.entropy_coef}",
        f"max_grad_norm={args.max_grad_norm}",
        f"normalize_advantages={args.normalize_advantages}",
        f"hidden={cfg.hidden}", f"beta={cfg.beta}", f"timesteps={cfg.timesteps}",
        f"frame_skip={args.frame_skip}", f"max_decisions={args.max_decisions}",
        f"config={args.config}",
        "engine_timeout_with_no_next_observation=zero bootstrap (finite episode horizon)",
        "grad_norm_preclip=norm before clipping",
    ]) + "\n", encoding="utf-8")

    columns = ["episode", "seed", "total_reward", "kill", "decisions", "terminated",
               "truncated", "timed_out", "rollouts", "rollout_steps",
               *[f"action_{n}" for n in ACTION_NAMES], "mean_spike_rate", "entropy",
               "actor_loss", "critic_loss", "total_loss", "grad_norm_preclip", "lr", "timestamp"]
    last_completed = start_episode - 1
    game, actions, vzd = build_game(args.config, visible=args.visible, seed=args.seed)
    try:
        with log_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            fh.flush()
            for ep in range(start_episode, start_episode + args.episodes):
                game.new_episode()
                state = model.reset_state()
                counts = {n: 0 for n in ACTION_NAMES}
                reward_sum, spike_sum, n_decisions = 0.0, 0.0, 0
                info_segments = []
                last = None
                while True:
                    remaining = (args.max_decisions - n_decisions) if args.max_decisions else 0
                    segment = collect_rollout(game, actions, vzd, model, state,
                                              frame_skip=args.frame_skip,
                                              rollout_steps=args.rollout_steps,
                                              remaining_decisions=remaining)
                    if not segment.transitions:
                        break
                    # No gradients through the bootstrap, but through all steps
                    # within this segment. Value from the CURRENT (pre-update) model.
                    value_next = bootstrap_value(model, segment.next_observation, segment.state)
                    metrics = a2c_update(model, optimizer, segment.transitions, value_next,
                                         args.gamma, args.gae_lambda, args.value_coef,
                                         args.entropy_coef, args.max_grad_norm,
                                         args.normalize_advantages)
                    info_segments.append((len(segment.transitions), metrics))
                    state = RSNNv2.detach_state(segment.state)
                    last = segment
                    reward_sum += segment.reward
                    spike_sum += segment.spike_sum
                    n_decisions += len(segment.transitions)
                    for key, count in segment.action_counts.items():
                        counts[key] += count
                    if segment.ended:
                        break
                killed = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT) > 0)
                denom = max(n_decisions, 1)
                def avg(key):
                    return sum(n * s[key] for n, s in info_segments) / denom
                rec = dict(episode=ep, seed=args.seed, total_reward=round(reward_sum, 4),
                           kill=killed, decisions=n_decisions,
                           terminated=int(last.terminated) if last else 0,
                           truncated=int(last.truncated) if last else 0,
                           timed_out=int(last.timeout) if last else 0,
                           rollouts=len(info_segments), rollout_steps=args.rollout_steps,
                           **{f"action_{n}": counts[n] for n in ACTION_NAMES},
                           mean_spike_rate=round(spike_sum / denom, 6),
                           entropy=round(avg("entropy"), 6),
                           actor_loss=round(avg("actor_loss"), 6),
                           critic_loss=round(avg("critic_loss"), 6),
                           total_loss=round(avg("total_loss"), 6),
                           grad_norm_preclip=round(avg("grad_norm_preclip"), 5),
                           lr=optimizer.param_groups[0]["lr"], timestamp=datetime.now().isoformat())
                writer.writerow(rec)
                fh.flush()
                last_completed = ep
                print(f"Ep {ep:04d}: reward={reward_sum:8.2f} kill={killed} "
                      f"dec={n_decisions:3d} rollouts={len(info_segments)} "
                      f"entropy={rec['entropy']:.3f} value_loss={rec['critic_loss']:.2f}")

                if ep % args.validate_every == 0 or ep == start_episode + args.episodes - 1:
                    success, val_reward = validate(model, args)
                    print(f"  Validation: kill={100*success:.1f}% reward={val_reward:.2f} "
                          f"across {len(args.val_seeds)*args.val_episodes} episodes")
                    if (success, val_reward) > (best_success, best_reward):
                        best_success, best_reward = success, val_reward
                        save_checkpoint(ckpt_dir / "best.pt", model, optimizer, ep, best_reward,
                                        extra={"best_val_success": best_success,
                                               "best_val_reward": best_reward,
                                               "train_config": str(cfg_file)})
                        print("  New best VALIDATION checkpoint saved")
                # Save at EVERY completed episode (small model). Never persist
                # partially updated weights under a completed episode number.
                save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, ep, best_reward,
                                extra={"best_val_success": best_success,
                                       "best_val_reward": best_reward,
                                       "train_config": str(cfg_file)})
    except KeyboardInterrupt:
        print("Interrupted: latest.pt still contains the last FULLY COMPLETED episode. "
              "Discarding any in-memory updates from the interrupted episode.")
    finally:
        game.close()
    print(f"Training log: {log_file}")
    print(f"Latest: {ckpt_dir / 'latest.pt'} | Best validation: {ckpt_dir / 'best.pt'}")


def parse_args():
    p = argparse.ArgumentParser(description="RSNN v2 corrected A2C/GAE training")
    p.add_argument("--mode", choices=("train", "smoke"), default="train")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=500, help="Additional episodes if resuming")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--visible", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--max-decisions", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=8)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--beta", type=float, default=0.5, choices=(0.25, 0.5, 0.75))
    p.add_argument("--timesteps", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--normalize-advantages", action="store_true",
                   help="Optional per-rollout norm; OFF by default")
    p.add_argument("--save-every", type=int, default=1,
                   help="Compatibility flag; latest checkpoint is saved every completed episode")
    p.add_argument("--validate-every", type=int, default=50)
    p.add_argument("--val-episodes", type=int, default=10)
    p.add_argument("--val-seeds", type=int, nargs="+", default=[1101, 1102])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "smoke":
        smoke_test(args)
    else:
        train(args)
