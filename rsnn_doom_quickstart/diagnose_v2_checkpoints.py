"""diagnose_v2_checkpoints.py — Diagnostic analysis of ep5 vs ep10 checkpoints.

Examines:
1. Model parameters (weights, biases, norms).
2. Action logits, softmax probabilities, entropy, margin on static frames.
3. Temporal evolution across consecutive decisions (evolving recurrent state).
4. ViZDoom greedy vs stochastic behavior.
"""
from __future__ import annotations

import pathlib
import sys
import torch
import numpy as np
from PIL import Image

from doom_env import frame_to_tensor, ACTION_NAMES
from rsnn_v2 import build_model_from_checkpoint, RSNNv2, RSNNv2Config


def inspect_checkpoint(name: str, path: str):
    print("=" * 70)
    print(f"CHECKPOINT: {name} ({path})")
    print("=" * 70)
    
    ckpt_path = pathlib.Path(path)
    if not ckpt_path.exists():
        print(f"File not found: {path}")
        return None, None
        
    model, ckpt = build_model_from_checkpoint(ckpt_path)
    model.eval()
    
    ep = ckpt.get("episode", "unknown")
    best_val_succ = ckpt.get("best_val_success", "unknown")
    best_val_rew = ckpt.get("best_val_reward", "unknown")
    print(f"Metadata: Episode={ep}, BestValSuccess={best_val_succ}, BestValReward={best_val_rew}")
    print(f"Model config: {model.cfg}")
    
    # Check parameter norms and biases
    print("\n--- Parameter Norms & Biases ---")
    print(f"  input_fc.weight:     norm={model.input_fc.weight.norm().item():.4f}, mean={model.input_fc.weight.mean().item():.6f}")
    print(f"  input_fc.bias:       norm={model.input_fc.bias.norm().item():.4f}")
    print(f"  recurrent_fc.weight: norm={model.recurrent_fc.weight.norm().item():.4f}, mean={model.recurrent_fc.weight.mean().item():.6f}")
    print(f"  actor_fc.weight:     norm={model.actor_fc.weight.norm().item():.4f}, mean={model.actor_fc.weight.mean().item():.6f}")
    print(f"  actor_fc.bias:       values={[round(b, 4) for b in model.actor_fc.bias.tolist()]}")
    print(f"  critic_fc.weight:    norm={model.critic_fc.weight.norm().item():.4f}, mean={model.critic_fc.weight.mean().item():.6f}")
    print(f"  critic_fc.bias:      value={model.critic_fc.bias.item():.4f}")
    
    # Test on various inputs
    test_inputs = {}
    if pathlib.Path("doom_frame.png").exists():
        test_inputs["doom_frame.png"] = frame_to_tensor(Image.open("doom_frame.png"))
    test_inputs["zeros (black)"] = torch.zeros(1, model.cfg.inputs)
    test_inputs["half (gray)"] = torch.full((1, model.cfg.inputs), 0.5)
    test_inputs["ones (white)"] = torch.ones(1, model.cfg.inputs)
    torch.manual_seed(123)
    test_inputs["random_noise"] = torch.rand(1, model.cfg.inputs)
    
    print("\n--- Static Decision Analysis (Single Step from Clean State) ---")
    for in_name, obs in test_inputs.items():
        state = model.reset_state()
        with torch.no_grad():
            logits, val, _, rate = model(obs, state)
        logits_1d = logits.squeeze(0)
        probs = torch.softmax(logits_1d, dim=-1)
        entropy = torch.distributions.Categorical(probs=probs).entropy().item()
        sorted_logits, sorted_indices = torch.sort(logits_1d, descending=True)
        top1_act = ACTION_NAMES[sorted_indices[0].item()]
        top2_act = ACTION_NAMES[sorted_indices[1].item()]
        margin = (sorted_logits[0] - sorted_logits[1]).item()
        
        print(f"\n  Input: {in_name}")
        print(f"    Spike rate: {rate.item():.4f} | Value V(s): {val.item():.4f} | Entropy: {entropy:.4f}")
        print(f"    Top-1: {top1_act} (logit={sorted_logits[0].item():+.4f}, prob={probs[sorted_indices[0]].item()*100:.1f}%)")
        print(f"    Top-2: {top2_act} (logit={sorted_logits[1].item():+.4f}, prob={probs[sorted_indices[1]].item()*100:.1f}%)")
        print(f"    Margin: {margin:.4f}")
        print("    Full distribution:")
        for act, prob, logit in zip(ACTION_NAMES, probs.tolist(), logits_1d.tolist()):
            print(f"      {act:<12}: prob={prob*100:5.2f}% | logit={logit:+.4f}")

    # Temporal test: run 15 consecutive steps with doom_frame.png (or random) to see if recurrent state shifts preferences
    print("\n--- Temporal Sequence (15 steps with doom_frame.png, state evolving) ---")
    obs = test_inputs.get("doom_frame.png", test_inputs["random_noise"])
    state = model.reset_state()
    for step in range(1, 16):
        with torch.no_grad():
            logits, val, state, rate = model(obs, state)
        logits_1d = logits.squeeze(0)
        probs = torch.softmax(logits_1d, dim=-1)
        entropy = torch.distributions.Categorical(probs=probs).entropy().item()
        top_idx = logits_1d.argmax().item()
        top_act = ACTION_NAMES[top_idx]
        sorted_logits, _ = torch.sort(logits_1d, descending=True)
        margin = (sorted_logits[0] - sorted_logits[1]).item()
        mem_norm = state[1].norm().item()
        spk_count = state[0].sum().item()
        print(f"    Step {step:2d}: top={top_act:<12} (prob={probs[top_idx].item()*100:5.2f}%, margin={margin:.4f}) | "
              f"entropy={entropy:.3f} | rate={rate.item():.3f} | val={val.item():6.2f} | mem_norm={mem_norm:.2f} spk={spk_count:.0f}")

    return model, ckpt


if __name__ == "__main__":
    inspect_checkpoint("Episode 5 (Best Backup)", "checkpoints/v2/best_ep5_backup.pt")
    inspect_checkpoint("Episode 10 (Latest Backup)", "checkpoints/v2/latest_ep10_backup.pt")
