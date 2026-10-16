# RSNN for ViZDoom BASIC — October 2026 quick prototype

## Status

This is a **proof of concept**, not a trained DOOM agent. It implements:

- Frame preprocessing (ViZDoom CHW RGB -> 32×24 grayscale).
- Deterministic rate spike encoding over 8 neural timesteps per game decision.
- A 96-neuron recurrent LIF layer, including explicit learnable spike-to-spike feedback.
- β=0.5 by default (permitted 0.25/0.5/0.75).
- Six DOOM action scores and autonomous action selection.
- Membrane/spike state persistence across game decisions and reset between episodes.
- Per-episode CSV (rewards, kills, decisions, mean hidden spike activity).
- Smoke test that checks surrogate-gradient flow to recurrent weights without ViZDoom.
- **Experimental** episode-based REINFORCE training mode (not DQN; convergence not assured).

This code uses **pure PyTorch** for the recurrent spiking neurons, so `snntorch` is not necessary. It does **not** implement the FPGA's 16-bit fixed-point arithmetic, confirmed synaptic mapping, or verified hardware action readout. The action output is analog and must be adapted to hardware if necessary.

## Install (same environment as your current ViZDoom project)

```bash
pip install torch numpy pillow vizdoom
```

## 1. Offline proof of RSNN forward and backward

Put your `doom_frame.png` beside the script (or pass its path):

```bash
python rsnn_doom.py --mode smoke --image doom_frame.png
```

This works without ViZDoom installed and reports recurrent-weight gradients.

## 2. Run the untrained RSNN inside DOOM

```bash
python rsnn_doom.py --mode demo --visible --episodes 3
```

If using your cloned repo instead of the installed scenarios:

```bash
python rsnn_doom.py --mode demo --config ../ViZDoom/scenarios/basic.cfg --visible --episodes 3
```

The generated actions are from an **untrained randomly initialized RSNN** and are not evidence of learning.

## 3. Optional: train a first learning baseline

```bash
python rsnn_doom.py --mode train --episodes 500
```

This mode implements exploratory REINFORCE with truncated backpropagation through time. It is an early training pipeline, not an optimized or proven method for ViZDoom BASIC. Check the learning curve before declaring success. Checkpoints are saved under `checkpoints/rsnn.pt`.

## 4. Evaluate an existing checkpoint

```bash
python rsnn_doom.py --mode evaluate --episodes 50 --checkpoint checkpoints/rsnn.pt
```

## What to report honestly

> A first recurrent SNN architecture and ViZDoom inference loop have been implemented. It uses 96 recurrent LIF neurons, beta 0.5, and 8 spiking timesteps per action, with 32×24 grayscale observations. The offline forward/backward smoke test verifies that feedback weights receive gradients. ViZDoom execution and gameplay learning still need testing on the laptop; 16-bit fixed-point/FPGA mapping is not verified.

## Next

1. Run the offline smoke test, then the ViZDoom integration test on the actual laptop.
2. Train/evaluate at least 50 episodes and compare against `random_baseline.py` using the same BASIC scenario and frame skip.
3. Track mean reward and kill rate. Investigate an ANN teacher if direct policy-gradient training does not improve.
4. Verify the exact SNAP-V recurrent connectivity limits and 16-bit fixed-point format before hardware claims.
