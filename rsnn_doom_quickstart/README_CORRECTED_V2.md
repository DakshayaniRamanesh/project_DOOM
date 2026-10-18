# PeraMorphIQ — RSNN v2 Corrected Training and Evaluation

This patch replaces the four uploaded **v2** Python files and supplies offline
regression tests. It does **not** change the RSNN v1 model architecture, v1
checkpoint, or any earlier training/evaluation CSVs.

## What was fixed

1. **Random policy** samples uniformly across all six actions even while the
   evaluation pipeline uses greedy action choice for neural policies.
2. **Evaluation isolation:** ViZDoom is newly constructed with the specified
   seed for each policy, rather than reusing the same game across policies.
3. **Entropy:** Uses `Categorical.entropy()` tensors from rollout collection;
   gradients correctly flow into action logits.
4. **A2C/GAE:** Uses detached GAE targets with gamma=0.99 and lambda=0.95;
   within-episode advantage normalization is **off** by default.
5. **Truncated BPTT:** Trainable recurrent states are retained across at most
   `--rollout-steps` successive DOOM decisions and detached after each update.
   Neural state **continues** within the same DOOM episode.
6. **Termination:** True episode ends and time-limit / manual truncations are
   recorded separately. For engine timeouts, ViZDoom does not expose a further
   legal observation, so bootstrap is set to zero at the finite episode end.
   At intermediate segment boundaries or user-imposed max-decision cutoffs,
   critic bootstrapping uses the *next observed screen*.
7. **Resume:** Constructs the model from the checkpoint config **before**
   constructing/loading its optimizer. Saves PyTorch, Python and NumPy RNG states.
   ViZDoom's internal RNG is **not** saved, so resumed gameplay is not bitwise
   identical to an uninterrupted run.
8. **Best checkpoint:** Selected on dedicated validation seeds by kill rate,
   then mean reward as a tie breaker. Never on a single training episode reward.
9. **Safety:** `latest.pt` is saved after each *completed* training episode.
   Interrupting mid-episode leaves the last complete checkpoint unchanged.
10. **Resources:** Current model has 96 hidden LIF neurons, 83,622 actor/core
    weights+biases, and 83,719 total actor-critic trainable parameters.

> Important: This is CPU/PyTorch floating-point simulation. The hardware map,
> true neuron/synapse resource consumption on SNAP-V and 16-bit fixed-point
> accuracy are not verified. No real-game improvement is claimed by this patch.

## Installation on Windows (PowerShell)

Run inside your `D:\PROJECT-DOOM\rsnn_doom_quickstart` folder.

**Before extracting:** back up `train_rsnn_v2.py`, `doom_env.py`,
`evaluate_rsnn_v2.py`, and `rsnn_v2.py` if you want to keep the originals.
Also back up the previous v2 checkpoint because a new v2 training run writes
`checkpoints\v2\latest.pt` and `checkpoints\v2\best.pt`.

```powershell
New-Item -ItemType Directory -Force checkpoints\v2_old | Out-Null
Copy-Item checkpoints\v2\*.pt checkpoints\v2_old\ -ErrorAction SilentlyContinue
```

Extract the ZIP's contents directly into `rsnn_doom_quickstart`, replacing only
its four v2 Python files. **Do not overwrite or delete `rsnn_doom.py` (v1).**
The ZIP includes no v1 model or checkpoint.

Install dependencies if needed:

```powershell
python -m pip install torch numpy pillow vizdoom pytest
```

## Test before any training

```powershell
python -m pytest -q tests/test_rsnn_v2_corrected.py
python train_rsnn_v2.py --mode smoke
```

The first tests require no ViZDoom package. The smoke test also needs no game.

## 10-episode ViZDoom integration run (new model)

```powershell
python train_rsnn_v2.py --episodes 10 --rollout-steps 8 --validate-every 5 --val-episodes 2 --val-seeds 1101 1102 --seed 42 --config ../ViZDoom/scenarios/basic.cfg
```

Inspect `results/v2/train_corrected_*.csv` and confirm that loss, entropy,
recurrent spike activity and action counts are finite and recorded.

## Train from scratch (only after the short integration test)

```powershell
python train_rsnn_v2.py --episodes 500 --rollout-steps 8 --validate-every 50 --val-episodes 10 --val-seeds 1101 1102 --seed 42 --config ../ViZDoom/scenarios/basic.cfg
```

`--episodes` is the number of episodes **to run in this invocation**. On resume
it is the number of **additional** episodes, not an absolute final index.

### Resume

```powershell
python train_rsnn_v2.py --episodes 100 --resume checkpoints/v2/latest.pt --hidden 96 --beta 0.5 --timesteps 8 --config ../ViZDoom/scenarios/basic.cfg
```

Set `--hidden`, `--beta`, `--timesteps` to the model config used in the checkpoint.
The script rejects incompatible checkpoint/model configurations rather than
silently creating an optimizer for the wrong model.

## Compare policies after training

Use test seeds that were NOT used for training or validation and, where
possible, were not inspected to tune the model. The previously inspected
seeds `43`, `100`, `200`, `300`, `400` are not a genuinely untouched test set.

```powershell
python evaluate_rsnn_v2.py --config ../ViZDoom/scenarios/basic.cfg --v1-checkpoint checkpoints/rsnn.pt --v2-checkpoint checkpoints/v2/best.pt --episodes 100 --seeds 3001 3002 3003
```

This compares random, always shoot, always left+shoot, v1 and v2. The random
agent should select *all* six actions. If it reports 100% `noop`, the wrong
old evaluator is being executed.

Report both kill success and mean reward, plus policy action frequencies.
The baseline policies on BASIC are already strong, so a spiking network that
mostly repeats one action is not automatically an adaptive agent.

## Files included

- `rsnn_v2.py`: recurrent spiking actor-critic core and safe checkpoint helpers.
- `doom_env.py`: game setup, preprocessing, bounded rollouts and inference.
- `train_rsnn_v2.py`: corrected n-step A2C/GAE and validation checkpointing.
- `evaluate_rsnn_v2.py`: corrected random and policy/seed isolation.
- `tests/test_rsnn_v2_corrected.py`: 22 offline tests, including mocked training,
  checkpoint resume, GAE, gradient checks and mock evaluation.
- `requirements_v2.txt`: dependency names (not a lockfile).

## What was actually verified before delivery

- Python syntax compilation on all modified modules.
- `pytest`: 22 passed, using mocked environments where necessary.
- `python train_rsnn_v2.py --mode smoke`: passed.
- **Actual ViZDoom gameplay/training:** NOT run in the preparation environment;
  `vizdoom` was not installed there. You must run the short integration test
  on your Windows machine before claiming the corrected model has trained.

This project still uses continuous actor/critic output heads. The critic is a
training component and can be removed for inference, but hardware compatibility
requires independent validation of encoding, state, synapses, quantization,
throughput, and 16-bit fixed-point accuracy.
