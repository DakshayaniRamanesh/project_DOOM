# Convolutional LIF-RSNN for ViZDoom BASIC

**Standalone, code-review-ready implementation. No performance is claimed for this new model; full training was not run.**

The agent uses RGB images → three convolutional LIF layers → one recurrent LIF layer → a six-action actor. A critic is present for reinforcement-learning updates. The architecture has no hidden mutable recurrent state; the caller explicitly stores and returns state across game decisions. It supports a teacher using ViZDoom object labels **only for creating demonstration targets**, behavioral cloning, DAgger with student-driven gameplay, and sequence-correct PPO/A2C.

## Setup (Ampere)

Copy/extract the project to `/new-home/e22/e22060/doom_rsnn_ampere` using `scp` from your own Windows machine, then SSH into `e22060@ampere` and run:

```bash
cd /new-home/e22/e22060/doom_rsnn_ampere
nvidia-smi
python3 --version
bash scripts/setup_ampere.sh
source .venv/bin/activate
```

Note: login nodes may disallow GPU jobs; use your university's approved interactive job scheduler/compute-node method. `nvidia-smi` on a login node is not proof GPU training is permitted there. If PyTorch is CPU-only, install a build compatible with the university's CUDA drivers following site instructions, then rerun tests.

The installed ViZDoom package usually includes `scenarios/basic.cfg`. If it is not found, pass `--scenario /path/to/basic.cfg` before the subcommand.

## Smoke checks (safe before training)

```bash
python -m pytest -q
python -m doomrsnn.cli --config configs/base.yaml smoke
python -m doomrsnn.cli --config configs/base.yaml inspect-env
```

The first two commands run without ViZDoom. The last requires installed ViZDoom and game assets. **Do not run full training until the review gate passes.**

## Commands available after review

```bash
# 1. Inspect teacher object detection without collecting data
python -m doomrsnn.cli --config configs/base.yaml inspect-env

# 2. Small real-environment demonstration test (no model learning)
python -m doomrsnn.cli --config configs/base.yaml collect --episodes 2 --output data/teacher_smoke

# 3. Later: collect a sufficient teacher dataset (requires explicit approval)
python -m doomrsnn.cli --config configs/base.yaml collect --episodes 1000 --output data/teacher

# 4. Inspect dataset
python -m doomrsnn.cli --config configs/base.yaml inspect-data --data data/teacher

# 5. Behavioral cloning; correct temporal replay, state persistence
python -m doomrsnn.cli --config configs/base.yaml bc --data data/teacher --epochs 30 --output checkpoints/bc_best.pt

# 6. DAgger: student drives; teacher only labels visited states
python -m doomrsnn.cli --config configs/base.yaml dagger --checkpoint checkpoints/bc_best.pt --teacher-data data/teacher --iterations 3 --episodes-per-iteration 50

# 7. PPO fine-tuning from BC checkpoint
python -m doomrsnn.cli --config configs/base.yaml rl --checkpoint checkpoints/bc_best.pt --episodes 500 --output checkpoints/rl_latest.pt

# 8. Evaluate real gameplay on held-out seeds (no dummy fallback)
python -m doomrsnn.cli --config configs/base.yaml evaluate --checkpoint checkpoints/bc_best.pt --episodes 300 --seed 900001

# 9. Export gameplay video after training
python -m doomrsnn.cli --config configs/base.yaml record --checkpoint checkpoints/bc_best.pt --episodes 3 --output runs/demo.mp4
```

**No command silently resumes an earlier run**. `rl` starts a new optimizer after loading the selected model weights; optimizer-state resume is an independent, future extension. The provided old 96-neuron `best_500ep.pt` is *not* compatible with the new convolutional architecture and must be evaluated in its original codebase.

## Architecture and state correctness

- All convolutions are followed by binary LIF spiking computations.
- Each observation is processed over `T=8` internal neural steps; this is distinct from the four game tics of action repetition.
- Membrane/spike states for all three convolutional layers and the recurrent layer carry between game decisions.
- All states are reset on game episode boundaries.
- Model `forward_step(image,state)` is pure with respect to recurrent state; the caller receives a new state rather than modifying one held inside the model.
- On-policy rollouts are collected without gradients, saving the initial state. Optimization replays the **ordered time sequence** with gradients from that state. The bootstrap value is computed from a temporary successor state without modifying the real state.
- DAgger records both the action actually executed by the student and the action recommended by the teacher from the same observation.
- The evaluation refuses to evaluate a student without an explicit checkpoint. No dummy environment fallback exists.

## Review/limitations before high-performance claims

1. The actual ViZDoom scenario, labels API, GPU environment and full end-to-end episodes **cannot be integration-tested in this environment** because `vizdoom` is unavailable here; run `inspect-env` and two teacher episodes on Ampere before proceeding.
2. The teacher is a simple bounding-box alignment controller and may need adjustment after visually inspecting actual labels and action directions. Teacher success is **not established**.
3. The new model has not been trained. There are no new kill-rate results or trained weights.
4. The original 81.3% BASIC number is a separate historical baseline using different files; don't merge scores without matching environment settings.
5. The three convolutional LIF layers have far more physical spiking units than the older 96-neuron flat RSNN. FPGA compression/mapping is future work.
6. PPO processes ordered trajectories in one sequence per optimization epoch. It is correct as a single-environment sequence replay design, but may be inefficient; recurrent minibatching and multiple environments can be implemented later.
7. Greedy validation uses a fixed development seed family; reserve fresh final-holdout seeds and repeat training across multiple initialization seeds.
8. ViZDoom reward/timeout semantics and engine seeding must be validated against the exact scenario version installed on Ampere.

## References

- https://github.com/imaadh-ifthi/doom — team research and baseline framework; this standalone implementation does **not** overwrite the repository.
- https://vizdoom.farama.org/ — ViZDoom documentation.
- Prior local 96-neuron RSNN model and 500-episode checkpoint are external baseline artifacts and are not bundled here.
