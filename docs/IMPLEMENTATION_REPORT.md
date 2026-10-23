# Implementation and local verification report

Date: 21 October 2026.

## Created components

- `ConvRSNN`: custom convolutional + recurrent LIF neuron model with explicit state, actor, critic, and temporal replay.
- `DoomBasic`: ViZDoom BASIC adapter for consistent RGB observations and six-action mapping; never falls back to a dummy environment.
- `BasicTeacher`: teacher-only object-label alignment policy.
- `dataset.py`: per-episode trajectory serialization and integrity checks.
- `train.py`: sequential imitation learning with truncation, student-driven DAgger collection, sequence-correct recurrent A2C/PPO rollout updates, GAE.
- `evaluate.py`: deterministic/stochastic real ViZDoom evaluation plus random and fixed-action baselines.
- `cli.py`: inspect, collect, BC, DAgger, RL, evaluation, MP4 recording, synthetic smoke.
- `scripts/setup_ampere.sh`: environment setup and safe validation workflow.

## Verified in the local container

- 25 automated tests passed, using small synthetic networks and fake/mock environment objects only.
- Synthetic model forward/backward smoke passed (all 13 trainable tensors had finite gradients).
- Syntax and CLI help verified.

## NOT verified here

- ViZDoom gameplay integration: ViZDoom was not installed in this container.
- Teacher gameplay success, screenshot shape/label availability for installed ViZDoom version.
- GPU performance, university server login, scheduler, or CUDA configuration.
- Convergence of behavioral cloning, DAgger, or PPO.
- Numerical fixed-point equivalence, SNAP-V model mapping or performance.

## Required before model training

Run on Ampere: tests, smoke, inspect-env, teacher 2-episode pilot, inspect-data, and then review teacher behavior and game configuration. Only after that collect larger demonstrations and start training.

## Experiment design notes

Historical 96-neuron RSNN BASIC result (81.3% kill, +9.2 mean reward on 300 episodes) is not a result of this codebase. A new model should be judged with fixed game settings, fresh validation and holdout seeds, actual teacher efficacy, and comparison against random and fixed-action controls. The current model is deliberately larger than SNAP-V capacity, consistent with the requested unconstrained LIF-first development phase.
