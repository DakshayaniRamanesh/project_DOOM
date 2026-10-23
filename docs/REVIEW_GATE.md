# Review gate — before substantial training

1. `python -m pytest -q` completes.
2. `python -m doomrsnn.cli --config configs/base.yaml smoke` succeeds.
3. `inspect-env` reports real ViZDoom, RGB buffer shape, target labels (especially `DoomImp`), and plausible teacher action.
4. A two-episode teacher collection run produces consistent trajectories and correct action labels, even if kills are not always successful.
5. Check actual BASIC `basic.cfg`: available buttons, reward values, episode timeout and version match intended benchmark. Runtime code overrides buttons and timeout, but the reward values still come from the configuration.
6. Check model state persistence and gradient values from tests; confirm RL replay uses `[time, batch, C, H, W]`.
7. Inspect a handful of saved demonstration frames with teacher labels to ensure LEFT and RIGHT corrections are not reversed.
8. Confirm GPU access with `nvidia-smi` on an allocated compute node; obtain approval for intensive training.
9. Collect and validate a moderate pilot teacher dataset before attempting 1,000 demos.
10. Measure teacher hit rate on validation seeds, then proceed to BC → DAgger → RL one phase at a time.

## Known unverified integration points

- Exact `vizdoom.GameVariable.KILLCOUNT` behavior at episode completion.
- Object-label names and screen buffer width during teacher inference.
- ViZDoom package version and scenario asset resolution on the university server.
- University's permitted GPU training allocation and job scheduling workflow.

Full training has not been run and no gameplay success rate is claimed.
