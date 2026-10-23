# PeraMorphIQ — RSNN v2 Corrected Training, Evaluation & Stability Diagnostics

This document details the diagnosis of the Episode-10 greedy action collapse and the stability improvements added to the RSNN v2 Advantage Actor-Critic (A2C + GAE) implementation.

---

## 1. Root Cause of Episode-10 Greedy "Noop" Collapse

### Empirical Diagnosis
- **The policy did NOT collapse to 100% `noop`:**
  - Softmax probability of `noop`: ~36%–43% (NOT 99% or 100%).
  - Softmax probability of shooting actions combined (`shoot`, `left_shoot`, `right_shoot`): ~50%–53%.
  - Policy entropy: $1.45$ to $1.56$ (healthy, well above 0; theoretical max for 6 actions is $\ln(6) \approx 1.7918$).
  - Logit margin between top action (`noop`) and 2nd action (`right_shoot`): only $\approx 0.30$ to $0.47$.
- **Greedy Mode Absorbing Loop:**
  - In ViZDoom BASIC, executing `noop` keeps the agent motionless and silent.
  - The screen buffer remains unchanged, and the recurrent LIF membrane potentials settle into an attractor state where `noop` continues to hold the highest logit.
  - In **greedy evaluation** (`argmax`), `noop` is selected at step 1 $\to$ screen stays identical $\to$ step 2 picks `noop` $\to$ absorbing loop for all 75 steps $\to$ timeout (0% kill rate, -300 reward).
  - In **stochastic evaluation** (`sample(Categorical)`), the agent samples from its 53% shooting probability and achieves an **80.0% kill rate**!
- **Unscaled Rewards & Critic Gradient Dominance:**
  - Raw ViZDoom rewards of $+95$ and $-395$ produce mean squared error targets of $(395)^2 \approx 156,000$ and $(95)^2 \approx 9,000$.
  - In baseline A2C, critic gradients on the shared RSNN exceeded actor gradients by up to **3.3x**, driving total preclip gradient norms to **677.2**.
  - Consecutive timeout episodes (Episodes 8 and 9) penalized frequently executed movement/shooting actions, allowing `noop` to edge ahead by a tiny 0.3 logit margin.

---

## 2. Stability Fixes Implemented

1. **Dual-Mode Evaluation with Network Probability Diagnostics (`evaluate_rsnn_v2.py`):**
   - Direct reporting of true softmax probabilities, logits, entropy, and margins.
   - Side-by-side `--eval-mode both` (or `--eval-mode greedy` / `--eval-mode stochastic`).
   - Optional bounded per-decision logging (`--log-decisions`).
2. **Reward Scaling (`--reward-scale 0.01`):**
   - Scales GAE targets and advantages consistently ($r_{\text{scaled}} = r \times 0.01$).
   - Reduces value loss from $\sim 9,000$ down to $\sim 1.0$.
   - Reduces pre-clip gradient norm from $677.2$ to $< 10.0$.
   - Preserves original unscaled rewards in evaluation logs and DOOM game scores.
3. **Huber / Smooth L1 Critic Loss (`--critic-loss huber`):**
   - Replaces quadratic MSE with linear penalty on large value errors ($|V - y| > 1$).
   - Achieves near-perfect $1:1$ balance between actor and critic shared gradient norms.
4. **Enhanced Multi-Seed Validation:**
   - Evaluates across 4 distinct validation seeds (`1101, 1102, 1103, 1104`), 5 episodes each (20 episodes total).
   - Selection metric: Primary kill rate, secondary mean reward, tertiary minimum decisions.

---

## 3. Verification & Benchmark Commands

### Run Full Test Suite (79 Tests)
```powershell
$env:OPENBLAS_NUM_THREADS=1; python -m pytest tests/ -v
```

### Run Dual-Mode Evaluation (Greedy vs Stochastic Comparison)
```powershell
$env:OPENBLAS_NUM_THREADS=1; python evaluate_rsnn_v2.py `
    --config ../ViZDoom/scenarios/basic.cfg `
    --v2-checkpoint checkpoints/v2/best_ep5_backup.pt `
    --compare-checkpoint checkpoints/v2/latest_ep10_backup.pt `
    --eval-mode both `
    --episodes 30 `
    --seeds 5101 5102 5103 `
    --log-decisions --max-logged-episodes 5
```

### Stable Training Run (Recommended Hyperparameters)
```powershell
$env:OPENBLAS_NUM_THREADS=1; python train_rsnn_v2.py `
    --episodes 500 `
    --reward-scale 0.01 `
    --critic-loss huber `
    --lr 0.0003 `
    --value-coef 0.5 `
    --entropy-coef 0.01 `
    --rollout-steps 8 `
    --max-grad-norm 0.5 `
    --validate-every 25 `
    --val-episodes 5 `
    --val-seeds 1101 1102 1103 1104 `
    --seed 42 `
    --config ../ViZDoom/scenarios/basic.cfg `
    --diagnose-grads
```
