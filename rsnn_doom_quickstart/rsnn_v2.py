"""RSNN v2 — Shared Actor-Critic Recurrent Spiking Neural Network.

Architecture
------------
Observation (32×24 grayscale)
  -> Spike encoding (T=8 deterministic rate)
  -> 96 recurrent LIF neurons  [shared feature extractor]
        |                  |
   Actor head          Critic head
   (6 logits)          (scalar V(s))

Key differences from v1 (rsnn_doom.py):
  - Separate actor and critic linear heads sharing the recurrent LIF layer.
  - forward() returns (logits, value, state, spike_rate).
  - The model tracks a hardware-oriented resource count at construction.
  - Entropy is easily computable from logits for A2C regularisation.

Hardware notes (SNAP-V / PeraMorphIQ):
  - Beta must be in {0.25, 0.5, 0.75}.
  - Neuron budget is tracked but not enforced here.
  - Critic head is for software training; it may be stripped for hardware inference.
  - No INT8/fixed-point conversion is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import pathlib
import random
import numpy as np

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Surrogate-gradient spike function (same formulation as v1)
# ---------------------------------------------------------------------------

class Spike(torch.autograd.Function):
    """Binary spike with fast-sigmoid surrogate gradient for BPTT."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Derivative of fast sigmoid: 1 / (1 + 2|x|)^2
        return grad_output / (1.0 + 2.0 * x.abs()).square()


# ---------------------------------------------------------------------------
# Architecture configuration
# ---------------------------------------------------------------------------

@dataclass
class RSNNv2Config:
    """All hyperparameters needed to reconstruct the model."""
    inputs: int = 768          # 32 x 24
    hidden: int = 96
    actions: int = 6
    beta: float = 0.5          # LIF decay — must be in {0.25, 0.5, 0.75}
    threshold: float = 0.4
    timesteps: int = 8

    def __post_init__(self):
        if self.beta not in (0.25, 0.5, 0.75):
            raise ValueError(f"beta must be 0.25, 0.5 or 0.75; got {self.beta}")
        if self.timesteps < 1:
            raise ValueError("timesteps must be >= 1")
        if self.hidden < 1 or self.inputs < 1 or self.actions < 1:
            raise ValueError("hidden, inputs, actions must be positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RSNNv2Config":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# RSNN v2 Actor-Critic module
# ---------------------------------------------------------------------------

class RSNNv2(nn.Module):
    """Recurrent Spiking Neural Network with shared actor and critic heads.

    State convention
    ----------------
    state = (spikes, membrane)  — both shape [batch, hidden]

    Call reset_state() at the start of each episode.
    Call detach_state() at rollout boundaries (not at every step).

    Forward returns
    ---------------
    logits     : [batch, actions]  — raw action scores (unnormalised log-probs)
    value      : [batch]           — critic V(s) estimate
    state      : (spikes, mem)     — updated recurrent state
    spike_rate : scalar tensor     — mean hidden activity [0, 1], detached
    """

    def __init__(self, cfg: Optional[RSNNv2Config] = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = RSNNv2Config(**kwargs)
        self.cfg = cfg

        # ---- Shared recurrent LIF layer ----
        self.input_fc = nn.Linear(cfg.inputs, cfg.hidden)
        self.recurrent_fc = nn.Linear(cfg.hidden, cfg.hidden, bias=False)

        # ---- Actor head (action logits) ----
        self.actor_fc = nn.Linear(cfg.hidden, cfg.actions)

        # ---- Critic head (state value V(s)) ----
        self.critic_fc = nn.Linear(cfg.hidden, 1)

        # Weight initialisation
        nn.init.xavier_uniform_(self.input_fc.weight)
        nn.init.zeros_(self.input_fc.bias)
        nn.init.normal_(self.recurrent_fc.weight, mean=0.0, std=0.06)
        nn.init.xavier_uniform_(self.actor_fc.weight)
        nn.init.zeros_(self.actor_fc.bias)
        nn.init.xavier_uniform_(self.critic_fc.weight)
        nn.init.zeros_(self.critic_fc.bias)

        # ---- Resource accounting (for SNAP-V budget tracking) ----
        self._resource_summary = self._count_resources()

    # ------------------------------------------------------------------
    # Resource accounting
    # ------------------------------------------------------------------

    def _count_resources(self) -> dict:
        c = self.cfg
        return {
            "spiking_neurons": c.hidden,
            "input_synapses": c.inputs * c.hidden,
            "recurrent_synapses": c.hidden * c.hidden,
            "actor_synapses": c.hidden * c.actions,
            "critic_synapses": c.hidden * 1,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "hardware_parameters": (
                c.inputs * c.hidden + c.hidden      # input_fc weight + bias
                + c.hidden * c.hidden               # recurrent_fc weight
                + c.hidden * c.actions + c.actions  # actor weight + bias
            ),
            "snn_timesteps_per_decision": c.timesteps,
            "note": (
                "Critic is software-only. Hardware uses actor + LIF layer. "
                "16-bit fixed-point mapping NOT verified."
            ),
        }

    def resource_summary(self) -> dict:
        return dict(self._resource_summary)

    def print_resources(self):
        r = self.resource_summary()
        print("=" * 56)
        print("  RSNN v2 Hardware Resource Summary")
        print("=" * 56)
        for k, v in r.items():
            print(f"  {k:<32}: {v}")
        print("=" * 56)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zeroed (spikes, membrane) tensors."""
        if device is None:
            device = self.input_fc.weight.device
        z = torch.zeros(batch_size, self.cfg.hidden, device=device)
        return z.clone(), z.clone()

    @staticmethod
    def detach_state(
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detach recurrent state at rollout boundaries to bound graph memory.

        Call this at the end of a rollout segment, NOT after every single
        game decision, so that gradients can flow through multiple decisions
        within a rollout.
        """
        return state[0].detach(), state[1].detach()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        observation: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """One DOOM decision: T LIF timesteps over a single observation.

        Parameters
        ----------
        observation : [batch, inputs]  — pixel values in [0, 1]
        state       : (spikes, membrane) from previous decision, or None

        Returns
        -------
        logits     : [batch, actions]
        value      : [batch]
        state      : (spikes, membrane)
        spike_rate : scalar — mean hidden spike activity (detached)
        """
        cfg = self.cfg
        if observation.ndim != 2 or observation.shape[1] != cfg.inputs:
            raise ValueError(
                f"Expected observation [batch, {cfg.inputs}], "
                f"got {tuple(observation.shape)}"
            )
        if state is None:
            state = self.reset_state(observation.shape[0], observation.device)

        spikes, mem = state
        pixel_acc = torch.zeros_like(observation)
        actor_acc = torch.zeros(observation.shape[0], cfg.actions,
                                device=observation.device)
        critic_acc = torch.zeros(observation.shape[0], 1,
                                 device=observation.device)
        total_spikes = torch.zeros((), device=observation.device)

        for _ in range(cfg.timesteps):
            # Deterministic rate encoding: brighter pixels spike sooner
            pixel_acc = pixel_acc + observation
            input_spikes = (pixel_acc >= 1.0).to(observation.dtype)
            pixel_acc = pixel_acc - input_spikes

            # LIF membrane dynamics with recurrent spike feedback
            current = self.input_fc(input_spikes) + self.recurrent_fc(spikes)
            mem = cfg.beta * mem + current - cfg.threshold * spikes
            spikes = Spike.apply(mem - cfg.threshold)

            # Accumulate head outputs (time-averaged)
            actor_acc = actor_acc + self.actor_fc(spikes)
            critic_acc = critic_acc + self.critic_fc(spikes)
            total_spikes = total_spikes + spikes.detach().sum()

        logits = actor_acc / cfg.timesteps
        value = (critic_acc / cfg.timesteps).squeeze(-1)     # [batch]
        spike_rate = total_spikes / (
            cfg.timesteps * observation.shape[0] * cfg.hidden
        )

        return logits, value, (spikes, mem), spike_rate


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path,
    model: RSNNv2,
    optimizer: torch.optim.Optimizer,
    episode: int,
    best_reward: float,
    extra: Optional[dict] = None,
):
    """Save model, optimizer and Python/PyTorch/NumPy RNG state.

    ViZDoom's internal RNG is not serialized. Resume at an episode boundary,
    but do not claim that a resumed run is bitwise identical to uninterrupted training.
    """
    payload = {
        "version": "rsnn_v2",
        "config": model.cfg.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
        "best_reward": best_reward,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": {
            "kind": np.random.get_state()[0],
            "keys": np.random.get_state()[1].tolist(),
            "position": np.random.get_state()[2],
            "has_gauss": np.random.get_state()[3],
            "gaussian": np.random.get_state()[4],
        },
    }
    if extra:
        payload.update(extra)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu") -> dict:
    """Load a v2 checkpoint dict. Returns raw dict for flexible use."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("version") != "rsnn_v2":
        raise ValueError(
            f"Checkpoint at {path} is not a v2 checkpoint "
            f"(version={ckpt.get('version')!r}). "
            "Use rsnn_doom.py to load v1 checkpoints."
        )
    return ckpt


def build_model_from_checkpoint(
    path, device="cpu"
) -> Tuple[RSNNv2, dict]:
    """Convenience: load checkpoint and reconstruct model with weights."""
    ckpt = load_checkpoint(path, device)
    cfg = RSNNv2Config.from_dict(ckpt["config"])
    model = RSNNv2(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def restore_rng_from_checkpoint(ckpt: dict) -> None:
    """Restore process RNGs at episode boundaries when states are available."""
    if "torch_rng" in ckpt:
        torch.set_rng_state(ckpt["torch_rng"])
    if "python_rng" in ckpt:
        random.setstate(ckpt["python_rng"])
    if "numpy_rng" in ckpt:
        d = ckpt["numpy_rng"]
        np.random.set_state((d["kind"], np.asarray(d["keys"], dtype=np.uint32),
                             int(d["position"]), int(d["has_gauss"]), float(d["gaussian"])))
