"""
Velocity Dynamics for Complexity architecture.

A simplified PID-like controller inspired by INL Dynamics (complexity-deep).
Provides smooth hidden state trajectories with momentum-based updates.

Architecture (like complexity-deep):
```
Input
  │
  ▼
[Attention] ─► [VelocityDynamics] ─► [Residual] ─► [MLP] ─► [Residual]
                    │
               (h, v, mu) ──────────────────────────────► next layer
```

Key features:
- Velocity: momentum-based state that accumulates across layers
- Mu: contextual signal that can guide attention in next layer
- Simpler than INL but same flow pattern

Usage:
    dynamics = VelocityDynamics(hidden_size=768)

    # In forward pass (across layers):
    velocity, mu = None, None
    for layer in layers:
        hidden, velocity, mu = layer(hidden, velocity=velocity, mu_prev=mu)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class VelocityDynamics(nn.Module):
    """
    Simplified PID-like dynamics for transformers.

    Inspired by INL Dynamics but simpler:
    - Velocity + momentum (like SGD with momentum)
    - Mu contextual signal (propagates to next layer)
    - Learnable scale (replaces gate)

    Returns (output, velocity, mu) like INL Dynamics.

    Args:
        hidden_size: Dimension of hidden states
        momentum: Momentum coefficient (default: 0.9)
    """

    def __init__(
        self,
        hidden_size: int,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.momentum = momentum

        # Learnable scaling (replaces gate in INL)
        self.scale = nn.Parameter(torch.ones(hidden_size) * 0.1)

        # Equilibrium projection (mu-like)
        self.eq_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.eq_proj.weight)

        # Mu projection: creates contextual mu for next layer
        # This is the key INL innovation - mu propagates across layers
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply velocity dynamics.

        Args:
            x: [batch, seq, hidden] - current hidden state (from attention)
            velocity: [batch, seq, hidden] - previous velocity
            mu_prev: [batch, seq, hidden] - mu from previous layer

        Returns:
            output: [batch, seq, hidden] - updated hidden state
            new_velocity: [batch, seq, hidden] - velocity for next layer
            mu_next: [batch, seq, hidden] - mu for next layer
        """
        # Initialize on first layer
        if velocity is None:
            velocity = torch.zeros_like(x)
        if mu_prev is None:
            mu_prev = torch.zeros_like(x)

        # Equilibrium = mu_prev + learned deviation
        # This lets mu guide the equilibrium point
        eq = mu_prev + self.eq_proj(x)

        # Error = deviation from equilibrium (P term)
        error = x - eq

        # Velocity update: momentum * v - error
        new_velocity = self.momentum * velocity - error

        # Clamp velocity to prevent explosion at ~400K steps
        new_velocity = new_velocity.clamp(-10.0, 10.0)

        # Output: x + scaled velocity
        output = x + self.scale * new_velocity

        # Contextual mu for next layer
        # mu accumulates context: mu_prev + projection of current state
        mu_next = mu_prev + self.mu_proj(output)

        # Clamp mu to prevent accumulation explosion
        mu_next = mu_next.clamp(-10.0, 10.0)

        return output, new_velocity, mu_next

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, momentum={self.momentum}"


class VelocityDynamicsV2(nn.Module):
    """
    Enhanced velocity dynamics with contextual momentum and gating.

    Closer to full INL Dynamics:
    - Contextual momentum (adapts per token)
    - Gated output
    - Full mu propagation

    This is the "full" version for when you want INL-like behavior.
    """

    def __init__(
        self,
        hidden_size: int,
        base_momentum: float = 0.9,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.base_momentum = base_momentum

        # Contextual momentum
        self.momentum_proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.momentum_proj.weight)
        nn.init.constant_(self.momentum_proj.bias, base_momentum)

        # Equilibrium projection
        self.eq_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.eq_proj.weight)

        # Gate (like INL)
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.gate_proj.weight)

        # Scale
        self.scale = nn.Parameter(torch.ones(hidden_size) * 0.1)

        # Mu projection
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply enhanced velocity dynamics with mu propagation.
        """
        if velocity is None:
            velocity = torch.zeros_like(x)
        if mu_prev is None:
            mu_prev = torch.zeros_like(x)

        # Contextual momentum
        momentum = torch.sigmoid(self.momentum_proj(x))  # [B, S, 1]

        # Equilibrium guided by mu_prev
        eq = mu_prev + self.eq_proj(x)

        # Error
        error = x - eq

        # Velocity update
        new_velocity = momentum * velocity - error

        # Clamp velocity to prevent explosion at ~400K steps
        new_velocity = new_velocity.clamp(-10.0, 10.0)

        # Gated output
        gate = torch.sigmoid(self.gate_proj(x))
        output = x + gate * self.scale * new_velocity

        # Contextual mu for next layer
        mu_next = mu_prev + self.mu_proj(output)

        # Clamp mu to prevent accumulation explosion
        mu_next = mu_next.clamp(-10.0, 10.0)

        return output, new_velocity, mu_next

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, base_momentum={self.base_momentum}"
