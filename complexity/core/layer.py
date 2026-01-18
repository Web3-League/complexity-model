"""
Transformer decoder layer for Complexity architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from complexity.core.normalization import RMSNorm
from complexity.core.attention import ComplexityAttention
from complexity.core.mlp import ComplexityMLP
from complexity.core.token_routed_mlp import TokenRoutedMLP
from complexity.core.dynamics import VelocityDynamics, VelocityDynamicsV2

# Try to import Triton-accelerated version (5-6x faster)
try:
    from complexity.cuda.triton_token_routed import TokenRoutedMLPTriton, HAS_TRITON
except ImportError:
    TokenRoutedMLPTriton = None
    HAS_TRITON = False


class ComplexityDecoderLayer(nn.Module):
    """
    Single transformer decoder layer.

    Architecture (Pre-LN):
        x = x + Attention(RMSNorm(x))
        x = x + MLP(RMSNorm(x))

    With Token-Routed MLP:
        x = x + Attention(RMSNorm(x))
        x = x + TokenRoutedMLP(RMSNorm(x), token_ids)  # Route based on token
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        # Token-Routed MLP params
        use_token_routed_mlp: bool = True,
        num_experts: int = 4,
        vocab_size: int = 100000,
        # 2024 innovations
        use_qk_norm: bool = True,
        sliding_window: int = None,
        use_sdpa: bool = True,
        # Velocity Dynamics (INL-inspired)
        use_velocity_dynamics: bool = True,
        dynamics_momentum: float = 0.9,
        dynamics_version: str = "v1",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_token_routed_mlp = use_token_routed_mlp
        self.use_velocity_dynamics = use_velocity_dynamics

        # Attention with 2024 innovations (Flash Attention, QK Norm, Sliding Window)
        self.self_attn = ComplexityAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            use_qk_norm=use_qk_norm,
            sliding_window=sliding_window,
            use_sdpa=use_sdpa,
        )

        # MLP - Token-Routed (Triton or PyTorch) or Standard
        if use_token_routed_mlp:
            # Use Triton-accelerated version if available (5-6x faster)
            if HAS_TRITON and TokenRoutedMLPTriton is not None:
                self.mlp = TokenRoutedMLPTriton(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    vocab_size=vocab_size,
                    hidden_act=hidden_act,
                    use_cggr=True,  # Enable CGGR optimization
                )
            else:
                self.mlp = TokenRoutedMLP(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_experts=num_experts,
                    vocab_size=vocab_size,
                    hidden_act=hidden_act,
                )
        else:
            self.mlp = ComplexityMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
            )

        # Layer norms (Pre-LN architecture)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Velocity Dynamics (INL-inspired PID control)
        if use_velocity_dynamics:
            if dynamics_version == "v2":
                self.dynamics = VelocityDynamicsV2(
                    hidden_size=hidden_size,
                    base_momentum=dynamics_momentum,
                )
            else:
                self.dynamics = VelocityDynamics(
                    hidden_size=hidden_size,
                    momentum=dynamics_momentum,
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        token_ids: Optional[torch.Tensor] = None,  # For Token-Routed MLP
        velocity: Optional[torch.Tensor] = None,   # For Velocity Dynamics
        mu_prev: Optional[torch.Tensor] = None,    # For Mu propagation (like INL)
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            past_key_value: Optional cached KV
            use_cache: Whether to return KV cache
            token_ids: [batch, seq_len] - for Token-Routed MLP routing
            velocity: [batch, seq_len, hidden_size] - for Velocity Dynamics
            mu_prev: [batch, seq_len, hidden_size] - mu from previous layer

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            past_key_value: Optional updated KV cache
            new_velocity: Optional velocity for next layer
            mu_next: Optional mu for next layer
        """
        # Self-attention with residual
        # INL 2025: Pass mu_prev to attention for mu-guided KQV
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_past_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            mu_prev=mu_prev,  # INL: mu guides K, Q, V
        )

        # Apply Velocity Dynamics after attention (before residual)
        # Returns (output, velocity, mu) like INL Dynamics
        new_velocity = None
        mu_next = None
        if self.use_velocity_dynamics:
            hidden_states, new_velocity, mu_next = self.dynamics(hidden_states, velocity, mu_prev)

        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Token-Routed MLP or Standard MLP
        # INL 2025: Pass mu to Token-Routed MLP for mu-guided expert routing
        if self.use_token_routed_mlp:
            # Use mu_next from dynamics (or mu_prev if dynamics disabled)
            mu_current = mu_next if mu_next is not None else mu_prev
            hidden_states = self.mlp(hidden_states, token_ids=token_ids, mu=mu_current)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, new_past_key_value, new_velocity, mu_next
