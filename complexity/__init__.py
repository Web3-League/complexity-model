"""
Complexity - A Llama-based Architecture
=======================================

Custom transformer architecture optimized for the Complexity-4 tokenizer.

Usage:
    from complexity import ComplexityConfig, ComplexityForCausalLM

    config = ComplexityConfig.complexity_base()
    model = ComplexityForCausalLM(config)
"""

from complexity.core import (
    RMSNorm,
    RotaryEmbedding,
    ComplexityAttention,
    ComplexityMLP,
    ComplexityDecoderLayer,
)

from complexity.models import (
    ComplexityConfig,
    ComplexityModel,
    ComplexityForCausalLM,
    create_complexity_model,
)

__version__ = "0.8.3"
__all__ = [
    # Core
    "RMSNorm",
    "RotaryEmbedding",
    "ComplexityAttention",
    "ComplexityMLP",
    "ComplexityDecoderLayer",
    # Models
    "ComplexityConfig",
    "ComplexityModel",
    "ComplexityForCausalLM",
    "create_complexity_model",
]
