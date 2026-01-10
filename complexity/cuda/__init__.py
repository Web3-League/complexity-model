"""
CUDA/Triton accelerated kernels for Complexity.

Provides CGGR (Coalesced Grouped Gemm with Ragged) optimization
for Token-Routed MLP.
"""

from .triton_token_routed import (
    TokenRoutedMLPTriton,
    sort_tokens_by_expert,
    HAS_TRITON,
)

__all__ = [
    "TokenRoutedMLPTriton",
    "sort_tokens_by_expert",
    "HAS_TRITON",
]
