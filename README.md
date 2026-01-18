# Complexity

A modern transformer architecture with **2024 optimizations**, **Token-Routed MLP**, and **Simplified PID**.

[![PyPI version](https://badge.fury.io/py/complexity.svg)](https://badge.fury.io/py/complexity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install complexity
```

## Innovations

### 1. Token-Routed MLP (Original)
Routes tokens to specialized experts based on token ID (modulo for uniform distribution):

```python
expert_id = token_id % num_experts  # Each expert gets ~25% of tokens
```

**INL 2025**: Mu-guided expert routing - mu from dynamics can override expert selection:
```python
# Base routing: deterministic by token ID
base_expert_ids = token_to_expert[token_ids]

# Mu override: mu can shift expert selection
mu_logits = mu_router(mu)
combined_logits = base_one_hot * 10.0 + mu_logits
expert_ids = combined_logits.argmax(dim=-1)
```

### 2. Simplified PID (INL 2025)
PID-like controller inspired by INL Dynamics (complexity-deep):

```
Input
  │
  ▼
[Attention] ←── mu_prev (guides K, Q, V)
  │
  ▼
[SimplifiedPID] ─► (h, velocity, mu_next)
  │
  ▼
[Residual]
  │
  ▼
[Token-Routed MLP] ←── mu_current (guides expert selection)
  │
  ▼
[Residual] ─► Output + (velocity, mu_next) ──► next layer
```

Key features:
- **Velocity**: momentum-based state that accumulates across layers
- **Mu**: contextual signal that guides attention and MLP routing
- **Mu-Guided KQV**: mu biases K, Q, V projections (fused concat for 2x speed)
- **Contextual momentum**: adapts per token
- **Gated output**: like INL Dynamics
- **Clamping**: prevents explosion at ~400K steps

### 3. Flash Attention (SDPA)
Uses PyTorch 2.0+ `scaled_dot_product_attention` for:
- 2-4x faster attention
- O(n) memory vs O(n^2)
- Automatic backend selection

### 4. QK Normalization (2024)
Normalizes Q and K before attention:
- Stabilizes training
- Prevents attention collapse
- Used in Gemma, Cohere, etc.

### 5. Sliding Window Attention (Optional)
Mistral-style local attention:
- Efficient for long sequences
- Configurable window size

## Usage

```python
from complexity import ComplexityConfig, ComplexityForCausalLM, create_complexity_model

# Create model by size
model = create_complexity_model("base")  # ~125M params

# Or with custom config
config = ComplexityConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    use_token_routed_mlp=True,
    num_experts=4,
    use_qk_norm=True,
    # INL 2025: Simplified PID
    use_simplified_pid=True,
    dynamics_momentum=0.9,
)
model = ComplexityForCausalLM(config)

# Forward pass
outputs = model(input_ids, labels=labels)
loss = outputs.loss
```

## Model Sizes

| Size | Params | Hidden | Layers | Experts |
|------|--------|--------|--------|---------|
| tiny | ~15M | 256 | 6 | 4 |
| 20m | ~20M | 320 | 8 | 4 |
| small | ~50M | 512 | 8 | 4 |
| 150m | ~150M | 768 | 12 | 4 |
| base | ~125M | 768 | 12 | 4 |
| medium | ~350M | 1024 | 24 | 4 |
| large | ~760M | 1536 | 24 | 4 |
| 1b | ~1B | 2048 | 24 | 4 |
| 3b | ~3B | 2560 | 32 | 4 |

## Architecture

```
complexity/
├── core/
│   ├── normalization.py    # RMSNorm
│   ├── rotary.py           # RoPE
│   ├── attention.py        # GQA + Flash + QK Norm + Mu-Guided KQV
│   ├── mlp.py              # Standard SwiGLU
│   ├── token_routed_mlp.py # Token-Routed MLP + Mu-Guided Routing
│   ├── dynamics.py         # SimplifiedPID
│   └── layer.py            # Decoder layer with dynamics
└── models/
    ├── config.py           # ComplexityConfig
    ├── modeling.py         # ComplexityForCausalLM
    └── utils.py            # create_complexity_model()
```

## Benefits

| Metric | Standard | Complexity |
|--------|----------|------------|
| Attention speed | 1x | 2-4x (Flash) |
| MLP compute/token | 100% | ~25% (1 expert) |
| Training stability | baseline | better (QK Norm + PID) |
| PPL | baseline | better (specialization + mu guidance) |
| Gradient flow | standard | smoother (velocity momentum) |

## Related Packages

- **complexity-deep** - Full INL Dynamics for robotics control
- **complexity-diffusion** - DiT for image generation
- **pyllm-inference** - Inference server with streaming

## License

MIT
