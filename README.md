# Complexity

A modern transformer architecture with **2024 optimizations** and **Token-Routed MLP** innovation.

[![PyPI version](https://badge.fury.io/py/complexity.svg)](https://badge.fury.io/py/complexity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install complexity
```

## Innovations

### 1. Token-Routed MLP (Original)
Routes tokens to specialized experts based on token ID:

```
Token IDs 0-25K     -> Expert 0 (frequent tokens)
Token IDs 25K-50K   -> Expert 1
Token IDs 50K-75K   -> Expert 2
Token IDs 75K-100K  -> Expert 3 (rare tokens)
```

### 2. Flash Attention (SDPA)
Uses PyTorch 2.0+ `scaled_dot_product_attention` for:
- 2-4x faster attention
- O(n) memory vs O(n^2)
- Automatic backend selection

### 3. QK Normalization (2024)
Normalizes Q and K before attention:
- Stabilizes training
- Prevents attention collapse
- Used in Gemma, Cohere, etc.

### 4. Sliding Window Attention (Optional)
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
│   ├── attention.py        # GQA + Flash + QK Norm
│   ├── mlp.py              # Standard SwiGLU
│   ├── token_routed_mlp.py # Token-Routed MLP
│   └── layer.py            # Decoder layer
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
| Training stability | baseline | better (QK Norm) |
| PPL | baseline | better (specialization) |

## Related Packages

- **complexity-deep** - Adds INL Dynamics for robotics control
- **complexity-diffusion** - DiT for image generation
- **pyllm-inference** - Inference server with streaming

## License

MIT
