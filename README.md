# Complexity Model

A Llama-based transformer architecture with **Token-Routed MLP** innovation.

## Innovation: Token-Routed MLP

Unlike standard transformers where all tokens go through the same MLP, Complexity routes tokens to specialized experts based on their token ID:

```
Token IDs 0-25K     → Expert 0 (frequent tokens)
Token IDs 25K-50K   → Expert 1
Token IDs 50K-75K   → Expert 2
Token IDs 75K-100K  → Expert 3 (rare tokens)
```

### Benefits

| Metric | Standard MLP | Token-Routed MLP |
|--------|--------------|------------------|
| Compute/token | 100% | ~25% (1 expert) |
| Training speed | 1x | ~1.5x faster |
| PPL | baseline | better (specialization) |
| Params | same | same |

## Architecture

```
complexity/
├── core/
│   ├── normalization.py    # RMSNorm
│   ├── rotary.py           # RoPE
│   ├── attention.py        # GQA attention
│   ├── mlp.py              # Standard SwiGLU
│   ├── token_routed_mlp.py # Token-Routed MLP (innovation)
│   └── layer.py            # Decoder layer
└── models/
    ├── config.py           # ComplexityConfig
    ├── modeling.py         # ComplexityForCausalLM
    └── utils.py            # create_complexity_model()
```

## Usage

```python
from complexity import create_complexity_model

# Create model with Token-Routed MLP (default)
model = create_complexity_model("base")

# Forward pass
outputs = model(input_ids, labels=labels)
loss = outputs.loss
```

## Model Sizes

| Size | Params | Hidden | Layers | Experts |
|------|--------|--------|--------|---------|
| tiny | ~15M | 256 | 6 | 4 |
| small | ~50M | 512 | 8 | 4 |
| base | ~125M | 768 | 12 | 4 |
| medium | ~350M | 1024 | 24 | 4 |
| large | ~760M | 1536 | 24 | 4 |
| 1b | ~1B | 2048 | 24 | 4 |

## Training

```bash
# Train tokenizer first
python train_tokenizer.py --dataset Pacific-Prime/mixed-inl --vocab-size 100000

# Train model
python train_complexity.py --size small --dataset Pacific-Prime/mixed-inl
```

## Comparison with Llama

| Component | Llama | Complexity |
|-----------|-------|------------|
| Attention | GQA + RoPE | GQA + RoPE |
| MLP | SwiGLU | **Token-Routed SwiGLU** |
| Norm | RMSNorm | RMSNorm |
| Innovation | - | Token-based expert routing |

## License

MIT
