#!/usr/bin/env python3
"""
Convert Complexity checkpoint to HuggingFace-compatible safetensors format.

Usage:
    python convert_to_safetensors.py --checkpoint checkpoints/last.pt --output complexity-hf
    python convert_to_safetensors.py --checkpoint checkpoints/step_10000.pt --output complexity-base-hf --size base
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


# Model config mapping (matches ComplexityConfig presets)
MODEL_CONFIGS = {
    "tiny": {
        "hidden_size": 256,
        "intermediate_size": 704,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    },
    "20m": {
        "hidden_size": 320,
        "intermediate_size": 896,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
    },
    "small": {
        "hidden_size": 512,
        "intermediate_size": 1408,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
    },
    "base": {
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
    },
    "medium": {
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
    },
    "large": {
        "hidden_size": 1536,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
    },
    "1b": {
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
    },
    "3b": {
        "hidden_size": 2560,
        "intermediate_size": 6912,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    },
}


def detect_model_size(state_dict: dict, config: dict = None) -> str:
    """Detect model size from state dict or config."""
    if config and "hidden_size" in config:
        hidden_size = config["hidden_size"]
        for size, cfg in MODEL_CONFIGS.items():
            if cfg["hidden_size"] == hidden_size:
                return size

    # Try to detect from embedding dimensions
    for key, tensor in state_dict.items():
        if "embed" in key.lower():
            if len(tensor.shape) == 2:
                hidden_size = tensor.shape[1]
                for size, cfg in MODEL_CONFIGS.items():
                    if cfg["hidden_size"] == hidden_size:
                        return size

    return "base"  # Default


def convert_to_safetensors(
    checkpoint_path: str,
    output_dir: str,
    tokenizer_dir: str = None,
    model_size: str = None,
):
    """Convert PyTorch checkpoint to safetensors format."""

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict and config
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            saved_config = checkpoint.get("config", {})
            vocab_size = checkpoint.get("vocab_size", 100000)
            step = checkpoint.get("step", 0)
        else:
            state_dict = checkpoint
            saved_config = {}
            vocab_size = 100000
            step = 0
    else:
        state_dict = checkpoint
        saved_config = {}
        vocab_size = 100000
        step = 0

    # Detect or use provided model size
    if model_size is None:
        model_size = detect_model_size(state_dict, saved_config)

    model_config = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS["base"])

    print(f"[OK] Detected model size: {model_size}")
    print(f"   - hidden_size: {model_config['hidden_size']}")
    print(f"   - num_layers: {model_config['num_hidden_layers']}")
    print(f"   - vocab_size: {vocab_size}")
    if step > 0:
        print(f"   - trained steps: {step:,}")

    # Convert tensors to float16 for smaller file size
    print("\n[...] Converting to safetensors...")
    converted_state_dict = {}
    for key, tensor in state_dict.items():
        if tensor.dtype == torch.float32:
            converted_state_dict[key] = tensor.half()
        else:
            converted_state_dict[key] = tensor

    # Save safetensors
    safetensors_path = output_dir / "model.safetensors"
    save_file(converted_state_dict, safetensors_path)
    print(f"[OK] Saved model.safetensors ({safetensors_path.stat().st_size / 1e9:.2f} GB)")

    # Create config.json (HuggingFace format)
    hf_config = {
        "architectures": ["ComplexityForCausalLM"],
        "model_type": "complexity",
        "vocab_size": vocab_size,
        "hidden_size": model_config["hidden_size"],
        "intermediate_size": model_config["intermediate_size"],
        "num_hidden_layers": model_config["num_hidden_layers"],
        "num_attention_heads": model_config["num_attention_heads"],
        "num_key_value_heads": model_config["num_key_value_heads"],
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float16",
        "transformers_version": "4.36.0",
        # Complexity-specific config
        "use_token_routed_mlp": True,
        "num_experts": 4,
        "use_qk_norm": True,
        "use_sdpa": True,
        "sliding_window": None,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"[OK] Saved config.json")

    # Create generation_config.json
    generation_config = {
        "_from_model_config": True,
        "bos_token_id": 2,
        "eos_token_id": 0,
        "pad_token_id": 1,
        "max_length": 2048,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }

    gen_config_path = output_dir / "generation_config.json"
    with open(gen_config_path, "w") as f:
        json.dump(generation_config, f, indent=2)
    print(f"[OK] Saved generation_config.json")

    # Copy tokenizer files if provided
    if tokenizer_dir:
        tokenizer_dir = Path(tokenizer_dir)
        if tokenizer_dir.exists():
            print(f"\n[*] Copying tokenizer files from {tokenizer_dir}...")

            tokenizer_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ]

            for fname in tokenizer_files:
                src = tokenizer_dir / fname
                if src.exists():
                    shutil.copy(src, output_dir / fname)
                    print(f"   [OK] Copied {fname}")

    # Create README.md
    readme_content = f"""---
license: apache-2.0
language:
- en
- fr
- code
tags:
- complexity
- token-routed-mlp
- flash-attention
- causal-lm
library_name: transformers
pipeline_tag: text-generation
---

# Complexity {model_size.upper()}

Complexity transformer model with Token-Routed MLP, Flash Attention, and QK Normalization.

## Model Details

- **Architecture**: Complexity with Token-Routed MLP
- **Size**: {model_size}
- **Hidden size**: {model_config['hidden_size']}
- **Layers**: {model_config['num_hidden_layers']}
- **Attention heads**: {model_config['num_attention_heads']} (KV heads: {model_config['num_key_value_heads']})
- **Vocabulary**: {vocab_size:,} tokens
- **Context length**: 2048 tokens
- **Training steps**: {step:,}

## Innovations

- **Token-Routed MLP**: Routes tokens to specialized experts based on token ID
- **Flash Attention**: Via PyTorch 2.0+ SDPA
- **QK Normalization**: Stabilizes training at scale

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Pacific-Prime/complexity-{model_size}")
model = AutoModelForCausalLM.from_pretrained("Pacific-Prime/complexity-{model_size}", trust_remote_code=True)

input_text = "def fibonacci(n):"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## License

Apache 2.0

## Citation

```bibtex
@misc{{complexity-{model_size},
  title={{Complexity: Token-Routed MLP Transformer}},
  author={{Pacific Prime}},
  year={{2025}},
  url={{https://huggingface.co/Pacific-Prime/complexity-{model_size}}}
}}
```
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"[OK] Saved README.md")

    print("\n" + "="*60)
    print("[OK] CONVERSION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1e9:
            size_str = f"{size/1e9:.2f} GB"
        elif size > 1e6:
            size_str = f"{size/1e6:.2f} MB"
        elif size > 1e3:
            size_str = f"{size/1e3:.2f} KB"
        else:
            size_str = f"{size} B"
        print(f"  - {f.name} ({size_str})")

    print(f"\nTo upload to HuggingFace:")
    print(f"  huggingface-cli upload Pacific-Prime/complexity-{model_size} {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Complexity checkpoint to safetensors")
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to PyTorch checkpoint (.pt)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for HuggingFace files"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        default="./tokenizer",
        help="Path to tokenizer directory (default: ./tokenizer)"
    )
    parser.add_argument(
        "--size", "-s",
        choices=["tiny", "20m", "small", "base", "medium", "large", "1b", "3b"],
        help="Model size (auto-detected if not provided)"
    )

    args = parser.parse_args()

    convert_to_safetensors(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        tokenizer_dir=args.tokenizer,
        model_size=args.size,
    )


if __name__ == "__main__":
    main()
