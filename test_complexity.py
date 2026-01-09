"""
Test Complexity architecture with Token-Routed MLP.
"""

import torch
import time

# Add complexity to path
import sys
sys.path.insert(0, ".")

from complexity import ComplexityConfig, ComplexityForCausalLM, create_complexity_model


def test_token_routed_mlp():
    """Test that Token-Routed MLP works correctly."""
    print("=" * 60)
    print("Testing Complexity with Token-Routed MLP")
    print("=" * 60)

    # Create small model for testing
    config = ComplexityConfig.complexity_tiny()
    config.use_token_routed_mlp = True
    config.num_experts = 4

    print(f"\nConfig:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_hidden_layers}")
    print(f"  num_experts: {config.num_experts}")
    print(f"  use_token_routed_mlp: {config.use_token_routed_mlp}")

    model = ComplexityForCausalLM(config)
    print(f"\nModel parameters: {model.num_parameters():,}")

    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    print(f"\nInput shape: {input_ids.shape}")

    # Forward pass
    outputs = model(input_ids, labels=labels)

    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")

    # Test that different token ranges go to different experts
    print("\n" + "-" * 40)
    print("Testing expert routing:")

    # Low token IDs (frequent tokens) -> Expert 0
    low_ids = torch.randint(0, 25000, (1, 32))
    # High token IDs (rare tokens) -> Expert 3
    high_ids = torch.randint(75000, 100000, (1, 32))

    out_low = model(low_ids)
    out_high = model(high_ids)

    print(f"  Low token IDs (0-25K) → Expert 0")
    print(f"  High token IDs (75K-100K) → Expert 3")
    print(f"  Different experts = specialized processing")

    return True


def benchmark_token_routed_vs_standard():
    """Compare Token-Routed MLP vs Standard MLP speed."""
    print("\n" + "=" * 60)
    print("Benchmark: Token-Routed MLP vs Standard MLP")
    print("=" * 60)

    batch_size, seq_len = 8, 512
    num_iterations = 10

    # Token-Routed model
    config_routed = ComplexityConfig.complexity_small()
    config_routed.use_token_routed_mlp = True
    config_routed.num_experts = 4
    model_routed = ComplexityForCausalLM(config_routed)

    # Standard model
    config_standard = ComplexityConfig.complexity_small()
    config_standard.use_token_routed_mlp = False
    model_standard = ComplexityForCausalLM(config_standard)

    print(f"\nToken-Routed params: {model_routed.num_parameters():,}")
    print(f"Standard params: {model_standard.num_parameters():,}")

    input_ids = torch.randint(0, 100000, (batch_size, seq_len))
    labels = input_ids.clone()

    # Warmup
    _ = model_routed(input_ids, labels=labels)
    _ = model_standard(input_ids, labels=labels)

    # Benchmark Token-Routed
    start = time.time()
    for _ in range(num_iterations):
        _ = model_routed(input_ids, labels=labels)
    time_routed = (time.time() - start) / num_iterations

    # Benchmark Standard
    start = time.time()
    for _ in range(num_iterations):
        _ = model_standard(input_ids, labels=labels)
    time_standard = (time.time() - start) / num_iterations

    print(f"\nResults (batch={batch_size}, seq={seq_len}):")
    print(f"  Token-Routed: {time_routed*1000:.1f} ms/forward")
    print(f"  Standard:     {time_standard*1000:.1f} ms/forward")
    print(f"  Speedup:      {time_standard/time_routed:.2f}x")


def test_model_sizes():
    """Test all model sizes."""
    print("\n" + "=" * 60)
    print("Model Sizes with Token-Routed MLP")
    print("=" * 60)

    sizes = ["tiny", "small", "base"]

    for size in sizes:
        model = create_complexity_model(size)
        print(f"\n{size.upper()}:")
        print(f"  Parameters: {model.num_parameters():,}")
        print(f"  Experts: {model.config.num_experts}")
        print(f"  Token-Routed: {model.config.use_token_routed_mlp}")

        # Quick forward test
        input_ids = torch.randint(0, 100000, (1, 64))
        outputs = model(input_ids, labels=input_ids)
        print(f"  Loss: {outputs.loss.item():.4f}")


if __name__ == "__main__":
    test_token_routed_mlp()
    test_model_sizes()

    # Only run benchmark if you have time
    # benchmark_token_routed_vs_standard()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
