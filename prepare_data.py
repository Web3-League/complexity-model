#!/usr/bin/env python3
"""
Prepare Pre-Tokenized Dataset for Fast Training
================================================

Downloads and pre-tokenizes a dataset, saving to disk in Arrow format.
Training then reads pre-tokenized data = ZERO tokenization overhead.

Usage:
    # Prepare C4 (English, ~50GB tokens = ~100B tokens)
    python prepare_data.py --dataset allenai/c4 --subset en --output ./data/c4

    # Prepare smaller dataset for testing
    python prepare_data.py --dataset roneneldan/TinyStories --output ./data/tinystories --max-samples 100000

    # Prepare with custom tokenizer
    python prepare_data.py --dataset allenai/c4 --tokenizer ./tokenizer --output ./data/c4

Then train:
    python train_complexity.py --data ./data/c4 --size small
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import multiprocessing as mp

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tqdm import tqdm


def tokenize_batch(args):
    """Tokenize a batch of texts (for multiprocessing)."""
    texts, tokenizer_path, max_length = args

    # Load tokenizer in worker
    if os.path.exists(tokenizer_path):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    samples = []
    buffer = []

    for text in texts:
        if not text:
            continue
        tokens = tokenizer.encode(text)
        buffer.extend(tokens)

        while len(buffer) >= max_length + 1:
            chunk = buffer[:max_length + 1]
            buffer = buffer[max_length:]
            samples.append(chunk)

    return samples


def prepare_dataset(
    dataset_name: str,
    output_dir: str,
    tokenizer_path: str = "gpt2",
    subset: Optional[str] = None,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    max_tokens: Optional[int] = None,  # e.g., 10_000_000_000 for 10B tokens
    num_workers: int = 8,
    batch_size: int = 1000,
    token: Optional[str] = None,
    text_field: str = "text",
):
    """
    Download, tokenize, and save dataset to disk.

    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Output directory for parquet files
        tokenizer_path: Path to tokenizer or HF model name
        subset: Dataset subset (e.g., "en" for C4)
        max_length: Sequence length
        max_samples: Max number of raw samples to process
        max_tokens: Max total tokens (stops when reached)
        num_workers: Number of parallel workers
        batch_size: Batch size for tokenization
        token: HuggingFace token
        text_field: Field containing text
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PREPARING PRE-TOKENIZED DATASET")
    print("=" * 60)
    print(f"Dataset: {dataset_name}" + (f" ({subset})" if subset else ""))
    print(f"Output: {output_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Sequence length: {max_length}")
    print(f"Workers: {num_workers}")
    if max_samples:
        print(f"Max samples: {max_samples:,}")
    if max_tokens:
        print(f"Max tokens: {max_tokens:,}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    if os.path.exists(tokenizer_path):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size:,}")

    # Load dataset (streaming to avoid downloading everything)
    print(f"\nLoading dataset (streaming)...")
    if subset:
        ds = load_dataset(dataset_name, subset, split="train", streaming=True, token=token)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True, token=token)

    # Process in batches
    print(f"\nProcessing...")

    all_samples = []
    total_tokens = 0
    samples_processed = 0
    shard_idx = 0
    samples_per_shard = 100_000  # ~50MB per shard

    batch_texts = []
    pbar = tqdm(desc="Processing", unit=" samples")

    for example in ds:
        # Extract text
        text = None
        for field in [text_field, "text", "content", "code"]:
            if field in example and example[field]:
                text = example[field]
                break

        if not text or len(text) < 50:
            continue

        batch_texts.append(text)
        samples_processed += 1
        pbar.update(1)

        # Process batch
        if len(batch_texts) >= batch_size:
            # Tokenize batch
            buffer = []
            for t in batch_texts:
                tokens = tokenizer.encode(t)
                buffer.extend(tokens)

                while len(buffer) >= max_length + 1:
                    chunk = buffer[:max_length + 1]
                    buffer = buffer[max_length:]
                    all_samples.append(chunk)
                    total_tokens += max_length

            batch_texts = []

            # Save shard if enough samples
            if len(all_samples) >= samples_per_shard:
                shard_path = output_path / f"shard_{shard_idx:05d}.parquet"
                save_shard(all_samples, shard_path)
                print(f"\nSaved {shard_path.name} ({len(all_samples):,} samples)")
                all_samples = []
                shard_idx += 1

        # Check limits
        if max_samples and samples_processed >= max_samples:
            print(f"\nReached max_samples limit ({max_samples:,})")
            break

        if max_tokens and total_tokens >= max_tokens:
            print(f"\nReached max_tokens limit ({max_tokens:,})")
            break

    pbar.close()

    # Process remaining batch
    if batch_texts:
        buffer = []
        for t in batch_texts:
            tokens = tokenizer.encode(t)
            buffer.extend(tokens)

            while len(buffer) >= max_length + 1:
                chunk = buffer[:max_length + 1]
                buffer = buffer[max_length:]
                all_samples.append(chunk)
                total_tokens += max_length

    # Save final shard
    if all_samples:
        shard_path = output_path / f"shard_{shard_idx:05d}.parquet"
        save_shard(all_samples, shard_path)
        print(f"Saved {shard_path.name} ({len(all_samples):,} samples)")
        shard_idx += 1

    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "subset": subset,
        "tokenizer": tokenizer_path,
        "vocab_size": vocab_size,
        "max_length": max_length,
        "total_samples": sum(1 for _ in output_path.glob("shard_*.parquet")),
        "total_tokens": total_tokens,
        "num_shards": shard_idx,
    }

    import json
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total samples: {total_tokens // max_length:,}")
    print(f"Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")
    print(f"Shards: {shard_idx}")
    print(f"Output: {output_path}")
    print("=" * 60)
    print(f"\nTo train:")
    print(f"  python train_complexity.py --data {output_path} --size small")


def save_shard(samples, path):
    """Save samples to parquet file."""
    # Convert to arrow table
    table = pa.table({
        "input_ids": [s[:-1] for s in samples],  # All but last
        "labels": [s[1:] for s in samples],      # All but first
    })
    pq.write_table(table, path, compression="snappy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare pre-tokenized dataset")

    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, default=None,
                        help="Dataset subset (e.g., 'en' for C4)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="Tokenizer path or HF model name")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max raw samples to process")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max total tokens (e.g., 10000000000 for 10B)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of workers")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for tokenization")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Text field name")

    args = parser.parse_args()

    prepare_dataset(
        dataset_name=args.dataset,
        output_dir=args.output,
        tokenizer_path=args.tokenizer,
        subset=args.subset,
        max_length=args.max_length,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        token=args.token,
        text_field=args.text_field,
    )
