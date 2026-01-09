"""
Complexity-4 Tokenizer Trainer
==============================

Train a BPE tokenizer from scratch, optimized for INL models.

Features:
- 100K vocabulary (better multilingual support)
- Improved handling of whitespace and special characters
- Better code tokenization

Usage:
    python train_tokenizer.py --vocab-size 100000 --dataset Pacific-Prime/mixed-inl
    python train_tokenizer.py --vocab-size 100000 --files ./data/*.txt
    python train_tokenizer.py --vocab-size 100000 --push Pacific-Prime/complexity-token
"""

from pathlib import Path
from typing import Iterator, Optional
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFC
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import argparse


def create_complexity4_tokenizer(vocab_size: int = 100000) -> tuple:
    """
    Create a Complexity-4 style BPE tokenizer (not trained yet).

    Complexity-4 improvements (inspired by cl100k_base):
    - 100K vocabulary for better multilingual
    - Better whitespace handling
    - Optimized for code
    """
    # BPE model
    tokenizer = Tokenizer(models.BPE())

    # Normalizer: NFC unicode normalization
    tokenizer.normalizer = NFC()

    # Pre-tokenizer: Byte-level with better splitting
    # GPT-4 uses regex patterns for better word boundaries
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            behavior="isolated",
            invert=False,
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # Decoder: Byte-level
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Trainer - Complexity-4 with 100K vocab
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<|endoftext|>",      # End of text
            "<|pad|>",            # Padding
            "<|startoftext|>",    # Start of text
        ],
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    return tokenizer, trainer




def text_iterator_from_dataset(
    dataset_name: str,
    text_field: str = "text",
    split: str = "train",
    max_samples: Optional[int] = None,
    token: Optional[str] = None,
) -> Iterator[str]:
    """Yield texts from a HuggingFace dataset."""
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split=split, streaming=True, token=token)

    for i, example in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        text = example.get(text_field, "")
        if text:
            yield text


def text_iterator_from_files(file_patterns: list) -> Iterator[str]:
    """Yield texts from local files."""
    for pattern in file_patterns:
        for file_path in Path(".").glob(pattern):
            print(f"Reading: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                yield f.read()


def train_tokenizer(
    vocab_size: int = 50257,
    dataset: Optional[str] = None,
    files: Optional[list] = None,
    text_field: str = "text",
    max_samples: Optional[int] = None,
    output_dir: str = "./output",
    token: Optional[str] = None,
):
    """
    Train a GPT-style tokenizer.

    Args:
        vocab_size: Size of vocabulary (GPT-2=50257, GPT-3=100k)
        dataset: HuggingFace dataset name
        files: List of file patterns (e.g., ["*.txt", "data/*.md"])
        text_field: Field name in dataset containing text
        max_samples: Max samples to use (None = all)
        output_dir: Where to save the tokenizer
        token: HuggingFace token
    """
    print("=" * 60)
    print("Complexity-4 Tokenizer Trainer")
    print("=" * 60)
    print(f"Vocab size: {vocab_size:,}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Create tokenizer and trainer
    tokenizer, trainer = create_complexity4_tokenizer(vocab_size)

    # Get text iterator
    if dataset:
        print(f"\nTraining from dataset: {dataset}")
        iterator = text_iterator_from_dataset(
            dataset, text_field, "train", max_samples, token
        )
    elif files:
        print(f"\nTraining from files: {files}")
        iterator = text_iterator_from_files(files)
    else:
        raise ValueError("Must provide --dataset or --files")

    # Train!
    print("\nTraining tokenizer...")
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw tokenizer
    tokenizer.save(str(output_path / "tokenizer.json"))

    # Also save as HuggingFace format (for easy loading with transformers)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token=None,  # Byte-level BPE has no UNK
    )
    hf_tokenizer.save_pretrained(output_dir)

    print(f"\nTokenizer saved to: {output_dir}")
    print(f"  - tokenizer.json (raw)")
    print(f"  - tokenizer_config.json (HuggingFace)")
    print(f"  - special_tokens_map.json")

    # Stats
    print(f"\nStats:")
    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")

    # Test
    test_texts = [
        "Hello, world!",
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "L'intégration de x² est x³/3 + C",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print(f"\nTest encoding:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"  '{text[:40]}...' -> {len(encoded.ids)} tokens")

    return tokenizer, hf_tokenizer


def push_to_hub(output_dir: str, repo_id: str, token: str):
    """Push tokenizer to HuggingFace Hub."""
    print(f"\nPushing to: {repo_id}")

    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(output_dir)
    hf_tokenizer.push_to_hub(repo_id, token=token)

    print(f"Tokenizer published: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Complexity-4 tokenizer")
    parser.add_argument("--vocab-size", type=int, default=100000,
                        help="Vocabulary size (default: 100K)")
    parser.add_argument("--dataset", type=str,
                        help="HuggingFace dataset to train on")
    parser.add_argument("--files", type=str, nargs="+",
                        help="Local files to train on (glob patterns)")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Text field name in dataset")
    parser.add_argument("--max-samples", type=int,
                        help="Max samples to use")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--token", type=str,
                        help="HuggingFace token")
    parser.add_argument("--push", type=str,
                        help="Push to HuggingFace repo (e.g., Pacific-Prime/gpt-tokenizer)")

    args = parser.parse_args()

    # Train
    tokenizer, hf_tokenizer = train_tokenizer(
        vocab_size=args.vocab_size,
        dataset=args.dataset,
        files=args.files,
        text_field=args.text_field,
        max_samples=args.max_samples,
        output_dir=args.output,
        token=args.token,
    )

    # Push if requested
    if args.push:
        if not args.token:
            print("\nERROR: --token required for --push")
        else:
            push_to_hub(args.output, args.push, args.token)
