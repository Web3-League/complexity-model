"""
Complexity Model Training Script
================================

Train a Complexity model (Llama-based) with Complexity-4 tokenizer.

Usage:
    # Train from scratch
    python train_complexity.py --size small --dataset Pacific-Prime/mixed-inl

    # Resume training
    python train_complexity.py --size base --resume ./checkpoints/last.pt

    # Push to HuggingFace
    python train_complexity.py --size base --push Pacific-Prime/complexity-base
"""

import os
import math
import time
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from complexity import ComplexityConfig, ComplexityForCausalLM, create_complexity_model


# ============================================================================
# DATASET
# ============================================================================

class StreamingTextDataset(IterableDataset):
    """Streaming dataset for large text corpora."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
        text_field: str = "text",
        split: str = "train",
        token: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field
        self.split = split
        self.token = token

    def __iter__(self):
        ds = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            token=self.token,
        )

        buffer = []
        for example in ds:
            text = example.get(self.text_field, "")
            if not text:
                continue

            # Tokenize
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)

            # Yield chunks
            while len(buffer) >= self.max_length + 1:
                chunk = buffer[: self.max_length + 1]
                buffer = buffer[self.max_length:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    """Collate batch of samples."""
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"input_ids": input_ids, "labels": labels}


# ============================================================================
# TRAINING
# ============================================================================

def train(
    model: ComplexityForCausalLM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: dict,
    device: torch.device,
):
    """Training loop."""
    model.train()
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = config.get("start_step", 0)
    total_loss = 0.0
    log_interval = config.get("log_interval", 100)
    save_interval = config.get("save_interval", 1000)
    max_steps = config.get("max_steps", 100000)

    start_time = time.time()
    pbar = tqdm(total=max_steps, initial=global_step, desc="Training")

    for batch in train_loader:
        if global_step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        global_step += 1
        pbar.update(1)

        # Logging
        if global_step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            tokens_per_sec = (global_step * config["batch_size"] * config["max_length"]) / elapsed

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "tok/s": f"{tokens_per_sec:.0f}",
            })

            total_loss = 0.0

        # Save checkpoint
        if global_step % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"step_{global_step}.pt"
            torch.save({
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }, checkpoint_path)

            # Also save as "last.pt"
            torch.save({
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }, checkpoint_dir / "last.pt")

            print(f"\nSaved checkpoint: {checkpoint_path}")

    pbar.close()
    return global_step


# ============================================================================
# MAIN
# ============================================================================

SIZE_CONFIGS = ["tiny", "20m", "small", "base", "medium", "large", "1b", "3b"]


def main():
    parser = argparse.ArgumentParser(description="Train Complexity model")

    # Model
    parser.add_argument("--size", type=str, default="small", choices=SIZE_CONFIGS,
                        help="Model size preset")

    # Data
    parser.add_argument("--dataset", type=str, default="Pacific-Prime/mixed-inl",
                        help="HuggingFace dataset")
    parser.add_argument("--tokenizer", type=str, default="./output",
                        help="Path to Complexity-4 tokenizer")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Text field in dataset")

    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--max-steps", type=int, default=100000,
                        help="Max training steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay")

    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="Save every N steps")

    # Other
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token")
    parser.add_argument("--push", type=str, default=None,
                        help="Push to HuggingFace repo")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("Complexity Model Training")
    print("=" * 60)
    print(f"Model size: {args.size}")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    if os.path.exists(args.tokenizer):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    else:
        print(f"Tokenizer not found at {args.tokenizer}")
        print("Train a tokenizer first: python train_tokenizer.py --dataset ...")
        return

    print(f"Vocab size: {tokenizer.vocab_size}")

    # Create model
    print(f"\nCreating model...")
    model = create_complexity_model(size=args.size, vocab_size=tokenizer.vocab_size)
    model = model.to(device)
    print(f"Parameters: {model.num_parameters():,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps,
        eta_min=args.lr * 0.1,
    )

    # Resume
    start_step = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"]
        print(f"Resumed at step {start_step}")

    # Dataset
    print(f"\nLoading dataset...")
    train_dataset = StreamingTextDataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_field=args.text_field,
        split="train",
        token=args.token,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # Streaming doesn't support multiprocessing
    )

    # Training config
    train_config = {
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "checkpoint_dir": args.checkpoint_dir,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "max_grad_norm": 1.0,
        "start_step": start_step,
    }

    # Train
    print(f"\nStarting training...")
    final_step = train(model, train_loader, optimizer, scheduler, train_config, device)
    print(f"\nTraining complete! Final step: {final_step}")

    # Save final model
    final_path = Path(args.checkpoint_dir) / "final.pt"
    torch.save({
        "step": final_step,
        "model_state_dict": model.state_dict(),
        "config": train_config,
    }, final_path)
    print(f"Final model saved: {final_path}")

    # Push to HuggingFace
    if args.push and args.token:
        print(f"\nPushing to {args.push}...")
        # TODO: Convert to HuggingFace format and push


if __name__ == "__main__":
    main()
