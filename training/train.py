"""
Training script for mini_gpt2.

This file does the basic GPT training loop:

    train.bin / val.bin
        -> dataloader gives x, y batches
        -> GPT predicts next tokens
        -> cross-entropy loss
        -> AdamW optimizer updates weights
        -> checkpoints are saved

Run:

    python -m training.train --config configs/gpt2_tiny.yaml
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from models import GPT, GPTConfig
from training.dataloader import BinaryTokenDataLoader


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------


def get_device(device_name: str) -> str:
    """
    Choose device.

    On your Mac, "auto" should pick MPS.
    """

    if device_name != "auto":
        return device_name

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def set_seed(seed: int) -> None:
    """
    Make random choices more reproducible.
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    """
    Load YAML config file.
    """

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_lr(
    iteration: int,
    *,
    learning_rate: float,
    min_lr: float,
    warmup_iters: int,
    max_iters: int,
) -> float:
    """
    GPT-style learning rate schedule.

    1. Warm up linearly.
    2. Decay smoothly with cosine schedule.
    """

    if iteration < warmup_iters:
        return learning_rate * (iteration + 1) / warmup_iters

    if iteration >= max_iters:
        return min_lr

    decay_ratio = (iteration - warmup_iters) / (max_iters - warmup_iters)
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + cosine_coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    dataloader: BinaryTokenDataLoader,
    eval_iters: int,
) -> dict[str, float]:
    """
    Estimate average train/val loss.

    We use multiple batches because one batch is noisy.
    """

    model.eval()
    losses: dict[str, float] = {}

    for split in ["train", "val"]:
        total = 0.0

        for _ in range(eval_iters):
            x, y = dataloader.get_batch(split)  # type: ignore[arg-type]
            _, loss = model(x, y)

            if loss is None:
                raise RuntimeError("Loss is None during evaluation.")

            total += loss.item()

        losses[split] = total / eval_iters

    model.train()
    return losses


def save_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    iteration: int,
    best_val_loss: float,
) -> None:
    """
    Save training state.

    We save both model and optimizer so training can be resumed later.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "iteration": iteration,
        "best_val_loss": best_val_loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[int, float]:
    """
    Load model + optimizer state.

    Returns:
        start iteration
        best validation loss
    """

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_iter = checkpoint["iteration"] + 1
    best_val_loss = checkpoint["best_val_loss"]

    return start_iter, best_val_loss


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------


def train(config_path: Path) -> None:
    config = load_config(config_path)

    seed = config["system"]["seed"]
    device = get_device(config["system"]["device"])

    set_seed(seed)

    print(f"Using device: {device}")

    # Build model config from YAML.
    model_cfg = GPTConfig(
        vocab_size=config["model"]["vocab_size"],
        block_size=config["model"]["block_size"],
        n_layer=config["model"]["n_layer"],
        n_head=config["model"]["n_head"],
        n_embd=config["model"]["n_embd"],
        dropout=config["model"]["dropout"],
    )

    # Dataloader reads prepared train.bin / val.bin.
    dataloader = BinaryTokenDataLoader(
        stage=config["data"]["stage"],
        block_size=model_cfg.block_size,
        batch_size=config["training"]["batch_size"],
        device=device,
    )

    dataloader.info()

    model = GPT(model_cfg).to(device)

    print(f"\nModel parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=config["training"]["weight_decay"],
    )

    out_dir = Path(config["checkpoint"]["out_dir"])
    last_ckpt = out_dir / "ckpt_last.pt"
    best_ckpt = out_dir / "ckpt_best.pt"

    max_iters = config["training"]["max_iters"]
    eval_interval = config["training"]["eval_interval"]
    eval_iters = config["training"]["eval_iters"]
    log_interval = config["training"]["log_interval"]

    # Start training from scratch or build upon previous stage checkpoint.
    best_val_loss = float("inf")
    start_iter = 0

    resume_from = config["checkpoint"].get("resume_from")

    if resume_from is not None:
        print(f"Loading checkpoint: {resume_from}")

        start_iter, best_val_loss = load_checkpoint(
            Path(resume_from),
            model=model,
            optimizer=optimizer,
            device=device,
        )

        print(f"Resuming from iteration {start_iter}")

    # Start the training loop.
    model.train()

    print("\nStarting training...\n")
    start_time = time.time()

    for iteration in range(start_iter, max_iters):
        # Set learning rate for this step.
        lr = get_lr(
            iteration,
            learning_rate=config["training"]["learning_rate"],
            min_lr=config["training"]["min_lr"],
            warmup_iters=config["training"]["warmup_iters"],
            max_iters=max_iters,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get one batch.
        x, y = dataloader.get_batch("train")

        # Forward + loss.
        _, loss = model(x, y)

        if loss is None:
            raise RuntimeError("Loss is None during training.")

        # Backprop.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping prevents occasional huge updates.
        grad_clip = config["training"]["grad_clip"]
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Light logging.
        if iteration % log_interval == 0:
            elapsed = time.time() - start_time

            print(
                f"iter {iteration:5d} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.2e} | "
                f"time {elapsed:.1f}s"
            )

        # Evaluation + checkpointing.
        if iteration % eval_interval == 0 or iteration == max_iters - 1:
            losses = estimate_loss(
                model=model,
                dataloader=dataloader,
                eval_iters=eval_iters,
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            print(
                f"\nEVAL iter {iteration:5d} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f}\n"
            )

            save_checkpoint(
                last_ckpt,
                model=model,
                optimizer=optimizer,
                config=config,
                iteration=iteration,
                best_val_loss=best_val_loss,
            )

            if config["checkpoint"]["save_best"] and val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    best_ckpt,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    iteration=iteration,
                    best_val_loss=best_val_loss,
                )

                print(f"Saved new best checkpoint: {best_ckpt}")

    print("\nTraining complete.")
    print(f"Last checkpoint: {last_ckpt}")
    print(f"Best checkpoint: {best_ckpt}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mini_gpt2.")

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
