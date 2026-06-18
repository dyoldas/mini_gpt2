"""
Text generation script for mini_gpt2.

This loads a trained checkpoint, encodes a prompt with GPT-2 BPE,
generates new tokens, and decodes them back into text.

Example run:

    python -m inference.generate \
        --checkpoint checkpoints/gpt2_50m_tinystories/ckpt_best.pt \
        --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from models import GPT, GPTConfig
from tokenizer import GPT2Tokenizer


def get_device(device_name: str) -> str:
    """
    Choose device automatically unless user specifies one.
    """

    if device_name != "auto":
        return device_name

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def build_model_config(checkpoint: dict[str, Any]) -> GPTConfig:
    """
    Rebuild GPTConfig from checkpoint.

    Our train.py saves the full YAML config under checkpoint["config"].
    This function extracts the model section.
    """

    if "config" in checkpoint:
        model_cfg = checkpoint["config"]["model"]

        return GPTConfig(
            vocab_size=model_cfg["vocab_size"],
            block_size=model_cfg["block_size"],
            n_layer=model_cfg["n_layer"],
            n_head=model_cfg["n_head"],
            n_embd=model_cfg["n_embd"],
            dropout=model_cfg["dropout"],
        )

    raise KeyError("Checkpoint does not contain model config.")


def load_model(
    checkpoint_path: Path,
    device: str,
) -> GPT:
    """
    Load GPT model from checkpoint.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
    )

    config = build_model_config(checkpoint)

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Model parameters: {model.num_parameters():,}")

    if "iteration" in checkpoint:
        print(f"Checkpoint iteration: {checkpoint['iteration']}")

    if "best_val_loss" in checkpoint:
        print(f"Best val loss: {checkpoint['best_val_loss']}")

    return model


@torch.no_grad()
def generate_text(
    *,
    model: GPT,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    """
    Generate text from a prompt.

    Steps:
    1. text prompt -> token IDs
    2. token IDs -> model.generate()
    3. generated token IDs -> text
    """

    token_ids = tokenizer.encode(prompt)

    if len(token_ids) == 0:
        token_ids = [tokenizer.eot_token]

    idx = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    return tokenizer.decode(out[0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from mini_gpt2.")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file, e.g. ckpt_best.pt.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Prompt text.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Number of tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Lower = safer, higher = more random.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Keep only top-k tokens during sampling. Use 0 to disable.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device(args.device)

    tokenizer = GPT2Tokenizer()

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    top_k = None if args.top_k <= 0 else args.top_k

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
    )

    print("\n--- GENERATED TEXT ---\n")
    print(text)


if __name__ == "__main__":
    main()
