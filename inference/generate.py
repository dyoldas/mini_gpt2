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
import json

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


def load_hf_gpt2_into_custom_model(
    model_dir: Path,
    device: str,
) -> GPT:
    """
    Load HuggingFace GPT-2 weights into our custom GPT model.

    HF GPT-2 uses names like:
        transformer.wte.weight
        transformer.h.0.attn.c_attn.weight

    Our model uses names like:
        token_embedding.weight
        blocks.0.attn.c_attn.weight

    Also important:
    HF Conv1D weights are stored transposed compared to nn.Linear,
    so attention/MLP projection weights need .T.
    """

    config_path = model_dir / "config.json"
    weights_path = model_dir / "pytorch_model.pt"

    with open(config_path, "r", encoding="utf-8") as f:
        hf_config = json.load(f)

    config = GPTConfig(
        vocab_size=hf_config["vocab_size"],
        block_size=hf_config["n_positions"],
        n_layer=hf_config["n_layer"],
        n_head=hf_config["n_head"],
        n_embd=hf_config["n_embd"],
        dropout=0.0,  # inference only
    )

    model = GPT(config).to(device)
    hf_sd = torch.load(weights_path, map_location="cpu")
    sd = model.state_dict()

    # Embeddings
    sd["token_embedding.weight"].copy_(hf_sd["transformer.wte.weight"])
    sd["position_embedding.weight"].copy_(hf_sd["transformer.wpe.weight"])

    # Transformer blocks
    for i in range(config.n_layer):
        prefix_hf = f"transformer.h.{i}"
        prefix_ours = f"blocks.{i}"

        # LayerNorms
        sd[f"{prefix_ours}.ln_1.weight"].copy_(hf_sd[f"{prefix_hf}.ln_1.weight"])
        sd[f"{prefix_ours}.ln_1.bias"].copy_(hf_sd[f"{prefix_hf}.ln_1.bias"])
        sd[f"{prefix_ours}.ln_2.weight"].copy_(hf_sd[f"{prefix_hf}.ln_2.weight"])
        sd[f"{prefix_ours}.ln_2.bias"].copy_(hf_sd[f"{prefix_hf}.ln_2.bias"])

        # Attention projections
        # HF stores Conv1D weights as (in, out), nn.Linear wants (out, in).
        sd[f"{prefix_ours}.attn.c_attn.weight"].copy_(
            hf_sd[f"{prefix_hf}.attn.c_attn.weight"].T
        )
        sd[f"{prefix_ours}.attn.c_attn.bias"].copy_(
            hf_sd[f"{prefix_hf}.attn.c_attn.bias"]
        )

        sd[f"{prefix_ours}.attn.c_proj.weight"].copy_(
            hf_sd[f"{prefix_hf}.attn.c_proj.weight"].T
        )
        sd[f"{prefix_ours}.attn.c_proj.bias"].copy_(
            hf_sd[f"{prefix_hf}.attn.c_proj.bias"]
        )

        # MLP projections
        sd[f"{prefix_ours}.mlp.c_fc.weight"].copy_(
            hf_sd[f"{prefix_hf}.mlp.c_fc.weight"].T
        )
        sd[f"{prefix_ours}.mlp.c_fc.bias"].copy_(hf_sd[f"{prefix_hf}.mlp.c_fc.bias"])

        sd[f"{prefix_ours}.mlp.c_proj.weight"].copy_(
            hf_sd[f"{prefix_hf}.mlp.c_proj.weight"].T
        )
        sd[f"{prefix_ours}.mlp.c_proj.bias"].copy_(
            hf_sd[f"{prefix_hf}.mlp.c_proj.bias"]
        )

    # Final LayerNorm
    sd["ln_f.weight"].copy_(hf_sd["transformer.ln_f.weight"])
    sd["ln_f.bias"].copy_(hf_sd["transformer.ln_f.bias"])

    # lm_head is tied to token_embedding in our model.
    model.load_state_dict(sd)
    model.eval()

    print(f"Loaded HF GPT-2 weights into custom GPT model: {model_dir}")
    print(f"Parameters: {model.num_parameters():,}")

    return model


def load_model(
    checkpoint_path: Path,
    device: str,
) -> GPT:
    """
    Load either:

    1. Our custom mini_gpt2 checkpoint:
        checkpoints/gpt2_50m_tinystories/ckpt_best.pt

    2. Real HuggingFace GPT-2 folder:
        checkpoints/gpt2
        checkpoints/gpt2-medium
        checkpoints/gpt2-large
        checkpoints/gpt2-xl

    Rule:
        file path   -> custom checkpoint
        folder path -> HF GPT-2 weights converted into our GPT class
    """

    if checkpoint_path.is_dir():
        return load_hf_gpt2_into_custom_model(
            model_dir=checkpoint_path,
            device=device,
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
    )

    config = build_model_config(checkpoint)

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded custom checkpoint: {checkpoint_path}")
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
