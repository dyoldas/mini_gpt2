"""
Script to download the real GPT-2 models from Hugging Face and save them in a format compatible with our training code.

This is useful for benchmarking and testing our implementation against the original models.
"""

from pathlib import Path

import torch
from transformers import GPT2LMHeadModel


MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
]

ROOT = Path("checkpoints")


for model_name in MODELS:
    print(f"Downloading {model_name}...")

    model = GPT2LMHeadModel.from_pretrained(model_name)

    out_dir = ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save pure weights
    torch.save(
        model.state_dict(),
        out_dir / "pytorch_model.pt",
    )

    # Save configuration
    with open(out_dir / "config.json", "w") as f:
        f.write(model.config.to_json_string())

    print(f"Saved to {out_dir}")
