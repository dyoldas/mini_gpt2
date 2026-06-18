"""
Binary token dataloader for mini_gpt2.

The data preparation script creates files like:

    data/processed/tinystories/train.bin
    data/processed/tinystories/val.bin

Each file is one long stream of GPT-2 BPE token IDs stored as uint16.

This dataloader turns that long token stream into random training batches:

    x = tokens[i : i + block_size]
    y = tokens[i + 1 : i + block_size + 1]

The model sees x and learns to predict y.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch


Split = Literal["train", "val"]


class BinaryTokenDataLoader:
    """
    Loads pre-tokenized GPT-2 data from train.bin / val.bin.

    This is not a PyTorch Dataset/DataLoader yet. It is a simple custom loader,
    which is enough for GPT-style training and similar to nanoGPT's approach.

    Example:
        loader = BinaryTokenDataLoader(
            stage="tinystories",
            block_size=256,
            batch_size=32,
            device="mps",
        )

        x, y = loader.get_batch("train")
    """

    def __init__(
        self,
        *,
        stage: str,
        block_size: int,
        batch_size: int,
        device: str,
    ) -> None:
        self.stage = stage
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        # Project root: training/dataloader.py -> mini_gpt2/
        self.project_root = Path(__file__).resolve().parents[1]

        self.data_dir = self.project_root / "data" / "processed" / stage
        self.train_path = self.data_dir / "train.bin"
        self.val_path = self.data_dir / "val.bin"

        if not self.train_path.exists():
            raise FileNotFoundError(f"Missing train file: {self.train_path}")

        if not self.val_path.exists():
            raise FileNotFoundError(f"Missing val file: {self.val_path}")

        # np.memmap does not load the whole file into RAM.
        # It gives array-like access to disk data.
        self.train_data = np.memmap(
            self.train_path,
            dtype=np.uint16,
            mode="r",
        )

        self.val_data = np.memmap(
            self.val_path,
            dtype=np.uint16,
            mode="r",
        )

        self._check_data_size()

    def _check_data_size(self) -> None:
        """
        Make sure both splits are large enough for the requested block size.

        We need block_size + 1 tokens because:
            x uses block_size tokens
            y is shifted by one token
        """

        min_required = self.block_size + 1

        if len(self.train_data) < min_required:
            raise ValueError(
                f"Train data too small. Need at least {min_required} tokens, "
                f"got {len(self.train_data)}."
            )

        if len(self.val_data) < min_required:
            raise ValueError(
                f"Val data too small. Need at least {min_required} tokens, "
                f"got {len(self.val_data)}."
            )

    def get_data(self, split: Split) -> np.memmap:
        """
        Return the selected token stream.
        """

        if split == "train":
            return self.train_data

        if split == "val":
            return self.val_data

        raise ValueError(f"Unknown split: {split}")

    def get_batch(self, split: Split) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create one random batch of x, y examples.

        Shapes:
            x: (batch_size, block_size)
            y: (batch_size, block_size)

        Example with block_size = 4:

            tokens = [10, 20, 30, 40, 50]

            x = [10, 20, 30, 40]
            y = [20, 30, 40, 50]
        """

        data = self.get_data(split)

        # Random start positions.
        # Last possible start must leave room for block_size + 1 tokens.
        max_start = len(data) - self.block_size - 1

        starts = torch.randint(
            low=0,
            high=max_start,
            size=(self.batch_size,),
        )

        # Build x and y examples.
        # Convert to int64 because PyTorch embedding layers expect LongTensor IDs.
        x_np = np.stack(
            [
                np.asarray(data[i : i + self.block_size], dtype=np.int64)
                for i in starts.tolist()
            ]
        )

        y_np = np.stack(
            [
                np.asarray(data[i + 1 : i + self.block_size + 1], dtype=np.int64)
                for i in starts.tolist()
            ]
        )

        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).to(self.device)

        return x, y

    def num_tokens(self, split: Split) -> int:
        """
        Return number of tokens in a split.
        """

        return len(self.get_data(split))

    def info(self) -> None:
        """
        Print basic dataloader information.
        """

        print("BinaryTokenDataLoader")
        print(f"Stage:       {self.stage}")
        print(f"Data dir:    {self.data_dir}")
        print(f"Train toks:  {len(self.train_data):,}")
        print(f"Val toks:    {len(self.val_data):,}")
        print(f"Block size:  {self.block_size}")
        print(f"Batch size:  {self.batch_size}")
        print(f"Device:      {self.device}")


if __name__ == "__main__":
    """
    Quick test:

        python training/dataloader.py

    Make sure you prepared data first:

        python -m data.prepare --stage tinystories --max_tokens 100000
    """

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    loader = BinaryTokenDataLoader(
        stage="tinystories",
        block_size=128,
        batch_size=4,
        device=device,
    )

    loader.info()

    x, y = loader.get_batch("train")

    print("\nx shape:", x.shape)
    print("y shape:", y.shape)

    print("\nFirst x example:")
    print(x[0])

    print("\nFirst y example:")
    print(y[0])
