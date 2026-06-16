"""
Prepare staged pretraining datasets for mini_gpt2.

This script creates GPT-2 BPE tokenized binary files:

    data/processed/<stage>/train.bin
    data/processed/<stage>/val.bin
    data/processed/<stage>/stats.json

Stages we will use:

    1. tinystories
       Easy first pretraining stage. Good for making the model coherent quickly.

    2. finewebedu
       Cleaner educational web text. More realistic than TinyStories.

    3. webmix
       FineWeb-Edu + OpenWebText-style mixture. Closest to mini GPT-2 flavor.

    4. local
       Optional local text file debugging.

Example commands:

    python -m data.prepare --stage tinystories --max_tokens 5000000

    python -m data.prepare --stage finewebedu --max_tokens 50000000

    python -m data.prepare --stage webmix --max_tokens 50000000

    python -m data.prepare --stage local --input_file data/raw/input.txt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from tokenizer import GPT2Tokenizer


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------
# Dataset stage configuration
# ---------------------------------------------------------------------

STAGE_CONFIGS = {
    "tinystories": {
        "description": "TinyStories: simple stories for first small-model pretraining.",
    },
    "finewebedu": {
        "description": "FineWeb-Edu: cleaner educational web text.",
    },
    "webmix": {
        "description": "70% FineWeb-Edu + 30% OpenWebText-style web text.",
        "finewebedu_ratio": 0.70,
    },
    "local": {
        "description": "Local text file for debugging.",
    },
}


# ---------------------------------------------------------------------
# Dataset iterators
# ---------------------------------------------------------------------


def iter_tinystories() -> Iterator[str]:
    """
    Stream TinyStories documents.

    TinyStories is ideal for the first stage because small models can learn
    simple grammar and story structure quickly.
    """

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )

    for row in dataset:
        text = row["text"].strip()  # type: ignore
        if text:
            yield text


def iter_finewebedu() -> Iterator[str]:
    """
    Stream FineWeb-Edu documents.

    We use the sample-10BT config because it is already a manageable subset
    compared with the full FineWeb-Edu corpus.

    This stage is more realistic than TinyStories because it contains broader
    educational web text.
    """

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    for row in dataset:
        text = row["text"].strip()  # type: ignore
        if text:
            yield text


def iter_openwebtext() -> Iterator[str]:
    """
    Stream OpenWebText-style documents.

    OpenWebText is a public attempt to imitate the style of GPT-2's WebText.
    It is not the exact GPT-2 dataset, but it is useful for GPT-2 like training.
    """

    dataset = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
    )

    for row in dataset:
        text = row["text"].strip()  # type: ignore
        if text:
            yield text


def iter_webmix(seed: int = 1337) -> Iterator[str]:
    """
    Stream a fixed mixture:

        70% FineWeb-Edu
        30% OpenWebText-style data

    This is the final mini GPT-2 pretraining stage:
    - FineWeb-Edu gives cleaner educational structure.
    - OpenWebText gives broader internet-text flavor.
    """

    rng = random.Random(seed)

    fineweb_iter = iter_finewebedu()
    openweb_iter = iter_openwebtext()

    fineweb_ratio = STAGE_CONFIGS["webmix"]["finewebedu_ratio"]

    while True:
        use_fineweb = rng.random() < fineweb_ratio

        try:
            if use_fineweb:
                yield next(fineweb_iter)
            else:
                yield next(openweb_iter)
        except StopIteration:
            break


def iter_local_text_file(input_file: Path) -> Iterator[str]:
    """
    Yield documents from a local text file.

    Blank lines separate documents.

    Example:

        data/raw/input.txt

    Then run:

        python -m data.prepare --stage local --input_file data/raw/input.txt
    """

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    text = input_file.read_text(encoding="utf-8")

    for block in text.split("\n\n"):
        block = block.strip()
        if block:
            yield block


def get_documents(stage: str, input_file: Path | None) -> Iterable[str]:
    """
    Return the document iterator for a given pretraining stage.
    """

    if stage == "tinystories":
        return iter_tinystories()

    if stage == "finewebedu":
        return iter_finewebedu()

    if stage == "webmix":
        return iter_webmix()

    if stage == "local":
        if input_file is None:
            raise ValueError("--input_file is required for --stage local")
        return iter_local_text_file(input_file)

    raise ValueError(f"Unknown stage: {stage}")


# ---------------------------------------------------------------------
# Token writing utilities
# ---------------------------------------------------------------------


def stage_dir(stage: str) -> Path:
    """
    Each stage gets its own processed directory.

    Example:

        data/processed/tinystories/train.bin
        data/processed/finewebedu/train.bin
        data/processed/webmix/train.bin

    This prevents later stages from overwriting earlier stages.
    """

    return PROCESSED_DIR / stage


def write_tokens_streaming(
    *,
    docs: Iterable[str],
    tokenizer: GPT2Tokenizer,
    output_file: Path,
    max_tokens: int,
) -> int:
    """
    Tokenize documents and write token IDs directly to disk.

    Important:
    We do NOT keep all tokens in a Python list. That would waste RAM for
    large datasets. Instead, we write chunks directly to a temporary file.

    The file stores uint16 token IDs because GPT-2 vocab size is 50257,
    and uint16 can store up to 65535.
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0

    with open(output_file, "wb") as f:
        progress = tqdm(total=max_tokens, desc="Tokenizing", unit="tok")

        for doc in docs:
            # Stop exactly at max_tokens.
            remaining = max_tokens - total_tokens
            if remaining <= 0:
                break

            # Append GPT-2's end-of-text token after every document.
            token_ids = tokenizer.encode_with_eot(doc)[:remaining]

            arr = np.array(token_ids, dtype=np.uint16)
            arr.tofile(f)

            total_tokens += len(token_ids)
            progress.update(len(token_ids))

        progress.close()

    return total_tokens


def copy_token_range(
    *,
    source_file: Path,
    target_file: Path,
    start: int,
    end: int,
    chunk_size: int = 10_000_000,
) -> None:
    """
    Copy a token range from source_file into target_file.

    This is used to split:

        all.bin -> train.bin + val.bin

    We copy in chunks to avoid loading the full token array into RAM.
    """

    source = np.memmap(source_file, dtype=np.uint16, mode="r")
    target_file.parent.mkdir(parents=True, exist_ok=True)

    with open(target_file, "wb") as f:
        for i in tqdm(
            range(start, end, chunk_size),
            desc=f"Writing {target_file.name}",
            unit="chunk",
        ):
            j = min(i + chunk_size, end)
            chunk = np.asarray(source[i:j], dtype=np.uint16)
            chunk.tofile(f)


def split_all_bin(
    *,
    all_bin: Path,
    train_bin: Path,
    val_bin: Path,
    total_tokens: int,
    val_fraction: float,
) -> tuple[int, int]:
    """
    Split one long token stream into train and validation files.

    We use the last val_fraction of tokens for validation.
    """

    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1.")

    val_tokens = int(total_tokens * val_fraction)
    train_tokens = total_tokens - val_tokens

    copy_token_range(
        source_file=all_bin,
        target_file=train_bin,
        start=0,
        end=train_tokens,
    )

    copy_token_range(
        source_file=all_bin,
        target_file=val_bin,
        start=train_tokens,
        end=total_tokens,
    )

    return train_tokens, val_tokens


def save_stats(
    *,
    stage: str,
    output_dir: Path,
    total_tokens: int,
    train_tokens: int,
    val_tokens: int,
    max_tokens: int,
    val_fraction: float,
    tokenizer: GPT2Tokenizer,
) -> None:
    """
    Save metadata so later training runs know what dataset was used.
    """

    stats = {
        "stage": stage,
        "description": STAGE_CONFIGS[stage]["description"],
        "total_tokens": total_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "requested_max_tokens": max_tokens,
        "val_fraction": val_fraction,
        "tokenizer": "gpt2",
        "vocab_size": tokenizer.vocab_size,
        "eot_token": tokenizer.eot_token,
        "dtype": "uint16",
        "train_bin": str((output_dir / "train.bin").relative_to(PROJECT_ROOT)),
        "val_bin": str((output_dir / "val.bin").relative_to(PROJECT_ROOT)),
    }

    if stage == "webmix":
        stats["finewebedu_ratio"] = STAGE_CONFIGS["webmix"]["finewebedu_ratio"]
        stats["openwebtext_ratio"] = 1.0 - STAGE_CONFIGS["webmix"]["finewebedu_ratio"]

    stats_file = output_dir / "stats.json"

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


# ---------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------


def prepare_stage(
    *,
    stage: str,
    max_tokens: int,
    val_fraction: float,
    input_file: Path | None,
    keep_all_bin: bool,
) -> None:
    """
    Prepare one pretraining stage.

    Output structure:

        data/processed/<stage>/
        ├── all.bin      optional temporary full stream
        ├── train.bin
        ├── val.bin
        └── stats.json
    """

    if stage not in STAGE_CONFIGS:
        raise ValueError(f"Unknown stage: {stage}")

    if max_tokens < 10_000:
        raise ValueError("Use at least 10,000 tokens for a meaningful dataset.")

    tokenizer = GPT2Tokenizer()
    docs = get_documents(stage=stage, input_file=input_file)

    out_dir = stage_dir(stage)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_bin = out_dir / "all.bin"
    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"

    print(f"\nPreparing stage: {stage}")
    print(STAGE_CONFIGS[stage]["description"])
    print(f"Output directory: {out_dir.relative_to(PROJECT_ROOT)}")
    print(f"Max tokens: {max_tokens:,}")

    total_tokens = write_tokens_streaming(
        docs=docs,
        tokenizer=tokenizer,
        output_file=all_bin,
        max_tokens=max_tokens,
    )

    if total_tokens < 10_000:
        raise RuntimeError(
            f"Only collected {total_tokens:,} tokens. Dataset is too small."
        )

    train_tokens, val_tokens = split_all_bin(
        all_bin=all_bin,
        train_bin=train_bin,
        val_bin=val_bin,
        total_tokens=total_tokens,
        val_fraction=val_fraction,
    )

    save_stats(
        stage=stage,
        output_dir=out_dir,
        total_tokens=total_tokens,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        max_tokens=max_tokens,
        val_fraction=val_fraction,
        tokenizer=tokenizer,
    )

    if not keep_all_bin:
        all_bin.unlink(missing_ok=True)

    print("\nDone.")
    print(f"Stage:        {stage}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Train tokens: {train_tokens:,}")
    print(f"Val tokens:   {val_tokens:,}")
    print(f"Saved:        {(train_bin).relative_to(PROJECT_ROOT)}")
    print(f"Saved:        {(val_bin).relative_to(PROJECT_ROOT)}")
    print(f"Saved:        {(out_dir / 'stats.json').relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """
    Minimal command line interface.

    Only expose what we actually need right now:
    - stage
    - max_tokens
    - local input file, only if needed
    """

    parser = argparse.ArgumentParser(
        description="Prepare staged GPT-2 BPE pretraining data."
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["tinystories", "finewebedu", "webmix", "local"],
        required=True,
        help="Which pretraining stage to prepare.",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10_000_000,
        help="Maximum number of GPT-2 BPE tokens to prepare.",
    )

    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.01,
        help="Fraction of tokens used for validation.",
    )

    parser.add_argument(
        "--input_file",
        type=Path,
        default=None,
        help="Only used with --stage local.",
    )

    parser.add_argument(
        "--keep_all_bin",
        action="store_true",
        help="Keep temporary all.bin file after creating train.bin/val.bin.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prepare_stage(
        stage=args.stage,
        max_tokens=args.max_tokens,
        val_fraction=args.val_fraction,
        input_file=args.input_file,
        keep_all_bin=args.keep_all_bin,
    )

    print("Dataset preparation complete.")
