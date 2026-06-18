# mini_gpt2

## Author: Deniz Yoldas

A small-scale recreation of the GPT-2 training pipeline for learning how token-level decoder-only language models are built, pretrained, continued on new datasets, and used for inference.

The goal is not to reproduce OpenAI's original GPT-2 training exactly. The goal is to build a clean educational GPT-2-style system from scratch.

---

## Project Goals

This project implements:

* GPT-2 BPE tokenization
* staged pretraining data preparation
* binary token dataloading
* GPT-2-style decoder Transformer
* next-token prediction training
* checkpoint saving and loading
* text generation from trained checkpoints
* native Tkinter popup chat window
* optional loading of real GPT-2 weights into the local infrastructure

---

## Project Structure

```
mini_gpt2/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── gpt2_50m_tinystories.yaml
│   ├── gpt2_tiny.yaml
│   └── ...
│
├── data/
|   ├── __init__.py
│   ├── raw/
│   ├── processed/
│   └── prepare.py
│
├── tokenizer/
│   ├── __init__.py
│   ├── gpt2_tokenizer.py
│
├── models/
│   ├── __init__.py
│   ├── gpt2.py
│   └── modules.py
│
├── training/
│   ├── __init__.py
│   ├── dataloader.py
│   └── train.py
│
├── inference/
│   ├── __init__.py
│   ├── generate.py
│   └── chat_window.py
│
├── scripts/
│   └── download_real_gpt2.py
|
└── checkpoints/
```

---

## Installation

Create and activate your environment, then install:

```bash
python -m pip install -r requirements.txt
```

Current dependencies:

```
torch
datasets
tqdm
pyyaml
transformers
tiktoken
numpy
huggingface-hub
matplotlib
```

`tkinter` is used for the popup chat window and is included with standard Python on macOS.

---

## Data Preparation

The data pipeline streams text datasets, tokenizes them with GPT-2 BPE, and saves binary token files.

Output format:

```text
data/processed/<stage>/
├── train.bin
├── val.bin
└── stats.json
```

Each `.bin` file stores GPT-2 token IDs as `uint16`.

### Pretraining Stages

| Stage         | Purpose                                       |
| ------------- | --------------------------------------------- |
| `tinystories` | First easy stage for coherent simple language |
| `finewebedu`  | Cleaner educational web-text pretraining      |
| `webmix`      | 70% FineWeb-Edu + 30% OpenWebText-style data  |
| `local`       | Debugging with a local text file              |

### Prepare TinyStories

Small debug run:

```bash
python -m data.prepare --stage tinystories --max_tokens 100000
```

First serious TinyStories run:

```bash
python -m data.prepare --stage tinystories --max_tokens 50000000
```

### Prepare Later Stages

FineWeb-Edu:

```bash
python -m data.prepare --stage finewebedu --max_tokens 100000000
```

WebMix:

```bash
python -m data.prepare --stage webmix --max_tokens 100000000
```

Local file:

```bash
python -m data.prepare --stage local --input_file data/raw/input.txt
```

---

## Training

Training uses:

```text
train.bin / val.bin
→ dataloader
→ GPT model
→ next-token prediction loss
→ AdamW update
→ checkpoint saving
```

### Tiny Debug Training

```bash
python -m training.train --config configs/gpt2_tiny.yaml
```

### Serious TinyStories Training

```bash
python -m training.train --config configs/gpt2_50m_tinystories.yaml
```

### Prevent macOS Sleep During Training

Use `caffeinate`:

```bash
caffeinate -dims python -m training.train --config configs/gpt2_50m_tinystories.yaml
```

This keeps the Mac awake while the training process runs.

---

## Checkpoints

Training saves checkpoints into:

```text
checkpoints/<run_name>/
├── ckpt_last.pt
└── ckpt_best.pt
```

| File           | Meaning                     |
| -------------- | --------------------------- |
| `ckpt_last.pt` | Latest saved training state |
| `ckpt_best.pt` | Best validation-loss model  |

Use `ckpt_best.pt` for generation.

Example:

```text
checkpoints/gpt2_50m_tinystories/ckpt_best.pt
```

---

## Progressive / Staged Pretraining

The intended training path is:

```text
TinyStories → FineWeb-Edu → WebMix
```

This is curriculum-style training. It is not exactly how GPT-2 was trained, but it is useful for learning.

### Stage 1: Train from Scratch on TinyStories

Config idea:

```yaml
data:
  stage: "tinystories"

checkpoint:
  out_dir: "checkpoints/gpt2_50m_tinystories"
  save_best: true
  resume_from: null
```

Run:

```bash
caffeinate -dims python -m training.train --config configs/gpt2_50m_tinystories.yaml
```

### Stage 2: Continue on FineWeb-Edu

Create a new config, for example:

```
configs/gpt2_50m_finewebedu.yaml
```

Use:

```yaml
data:
  stage: "finewebedu"

checkpoint:
  out_dir: "checkpoints/gpt2_50m_finewebedu"
  save_best: true
  resume_from: "checkpoints/gpt2_50m_tinystories/ckpt_best.pt"
```

Run:

```bash
caffeinate -dims python -m training.train --config configs/gpt2_50m_finewebedu.yaml
```

### Stage 3: Continue on WebMix

Create:

```
configs/gpt2_50m_webmix.yaml
```

Use:

```yaml
data:
  stage: "webmix"

checkpoint:
  out_dir: "checkpoints/gpt2_50m_webmix"
  save_best: true
  resume_from: "checkpoints/gpt2_50m_finewebedu/ckpt_best.pt"
```

Run:

```bash
caffeinate -dims python -m training.train --config configs/gpt2_50m_webmix.yaml
```

---

## Text Generation

Generate from your trained checkpoint:

```bash
python -m inference.generate \
  --checkpoint checkpoints/gpt2_50m_tinystories/ckpt_best.pt \
  --prompt "Once upon a time" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_k 50
```

Useful TinyStories prompts:

```text
Once upon a time
Lily went to the forest and saw
The little robot wanted to learn
Tom found a strange box
The dragon was afraid because
```

---

## Popup Chat Window

Run the native Tkinter chat window:

```bash
python -m inference.chat_window \
  --checkpoint checkpoints/gpt2_50m_tinystories/ckpt_best.pt
```

With custom sampling:

```bash
python -m inference.chat_window \
  --checkpoint checkpoints/gpt2_50m_tinystories/ckpt_best.pt \
  --temperature 0.7 \
  --top_k 40 \
  --max_new_tokens 200
```

---

## Using Real GPT-2 Weights

You can also store real GPT-2 weights under `checkpoints/` and use them through the same generation/chat infrastructure.

Run the following script to download original weights from huggingface (***Total download size is ~10Gb***)

```bash
python scripts/download_real_gpt2.py
```

Expected format:

```text
checkpoints/
├── gpt2/
│   ├── pytorch_model.pt   # 117 M
│   └── config.json
├── gpt2-medium/
│   ├── pytorch_model.pt   # 345 M
│   └── config.json
├── gpt2-large/
│   ├── pytorch_model.pt   # 762 M
│   └── config.json
└── gpt2-xl/
    ├── pytorch_model.pt   # 1542 M
    └── config.json
```

In this project:

```text
checkpoint path is a file   → load custom mini_gpt2 checkpoint
checkpoint path is a folder → load real GPT-2 weights into custom GPT class
```

### Generate with Real GPT-2

```bash
python -m inference.generate \
  --checkpoint checkpoints/gpt2-large \
  --prompt "The future of artificial intelligence is" \
  --max_new_tokens 200
```

### Chat with Real GPT-2

```bash
python -m inference.chat_window \
  --checkpoint checkpoints/gpt2-large
```

For the strongest GPT-2 model:

```bash
python -m inference.chat_window \
  --checkpoint checkpoints/gpt2-xl
```

---

## Important Notes

* `checkpoints/` stores model weights.
* `data/processed/` stores tokenized datasets.
* Tokenized data and large checkpoints should usually not be committed to normal Git.
* This project is GPT-2-style, but not an exact OpenAI GPT-2 reproduction.
* The original GPT-2 dataset, WebText, was not publicly released.

---

## Recommended Learning Path

```
1. Train tiny model on TinyStories
2. Train 50M model on TinyStories
3. Generate samples and inspect behavior
4. Continue training on FineWeb-Edu
5. Continue training on WebMix
6. Compare your model to real GPT-2 weights
7. Study GPT-3, scaling laws, Chinchilla, and LLaMA
```

Core intuition:

```
From char-level nanoGPT to:
→ GPT-2 BPE token model
→ staged pretraining
→ scaling experiments
→ real GPT-2 comparison
```

---

## Related Papers

- Attention Is All You Need (Transformer, Google, 2017)
- -Improving Language Understanding by Generative Pre-Training (GPT1, OpenAI, 2018)
- Language Models are Unsupervised Multitask Learners (GPT2, OpenAI, 2019)
- Language Models Are Few-Shot Learners (GPT3, OpenAI, 2020)
- Scaling Laws for Neural Language Models (Scaling, OpenAI, 2020)
- Training Compute-Optimal Large Language Models (Chinchilla, Google, 2022)
- LLama: Open and Efficient Foundation Language Models (LLama, Meta, 2023)