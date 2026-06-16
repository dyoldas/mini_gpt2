# mini_gpt2

## Author: Deniz Yoldas

Recreating the OpenAI's GPT2 model on a smaller scale.

---

## Project Structure

```
mini_gpt2/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── gpt2_tiny.yaml        # 10M-ish debug model
│   ├── gpt2_small.yaml       # 30M-60M serious local model
│   └── gpt2_124m.yaml        # GPT-2 small style config
│
├── data/
│   ├── raw/                  # downloaded/raw text files
│   ├── processed/            # tokenized train.bin / val.bin
│   └── prepare.py            # dataset download/clean/tokenize
│
├── tokenizer/
│   ├── __init__.py
│   ├── gpt2_tokenizer.py     # tiktoken GPT-2 BPE wrapper
│   └── train_bpe.py          # training BPE from scratch
│
├── models/
│   ├── __init__.py
│   ├── gpt2.py               # GPT-2 style model
│   └── modules.py            # attention, MLP, block, layernorm helpers
│
├── training/
│   ├── train.py              # main pretraining script
│   ├── trainer.py            # training loop logic
│   ├── dataloader.py         # binary token batch loader
│   └── utils.py              # seed, device, checkpoint helpers
│
├── inference/
│   ├── generate.py           # sample text from checkpoint
│   └── chat.py               # simple prompt/completion interface
│
├── evaluation/
│   ├── eval_loss.py          # validation loss/perplexity
│   ├── sample_prompts.py     # fixed qualitative prompts
│   └── compare_runs.py       # compare 10M/30M/60M runs
│
├── checkpoints/
│   └── .gitkeep
│
├── runs/
│   └── .gitkeep             # logs, loss curves, samples
│
├── scripts/
│   ├── train_tiny.sh
│   ├── train_small.sh
│   └── generate.sh
│
└── notebooks/
    └── inspect_model.ipynb   # optional analysis/plots
```

---

## Data Preparation

This project uses staged GPT-2 BPE pretraining data. Raw text is streamed from datasets, tokenized with the GPT-2 tokenizer, and saved as binary token files.

Output format:

```
data/processed/<stage>/
├── train.bin
├── val.bin
└── stats.json
```

Each `.bin` file stores GPT-2 token IDs as `uint16`.

### Pretraining stages

| Stage         | Purpose                                               |
| ------------- | ----------------------------------------------------- |
| `tinystories` | First easy pretraining stage for simple coherent text |
| `finewebedu`  | Cleaner educational web-text pretraining              |
| `webmix`      | 70% FineWeb-Edu + 30% OpenWebText-style data          |
| `local`       | Debug using a local text file                         |

### Commands

Prepare TinyStories first:

```bash
python -m data.prepare --stage tinystories --max_tokens 5000000
```

Prepare FineWeb-Edu later:

```bash
python -m data.prepare --stage finewebedu --max_tokens 50000000
```

Prepare mixed web-text stage:

```bash
python -m data.prepare --stage webmix --max_tokens 50000000
```

Prepare a local text file:

```bash
python -m data.prepare --stage local --input_file data/raw/input.txt
```

### Training progression

The intended pretraining order is:

```
TinyStories -> FineWeb-Edu -> FineWeb-Edu/OpenWebText mix
```

This lets the model first learn simple language structure, then move toward broader GPT-2-style web-text pretraining.

---

## Related Papers

- Language Models are Unsupervised Multitask Learners (GPT2, OpenAI)
- Attention Is All You Need (Transformer, Google)
- Language Models Are Few-Shot Learners (GPT3, OpenAI)
- Scaling Laws for Neural Language Models (Scaling, OpenAI)
- Training Compute-Optimal Large Language Models (Optimal Scaling, Google)
- LLama: Open and Efficient Foundation Language Models (LLama, Meta)