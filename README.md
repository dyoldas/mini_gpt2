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

## Related Papers

- Language Models are Unsupervised Multitask Learners (GPT2, OpenAI)
- Attention Is All You Need (Transformer, Google)
- Language Models Are Few-Shot Learners (GPT3, OpenAI)
- Scaling Laws for Neural Language Models (Scaling, OpenAI)
- Training Compute-Optimal Large Language Models (Optimal Scaling, Google)
- LLama: Open and Efficient Foundation Language Models (LLama, Meta)