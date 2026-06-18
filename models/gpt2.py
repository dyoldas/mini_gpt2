"""
GPT-2 style decoder-only language model.

Important GPT-2 details included:
- GPT-2 BPE vocab size
- learned absolute positional embeddings
- packed QKV attention
- pre-LayerNorm Transformer blocks
- GELU MLP
- weight tying
- residual projection scaling
- next-token prediction loss
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import TransformerBlock


@dataclass
class GPTConfig:
    """
    Model size/configuration.

    GPT-2 small style:
        vocab_size = 50257
        block_size = 1024
        n_layer = 12
        n_head = 12
        n_embd = 768
    """

    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class GPT(nn.Module):
    """
    Decoder-only GPT language model.

    Training:
        input  = [x0, x1, x2, ...]
        target = [x1, x2, x3, ...]

    The model learns next-token prediction.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.config = config

        # Token embeddings convert token IDs into vectors.
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        # GPT-2 uses learned absolute position embeddings.
        # Position 0, 1, 2, ... each has its own learned vector.
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        # Maps hidden states back to vocabulary logits.
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        # Initialize all parameters.
        self.apply(self._init_weights)

        # Weight tying:
        # reuse the input token embedding matrix as the output vocabulary classifier.
        # h @ lm_head.weight.T, maps the embedding dimension back to vocab size.
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        """
        GPT-style initialization.

        Normal layers use std=0.02.

        Residual output projections use smaller std: 0.02 / sqrt(2 * n_layer)

        Head scaling:
        where?  attention score computation
        when?   every forward pass
        why?    stabilize softmax attention

        Residual scaling:
        where?  weight initialization of residual output projections
        when?   once, before training
        why?    reduces the chance that activations/gradients blow up at the start of training

        Note: we scale only the residual-output projections because those are the parts repeatedly added into the residual stream,
        making them smaller prevents layer-by-layer variance growth without weakening all internal attention and mlp computations.
        """

        if isinstance(module, nn.Linear):
            std = 0.02

            # modules.py marks attention/MLP output projections with this flag.
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5

            nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        idx:
            token IDs, shape (B, T)

        targets:
            next-token labels, shape (B, T)
        """

        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError("Input sequence exceeds block_size.")

        tok_emb = self.token_embedding(idx)  # (B, T, C)

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        pos_emb = self.position_embedding(pos)  # (T, C)

        # Token identity + position.
        x = tok_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        At each step:
        1. crop context to block_size
        2. predict next-token logits
        3. sample one token
        4. append it
        """

        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]

            logits, _ = self(idx_cond)

            # Use only the final position to sample the next token.
            logits = logits[:, -1, :]

            # Lower temperature = sharper distribution.
            # Higher temperature = more random sampling.
            logits = logits / temperature

            # Top-k keeps only the k most likely tokens.
            if top_k is not None:
                values, _ = torch.topk(logits, k=top_k)
                cutoff = values[:, [-1]]

                logits = torch.where(
                    logits < cutoff,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def num_parameters(self, non_embedding: bool = False) -> int:
        """
        Count parameters.

        non_embedding=True excludes positional embeddings.
        This is sometimes used when comparing GPT parameter counts.
        """

        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n_params -= self.position_embedding.weight.numel()

        return n_params


if __name__ == "__main__":
    """
    Quick test:

        python -m models.gpt2
    """

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    config = GPTConfig(
        vocab_size=50257,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1,
    )

    model = GPT(config).to(device)

    print("Model created.")
    print("Device:", device)
    print("Parameters:", f"{model.num_parameters():,}")

    x = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(4, config.block_size),
        device=device,
    )

    y = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(4, config.block_size),
        device=device,
    )

    logits, loss = model(x, y)

    print("logits shape:", logits.shape)
    print("loss:", loss.item() if loss is not None else None)

    start = torch.tensor([[15496]], dtype=torch.long, device=device)  # "Hello"
    out = model.generate(
        start,
        max_new_tokens=10,
        temperature=1.0,
        top_k=50,
    )

    print("generated token ids:", out[0].tolist())
