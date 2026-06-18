from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    GPT needs causal attention because token t can only use tokens <= t.
    """

    def __init__(
        self,
        *,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        # GPT-2 uses one packed projection for Q, K, V.
        # This is mathematically the same as three Linear layers, but faster/cleaner.
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)

        # After all heads are combined, project back into the residual stream.
        # This projection receives GPT-2 residual scaling during init.
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.GPT_SCALE_INIT = True  # type: ignore[attr-defined]

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Store causal mask inside the module.
        # register_buffer means:
        # - it moves with model.to(device)
        # - it is saved in state_dict
        # - it is not trainable
        # Mask shape is (1, 1, block_size, block_size) so it can be broadcasted
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        if T > self.block_size:
            raise ValueError("Sequence length exceeds block_size.")

        # Packed QKV projection:
        # x:   (B, T, C)
        # qkv: (B, T, 3C)
        qkv = self.c_attn(x)

        # Split packed projection into query, key, value.
        # So q, k, v are each (B, T, C).
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape into multiple heads:
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores:
        # each token compares its query against all keys.
        att = q @ k.transpose(-2, -1)

        # Scale by sqrt(head_dim).
        # Without this, dot products grow with dimension and softmax becomes too sharp.
        att = att / math.sqrt(self.head_dim)

        # Mask future positions.
        # Future tokens get -inf, so softmax turns them into probability 0.
        att = att.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,  # type: ignore
            float("-inf"),
        )

        # Row-wise softmax to get attention probabilities.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values.
        y = att @ v  # (B, n_head, T, head_dim)

        # Transpose swaps the head and sequence dimensions,
        # contiguous makes the tensor's memory layout in transpose form,
        # and view reshapes it to combine all heads into a single (B, T, C) tensor.
        # (B, n_head, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y


class MLP(nn.Module):
    """
    GPT-2 feed-forward network.

    Expansion ratio is 4x:
        n_embd -> 4*n_embd -> n_embd
    """

    def __init__(self, *, n_embd: int, dropout: float) -> None:
        super().__init__()

        self.c_fc = nn.Linear(n_embd, 4 * n_embd)

        # GPT-2 uses GELU. approximate="tanh" is the common fast approximation.
        self.gelu = nn.GELU(approximate="tanh")

        # Projection back into residual stream.
        # This also receives GPT-2 residual scaling.
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.GPT_SCALE_INIT = True  # type: ignore[attr-defined]

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    One GPT-2 Transformer block.

    GPT-2 uses pre-LayerNorm:

        x = x + attention(layernorm(x))
        x = x + mlp(layernorm(x))

    The residual stream carries information through the whole network.
    """

    def __init__(
        self,
        *,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd)

        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
        )

        self.ln_2 = nn.LayerNorm(n_embd)

        self.mlp = MLP(
            n_embd=n_embd,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
