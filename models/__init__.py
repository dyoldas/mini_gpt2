"""
Model package for mini_gpt2.

Main exports:
- GPTConfig: configuration object for GPT model size
- GPT: decoder-only GPT-2 style language model
"""

from .gpt2 import GPT, GPTConfig

__all__ = ["GPT", "GPTConfig"]
