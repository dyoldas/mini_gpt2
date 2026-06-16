"""
GPT-2 BPE tokenizer wrapper.

nano_gpt project used character-level tokenization:

    text -> characters -> integer IDs

This file upgrades the project to GPT-2-style BPE tokenization:

    text -> subword tokens -> integer IDs

We use `tiktoken`, which provides OpenAI-compatible tokenizers.
"""

from __future__ import annotations

from typing import List

import tiktoken  # type: ignore


class GPT2Tokenizer:
    """
    Small wrapper around the GPT-2 BPE tokenizer.

    This keeps tokenizer usage clean across the project.
    When we train our own BPE tokenizer, we can swap it here
    without changing the training code.
    """

    def __init__(self) -> None:
        # "gpt2" is the original GPT-2 BPE encoding.
        self.encoding = tiktoken.get_encoding("gpt2")

        # GPT-2 vocabulary size is 50257.
        self.vocab_size = self.encoding.n_vocab

        # GPT-2 uses token 50256 as the end-of-text token.
        self.eot_token = self.encoding.eot_token

    def encode(self, text: str) -> List[int]:
        """
        Convert text into token IDs.

        Example:
            "Hello world" -> [15496, 995]
        """
        return self.encoding.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back into text.

        Example:
            [15496, 995] -> "Hello world"
        """
        return self.encoding.decode(token_ids)

    def encode_with_eot(self, text: str) -> List[int]:
        """
        Encode text and append GPT-2's end-of-text token.

        Useful when combining documents during pretraining.
        """
        return self.encode(text) + [self.eot_token]

    def count_tokens(self, text: str) -> int:
        """
        Return number of BPE tokens in a string.
        """
        return len(self.encode(text))


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer()

    text = "This is a mini GPT-2 tokenizer test."

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("Original text:")
    print(text)

    print("\nToken IDs:")
    print(ids)

    print("\nDecoded text:")
    print(decoded)

    print("\nVocab size:")
    print(tokenizer.vocab_size)

    print("\nEnd-of-text token:")
    print(tokenizer.eot_token)

    print("\nNumber of tokens:")
    print(tokenizer.count_tokens(text))
