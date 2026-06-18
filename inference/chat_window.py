"""
Native Tkinter popup chat window for mini_gpt2.

This is a simple desktop interface:
- type a prompt
- click Generate
- model continues the text
- response appears in the chat box

Example run:

    python -m inference.chat_window \
        --checkpoint checkpoints/gpt2_50m_tinystories/ckpt_best.pt
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import tkinter as tk
from tkinter import scrolledtext

from inference.generate import (
    generate_text,
    get_device,
    load_model,
)
from tokenizer import GPT2Tokenizer


class MiniGPT2ChatApp:
    """
    Small Tkinter app for prompting the trained model.

    Generation runs in a background thread so the window does not freeze.
    """

    def __init__(
        self,
        root: tk.Tk,
        *,
        checkpoint: Path,
        device_name: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> None:
        self.root = root
        self.root.title("mini_gpt2 Chat")
        self.root.geometry("850x650")

        self.device = get_device(device_name)
        self.tokenizer = GPT2Tokenizer()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

        # Load model once at startup.
        self.model = load_model(
            checkpoint_path=checkpoint,
            device=self.device,
        )

        self._build_ui()

    def _build_ui(self) -> None:
        """
        Build the chat window layout.
        """

        self.chat_box = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Menlo", 13),
        )
        self.chat_box.pack(
            padx=12,
            pady=12,
            fill=tk.BOTH,
            expand=True,
        )

        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(
            padx=12,
            pady=(0, 12),
            fill=tk.X,
        )

        self.prompt_entry = tk.Text(
            bottom_frame,
            height=4,
            font=("Menlo", 13),
        )
        self.prompt_entry.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
        )

        self.generate_button = tk.Button(
            bottom_frame,
            text="Generate",
            command=self.on_generate_clicked,
            width=12,
        )
        self.generate_button.pack(
            side=tk.RIGHT,
            padx=(10, 0),
            fill=tk.Y,
        )

        self._append_system_message(
            "Loaded mini_gpt2. Type a prompt and press Generate.\n"
        )

    def _append_text(self, text: str) -> None:
        """
        Append text to the chat box.
        """

        self.chat_box.insert(tk.END, text)
        self.chat_box.see(tk.END)

    def _append_system_message(self, text: str) -> None:
        self._append_text(f"[system] {text}\n")

    def on_generate_clicked(self) -> None:
        """
        Start generation from the current prompt.
        """

        prompt = self.prompt_entry.get("1.0", tk.END).strip()

        if not prompt:
            return

        self.prompt_entry.delete("1.0", tk.END)

        self._append_text(f"\nYou:\n{prompt}\n\n")
        self._append_text("mini_gpt2:\n")

        self.generate_button.config(state=tk.DISABLED)

        thread = threading.Thread(
            target=self._generate_in_background,
            args=(prompt,),
            daemon=True,
        )
        thread.start()

    def _generate_in_background(self, prompt: str) -> None:
        """
        Generate text in a background thread.

        Tkinter UI updates must happen on the main thread, so we use
        root.after(...) to safely update the chat box.
        """

        try:
            output = generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=self.device,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
            )

            # Show only continuation after the prompt if possible.
            if output.startswith(prompt):
                output = output[len(prompt) :].lstrip()

            self.root.after(
                0,
                lambda: self._finish_generation(output),
            )

        except Exception as exc:
            error_msg = f"[error] {exc}"
            self.root.after(0, lambda: self._finish_generation(error_msg))

    def _finish_generation(self, output: str) -> None:
        """
        Display generated output and re-enable button.
        """

        self._append_text(output + "\n")
        self.generate_button.config(state=tk.NORMAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tkinter chat for mini_gpt2.")

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint file.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Number of tokens to generate per prompt.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling. Use 0 to disable.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    top_k = None if args.top_k == 0 else args.top_k

    root = tk.Tk()

    MiniGPT2ChatApp(
        root,
        checkpoint=args.checkpoint,
        device_name=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
