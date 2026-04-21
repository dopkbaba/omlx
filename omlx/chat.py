"""Interactive chat session support for omlx.

Provides a readline-based REPL that keeps a llama.cpp process alive
for multi-turn conversations, passing the accumulated prompt context
between turns.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from .model import get_model
from .run import _find_llama_cpp

# Sentinel that the user types to exit the chat loop
_EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", "bye"}

# Default system prompt injected at the start of every session
_DEFAULT_SYSTEM = (
    "You are a helpful, concise assistant. "
    "Answer the user's questions accurately and briefly."
)


def _build_prompt(history: List[dict], system: str) -> str:
    """Render the conversation history into a plain-text prompt.

    Uses the simple "### Human / ### Assistant" format understood by most
    instruction-tuned GGUF models.
    """
    lines: List[str] = []
    if system:
        lines.append(f"### System\n{system}\n")
    for turn in history:
        role = "Human" if turn["role"] == "user" else "Assistant"
        lines.append(f"### {role}\n{turn['content']}\n")
    # Leave the assistant turn open so the model completes it
    lines.append("### Assistant\n")
    return "\n".join(lines)


def chat_model(
    model_name: str,
    *,
    system: Optional[str] = None,
    ctx_size: int = 2048,
    temperature: float = 0.7,
    extra_args: Optional[List[str]] = None,
) -> None:
    """Start an interactive chat session with *model_name*.

    Parameters
    ----------
    model_name:
        Registered model alias (looked up via :func:`omlx.model.get_model`).
    system:
        System prompt to prepend. Defaults to :data:`_DEFAULT_SYSTEM`.
    ctx_size:
        Context window size passed to llama.cpp (``-c`` flag).
    temperature:
        Sampling temperature (``--temp`` flag).
    extra_args:
        Additional raw arguments forwarded to the llama.cpp binary.
    """
    llama_bin = _find_llama_cpp()
    if llama_bin is None:
        print(
            "error: llama.cpp binary not found. "
            "Install llama.cpp and make sure 'llama-cli' or 'main' is on PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    model_info = get_model(model_name)
    if model_info is None:
        print(f"error: model '{model_name}' not found. Run 'omlx pull {model_name}' first.",
              file=sys.stderr)
        sys.exit(1)

    model_path = Path(model_info["path"])
    if not model_path.exists():
        print(f"error: model file missing at {model_path}", file=sys.stderr)
        sys.exit(1)

    system_prompt = system or _DEFAULT_SYSTEM
    history: List[dict] = []

    print(textwrap.dedent(f"""\
        omlx chat  —  model: {model_name}
        Type 'exit' or 'quit' to end the session.
        ─────────────────────────────────────────"""))

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in _EXIT_COMMANDS:
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})
        prompt = _build_prompt(history, system_prompt)

        cmd = [
            llama_bin,
            "-m", str(model_path),
            "-c", str(ctx_size),
            "--temp", str(temperature),
            "-p", prompt,
            "--no-display-prompt",  # suppress echoing the prompt
            "-n", "512",            # max tokens per reply
        ]
        if extra_args:
            cmd.extend(extra_args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            reply = result.stdout.strip()
        except subprocess.CalledProcessError as exc:
            print(f"error: llama.cpp exited with code {exc.returncode}", file=sys.stderr)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)
            break

        print(f"Assistant: {reply}\n")
        history.append({"role": "assistant", "content": reply})


def cmd_chat(args) -> None:  # noqa: ANN001
    """Entry point wired up by the argument parser in :mod:`omlx.main`."""
    chat_model(
        args.model,
        system=getattr(args, "system", None),
        ctx_size=getattr(args, "ctx_size", 2048),
        temperature=getattr(args, "temperature", 0.7),
        extra_args=getattr(args, "extra_args", None),
    )
