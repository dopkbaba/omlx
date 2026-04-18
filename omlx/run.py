"""Run a downloaded model using the appropriate backend."""

import os
import subprocess
import sys
from pathlib import Path

from .model import get_model, get_data_dir


def _find_llama_cpp() -> str | None:
    """Locate llama-cpp main binary in common install paths."""
    candidates = [
        "llama-cli",
        "llama.cpp/main",
        os.path.expanduser("~/.local/bin/llama-cli"),
        "/usr/local/bin/llama-cli",
        "/opt/homebrew/bin/llama-cli",
    ]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def run_model(
    name: str,
    prompt: str | None = None,
    n_ctx: int = 2048,
    n_predict: int = 512,
    interactive: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    """Run a model by name.

    Args:
        name: Model name or alias registered in the index.
        prompt: Optional prompt string. If None and not interactive, reads stdin.
        n_ctx: Context window size.
        n_predict: Max tokens to predict.
        interactive: Launch in interactive/chat mode.
        extra_args: Additional arguments forwarded to the backend binary.

    Returns:
        Exit code of the backend process.
    """
    model = get_model(name)
    if model is None:
        print(f"omlx: model '{name}' not found. Run 'omlx pull {name}' first.",
              file=sys.stderr)
        return 1

    model_path = Path(get_data_dir()) / model["file"]
    if not model_path.exists():
        print(f"omlx: model file not found at {model_path}", file=sys.stderr)
        return 1

    backend = _find_llama_cpp()
    if backend is None:
        print(
            "omlx: llama-cli not found. Install llama.cpp and ensure it is on PATH.",
            file=sys.stderr,
        )
        return 1

    cmd: list[str] = [
        backend,
        "--model", str(model_path),
        "--ctx-size", str(n_ctx),
        "--n-predict", str(n_predict),
    ]

    if interactive:
        cmd.append("--interactive-first")
    elif prompt:
        cmd.extend(["--prompt", prompt])

    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except KeyboardInterrupt:
        return 0


def cmd_run(args) -> int:
    """Entry point for the 'omlx run' sub-command."""
    return run_model(
        name=args.name,
        prompt=args.prompt,
        n_ctx=args.ctx_size,
        n_predict=args.n_predict,
        interactive=args.interactive,
        extra_args=args.extra or [],
    )
