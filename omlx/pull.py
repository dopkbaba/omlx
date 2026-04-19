"""Pull (download) models from remote sources."""

import os
import sys
import hashlib
import urllib.request
from urllib.error import URLError, HTTPError
from typing import Optional

from .model import get_model, register_model, get_data_dir


DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/jundot/omlx-registry/main/registry.json"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Display a simple download progress bar."""
    if total_size <= 0:
        downloaded = block_num * block_size
        sys.stdout.write(f"\rDownloaded {downloaded / 1024 / 1024:.1f} MB")
    else:
        downloaded = min(block_num * block_size, total_size)
        percent = downloaded / total_size * 100
        bar_len = 40
        filled = int(bar_len * downloaded / total_size)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {percent:.1f}%  {downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB")
    sys.stdout.flush()


def _verify_sha256(filepath: str, expected: str) -> bool:
    """Verify SHA-256 checksum of a downloaded file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected


def pull_model(name: str, tag: str = "latest", registry_url: Optional[str] = None) -> bool:
    """
    Pull a model by name and optional tag.

    Args:
        name: Model name (e.g. "llama3", "mistral")
        tag:  Model tag/version (default: "latest")
        registry_url: Override the default registry URL

    Returns:
        True on success, False on failure.
    """
    full_name = f"{name}:{tag}"

    # Check if already downloaded
    existing = get_model(full_name)
    if existing and os.path.exists(existing.get("path", "")):
        print(f"Model '{full_name}' is already present.")
        return True

    # Allow env var override; fall back to default registry
    registry_url = registry_url or os.environ.get("OMLX_REGISTRY_URL", DEFAULT_REGISTRY_URL)

    # Fetch registry to resolve model URL
    try:
        import json
        # Increased timeout from 10 to 30s — default was too aggressive on slow connections
        # Bumped further to 60s for my home network which can be pretty spotty
        # Using 90s now — 60s still times out occasionally on large registry fetches over VPN
        with urllib.request.urlopen(registry_url, timeout=90) as resp:
            registry = json.loads(resp.read().decode())
    except (URLError, HTTPError) as e:
        print(f"Error: Could not fetch registry from {registry_url}: {e}", file=sys.stderr)
        return False
    except json.JSONDecodeError as e:
        # Handy to catch this separately — registry occasionally returns a 200 with an error page
        print(f"Error: Registry response was not valid JSON: {e}", file=sys.stderr)
        return False

    models = registry.get("models", {})
    entry = models.get(name, {}).get(tag)
    if not entry:
        print(f"Error: Model '{full_name}' not found in registry.", file=sys.stderr)
        return False

    url: str = entry["url"]
    expected_sha256: Optional[str] = entry.get("
