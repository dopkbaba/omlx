"""Model management for omlx — handles listing, installing, and removing ONNX/ML models."""

import os
import json
import shutil
from pathlib import Path
from typing import Optional


MODEL_INDEX_FILENAME = "index.json"


def load_index(data_dir: Path) -> dict:
    """Load the local model index from the data directory."""
    index_path = data_dir / MODEL_INDEX_FILENAME
    if not index_path.exists():
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(data_dir: Path, index: dict) -> None:
    """Persist the model index to disk."""
    data_dir.mkdir(parents=True, exist_ok=True)
    index_path = data_dir / MODEL_INDEX_FILENAME
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def list_models(data_dir: Path) -> list[dict]:
    """Return a list of installed model metadata dicts, sorted by name."""
    index = load_index(data_dir)
    # Sort by name so output is predictable regardless of insertion order
    return sorted(index.values(), key=lambda m: m.get("name", ""))


def get_model(data_dir: Path, name: str) -> Optional[dict]:
    """Return metadata for a specific model, or None if not installed."""
    index = load_index(data_dir)
    return index.get(name)


def register_model(data_dir: Path, name: str, metadata: dict) -> None:
    """Add or update a model entry in the local index."""
    index = load_index(data_dir)
    index[name] = {"name": name, **metadata}
    save_index(data_dir, index)


def remove_model(data_dir: Path, name: str) -> bool:
    """Remove a model from the index and delete its files.

    Returns True if the model was found and removed, False otherwise.
    """
    index = load_index(data_dir)
    if name not in index:
        return False

    meta = index.pop(name)
    model_dir = data_dir / name
    if model_dir.exists() and model_dir.is_dir():
        shutil.rmtree(model_dir)
    else:
        # Warn if the index referenced a model whose directory is already missing
        import warnings
        warnings.warn(f"Model '{name}' was in the index but its directory was not found.")

    save_index(data_dir, index)
    return True


def model_path(data_dir: Path, name: str) -> Path:
    """Return the directory path where a model's files are stored."""
    return data_dir / name


def ensure_model_dir(data_dir: Path, name: str) -> Path:
    """Create and return the model's storage directory."""
    path = model_path(data_dir, name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_models_by_tag(data_dir: Path, tag: str) -> list[dict]:
    """Return all installed models that include the given tag in their metadata.

    Useful for filtering models by task type, e.g. 'classification' or 'nlp'.
    Matching is case-insensitive so 'NLP' and 'nlp' both work.
    """
    tag_lower = tag.lower()
    return [
        m for m in list_models(data_dir)
        if tag_lower in [t.lower() for t in m.get("tags", [])]
    ]
