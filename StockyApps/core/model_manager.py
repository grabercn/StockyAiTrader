"""
Model Manager — tracks, downloads, and manages ML models used by StockyAiTrader.

Provides status information about each required model:
- Whether it's downloaded
- How much disk space it uses
- Download progress when fetching

Currently manages:
- FinBERT (ProsusAI/finbert) — ~300MB, used for financial sentiment analysis
- LightGBM per-ticker models — small (<1MB each), created during training
"""

import os
import shutil
import threading
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path


@dataclass
class ModelInfo:
    """Information about a managed model."""
    name: str
    description: str
    hf_repo: Optional[str]  # HuggingFace repo ID, or None for local-only models
    size_estimate: str       # Human-readable expected size
    required: bool           # Whether the app needs this to function


# ─── Registry of all models we use ──────────────────────────────────────────

MANAGED_MODELS = [
    ModelInfo(
        name="FinBERT",
        description="Financial sentiment analysis (core)",
        hf_repo="ProsusAI/finbert",
        size_estimate="~300 MB",
        required=True,
    ),
    ModelInfo(
        name="FinBERT-Tone",
        description="Analyst report sentiment (addon)",
        hf_repo="yiyanghkust/finbert-tone",
        size_estimate="~420 MB",
        required=False,
    ),
    ModelInfo(
        name="Twitter-RoBERTa",
        description="Social media sentiment (addon)",
        hf_repo="cardiffnlp/twitter-roberta-base-sentiment-latest",
        size_estimate="~500 MB",
        required=False,
    ),
]


def get_hf_cache_dir():
    """Path to the HuggingFace model cache directory."""
    return os.path.join(Path.home(), ".cache", "huggingface", "hub")


def get_lgbm_model_dir():
    """Path to our trained LightGBM models."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "models")


def _hf_model_cache_path(repo_id):
    """Convert a HuggingFace repo ID to its cache directory name."""
    # HuggingFace stores models as "models--org--name"
    return os.path.join(get_hf_cache_dir(), f"models--{repo_id.replace('/', '--')}")


def _dir_size_bytes(path):
    """Total size of a directory in bytes."""
    total = 0
    if os.path.isdir(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    return total


def _format_size(bytes_val):
    """Convert bytes to human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024 ** 2:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 ** 3:
        return f"{bytes_val / 1024**2:.1f} MB"
    else:
        return f"{bytes_val / 1024**3:.2f} GB"


# ─── Public API ──────────────────────────────────────────────────────────────

def get_model_status(model_info):
    """
    Check if a model is downloaded and how much space it uses.

    Returns:
        (is_downloaded: bool, size_str: str)
    """
    if model_info.hf_repo:
        cache_path = _hf_model_cache_path(model_info.hf_repo)
        if os.path.isdir(cache_path):
            size = _dir_size_bytes(cache_path)
            return True, _format_size(size)
        return False, "Not downloaded"
    return False, "N/A"


def get_lgbm_models():
    """
    List all trained LightGBM ticker models.

    Returns:
        List of (filename, size_str) tuples
    """
    model_dir = get_lgbm_model_dir()
    if not os.path.isdir(model_dir):
        return []

    models = []
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(".txt"):
            path = os.path.join(model_dir, f)
            size = os.path.getsize(path)
            models.append((f, _format_size(size)))
    return models


def download_model(model_info, progress_callback=None):
    """
    Download a HuggingFace model in a background thread.

    Args:
        model_info:        ModelInfo to download
        progress_callback: Called with (status_str) on progress updates

    The model is downloaded by importing and loading it via transformers,
    which caches it to ~/.cache/huggingface/hub/.
    """
    def _download():
        try:
            if progress_callback:
                progress_callback(f"Downloading {model_info.name}...")

            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            if progress_callback:
                progress_callback(f"Downloading {model_info.name} tokenizer...")
            AutoTokenizer.from_pretrained(model_info.hf_repo)
            if progress_callback:
                progress_callback(f"Downloading {model_info.name} model weights...")
            AutoModelForSequenceClassification.from_pretrained(model_info.hf_repo)

            if progress_callback:
                progress_callback(f"{model_info.name} downloaded successfully!")

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error downloading {model_info.name}: {e}")

    thread = threading.Thread(target=_download, daemon=True)
    thread.start()
    return thread


def delete_model(model_info):
    """
    Delete a cached HuggingFace model.

    Returns:
        (success: bool, message: str)
    """
    if model_info.hf_repo:
        cache_path = _hf_model_cache_path(model_info.hf_repo)
        if os.path.isdir(cache_path):
            shutil.rmtree(cache_path)
            return True, f"{model_info.name} deleted."
        return False, f"{model_info.name} not found in cache."
    return False, "Cannot delete non-HuggingFace model."


def delete_lgbm_model(filename):
    """Delete a single trained LightGBM model file."""
    path = os.path.join(get_lgbm_model_dir(), filename)
    if os.path.isfile(path):
        os.remove(path)
        return True, f"Deleted {filename}"
    return False, f"{filename} not found."


def delete_all_lgbm_models():
    """Delete all trained LightGBM models."""
    model_dir = get_lgbm_model_dir()
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        return True, "All LightGBM models deleted."
    return False, "Model directory not found."
