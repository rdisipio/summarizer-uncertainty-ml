"""Helpers for provisioning NLTK resources in local and container environments."""

from __future__ import annotations

import os
from pathlib import Path

import nltk


_PUNKT_RESOURCE_CANDIDATES = (
    "tokenizers/punkt_tab/english",
    "tokenizers/punkt/english.pickle",
)


def ensure_sentence_tokenizer(download: bool = False) -> None:
    """Ensure sentence-tokenizer resources are available for NLTK.

    When ``download`` is true, missing resources are downloaded into the
    directory pointed to by ``NLTK_DATA`` when present, otherwise the default
    NLTK data path is used.
    """

    if _has_sentence_tokenizer():
        return

    if not download:
        raise RuntimeError(
            "NLTK sentence tokenization data is not installed. "
            "Set download=True during image build or install the punkt resources manually."
        )

    download_dir = _resolve_download_dir()
    if download_dir is not None:
        download_dir.mkdir(parents=True, exist_ok=True)

    # Newer NLTK releases may require punkt_tab; older ones may still use punkt.
    nltk.download("punkt_tab", download_dir=str(download_dir) if download_dir else None, quiet=True)
    nltk.download("punkt", download_dir=str(download_dir) if download_dir else None, quiet=True)

    if not _has_sentence_tokenizer():
        raise RuntimeError("Failed to download NLTK sentence tokenization data.")


def _has_sentence_tokenizer() -> bool:
    """Return whether any supported NLTK sentence tokenizer resource exists."""

    for resource_name in _PUNKT_RESOURCE_CANDIDATES:
        try:
            nltk.data.find(resource_name)
            return True
        except LookupError:
            continue
    return False


def _resolve_download_dir() -> Path | None:
    """Return the preferred NLTK download directory, if configured."""

    nltk_data = os.environ.get("NLTK_DATA")
    if not nltk_data:
        return None
    return Path(nltk_data)


if __name__ == "__main__":
    ensure_sentence_tokenizer(download=True)

