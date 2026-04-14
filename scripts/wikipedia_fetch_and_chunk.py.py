#!/usr/bin/env python3
"""
wikipedia_fetch_and_chunk.py

Fetch Wikipedia pages and split them into ~250-word chunks preserving sentence
boundaries.  Output is a JSONL file, one object per chunk.

Usage:
    python scripts/wikipedia_fetch_and_chunk.py.py \\
        --titles-file data/wikipedia.lst \\
        --out data/wikipedia_chunks.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_pipeline import build_session, fetch_and_chunk_titles, write_jsonl
from src.nltk_setup import ensure_sentence_tokenizer


def main(args: argparse.Namespace) -> None:
    ensure_sentence_tokenizer(download=True)

    with open(args.titles_file, encoding="utf-8") as f:
        titles = [line.strip() for line in f if line.strip()]

    session = build_session(cert_path=args.ssl_cert)
    written = 0

    for chunk in tqdm(
        fetch_and_chunk_titles(titles, approx_words=args.approx_words, session=session),
        desc="chunking",
    ):
        write_jsonl(args.out, chunk)
        written += 1

    print(f"Done. Wrote {written} chunks to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fetch and chunk Wikipedia articles.")
    p.add_argument("--titles-file", required=True, help="One Wikipedia title per line")
    p.add_argument("--out", required=True, help="Output JSONL path (use .gz to compress)")
    p.add_argument("--approx-words", type=int, default=250, help="Target words per chunk")
    p.add_argument("--ssl-cert", help="Path to a CA bundle for HTTPS verification")
    main(p.parse_args())
