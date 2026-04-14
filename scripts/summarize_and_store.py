#!/usr/bin/env python3
"""
summarize_and_store.py

Read chunk JSONL produced by wikipedia_fetch_and_chunk.py, call the OpenRouter
chat completions API, and for each chunk produce three summaries:
shorten / professional / informal.

Usage:
    python scripts/summarize_and_store.py \\
        --infile data/wikipedia_chunks.jsonl \\
        --outfile data/summaries_v3.jsonl \\
        --model openai/gpt-4o-mini

Requires OPENROUTER_API_KEY in the environment or a .env file at the project root.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
sys.path.insert(0, str(ROOT_DIR))

from src.data_pipeline import (
    SUMMARY_STYLES,
    build_session,
    read_jsonl,
    summarize_chunks,
    write_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    chunks = list(read_jsonl(args.infile))
    if args.n_max is not None:
        chunks = chunks[: args.n_max]
    logger.info("Loaded %d chunks from %r", len(chunks), args.infile)

    session = build_session(cert_path=args.ssl_cert)
    written = errors = 0

    for result in tqdm(
        summarize_chunks(
            chunks,
            model=args.model,
            styles=SUMMARY_STYLES,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=api_key,
            workers=args.workers,
            session=session,
        ),
        total=len(chunks) * len(SUMMARY_STYLES),
        desc="summarizing",
    ):
        if "error" in result:
            logger.error("Summarization error: %s", result)
            errors += 1
        else:
            write_jsonl(args.outfile, result)
            written += 1

    logger.info("Done. Wrote %d summaries, %d errors -> %r", written, errors, args.outfile)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Summarize chunks via OpenRouter.")
    p.add_argument("--infile", required=True, help="Input JSONL from wikipedia_fetch_and_chunk.py")
    p.add_argument("--outfile", required=True, help="Output JSONL (gz ok)")
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-max", type=int, default=None, help="Max number of chunks to process")
    p.add_argument("--ssl-cert", help="Path to a CA bundle for HTTPS verification")
    main(p.parse_args())
