#!/usr/bin/env python3
"""
compute_uncertainty_scores.py

Read (source, summary) pairs from a JSONL file produced by summarize_and_store.py,
score each style variant with the MC Dropout backend, and write one output record
per base chunk with sentence-level uncertainty values averaged across all style variants.

Grouping by base chunk ID ensures the calibration distribution reflects content
uncertainty rather than style variation.

Usage:
    python scripts/compute_uncertainty_scores.py \
        --infile data/summaries.jsonl \
        --outfile data/uncertainty_scores.jsonl \
        --model sshleifer/distilbart-cnn-12-6 \
        --sample-count 20 \
        --n-max 500

Output schema (one JSON object per base chunk):
    {
        "id": "wikipedia|Clocks|chunk_0000",
        "sentence_scores": [
            {"sentence_index": 0, "sentence_text": "...", "uncertainty": 0.043, "summary_style": "shorten"},
            {"sentence_index": 0, "sentence_text": "...", "uncertainty": 0.051, "summary_style": "professional"},
            {"sentence_index": 0, "sentence_text": "...", "uncertainty": 0.038, "summary_style": "informal"},
            ...
        ],
        "uncertainty_avg": 0.044
    }
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.mc_dropout_backend import build_mc_dropout_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def read_jsonl(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, obj: dict) -> None:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def base_chunk_id(record_id: str) -> str:
    """Strip the trailing |style suffix added by summarize_and_store.py."""
    return record_id.rsplit("|", 1)[0]


def main(args: argparse.Namespace) -> None:
    logger.info("Loading scorer (model=%r, device=%r)", args.model, args.device)
    scorer = build_mc_dropout_scorer(model_name=args.model, device=args.device)
    logger.info("Scorer ready")

    # Group all records by base chunk ID so we can score all style variants
    # together and average out the style contribution.
    logger.info("Reading and grouping records from %r", args.infile)
    chunk_groups: dict[str, list[dict]] = defaultdict(list)
    for record in read_jsonl(args.infile):
        chunk_groups[base_chunk_id(record["id"])].append(record)
    logger.info("%d unique chunk(s) loaded", len(chunk_groups))

    processed = 0
    skipped = 0

    for base_id, style_records in tqdm(chunk_groups.items(), desc="scoring", unit="chunk"):
        if args.n_max is not None and processed >= args.n_max:
            break

        all_sentence_scores: list[dict] = []

        for record in style_records:
            source = record.get("paragraph_text", "").strip()
            summary = record.get("summary", "").strip()
            style = record.get("summary_style", "unknown")

            if not source or not summary:
                logger.warning("Skipping style %r of chunk %r: missing source or summary", style, base_id)
                skipped += 1
                continue

            try:
                result = scorer.score_summary(
                    source=source,
                    summary=summary,
                    sample_count=args.sample_count,
                    seed=args.seed,
                )
            except Exception as exc:
                logger.error("Failed to score chunk %r style %r: %s", base_id, style, exc)
                skipped += 1
                continue

            for s in result.sentence_results:
                all_sentence_scores.append({
                    "sentence_index": s.sentence_index,
                    "sentence_text": s.sentence_text,
                    "uncertainty": s.uncertainty,
                    "summary_style": style,
                })

        if not all_sentence_scores:
            skipped += 1
            continue

        uncertainty_avg = sum(s["uncertainty"] for s in all_sentence_scores) / len(all_sentence_scores)

        write_jsonl(args.outfile, {
            "id": base_id,
            "sentence_scores": all_sentence_scores,
            "uncertainty_avg": uncertainty_avg,
        })
        processed += 1

    logger.info("Done. Scored %d chunk(s), skipped %d.", processed, skipped)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Score (source, summary) pairs with MC Dropout, grouped by chunk.")
    p.add_argument("--infile", required=True, help="Input JSONL from summarize_and_store.py (gz ok)")
    p.add_argument("--outfile", required=True, help="Output JSONL with raw uncertainty scores (gz ok)")
    p.add_argument("--model", default="sshleifer/distilbart-cnn-12-6", help="HuggingFace model identifier")
    p.add_argument("--device", default=None, help="Torch device (cpu, cuda; auto-detected if unset)")
    p.add_argument("--sample-count", type=int, default=20, help="MC Dropout forward passes per summary")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed for reproducibility")
    p.add_argument("--n-max", type=int, default=None, help="Maximum number of chunks to score")
    args = p.parse_args()
    main(args)
