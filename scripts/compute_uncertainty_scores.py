#!/usr/bin/env python3
"""
compute_uncertainty_scores.py

Read (source, summary) pairs from a JSONL file produced by summarize_and_store.py,
score each pair with the MC Dropout backend, and write the sentence-level raw
uncertainty values to an output JSONL file.

The output is used by fit_quantiles.py to calibrate the normalizer boundaries.

Usage:
    python scripts/compute_uncertainty_scores.py \
        --infile data/summaries.json \
        --outfile data/uncertainty_scores.jsonl \
        --model sshleifer/distilbart-cnn-12-6 \
        --sample-count 20 \
        --n-max 500

Output schema (one JSON object per input record):
    {
        "id": "wikipedia|Clocks|chunk_0000",
        "sentence_scores": [
            {"sentence_index": 0, "sentence_text": "...", "uncertainty": 0.043},
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
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


def main(args: argparse.Namespace) -> None:
    logger.info("Loading scorer (model=%r, device=%r)", args.model, args.device)
    scorer = build_mc_dropout_scorer(model_name=args.model, device=args.device)
    logger.info("Scorer ready")

    processed = 0
    skipped = 0

    for record in tqdm(read_jsonl(args.infile), desc="scoring", unit="pair"):
        if args.n_max is not None and processed >= args.n_max:
            break

        source = record.get("paragraph_text", "").strip()
        summary = record.get("summary", "").strip()
        record_id = record.get("id", f"record_{processed}")

        if not source or not summary:
            logger.warning("Skipping record %r: missing source or summary", record_id)
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
            logger.error("Failed to score record %r: %s", record_id, exc)
            skipped += 1
            continue

        out = {
            "id": record_id,
            "sentence_scores": [
                {
                    "sentence_index": s.sentence_index,
                    "sentence_text": s.sentence_text,
                    "uncertainty": s.uncertainty,
                }
                for s in result.sentence_results
            ],
        }
        write_jsonl(args.outfile, out)
        processed += 1

    logger.info("Done. Scored %d record(s), skipped %d.", processed, skipped)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Score (source, summary) pairs with MC Dropout.")
    p.add_argument("--infile", required=True, help="Input JSONL from summarize_and_store.py (gz ok)")
    p.add_argument("--outfile", required=True, help="Output JSONL with raw uncertainty scores (gz ok)")
    p.add_argument("--model", default="sshleifer/distilbart-cnn-12-6", help="HuggingFace model identifier")
    p.add_argument("--device", default=None, help="Torch device (cpu, cuda; auto-detected if unset)")
    p.add_argument("--sample-count", type=int, default=20, help="MC Dropout forward passes per pair")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed for reproducibility")
    p.add_argument("--n-max", type=int, default=None, help="Maximum number of pairs to score")
    args = p.parse_args()
    main(args)
