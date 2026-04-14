#!/usr/bin/env python3
"""
compute_uncertainty_scores_mc_dropout.py

Score (source, summary) pairs with the MC Dropout backend and write
sentence-level uncertainty results to a JSONL file, grouped by base chunk ID.

Usage:
    python scripts/compute_uncertainty_scores_mc_dropout.py \\
        --infile data/summaries_v3.jsonl \\
        --outfile data/uncertainty_scores_mc_dropout.jsonl \\
        --model sshleifer/distilbart-cnn-12-6 \\
        --sample-count 20

Output schema (one JSON object per base chunk):
    {
        "id": "wikipedia|Clocks|chunk_0000",
        "sentence_scores": [
            {"sentence_index": 0, "sentence_text": "...", "uncertainty": 0.043, "summary_style": "shorten"},
            ...
        ],
        "uncertainty_avg": 0.044
    }
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_pipeline import read_jsonl, write_jsonl
from src.mc_dropout_backend import build_mc_dropout_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def base_chunk_id(record_id: str) -> str:
    return record_id.rsplit("|", 1)[0]


def main(args: argparse.Namespace) -> None:
    logger.info("Loading scorer (model=%r, device=%r)", args.model, args.device)
    scorer = build_mc_dropout_scorer(model_name=args.model, device=args.device)

    chunk_groups: dict[str, list[dict]] = defaultdict(list)
    for record in read_jsonl(args.infile):
        chunk_groups[base_chunk_id(record["id"])].append(record)
    logger.info("%d unique chunk(s) loaded", len(chunk_groups))

    processed = skipped = 0

    for base_id, style_records in tqdm(chunk_groups.items(), desc="scoring", unit="chunk"):
        if args.n_max is not None and processed >= args.n_max:
            break

        all_sentence_scores: list[dict] = []

        for record in style_records:
            source = record.get("paragraph_text", "").strip()
            summary = record.get("summary", "").strip()
            style = record.get("summary_style", "unknown")
            if not source or not summary:
                skipped += 1
                continue
            try:
                result = scorer.score_summary(
                    source=source, summary=summary,
                    sample_count=args.sample_count, seed=args.seed,
                )
            except Exception as exc:
                logger.error("Failed chunk %r style %r: %s", base_id, style, exc)
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
        write_jsonl(args.outfile, {"id": base_id, "sentence_scores": all_sentence_scores,
                                   "uncertainty_avg": uncertainty_avg})
        processed += 1

    logger.info("Done. Scored %d chunk(s), skipped %d.", processed, skipped)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=True)
    p.add_argument("--outfile", required=True)
    p.add_argument("--model", default="sshleifer/distilbart-cnn-12-6")
    p.add_argument("--device", default=None)
    p.add_argument("--sample-count", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-max", type=int, default=None)
    main(p.parse_args())
