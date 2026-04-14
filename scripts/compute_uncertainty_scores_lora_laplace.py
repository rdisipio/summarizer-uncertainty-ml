#!/usr/bin/env python3
"""
compute_uncertainty_scores_lora_laplace.py

Score (source, summary) pairs with the LoRA + Laplace Approximation backend
and write sentence-level uncertainty results to a JSONL file.

Runs in two phases:
  1. Calibration — fit the diagonal Laplace approximation on a held-out subset.
  2. Scoring     — run posterior-sampled forward passes and write results.

Usage:
    python scripts/compute_uncertainty_scores_lora_laplace.py \\
        --infile data/summaries_v3.jsonl \\
        --outfile data/uncertainty_scores_lora_laplace.jsonl \\
        --base-model facebook/bart-large-xsum \\
        --adapter-path models/bart-large-xsum-lora \\
        --sample-count 20 \\
        --save-sampler models/bart-large-xsum-lora/laplace_sampler.npz
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_pipeline import read_jsonl, write_jsonl
from src.lora_laplace_backend import build_lora_laplace_scorer, save_laplace_sampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def base_chunk_id(record_id: str) -> str:
    return record_id.rsplit("|", 1)[0]


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    all_records = [
        r for r in read_jsonl(args.infile)
        if r.get("paragraph_text") and r.get("summary")
    ]
    logger.info("%d valid records loaded", len(all_records))
    if not all_records:
        logger.error("No valid records found. Aborting.")
        sys.exit(1)

    random.shuffle(all_records)
    cal_size = max(1, int(len(all_records) * args.calibration_split))
    calibration_data = [
        (r["paragraph_text"].strip(), r["summary"].strip())
        for r in all_records[:cal_size]
    ]
    scoring_records = all_records[cal_size:]
    logger.info("Calibration: %d  Scoring: %d", len(calibration_data), len(scoring_records))

    scorer = build_lora_laplace_scorer(
        base_model_name=args.base_model,
        peft_model_path=args.adapter_path,
        calibration_data=calibration_data,
        prior_precision=args.prior_precision,
        device=args.device,
    )

    if args.save_sampler:
        save_laplace_sampler(scorer._posterior_sampler, args.save_sampler)
        logger.info("Laplace sampler saved to %r", args.save_sampler)

    chunk_groups: dict[str, list[dict]] = defaultdict(list)
    for record in scoring_records:
        chunk_groups[base_chunk_id(record["id"])].append(record)
    logger.info("%d unique chunk(s) to score", len(chunk_groups))

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
    p.add_argument("--base-model", default="facebook/bart-large-xsum")
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--calibration-split", type=float, default=0.1)
    p.add_argument("--prior-precision", type=float, default=1.0)
    p.add_argument("--sample-count", type=int, default=20)
    p.add_argument("--save-sampler", default=None, metavar="PATH")
    p.add_argument("--device", default=None)
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
