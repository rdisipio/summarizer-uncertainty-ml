#!/usr/bin/env python3
"""
compute_uncertainty_scores_lora_laplace.py

Score (source, summary) pairs with the LoRA + Laplace Approximation backend
and write sentence-level uncertainty results to a JSONL file.

The script runs in two phases:
  1. Calibration — fit the diagonal Laplace approximation over the LoRA
     adapter parameters using a held-out subset of the input data.
  2. Scoring — run posterior-sampled forward passes over the remaining pairs
     and write sentence-level uncertainty scores.

The output schema matches compute_uncertainty_scores_mc_dropout.py so that
fit_quantiles.py can be used unchanged downstream.

Usage:
    python scripts/compute_uncertainty_scores_lora_laplace.py \\
        --infile data/summaries_v3.jsonl \\
        --outfile data/uncertainty_scores_lora_laplace.jsonl \\
        --base-model facebook/bart-large-xsum \\
        --adapter-path models/bart-large-xsum-lora \\
        --sample-count 20 \\
        --calibration-split 0.1 \\
        --prior-precision 1.0

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
import gzip
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.lora_laplace_backend import build_lora_laplace_scorer, save_laplace_sampler

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
    random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load and partition data
    # ------------------------------------------------------------------
    logger.info("Reading records from %r", args.infile)
    all_records = [r for r in read_jsonl(args.infile) if r.get("paragraph_text") and r.get("summary")]
    logger.info("%d valid records loaded", len(all_records))

    if not all_records:
        logger.error("No valid records found. Aborting.")
        sys.exit(1)

    random.shuffle(all_records)
    cal_size = max(1, int(len(all_records) * args.calibration_split))
    calibration_records = all_records[:cal_size]
    scoring_records = all_records[cal_size:]

    calibration_data = [
        (r["paragraph_text"].strip(), r["summary"].strip())
        for r in calibration_records
    ]
    logger.info(
        "Partition: %d calibration records, %d scoring records",
        len(calibration_data),
        len(scoring_records),
    )

    # ------------------------------------------------------------------
    # Build scorer (loads model, fits Laplace approximation)
    # ------------------------------------------------------------------
    logger.info(
        "Building LoRA-Laplace scorer (base=%r, adapter=%r)",
        args.base_model,
        args.adapter_path,
    )
    scorer = build_lora_laplace_scorer(
        base_model_name=args.base_model,
        peft_model_path=args.adapter_path,
        calibration_data=calibration_data,
        prior_precision=args.prior_precision,
        device=args.device,
    )

    # Optionally persist the fitted sampler so calibration need not be re-run.
    if args.save_sampler:
        save_laplace_sampler(scorer._posterior_sampler, args.save_sampler)
        logger.info("Laplace sampler saved to %r", args.save_sampler)

    # ------------------------------------------------------------------
    # Score — group by base chunk ID to average over style variants
    # ------------------------------------------------------------------
    chunk_groups: dict[str, list[dict]] = defaultdict(list)
    for record in scoring_records:
        chunk_groups[base_chunk_id(record["id"])].append(record)
    logger.info("%d unique chunk(s) to score", len(chunk_groups))

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
                logger.warning(
                    "Skipping style %r of chunk %r: missing source or summary",
                    style,
                    base_id,
                )
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
                logger.error(
                    "Failed to score chunk %r style %r: %s", base_id, style, exc
                )
                skipped += 1
                continue

            for s in result.sentence_results:
                all_sentence_scores.append(
                    {
                        "sentence_index": s.sentence_index,
                        "sentence_text": s.sentence_text,
                        "uncertainty": s.uncertainty,
                        "summary_style": style,
                    }
                )

        if not all_sentence_scores:
            skipped += 1
            continue

        uncertainty_avg = sum(s["uncertainty"] for s in all_sentence_scores) / len(
            all_sentence_scores
        )
        write_jsonl(
            args.outfile,
            {
                "id": base_id,
                "sentence_scores": all_sentence_scores,
                "uncertainty_avg": uncertainty_avg,
            },
        )
        processed += 1

    logger.info("Done. Scored %d chunk(s), skipped %d.", processed, skipped)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Score (source, summary) pairs with LoRA+Laplace, grouped by chunk."
    )
    p.add_argument(
        "--infile",
        required=True,
        help="Input JSONL from summarize_and_store.py (gz ok)",
    )
    p.add_argument(
        "--outfile",
        required=True,
        help="Output JSONL with sentence-level uncertainty scores (gz ok)",
    )
    p.add_argument(
        "--base-model",
        default="facebook/bart-large-xsum",
        help="HuggingFace base model identifier",
    )
    p.add_argument(
        "--adapter-path",
        required=True,
        help="Path to the PEFT/LoRA adapter checkpoint directory",
    )
    p.add_argument(
        "--calibration-split",
        type=float,
        default=0.1,
        help="Fraction of input records used to fit the Laplace approximation (default: 0.1)",
    )
    p.add_argument(
        "--prior-precision",
        type=float,
        default=1.0,
        help="Isotropic prior precision for the diagonal Laplace (default: 1.0)",
    )
    p.add_argument(
        "--sample-count",
        type=int,
        default=20,
        help="Posterior samples per summary (default: 20)",
    )
    p.add_argument(
        "--save-sampler",
        default=None,
        metavar="PATH",
        help="If set, persist the fitted Laplace sampler to this .npz path",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device (cpu, cuda, mps — auto-detected if unset)",
    )
    p.add_argument(
        "--n-max",
        type=int,
        default=None,
        help="Maximum number of chunks to score (useful for testing)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
