#!/usr/bin/env python3
"""
rescale_and_recalibrate.py

Rescale the diagonal Laplace posterior_std to a target mean magnitude,
re-score all (source, summary) pairs, and refit all three quantile
normalizer config files.

The Laplace Fisher diagonal is often dominated by the isotropic prior
(F_ii << lambda), producing posterior_std ≈ 1/sqrt(lambda) ≈ 1.0 for all
parameters regardless of the data.  This dwarfs typical LoRA weight
magnitudes (~0.02-0.04) and randomizes the model at every forward pass,
collapsing per-sentence MI to near-zero.

This script rescales posterior_std uniformly so that its mean equals
--target-std (default 0.1), then scores the full dataset and refits
the quantile boundaries used by the API server.

Usage:
    pipenv run python scripts/rescale_and_recalibrate.py \\
        --infile data/summaries_v4.jsonl \\
        --outfile data/uncertainty_scores_lora_laplace_recalibrated.jsonl \\
        --sampler-path models/bart-large-xsum-lora/laplace_sampler.npz \\
        --adapter-path models/bart-large-xsum-lora \\
        --target-std 0.1 \\
        --sample-count 20
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import gzip
import json

import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.lora_laplace_backend import (
    LoraLaplaceBackend,
    DiagonalLaplacePosteriorSampler,
    load_laplace_sampler,
    save_laplace_sampler,
)
from src.lora_training import fit_quantiles, save_quantiles
from src.scorer import SummaryUncertaintyScorer

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
    return record_id.rsplit("|", 1)[0]


def rescale_sampler(
    sampler: DiagonalLaplacePosteriorSampler,
    target_std: float,
) -> DiagonalLaplacePosteriorSampler:
    """Return a new sampler with posterior_std scaled to mean == target_std."""
    current_std = sampler._posterior_std
    current_mean = float(current_std.mean())
    scale = target_std / current_mean
    logger.info(
        "Rescaling posterior_std: current mean=%.6f → target=%.6f (scale=%.4f)",
        current_mean, target_std, scale,
    )
    new_std = current_std * scale
    logger.info(
        "New posterior_std — min: %.6f  mean: %.6f  max: %.6f",
        float(new_std.min()), float(new_std.mean()), float(new_std.max()),
    )
    return DiagonalLaplacePosteriorSampler(
        param_names=sampler._param_names,
        param_shapes=sampler._param_shapes,
        posterior_variance=new_std ** 2,
    )


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    # ------------------------------------------------------------------
    # 1. Rescale sampler
    # ------------------------------------------------------------------
    logger.info("Loading sampler from %r", args.sampler_path)
    sampler = load_laplace_sampler(args.sampler_path)
    sampler = rescale_sampler(sampler, args.target_std)

    logger.info("Saving rescaled sampler back to %r", args.sampler_path)
    save_laplace_sampler(sampler, args.sampler_path)

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info("Loading tokenizer and base model %r", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    logger.info("Loading PEFT adapter from %r", args.adapter_path)
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=True)

    backend = LoraLaplaceBackend(
        peft_model=peft_model,
        tokenizer=tokenizer,
        device=args.device,
    )
    scorer = SummaryUncertaintyScorer(backend=backend, posterior_sampler=sampler)

    # ------------------------------------------------------------------
    # 3. Load and score records
    # ------------------------------------------------------------------
    all_records = [
        r for r in read_jsonl(args.infile)
        if r.get("paragraph_text") and r.get("summary")
    ]
    logger.info("%d valid records loaded", len(all_records))
    if not all_records:
        logger.error("No valid records found. Aborting.")
        sys.exit(1)

    random.shuffle(all_records)
    if args.n_max is not None:
        all_records = all_records[: args.n_max]
        logger.info("Capped at %d records", len(all_records))

    chunk_groups: dict[str, list[dict]] = defaultdict(list)
    for record in all_records:
        chunk_groups[base_chunk_id(record["id"])].append(record)
    logger.info("%d unique chunk(s) to score", len(chunk_groups))

    uncertainty_scores: list[float] = []
    expected_entropy_scores: list[float] = []
    mean_logprob_scores: list[float] = []

    processed = skipped = 0
    outfile = args.outfile

    for base_id, style_records in tqdm(chunk_groups.items(), desc="scoring", unit="chunk"):
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
                    source=source,
                    summary=summary,
                    sample_count=args.sample_count,
                    seed=args.seed,
                )
            except Exception as exc:
                logger.error("Failed chunk %r style %r: %s", base_id, style, exc)
                skipped += 1
                continue

            for s in result.sentence_results:
                sentence_entry = {
                    "sentence_index": s.sentence_index,
                    "sentence_text": s.sentence_text,
                    "uncertainty": s.uncertainty,
                    "expected_entropy": s.expected_entropy,
                    "mean_logprob": s.mean_logprob,
                    "summary_style": style,
                }
                all_sentence_scores.append(sentence_entry)
                uncertainty_scores.append(s.uncertainty)
                expected_entropy_scores.append(s.expected_entropy)
                # consistency uses −mean_logprob; store raw mean_logprob
                mean_logprob_scores.append(s.mean_logprob)

        if not all_sentence_scores:
            skipped += 1
            continue

        uncertainty_avg = sum(s["uncertainty"] for s in all_sentence_scores) / len(all_sentence_scores)
        write_jsonl(
            outfile,
            {
                "id": base_id,
                "sentence_scores": all_sentence_scores,
                "uncertainty_avg": uncertainty_avg,
            },
        )
        processed += 1

    logger.info("Done. Scored %d chunk(s), skipped %d.", processed, skipped)

    if not uncertainty_scores:
        logger.error("No scores collected — cannot fit quantile normalizers.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Refit quantile normalizers
    # ------------------------------------------------------------------
    logger.info(
        "Fitting uncertainty quantiles from %d sentence scores", len(uncertainty_scores)
    )
    uncertainty_boundaries = fit_quantiles(uncertainty_scores)
    save_quantiles(uncertainty_boundaries, args.uncertainty_config)

    logger.info(
        "Fitting ambiguity quantiles from %d expected_entropy values",
        len(expected_entropy_scores),
    )
    ambiguity_boundaries = fit_quantiles(expected_entropy_scores)
    save_quantiles(ambiguity_boundaries, args.ambiguity_config)

    # Consistency normalizer uses −mean_logprob (higher logprob → lower raw_score → higher display score).
    neg_logprob_scores = [-v for v in mean_logprob_scores]
    logger.info(
        "Fitting consistency quantiles from %d −mean_logprob values",
        len(neg_logprob_scores),
    )
    consistency_boundaries = fit_quantiles(neg_logprob_scores)
    save_quantiles(consistency_boundaries, args.consistency_config)

    logger.info("All quantile config files updated.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Rescale Laplace posterior_std and refit quantile normalizers."
    )
    p.add_argument("--infile", default="data/summaries_v4.jsonl")
    p.add_argument(
        "--outfile",
        default="data/uncertainty_scores_lora_laplace_recalibrated.jsonl",
    )
    p.add_argument(
        "--sampler-path",
        default="models/bart-large-xsum-lora/laplace_sampler.npz",
    )
    p.add_argument(
        "--adapter-path",
        default="models/bart-large-xsum-lora",
    )
    p.add_argument(
        "--base-model",
        default="facebook/bart-large-xsum",
    )
    p.add_argument(
        "--target-std",
        type=float,
        default=0.1,
        help="Target mean posterior_std after rescaling (default: 0.1)",
    )
    p.add_argument(
        "--sample-count",
        type=int,
        default=20,
        help="Posterior samples per summary (default: 20)",
    )
    p.add_argument(
        "--uncertainty-config",
        default="config/uncertainty_quantiles_lora_laplace.json",
    )
    p.add_argument(
        "--ambiguity-config",
        default="config/ambiguity_quantiles_lora_laplace.json",
    )
    p.add_argument(
        "--consistency-config",
        default="config/consistency_quantiles_lora_laplace.json",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--n-max", type=int, default=None, metavar="N",
                   help="Cap the number of records to score (default: all)")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
