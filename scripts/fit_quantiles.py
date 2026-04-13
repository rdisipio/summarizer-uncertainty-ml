#!/usr/bin/env python3
"""
fit_quantiles.py

Read sentence-level raw uncertainty scores produced by compute_uncertainty_scores.py,
compute quantiles across all sentences, and write the boundaries to the normalizer
config file (config/uncertainty_quantiles.json).

Usage:
    python scripts/fit_quantiles.py \
        --infile data/uncertainty_scores.jsonl \
        --outfile config/uncertainty_quantiles.json \
        --quantiles 0.0 0.25 0.5 0.75 1.0

The default quantile points (0%, 25%, 50%, 75%, 100%) spread the 0-100 display
scale evenly across the observed distribution.  Adjust if you want finer resolution
in a particular range (e.g. add 0.9 to stretch out the high-uncertainty tail).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np

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


def main(args: argparse.Namespace) -> None:
    scores: list[float] = []

    for record in read_jsonl(args.infile):
        for sentence in record.get("sentence_scores", []):
            value = sentence.get("uncertainty")
            if value is not None:
                scores.append(float(value))

    if not scores:
        logger.error("No uncertainty scores found in %r. Aborting.", args.infile)
        sys.exit(1)

    logger.info("Collected %d sentence-level scores", len(scores))

    scores_array = np.array(scores, dtype=np.float64)
    logger.info(
        "Score summary — min: %.6f  median: %.6f  max: %.6f  mean: %.6f  std: %.6f",
        scores_array.min(),
        float(np.median(scores_array)),
        scores_array.max(),
        scores_array.mean(),
        scores_array.std(),
    )

    quantile_points = sorted(set(args.quantiles))
    boundaries = [float(np.quantile(scores_array, q)) for q in quantile_points]
    logger.info("Quantile points: %s", quantile_points)
    logger.info("Boundaries:      %s", [f"{b:.6f}" for b in boundaries])

    if boundaries[0] == boundaries[-1]:
        logger.error("All boundaries are equal (%.6f). The score distribution has no spread.", boundaries[0])
        sys.exit(1)

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        json.dump({"boundaries": boundaries}, f, indent=2)
        f.write("\n")

    logger.info("Boundaries written to %s", outfile)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fit normalizer quantile boundaries from scored pairs.")
    p.add_argument("--infile", required=True, help="Input JSONL from compute_uncertainty_scores.py (gz ok)")
    p.add_argument(
        "--outfile",
        default=str(Path(__file__).resolve().parent.parent / "config" / "uncertainty_quantiles.json"),
        help="Output JSON config file (default: config/uncertainty_quantiles.json)",
    )
    p.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Quantile points to use as boundaries (default: 0.0 0.25 0.5 0.75 1.0)",
    )
    args = p.parse_args()
    main(args)
