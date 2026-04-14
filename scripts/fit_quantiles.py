#!/usr/bin/env python3
"""
fit_quantiles.py

Read sentence-level raw uncertainty scores produced by a
compute_uncertainty_scores_*.py script, compute quantile boundaries, and write
them to a normalizer config file.  The output path should reflect the backend:
  config/uncertainty_quantiles_mc_dropout.json
  config/uncertainty_quantiles_lora_laplace.json

Usage:
    python scripts/fit_quantiles.py \\
        --infile data/uncertainty_scores_mc_dropout.jsonl \\
        --outfile config/uncertainty_quantiles_mc_dropout.json \\
        --quantiles 0.0 0.25 0.5 0.75 1.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.lora_training import collect_scores_from_jsonl, fit_quantiles, save_quantiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    scores = collect_scores_from_jsonl(args.infile)
    if not scores:
        logger.error("No uncertainty scores found in %r. Aborting.", args.infile)
        sys.exit(1)
    logger.info("Collected %d sentence-level scores", len(scores))

    boundaries = fit_quantiles(scores, quantile_points=args.quantiles)
    logger.info("Boundaries: %s", [f"{b:.6f}" for b in boundaries])

    save_quantiles(boundaries, args.outfile)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fit quantile normalizer from scored pairs.")
    p.add_argument("--infile", required=True)
    p.add_argument("--outfile", required=True,
                   help="e.g. config/uncertainty_quantiles_lora_laplace.json")
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    main(p.parse_args())
