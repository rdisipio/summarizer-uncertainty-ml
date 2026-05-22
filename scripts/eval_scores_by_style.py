#!/usr/bin/env python3
"""
eval_scores_by_style.py

Score a random sample of (source, summary) pairs from each summary style
and check whether the resulting uncertainty/consistency scores are
meaningfully differentiated across styles.

Scores are fetched from a running API instance (local or deployed).
The Kruskal-Wallis H-test is used to check for a statistically significant
difference in medians across the three styles; per-pair Mann-Whitney U tests
check the expected orderings.

Expected orderings (heuristic):
  uncertainty:  informal >= professional >= shorten
  consistency:  shorten >= professional >= informal

Usage:
    pipenv run python scripts/eval_scores_by_style.py \\
        --api-url http://localhost:7860 \\
        --n-per-style 50 \\
        --sample-count 3
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import requests
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_by_style(path: str, styles: list[str]) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            style = r.get("summary_style", "unknown")
            if style in styles and r.get("paragraph_text") and r.get("summary"):
                records[style].append(r)
    return records


def score_record(api_url: str, record: dict, sample_count: int, api_token: str | None) -> dict | None:
    payload = {
        "source": record["paragraph_text"],
        "summary": record["summary"],
        "sample_count": sample_count,
        "seed": 42,
    }
    headers = {}
    if api_token:
        headers["X-Api-Token"] = api_token
    try:
        resp = requests.post(f"{api_url}/score", json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Request failed: %s", exc)
        return None


def extract_sentence_scores(result: dict) -> list[dict]:
    return result.get("sentence_results", [])


def report(metric: str, scores_by_style: dict[str, list[float]], expected_order: list[str]) -> None:
    print(f"\n── {metric} ──")
    medians = {}
    for style in expected_order:
        vals = scores_by_style.get(style, [])
        if not vals:
            print(f"  {style:12s}  n=0  (no data)")
            continue
        med = sorted(vals)[len(vals) // 2]
        medians[style] = med
        print(f"  {style:12s}  n={len(vals):4d}  median={med:6.1f}  mean={sum(vals)/len(vals):6.1f}")

    groups = [scores_by_style.get(s, []) for s in expected_order if scores_by_style.get(s)]
    if len(groups) >= 2:
        h_stat, p_val = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis H={h_stat:.2f}  p={p_val:.4f}", end="")
        print("  ✓ significant" if p_val < 0.05 else "  ✗ not significant")

    # Pairwise ordering check for adjacent pairs in expected_order
    styles_with_data = [s for s in expected_order if scores_by_style.get(s)]
    for i in range(len(styles_with_data) - 1):
        a, b = styles_with_data[i], styles_with_data[i + 1]
        va, vb = scores_by_style[a], scores_by_style[b]
        u_stat, p = stats.mannwhitneyu(va, vb, alternative="greater")
        direction = ">" if medians.get(a, 0) >= medians.get(b, 0) else "<"
        ok = "✓" if p < 0.05 and medians.get(a, 0) >= medians.get(b, 0) else "✗"
        print(f"  {a} {direction} {b}:  U={u_stat:.0f}  p={p:.4f}  {ok}")


def main(args: argparse.Namespace) -> None:
    styles = ["informal", "professional", "shorten"]
    logger.info("Loading data from %s", args.infile)
    by_style = load_by_style(args.infile, styles)

    random.seed(args.seed)
    sample: dict[str, list[dict]] = {}
    for style in styles:
        pool = by_style.get(style, [])
        random.shuffle(pool)
        sample[style] = pool[: args.n_per_style]
        logger.info("%s: %d records sampled", style, len(sample[style]))

    uncertainty: dict[str, list[float]] = defaultdict(list)
    ambiguity:   dict[str, list[float]] = defaultdict(list)
    consistency: dict[str, list[float]] = defaultdict(list)

    total = sum(len(v) for v in sample.values())
    done = 0
    for style, records in sample.items():
        for record in records:
            result = score_record(args.api_url, record, args.sample_count, args.api_token)
            done += 1
            if result is None:
                continue
            for sent in extract_sentence_scores(result):
                if "uncertainty_score" in sent:
                    uncertainty[style].append(sent["uncertainty_score"])
                if "ambiguity_score" in sent:
                    ambiguity[style].append(sent["ambiguity_score"])
                if "consistency_score" in sent:
                    consistency[style].append(sent["consistency_score"])
            if done % 10 == 0:
                logger.info("%d / %d records scored", done, total)

    logger.info("Scoring complete. %d / %d records returned results.", done, total)

    print("\n═══ Score distributions by style ═══")
    # informal tends to be more uncertain/ambiguous; shorten tends to be more consistent
    report("uncertainty_score",  uncertainty,  ["informal", "professional", "shorten"])
    report("ambiguity_score",    ambiguity,    ["informal", "professional", "shorten"])
    report("consistency_score",  consistency,  ["shorten",  "professional", "informal"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate score distributions across summary styles.")
    p.add_argument("--infile",       default="data/summaries_v4.jsonl")
    p.add_argument("--api-url",      default="http://localhost:7860",
                   help="Base URL of the scoring API (default: http://localhost:7860)")
    p.add_argument("--api-token",    default=None, metavar="TOKEN")
    p.add_argument("--n-per-style",  type=int, default=50,
                   help="Records to sample per style (default: 50)")
    p.add_argument("--sample-count", type=int, default=3,
                   help="Posterior samples per request (default: 3)")
    p.add_argument("--seed",         type=int, default=42)
    main(p.parse_args())
