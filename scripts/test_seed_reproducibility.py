#!/usr/bin/env python3
"""Quick CPU test: verify that identical seeds produce identical scores across runs,
and that different seed offsets produce different (but close) scores."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.mc_dropout_backend import build_mc_dropout_scorer

SOURCE = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
    "Constructed from 1887 to 1889, it was initially criticised by some of France's leading artists "
    "and intellectuals for its design, but it has become a global cultural icon of France."
)
SUMMARY = (
    "The Eiffel Tower, built between 1887 and 1889, is a wrought-iron tower in Paris. "
    "Despite early criticism, it became a global icon of France."
)

MODEL = "sshleifer/distilbart-cnn-12-6"
SAMPLE_COUNT = 10

print(f"Loading model {MODEL!r} on cpu ...")
scorer = build_mc_dropout_scorer(model_name=MODEL, device="cpu")

print(f"\nRun A (seed=0) ...")
result_a1 = scorer.score_summary(SOURCE, SUMMARY, sample_count=SAMPLE_COUNT, seed=0)
scores_a1 = [s.uncertainty for s in result_a1.sentence_results]

print(f"Run A again (seed=0) ...")
result_a2 = scorer.score_summary(SOURCE, SUMMARY, sample_count=SAMPLE_COUNT, seed=0)
scores_a2 = [s.uncertainty for s in result_a2.sentence_results]

print(f"\nRun B (seed={SAMPLE_COUNT}) ...")
result_b = scorer.score_summary(SOURCE, SUMMARY, sample_count=SAMPLE_COUNT, seed=SAMPLE_COUNT)
scores_b = [s.uncertainty for s in result_b.sentence_results]

print("\n--- Results ---")
for i, (s1, s2, sb) in enumerate(zip(scores_a1, scores_a2, scores_b)):
    match = "OK" if abs(s1 - s2) < 1e-9 else "MISMATCH"
    print(f"  sentence {i}: run_A1={s1:.6f}  run_A2={s2:.6f}  [{match}]  run_B={sb:.6f}")

all_match = all(abs(s1 - s2) < 1e-9 for s1, s2 in zip(scores_a1, scores_a2))
print(f"\nReproducibility (A1==A2): {'PASS' if all_match else 'FAIL'}")
any_differ = any(abs(s1 - sb) > 1e-9 for s1, sb in zip(scores_a1, scores_b))
print(f"Independence   (A!=B):    {'PASS' if any_differ else 'UNEXPECTED (same scores with different seeds)'}")
