"""Normalization helpers for mapping raw uncertainty to a 0..100 display score."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QuantileNormalizer:
    """Map a non-negative raw uncertainty score onto a 0..100 display scale."""

    boundaries: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.boundaries) < 2:
            raise ValueError("Quantile boundaries must contain at least min and max.")
        if any(boundary < 0.0 for boundary in self.boundaries):
            raise ValueError("Quantile boundaries must be non-negative.")
        if tuple(self.boundaries) != tuple(sorted(self.boundaries)):
            raise ValueError("Quantile boundaries must be sorted in ascending order.")
        if self.boundaries[0] == self.boundaries[-1]:
            raise ValueError("Quantile boundaries must span a non-zero range.")

    def normalize(self, raw_score: float) -> float:
        """Map a raw uncertainty value onto a 0..100 scale with clamping."""

        if raw_score < 0.0:
            raise ValueError("raw_score must be non-negative.")

        quantile_positions = np.linspace(0.0, 100.0, num=len(self.boundaries))
        normalized_score = np.interp(raw_score, self.boundaries, quantile_positions)
        return float(np.clip(normalized_score, 0.0, 100.0))

    def band(self, raw_score: float) -> str:
        """Map a raw uncertainty value onto a coarse display band."""

        normalized_score = self.normalize(raw_score)
        if normalized_score < (100.0 / 3.0):
            return "low"
        if normalized_score < (200.0 / 3.0):
            return "mid"
        return "high"


def load_quantile_normalizer(path: str | Path) -> QuantileNormalizer:
    """Load a normalizer from a JSON file containing ordered boundaries."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        payload = json.load(config_file)

    boundaries = payload.get("boundaries")
    if not isinstance(boundaries, list):
        raise ValueError("Normalization config must contain a 'boundaries' list.")

    return QuantileNormalizer(boundaries=tuple(float(boundary) for boundary in boundaries))
