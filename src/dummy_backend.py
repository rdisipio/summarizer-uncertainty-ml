"""Dummy inference backend for exercising the API before model training exists."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Sequence

import numpy as np

from .scorer import (
    PosteriorSampler,
    PreparedSummary,
    RuleBasedSentenceBackend,
    SampledSentenceDistributions,
    SummaryUncertaintyScorer,
)


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_VOCAB_SIZE = 64
_AMBIGUITY_HINTS = frozenset(
    {
        "allegedly",
        "apparently",
        "approximate",
        "approximately",
        "assumed",
        "believed",
        "could",
        "estimate",
        "estimated",
        "likely",
        "maybe",
        "may",
        "might",
        "possible",
        "possibly",
        "reportedly",
        "roughly",
        "suggests",
        "unclear",
        "unknown",
    }
)


@dataclass(frozen=True)
class DummyPosteriorSample:
    """A lightweight posterior sample identifier."""

    sample_index: int


class DummyPosteriorSampler(PosteriorSampler):
    """Repeatable posterior sampler for the dummy backend."""

    def sample(self, seed: int | None = None) -> DummyPosteriorSample:
        """Return a deterministic sample id."""

        return DummyPosteriorSample(sample_index=0 if seed is None else seed)


class DummySummaryScoringBackend(RuleBasedSentenceBackend):
    """Token-distribution backend that mimics posterior disagreement."""

    def prepare_summary(
        self,
        source: str,
        summary: str,
        sentences: Sequence[str] | None = None,
    ) -> PreparedSummary:
        """Prepare summary text and cache lightweight token metadata."""

        prepared_summary = super().prepare_summary(source, summary, sentences)
        sentence_tokens = {
            sentence.sentence_index: _tokenize_sentence(sentence.text)
            for sentence in prepared_summary.sentences
        }
        metadata = dict(prepared_summary.metadata)
        metadata["sentence_tokens"] = sentence_tokens
        return PreparedSummary(
            source=prepared_summary.source,
            summary=prepared_summary.summary,
            sentences=prepared_summary.sentences,
            metadata=metadata,
        )

    def score_posterior_sample(
        self,
        prepared_summary: PreparedSummary,
        posterior_sample: DummyPosteriorSample,
    ) -> Sequence[SampledSentenceDistributions]:
        """Generate token distributions with deterministic posterior spread."""

        sentence_tokens = prepared_summary.metadata["sentence_tokens"]
        sentence_distributions: list[SampledSentenceDistributions] = []

        for sentence in prepared_summary.sentences:
            tokens = sentence_tokens[sentence.sentence_index]
            target_token_ids = np.array(
                [_token_to_id(token) for token in tokens],
                dtype=np.int64,
            )
            token_probabilities = np.stack(
                [
                    _token_distribution(
                        token=token,
                        token_id=int(token_id),
                        sentence=sentence.text,
                        source=prepared_summary.source,
                        sample_index=posterior_sample.sample_index,
                    )
                    for token, token_id in zip(tokens, target_token_ids, strict=True)
                ],
                axis=0,
            )
            sentence_distributions.append(
                SampledSentenceDistributions(
                    sentence_index=sentence.sentence_index,
                    target_token_ids=target_token_ids,
                    token_probabilities=token_probabilities,
                )
            )

        return sentence_distributions


def build_dummy_scorer() -> SummaryUncertaintyScorer:
    """Create the default dummy scorer used for local and Docker smoke tests."""

    return SummaryUncertaintyScorer(
        backend=DummySummaryScoringBackend(),
        posterior_sampler=DummyPosteriorSampler(),
    )


def _tokenize_sentence(sentence: str) -> tuple[str, ...]:
    """Split a sentence into simple display-aligned tokens."""

    tokens = tuple(_TOKEN_PATTERN.findall(sentence))
    return tokens or (sentence,)


def _token_to_id(token: str) -> int:
    """Map a token string into a small fixed vocabulary id."""

    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:2], byteorder="big") % _VOCAB_SIZE


def _token_distribution(
    *,
    token: str,
    token_id: int,
    sentence: str,
    source: str,
    sample_index: int,
) -> np.ndarray:
    """Return a deterministic pseudo-posterior token distribution."""

    distribution = np.full(_VOCAB_SIZE, 0.0, dtype=np.float64)

    base_confidence = 0.78
    ambiguity_boost = _ambiguity_boost(token=token, sentence=sentence)
    source_overlap_boost = _source_overlap_boost(token=token, source=source)
    is_ambiguous = ambiguity_boost >= 0.12
    if is_ambiguous:
        target_probability = 0.78 if sample_index % 2 == 0 else 0.32
    else:
        sample_variation = _sample_variation(token=token, sample_index=sample_index)
        target_probability = (
            base_confidence - ambiguity_boost + source_overlap_boost - sample_variation
        )
        target_probability = float(np.clip(target_probability, 0.2, 0.94))

    remaining_mass = 1.0 - target_probability
    if remaining_mass <= 0.0:
        distribution[token_id] = 1.0
        return distribution

    non_target_indices = [index for index in range(_VOCAB_SIZE) if index != token_id]
    focus_index = _alternative_focus_index(
        token=token,
        sample_index=sample_index,
        token_id=token_id,
    )
    focus_probability = min(
        remaining_mass * (0.75 if is_ambiguous else 0.35 + ambiguity_boost),
        remaining_mass * 0.85,
    )
    background_probability = (remaining_mass - focus_probability) / (len(non_target_indices) - 1)

    for index in non_target_indices:
        distribution[index] = background_probability
    distribution[focus_index] = focus_probability
    distribution[token_id] = target_probability
    return distribution


def _ambiguity_boost(*, token: str, sentence: str) -> float:
    """Increase posterior spread for uncertain-looking text."""

    normalized_token = token.lower()
    boost = 0.0
    if normalized_token in _AMBIGUITY_HINTS:
        boost += 0.18
    if any(character.isdigit() for character in normalized_token):
        boost += 0.04
    if "%" in sentence or "percent" in sentence.lower():
        boost += 0.03
    return boost


def _source_overlap_boost(*, token: str, source: str) -> float:
    """Reduce uncertainty slightly when the token appears in the source."""

    normalized_token = token.lower()
    if not normalized_token.isalpha():
        return 0.0
    return 0.08 if normalized_token in source.lower() else 0.0


def _sample_variation(*, token: str, sample_index: int) -> float:
    """Create deterministic posterior movement across samples."""

    digest = hashlib.sha256(f"{token}:{sample_index}".encode("utf-8")).digest()
    raw_value = int.from_bytes(digest[:2], byteorder="big") / 65535.0
    return (raw_value - 0.5) * 0.16


def _alternative_focus_index(*, token: str, sample_index: int, token_id: int) -> int:
    """Select a stable competing token id for non-target mass."""

    digest = hashlib.sha256(f"alt:{token}".encode("utf-8")).digest()
    candidate_offsets = (
        int.from_bytes(digest[:2], byteorder="big") % 7 + 1,
        int.from_bytes(digest[2:4], byteorder="big") % 11 + 8,
        int.from_bytes(digest[4:6], byteorder="big") % 13 + 20,
    )
    candidate = (token_id + candidate_offsets[sample_index % len(candidate_offsets)]) % _VOCAB_SIZE
    if candidate == token_id:
        return (candidate + 1) % _VOCAB_SIZE
    return candidate
