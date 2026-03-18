"""Runtime orchestration for summary-prefix-aware uncertainty scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol, Sequence

import numpy as np


_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
_EPSILON = 1e-12


@dataclass(frozen=True)
class SentenceSpec:
    """A sentence extracted from a summary with stable character offsets."""

    sentence_index: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class PreparedSummary:
    """Backend-specific representation of a summary ready for scoring."""

    source: str
    summary: str
    sentences: tuple[SentenceSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampledSentenceDistributions:
    """Token distributions for one sentence under one posterior sample."""

    sentence_index: int
    target_token_ids: np.ndarray
    token_probabilities: np.ndarray

    def __post_init__(self) -> None:
        if self.target_token_ids.ndim != 1:
            raise ValueError("target_token_ids must be a 1D array.")
        if self.token_probabilities.ndim != 2:
            raise ValueError("token_probabilities must be a 2D array.")
        if self.token_probabilities.shape[0] != self.target_token_ids.shape[0]:
            raise ValueError(
                "token_probabilities rows must match target_token_ids length."
            )


@dataclass(frozen=True)
class TokenScore:
    """Posterior predictive statistics for a single observed token."""

    token_index: int
    target_token_id: int
    mean_logprob: float
    predictive_entropy: float
    expected_entropy: float
    epistemic_mi: float


@dataclass(frozen=True)
class SentenceScore:
    """Aggregated uncertainty statistics for one summary sentence."""

    sentence_index: int
    sentence_text: str
    uncertainty: float
    mean_logprob: float
    epistemic_mi: float
    predictive_entropy: float
    expected_entropy: float
    token_scores: tuple[TokenScore, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "sentence_index": self.sentence_index,
            "sentence_text": self.sentence_text,
            "uncertainty": self.uncertainty,
            "mean_logprob": self.mean_logprob,
            "epistemic_mi": self.epistemic_mi,
            "predictive_entropy": self.predictive_entropy,
            "expected_entropy": self.expected_entropy,
            "token_scores": [
                {
                    "token_index": token_score.token_index,
                    "target_token_id": token_score.target_token_id,
                    "mean_logprob": token_score.mean_logprob,
                    "predictive_entropy": token_score.predictive_entropy,
                    "expected_entropy": token_score.expected_entropy,
                    "epistemic_mi": token_score.epistemic_mi,
                }
                for token_score in self.token_scores
            ],
        }


@dataclass(frozen=True)
class SummaryScore:
    """Sentence-level uncertainty results for a full summary."""

    source: str
    summary: str
    sentence_results: tuple[SentenceScore, ...]
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "source": self.source,
            "summary": self.summary,
            "sample_count": self.sample_count,
            "sentence_results": [
                sentence_result.to_dict() for sentence_result in self.sentence_results
            ],
        }


class PosteriorSampler(Protocol):
    """Generates posterior samples over LoRA parameters."""

    def sample(self, seed: int | None = None) -> Any:
        """Return one sampled posterior state."""


class SummaryScoringBackend(Protocol):
    """Backend that preserves full summary-prefix context during scoring."""

    def prepare_summary(
        self,
        source: str,
        summary: str,
        sentences: Sequence[str] | None = None,
    ) -> PreparedSummary:
        """Tokenize and map summary sentences for prefix-aware scoring."""

    def score_posterior_sample(
        self,
        prepared_summary: PreparedSummary,
        posterior_sample: Any,
    ) -> Sequence[SampledSentenceDistributions]:
        """Return per-sentence token distributions for one posterior sample."""


class SummaryUncertaintyScorer:
    """Orchestrates posterior sampling and sentence-level aggregation."""

    def __init__(
        self,
        backend: SummaryScoringBackend,
        posterior_sampler: PosteriorSampler,
    ) -> None:
        self._backend = backend
        self._posterior_sampler = posterior_sampler

    def score_summary(
        self,
        source: str,
        summary: str,
        sentences: Sequence[str] | None = None,
        sample_count: int = 40,
        top_k_tokens: int | None = None,
        seed: int | None = None,
    ) -> SummaryScore:
        """Score an existing summary without re-generating it."""

        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")

        prepared_summary = self._backend.prepare_summary(source, summary, sentences)
        if not prepared_summary.sentences:
            return SummaryScore(
                source=source,
                summary=summary,
                sentence_results=tuple(),
                sample_count=sample_count,
            )

        sentence_sample_map: dict[int, list[SampledSentenceDistributions]] = {
            sentence.sentence_index: [] for sentence in prepared_summary.sentences
        }

        for sample_index in range(sample_count):
            sample_seed = None if seed is None else seed + sample_index
            posterior_sample = self._posterior_sampler.sample(seed=sample_seed)
            sentence_distributions = self._backend.score_posterior_sample(
                prepared_summary=prepared_summary,
                posterior_sample=posterior_sample,
            )
            for sentence_distribution in sentence_distributions:
                sentence_sample_map[sentence_distribution.sentence_index].append(
                    sentence_distribution
                )

        sentence_results = tuple(
            self._aggregate_sentence(
                sentence=sentence,
                sentence_samples=sentence_sample_map[sentence.sentence_index],
                top_k_tokens=top_k_tokens,
            )
            for sentence in prepared_summary.sentences
        )

        return SummaryScore(
            source=source,
            summary=summary,
            sentence_results=sentence_results,
            sample_count=sample_count,
        )

    def _aggregate_sentence(
        self,
        sentence: SentenceSpec,
        sentence_samples: Sequence[SampledSentenceDistributions],
        top_k_tokens: int | None,
    ) -> SentenceScore:
        """Aggregate posterior sample distributions into sentence-level metrics."""

        if not sentence_samples:
            raise ValueError(
                f"No posterior samples were collected for sentence {sentence.sentence_index}."
            )

        token_id_reference = sentence_samples[0].target_token_ids
        probability_stack = []
        for sentence_sample in sentence_samples:
            if not np.array_equal(sentence_sample.target_token_ids, token_id_reference):
                raise ValueError(
                    "All posterior samples must target the same observed token ids."
                )
            probability_stack.append(_normalize_probabilities(sentence_sample.token_probabilities))

        sample_probabilities = np.stack(probability_stack, axis=0)
        predictive_distribution = np.mean(sample_probabilities, axis=0)
        predictive_distribution = _normalize_probabilities(predictive_distribution)

        predictive_entropy = _entropy(predictive_distribution)
        expected_entropy = np.mean(_entropy(sample_probabilities), axis=0)
        epistemic_mi = np.clip(
            predictive_entropy - expected_entropy,
            a_min=0.0,
            a_max=None,
        )

        token_target_probabilities = np.take_along_axis(
            sample_probabilities,
            token_id_reference[np.newaxis, :, np.newaxis],
            axis=2,
        ).squeeze(axis=2)
        mean_logprob = np.mean(np.log(np.clip(token_target_probabilities, _EPSILON, 1.0)), axis=0)

        token_scores = tuple(
            TokenScore(
                token_index=token_index,
                target_token_id=int(target_token_id),
                mean_logprob=float(mean_logprob[token_index]),
                predictive_entropy=float(predictive_entropy[token_index]),
                expected_entropy=float(expected_entropy[token_index]),
                epistemic_mi=float(epistemic_mi[token_index]),
            )
            for token_index, target_token_id in enumerate(token_id_reference)
        )

        token_uncertainty_values = np.array(
            [token_score.epistemic_mi for token_score in token_scores],
            dtype=np.float64,
        )
        uncertainty = _aggregate_uncertainty(token_uncertainty_values, top_k_tokens)

        return SentenceScore(
            sentence_index=sentence.sentence_index,
            sentence_text=sentence.text,
            uncertainty=uncertainty,
            mean_logprob=float(np.mean(mean_logprob)),
            epistemic_mi=float(np.mean(epistemic_mi)),
            predictive_entropy=float(np.mean(predictive_entropy)),
            expected_entropy=float(np.mean(expected_entropy)),
            token_scores=token_scores,
        )


class RuleBasedSentenceBackend:
    """Default text preparation backend with explicit sentence alignment."""

    def prepare_summary(
        self,
        source: str,
        summary: str,
        sentences: Sequence[str] | None = None,
    ) -> PreparedSummary:
        """Build a prepared summary object from raw text."""

        del source
        sentence_texts = tuple(sentences) if sentences is not None else split_sentences(summary)
        sentence_specs = tuple(
            align_sentences(summary=summary, sentences=sentence_texts)
        )
        return PreparedSummary(
            source=source,
            summary=summary,
            sentences=sentence_specs,
            metadata={},
        )

    def score_posterior_sample(
        self,
        prepared_summary: PreparedSummary,
        posterior_sample: Any,
    ) -> Sequence[SampledSentenceDistributions]:
        """Signal that model-specific scoring must be provided by the caller."""

        del prepared_summary
        del posterior_sample
        raise NotImplementedError(
            "RuleBasedSentenceBackend only prepares summaries. "
            "Provide a model-specific backend for posterior scoring."
        )


def split_sentences(text: str) -> tuple[str, ...]:
    """Split summary text into sentences while preserving display order."""

    stripped_text = text.strip()
    if not stripped_text:
        return tuple()

    raw_sentences = _SENTENCE_BOUNDARY_PATTERN.split(stripped_text)
    normalized_sentences = tuple(sentence.strip() for sentence in raw_sentences if sentence.strip())
    if normalized_sentences:
        return normalized_sentences
    return (stripped_text,)


def align_sentences(summary: str, sentences: Sequence[str]) -> tuple[SentenceSpec, ...]:
    """Align sentence strings back to summary character offsets."""

    aligned_sentences: list[SentenceSpec] = []
    cursor = 0

    for sentence_index, sentence in enumerate(sentences):
        sentence_text = sentence.strip()
        if not sentence_text:
            continue

        char_start = summary.find(sentence_text, cursor)
        if char_start < 0:
            raise ValueError(
                f"Could not align sentence {sentence_index} back to the summary text."
            )
        char_end = char_start + len(sentence_text)
        aligned_sentences.append(
            SentenceSpec(
                sentence_index=sentence_index,
                text=sentence_text,
                char_start=char_start,
                char_end=char_end,
            )
        )
        cursor = char_end

    return tuple(aligned_sentences)


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clip and normalize probability rows."""

    clipped = np.clip(probabilities.astype(np.float64), _EPSILON, None)
    row_sums = clipped.sum(axis=-1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Probability rows must have positive mass.")
    return clipped / row_sums


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    """Compute entropy along the last axis."""

    safe_probabilities = np.clip(probabilities, _EPSILON, 1.0)
    return -np.sum(safe_probabilities * np.log(safe_probabilities), axis=-1)


def _aggregate_uncertainty(
    token_uncertainty_values: np.ndarray,
    top_k_tokens: int | None,
) -> float:
    """Aggregate token uncertainty into one sentence score."""

    if token_uncertainty_values.size == 0:
        return 0.0

    if top_k_tokens is None or top_k_tokens <= 0:
        return float(np.mean(token_uncertainty_values))

    capped_top_k = min(top_k_tokens, token_uncertainty_values.size)
    sorted_uncertainty = np.sort(token_uncertainty_values)
    return float(np.mean(sorted_uncertainty[-capped_top_k:]))


class DeterministicPosteriorSampler:
    """Posterior sampler implementation for tests and local demos."""

    def __init__(self, samples: Sequence[Any]) -> None:
        if not samples:
            raise ValueError("samples must not be empty.")
        self._samples = tuple(samples)

    def sample(self, seed: int | None = None) -> Any:
        """Return a repeatable sample selection."""

        if seed is None:
            return self._samples[0]
        return self._samples[seed % len(self._samples)]


def build_demo_backend(
    sentence_probabilities: Sequence[Sequence[np.ndarray]],
) -> SummaryScoringBackend:
    """Construct a fixed-output backend for tests and integration wiring."""

    return _DemoScoringBackend(sentence_probabilities)


class _DemoScoringBackend(RuleBasedSentenceBackend):
    """Backend that returns caller-provided token distributions."""

    def __init__(self, sentence_probabilities: Sequence[Sequence[np.ndarray]]) -> None:
        self._sentence_probabilities = tuple(
            tuple(probabilities for probabilities in sentence_group)
            for sentence_group in sentence_probabilities
        )

    def score_posterior_sample(
        self,
        prepared_summary: PreparedSummary,
        posterior_sample: Any,
    ) -> Sequence[SampledSentenceDistributions]:
        """Return fixed distributions indexed by posterior_sample."""

        sample_index = int(posterior_sample)
        sentence_distributions: list[SampledSentenceDistributions] = []
        for sentence in prepared_summary.sentences:
            probabilities = self._sentence_probabilities[sentence.sentence_index][sample_index]
            target_token_ids = np.argmax(probabilities, axis=1)
            sentence_distributions.append(
                SampledSentenceDistributions(
                    sentence_index=sentence.sentence_index,
                    target_token_ids=target_token_ids,
                    token_probabilities=probabilities,
                )
            )
        return sentence_distributions
