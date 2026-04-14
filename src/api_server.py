"""FastAPI wrapper around the summary uncertainty scoring service."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any, Protocol, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .dummy_backend import build_dummy_scorer
from .normalization import QuantileNormalizer, load_quantile_normalizer
from .scorer import SummaryScore


class ScoringService(Protocol):
    """Minimal service interface required by the API layer."""

    def score_summary(
        self,
        source: str,
        summary: str,
        sentences: Sequence[str] | None = None,
        sample_count: int = 40,
        top_k_tokens: int | None = None,
        seed: int | None = None,
    ) -> SummaryScore:
        """Score a displayed summary and return sentence-level uncertainty."""


class ScoreRequest(BaseModel):
    """Request payload for uncertainty scoring."""

    source: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    sentences: list[str] | None = None
    sample_count: int = Field(default=20, ge=1, le=100)
    top_k_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None

    @field_validator("sentences")
    @classmethod
    def validate_sentences(cls, sentences: list[str] | None) -> list[str] | None:
        """Reject blank sentence strings at the API boundary."""

        if sentences is None:
            return None
        normalized_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not normalized_sentences:
            raise ValueError("sentences must contain at least one non-empty sentence.")
        return normalized_sentences


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


def create_app(
    scoring_service: ScoringService,
    *,
    normalizer: QuantileNormalizer,
    title: str = "Summary Uncertainty API",
) -> FastAPI:
    """Create the FastAPI application with an injected scoring service."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        app.state.scoring_service = scoring_service
        app.state.normalizer = normalizer
        yield

    app = FastAPI(title=title, lifespan=lifespan)

    @app.get("/")
    async def root() -> dict[str, Any]:
        """Return a brief description of the service and its endpoints."""

        return {
            "service": title,
            "description": (
                "Estimates per-sentence epistemic uncertainty for a given "
                "source text and its summary."
            ),
            "endpoints": {
                "GET /health": "Liveness check.",
                "POST /score": (
                    "Score a summary. Required fields: source (str), summary (str). "
                    "Optional: sample_count (int, 1-100), sentences (list[str]), "
                    "top_k_tokens (int), seed (int)."
                ),
            },
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return API liveness information."""

        return HealthResponse(status="ok")

    @app.post("/score")
    async def score_summary(request: ScoreRequest) -> dict[str, Any]:
        """Score the displayed summary without re-generating it."""

        try:
            result = app.state.scoring_service.score_summary(
                source=request.source,
                summary=request.summary,
                sentences=request.sentences,
                sample_count=request.sample_count,
                top_k_tokens=request.top_k_tokens,
                seed=request.seed,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except NotImplementedError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

        return _serialize_summary_score(result, app.state.normalizer)

    return app


class UnconfiguredScoringService:
    """Fallback scoring service that forces explicit model wiring."""

    def score_summary(
        self,
        source: str,
        summary: str,
        sentences: Sequence[str] | None = None,
        sample_count: int = 40,
        top_k_tokens: int | None = None,
        seed: int | None = None,
    ) -> SummaryScore:
        """Raise until a real posterior-scoring backend is configured."""

        del source
        del summary
        del sentences
        del sample_count
        del top_k_tokens
        del seed
        raise NotImplementedError(
            "No scoring service has been configured. "
            "Inject a SummaryUncertaintyScorer backed by the trained model and Laplace sampler."
        )


def _build_default_service() -> ScoringService:
    """Build the default scoring service from environment configuration.

    Recognised SCORING_BACKEND values:
    - ``dummy``       – rule-based mock, no model required (default)
    - ``mc_dropout``  – teacher-forced MC Dropout over a HuggingFace seq2seq model
    - ``unconfigured``– raises on every request (forces explicit wiring)

    MC Dropout environment variables:
    - ``MC_DROPOUT_MODEL``  – HuggingFace model identifier (default: facebook/bart-large-cnn)
    - ``MC_DROPOUT_DEVICE`` – torch device string, e.g. ``cpu`` or ``cuda`` (auto-detected when unset)
    """

    backend_name = os.environ.get("SCORING_BACKEND", "dummy").strip().lower()
    if backend_name == "dummy":
        return build_dummy_scorer()
    if backend_name == "mc_dropout":
        from .mc_dropout_backend import build_mc_dropout_scorer

        model_name = os.environ.get("MC_DROPOUT_MODEL", "facebook/bart-large-cnn")
        device = os.environ.get("MC_DROPOUT_DEVICE") or None
        return build_mc_dropout_scorer(model_name=model_name, device=device)
    if backend_name == "unconfigured":
        return UnconfiguredScoringService()
    raise RuntimeError(f"Unsupported SCORING_BACKEND value: {backend_name}")


def _build_default_normalizer() -> QuantileNormalizer:
    """Load the configured uncertainty normalizer."""

    default_path = Path(__file__).resolve().parent.parent / "config" / "uncertainty_quantiles.json"
    config_path = os.environ.get("QUANTILE_CONFIG_PATH", str(default_path))
    return load_quantile_normalizer(config_path)


def _serialize_summary_score(
    summary_score: SummaryScore,
    normalizer: QuantileNormalizer,
) -> dict[str, Any]:
    """Serialize a summary score and attach display-oriented uncertainty values."""

    payload = summary_score.to_dict()
    payload["normalization"] = {"boundaries": list(normalizer.boundaries)}

    for sentence_result in payload["sentence_results"]:
        raw_uncertainty = float(sentence_result["uncertainty"])
        sentence_result["uncertainty_raw"] = raw_uncertainty
        sentence_result["uncertainty_score"] = normalizer.normalize(raw_uncertainty)
        sentence_result["uncertainty_band"] = normalizer.band(raw_uncertainty)

    return payload


app = create_app(
    _build_default_service(),
    normalizer=_build_default_normalizer(),
)
