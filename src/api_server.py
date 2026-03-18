"""FastAPI wrapper around the summary uncertainty scoring service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Protocol, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

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
    title: str = "Summary Uncertainty API",
) -> FastAPI:
    """Create the FastAPI application with an injected scoring service."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        app.state.scoring_service = scoring_service
        yield

    app = FastAPI(title=title, lifespan=lifespan)

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

        return result.to_dict()

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


app = create_app(UnconfiguredScoringService())
