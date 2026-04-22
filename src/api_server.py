"""FastAPI wrapper around the summary uncertainty scoring service."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

logger = logging.getLogger(__name__)

from fastapi import FastAPI, Header, HTTPException
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
    compute_consistency: bool = True

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


class WakeResponse(BaseModel):
    """Wake-up response."""

    status: str


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool




def create_app(
    scoring_service_factory: Callable[[], ScoringService],
    *,
    normalizer: QuantileNormalizer,
    ambiguity_normalizer: QuantileNormalizer,
    consistency_normalizer: QuantileNormalizer,
    api_token: str | None = None,
    title: str = "Summary Uncertainty API",
) -> FastAPI:
    """Create the FastAPI application with an injected scoring service factory.

    The factory is called in a background thread during lifespan startup so the
    server can accept /wake and /is-ready requests while the model loads.
    """

    async def _load_service(app: FastAPI) -> None:
        loop = asyncio.get_event_loop()
        app.state.scoring_service = await loop.run_in_executor(None, scoring_service_factory)
        app.state.ready = True
        logger.info("Scoring service ready")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        app.state.ready = False
        app.state.scoring_service = None
        app.state.normalizer = normalizer
        app.state.ambiguity_normalizer = ambiguity_normalizer
        app.state.consistency_normalizer = consistency_normalizer
        asyncio.create_task(_load_service(app))
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
                "GET /wake": "Wake-up ping; call on frontend start to trigger cold-start recovery.",
                "GET /is-ready": "Returns {ready: true} once the scoring model is fully loaded.",
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

    @app.get("/wake", response_model=WakeResponse)
    async def wake() -> WakeResponse:
        """Wake-up ping for cold-start scenarios.

        Call this endpoint when the frontend starts so that the API server
        is fully warmed up by the time the user submits their first request.
        """

        logger.info("GET /wake — server awake")
        return WakeResponse(status="awake")

    @app.get("/is-ready", response_model=ReadinessResponse)
    async def is_ready() -> ReadinessResponse:
        """Return whether the scoring model has finished loading."""

        return ReadinessResponse(ready=getattr(app.state, "ready", False))

    @app.post("/score")
    async def score_summary(
        request: ScoreRequest,
        x_api_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Score the displayed summary without re-generating it."""

        if api_token and x_api_token != api_token.strip():
            raise HTTPException(status_code=401, detail="Invalid or missing API token.")

        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Scoring service is still loading.")

        logger.info(
            "POST /score — sample_count=%d top_k_tokens=%s seed=%s",
            request.sample_count,
            request.top_k_tokens,
            request.seed,
        )
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

        return _serialize_summary_score(
            result,
            app.state.normalizer,
            app.state.ambiguity_normalizer,
            app.state.consistency_normalizer,
            compute_consistency=request.compute_consistency,
        )

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
    - ``dummy``        – rule-based mock, no model required (default)
    - ``mc_dropout``   – teacher-forced MC Dropout over a HuggingFace seq2seq model
    - ``lora_laplace`` – LoRA + diagonal Laplace approximation
    - ``unconfigured`` – raises on every request (forces explicit wiring)

    MC Dropout environment variables:
    - ``MC_DROPOUT_MODEL``  – HuggingFace model identifier (default: facebook/bart-large-cnn)
    - ``MC_DROPOUT_DEVICE`` – torch device string, e.g. ``cpu`` or ``cuda`` (auto-detected when unset)

    LoRA-Laplace environment variables:
    - ``LORA_BASE_MODEL``   – HuggingFace base model identifier (default: facebook/bart-large-xsum)
    - ``LORA_ADAPTER_PATH`` – path to the PEFT adapter checkpoint directory (required)
    - ``LORA_SAMPLER_PATH`` – path to a pre-fitted laplace_sampler.npz (required);
                              fit offline with compute_uncertainty_scores_lora_laplace.py --save-sampler
    - ``LORA_DEVICE``       – torch device string (auto-detected when unset)
    """

    backend_name = os.environ.get("SCORING_BACKEND", "dummy").strip().lower()
    if backend_name == "dummy":
        return build_dummy_scorer()
    if backend_name == "mc_dropout":
        from .mc_dropout_backend import build_mc_dropout_scorer

        model_name = os.environ.get("MC_DROPOUT_MODEL", "facebook/bart-large-cnn")
        device = os.environ.get("MC_DROPOUT_DEVICE") or None
        return build_mc_dropout_scorer(model_name=model_name, device=device)
    if backend_name == "lora_laplace":
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from .lora_laplace_backend import LoraLaplaceBackend, load_laplace_sampler
        from .scorer import SummaryUncertaintyScorer

        base_model_name = os.environ.get("LORA_BASE_MODEL", "facebook/bart-large-xsum")
        adapter_path = os.environ.get("LORA_ADAPTER_PATH", "")
        sampler_path = os.environ.get("LORA_SAMPLER_PATH", "")
        device = os.environ.get("LORA_DEVICE") or None

        if not adapter_path:
            raise RuntimeError("LORA_ADAPTER_PATH must be set for the lora_laplace backend.")
        if not sampler_path:
            raise RuntimeError(
                "LORA_SAMPLER_PATH must be set for the lora_laplace backend. "
                "Fit the sampler offline with compute_uncertainty_scores_lora_laplace.py "
                "--save-sampler and provide the resulting .npz path here."
            )

        import torch

        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        peft_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        backend = LoraLaplaceBackend(peft_model=peft_model, tokenizer=tokenizer, device=device)
        sampler = load_laplace_sampler(sampler_path)
        return SummaryUncertaintyScorer(backend=backend, posterior_sampler=sampler)
    if backend_name == "unconfigured":
        return UnconfiguredScoringService()
    raise RuntimeError(f"Unsupported SCORING_BACKEND value: {backend_name}")


def _build_default_normalizer() -> QuantileNormalizer:
    """Load the configured uncertainty normalizer."""

    default_path = Path(__file__).resolve().parent.parent / "config" / "uncertainty_quantiles_mc_dropout.json"
    config_path = os.environ.get("QUANTILE_CONFIG_PATH", str(default_path))
    normalizer = load_quantile_normalizer(config_path)
    logger.info(
        "Quantile normalizer loaded from %r — boundaries: %s",
        config_path,
        [f"{b:.4f}" for b in normalizer.boundaries],
    )
    return normalizer


def _build_default_consistency_normalizer() -> QuantileNormalizer:
    """Load the configured consistency normalizer, falling back to the uncertainty normalizer.

    Boundaries are over -mean_logprob (a positive value; higher = less consistent).
    Fit them from a calibration corpus with compute_uncertainty_scores_lora_laplace.py
    and save to the path set in CONSISTENCY_QUANTILE_CONFIG_PATH.
    """

    default_path = Path(__file__).resolve().parent.parent / "config" / "consistency_quantiles_lora_laplace.json"
    config_path = os.environ.get("CONSISTENCY_QUANTILE_CONFIG_PATH", str(default_path))
    if not Path(config_path).exists():
        logger.warning(
            "Consistency quantile config not found at %r — falling back to uncertainty normalizer.",
            config_path,
        )
        return _build_default_normalizer()
    normalizer = load_quantile_normalizer(config_path)
    logger.info("Consistency normalizer loaded from %r", config_path)
    return normalizer


def _build_default_ambiguity_normalizer() -> QuantileNormalizer:
    """Load the configured ambiguity normalizer, falling back to the uncertainty normalizer."""

    default_path = Path(__file__).resolve().parent.parent / "config" / "ambiguity_quantiles_mc_dropout.json"
    config_path = os.environ.get("AMBIGUITY_QUANTILE_CONFIG_PATH", str(default_path))
    if not Path(config_path).exists():
        logger.warning(
            "Ambiguity quantile config not found at %r — falling back to uncertainty normalizer.",
            config_path,
        )
        return _build_default_normalizer()
    normalizer = load_quantile_normalizer(config_path)
    logger.info(
        "Ambiguity normalizer loaded from %r — boundaries: %s",
        config_path,
        [f"{b:.4f}" for b in normalizer.boundaries],
    )
    return normalizer


def _serialize_summary_score(
    summary_score: SummaryScore,
    normalizer: QuantileNormalizer,
    ambiguity_normalizer: QuantileNormalizer,
    consistency_normalizer: QuantileNormalizer,
    compute_consistency: bool = True,
) -> dict[str, Any]:
    """Serialize a summary score and attach display-oriented uncertainty values."""

    payload = summary_score.to_dict()
    normalization: dict[str, Any] = {
        "boundaries": list(normalizer.boundaries),
        "ambiguity_boundaries": list(ambiguity_normalizer.boundaries),
    }
    if compute_consistency:
        normalization["consistency_boundaries"] = list(consistency_normalizer.boundaries)
    payload["normalization"] = normalization

    for sentence_result in payload["sentence_results"]:
        raw_uncertainty = float(sentence_result["uncertainty"])
        sentence_result["uncertainty_raw"] = raw_uncertainty
        sentence_result["uncertainty_score"] = normalizer.normalize(raw_uncertainty)
        sentence_result["uncertainty_band"] = normalizer.band(raw_uncertainty)

        raw_ambiguity = float(sentence_result["expected_entropy"])
        sentence_result["ambiguity_score"] = ambiguity_normalizer.normalize(raw_ambiguity)
        sentence_result["ambiguity_band"] = ambiguity_normalizer.band(raw_ambiguity)

        if compute_consistency:
            # consistency_score: higher = more consistent with the source.
            # mean_logprob is negative; negate it so higher raw = less consistent,
            # then invert the 0-100 scale so the final score reads naturally.
            raw_inconsistency = -float(sentence_result["mean_logprob"])
            inconsistency_normalized = consistency_normalizer.normalize(raw_inconsistency)
            sentence_result["consistency_score"] = round(100.0 - inconsistency_normalized, 4)
            sentence_result["consistency_band"] = _invert_band(
                consistency_normalizer.band(raw_inconsistency)
            )

    return payload


def _invert_band(band: str) -> str:
    if band == "low":
        return "high"
    if band == "high":
        return "low"
    return band


_api_token = os.environ.get("API_TOKEN") or None

app = create_app(
    _build_default_service,
    normalizer=_build_default_normalizer(),
    ambiguity_normalizer=_build_default_ambiguity_normalizer(),
    consistency_normalizer=_build_default_consistency_normalizer(),
    api_token=_api_token,
)
