"""Gradio + ZeroGPU entry point for HuggingFace Spaces deployment.

On ZeroGPU Spaces (SPACES_ZERO_GPU=1 injected by HF infrastructure), the model
lives on CPU between requests and is moved to CUDA only for the duration of the
@spaces.GPU-decorated call.  Locally (Mac MPS or CPU-only), @spaces.GPU is a
no-op and the model stays on the auto-detected device throughout.

Local usage:
    LORA_BASE_MODEL=... LORA_ADAPTER_PATH=... LORA_SAMPLER_PATH=... pipenv run python app.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import gradio as gr
import spaces
import torch
from fastapi import Header, HTTPException
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.api_server import ScoreRequest, _serialize_summary_score
from src.lora_laplace_backend import LoraLaplaceBackend, load_laplace_sampler
from src.normalization import load_quantile_normalizer
from src.scorer import SummaryUncertaintyScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ZeroGPU sets this env var; absent locally and in plain Docker spaces.
_IS_ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "0") == "1"

if _IS_ZERO_GPU:
    _init_device = "cpu"
elif torch.cuda.is_available():
    _init_device = "cuda"
elif torch.backends.mps.is_available():
    _init_device = "mps"
else:
    _init_device = "cpu"

logger.info("ZeroGPU=%s  init_device=%s", _IS_ZERO_GPU, _init_device)

# ---------------------------------------------------------------------------
# Model loading (runs once at startup, on CPU for ZeroGPU)
# ---------------------------------------------------------------------------

_base_model_name = os.environ.get("LORA_BASE_MODEL", "facebook/bart-base")
_adapter_path = os.environ.get("LORA_ADAPTER_PATH", "")
_sampler_path = os.environ.get("LORA_SAMPLER_PATH", "")

if not _adapter_path:
    raise RuntimeError("LORA_ADAPTER_PATH must be set.")
if not _sampler_path:
    raise RuntimeError("LORA_SAMPLER_PATH must be set.")

_base_model = AutoModelForSeq2SeqLM.from_pretrained(_base_model_name)
_peft_model = PeftModel.from_pretrained(_base_model, _adapter_path, is_trainable=True)
_tokenizer = AutoTokenizer.from_pretrained(_base_model_name)

backend = LoraLaplaceBackend(peft_model=_peft_model, tokenizer=_tokenizer, device=_init_device)
sampler = load_laplace_sampler(_sampler_path)
scorer = SummaryUncertaintyScorer(backend=backend, posterior_sampler=sampler)

_cfg = Path(__file__).parent / "config"
_normalizer = load_quantile_normalizer(
    os.environ.get("QUANTILE_CONFIG_PATH", str(_cfg / "uncertainty_quantiles_lora_laplace.json"))
)
_amb_normalizer = load_quantile_normalizer(
    os.environ.get("AMBIGUITY_QUANTILE_CONFIG_PATH", str(_cfg / "ambiguity_quantiles_lora_laplace.json"))
)
_con_normalizer = load_quantile_normalizer(
    os.environ.get("CONSISTENCY_QUANTILE_CONFIG_PATH", str(_cfg / "consistency_quantiles_lora_laplace.json"))
)

# ---------------------------------------------------------------------------
# GPU-gated scoring
# ---------------------------------------------------------------------------

@spaces.GPU
def _score(
    source: str,
    summary: str,
    sample_count: int,
    seed: int | None,
    compute_consistency: bool,
) -> dict:
    """Run scoring on GPU (ZeroGPU) or the current device (local)."""
    if _IS_ZERO_GPU:
        backend.to("cuda")
    try:
        result = scorer.score_summary(
            source=source,
            summary=summary,
            sample_count=sample_count,
            seed=seed,
        )
    finally:
        if _IS_ZERO_GPU:
            backend.to("cpu")
    return _serialize_summary_score(
        result, _normalizer, _amb_normalizer, _con_normalizer,
        compute_consistency=compute_consistency,
    )

# ---------------------------------------------------------------------------
# Gradio UI
# In Gradio 6, demo.app (a FastAPI subclass) is created when the `with` block
# exits, so custom routes can be added to it before calling demo.launch().
# ---------------------------------------------------------------------------

with gr.Blocks(title="Stylo — Summary Uncertainty") as demo:
    gr.Markdown("## Summary Uncertainty Scorer")
    gr.Markdown(
        "Score per-sentence epistemic uncertainty for a summary relative to its source text. "
        "Each sentence receives an uncertainty band: **low / medium / high**."
    )
    with gr.Row():
        src_box = gr.Textbox(
            label="Source text",
            lines=8,
            placeholder="Paste the original article or paragraph here...",
        )
        smr_box = gr.Textbox(
            label="Summary",
            lines=4,
            placeholder="Paste the summary to score here...",
        )
    run_btn = gr.Button("Score", variant="primary")
    out_box = gr.JSON(label="Sentence uncertainty results")
    run_btn.click(
        fn=lambda src, smr: _score(src, smr, 10, None, True),
        inputs=[src_box, smr_box],
        outputs=out_box,
    )

# demo.app is a FastAPI-subclass instance created when the with-block exits.
# Add /score, /health, /wake so the browser extension keeps working unchanged.

_api_token = os.environ.get("API_TOKEN") or None


@demo.app.post("/score")
async def score_endpoint(
    request: ScoreRequest,
    x_api_token: str | None = Header(default=None),
) -> dict:
    if _api_token and x_api_token != _api_token.strip():
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")
    return _score(
        request.source,
        request.summary,
        request.sample_count,
        request.seed,
        request.compute_consistency,
    )


@demo.app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@demo.app.get("/wake")
async def wake() -> dict:
    return {"status": "awake"}


demo.queue().launch()
