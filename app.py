"""Gradio + ZeroGPU entry point for HuggingFace Spaces deployment.

On ZeroGPU Spaces (SPACES_ZERO_GPU=1 injected by HF infrastructure), the model
lives on CPU between requests and is moved to CUDA only for the duration of the
@spaces.GPU-decorated call.  Locally (Mac MPS or CPU-only), @spaces.GPU is a
no-op and the model stays on the auto-detected device throughout.

Model artifacts are loaded from HuggingFace Hub at startup.  The variant is
selected via the LORA_MODEL_VARIANT env var (default: bart-base-lora-laplace).

Local usage:
    LORA_MODEL_VARIANT=bart-base-lora-laplace pipenv run python app.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import gradio as gr
import spaces
import torch
from fastapi import Header, HTTPException
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.api_server import ScoreRequest, _serialize_summary_score
from src.lora_laplace_backend import LoraLaplaceBackend, load_laplace_sampler
from src.nltk_setup import ensure_sentence_tokenizer
from src.normalization import load_quantile_normalizer
from src.scorer import SummaryUncertaintyScorer

ensure_sentence_tokenizer(download=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
# Download model artifacts from HF Hub
# ---------------------------------------------------------------------------

_MODEL_REPO = "rdisipio/summarizer-uncertainty-models"
_MODEL_VARIANT = os.environ.get("LORA_MODEL_VARIANT", "bart-base-lora-laplace")

logger.info("Downloading model variant %r from %s", _MODEL_VARIANT, _MODEL_REPO)
_snapshot_dir = Path(
    snapshot_download(
        repo_id=_MODEL_REPO,
        allow_patterns=f"{_MODEL_VARIANT}/*",
    )
)
_variant_dir = _snapshot_dir / _MODEL_VARIANT
logger.info("Model artifacts at %s", _variant_dir)

# Read the base model name directly from the adapter config.
with open(_variant_dir / "adapter_config.json") as _f:
    _base_model_name = json.load(_f)["base_model_name_or_path"]
logger.info("Base model: %s", _base_model_name)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

_base_model = AutoModelForSeq2SeqLM.from_pretrained(_base_model_name)
_peft_model = PeftModel.from_pretrained(_base_model, str(_variant_dir), is_trainable=True)
_tokenizer = AutoTokenizer.from_pretrained(str(_variant_dir))

backend = LoraLaplaceBackend(peft_model=_peft_model, tokenizer=_tokenizer, device=_init_device)
sampler = load_laplace_sampler(str(_variant_dir / "laplace_sampler.npz"))
scorer = SummaryUncertaintyScorer(backend=backend, posterior_sampler=sampler)

# Quantile configs live in the variant dir on the Hub; fall back to local copies.
def _load_normalizer(hub_filename: str, env_var: str, local_filename: str) -> object:
    env_path = os.environ.get(env_var)
    if env_path:
        return load_quantile_normalizer(env_path)
    hub_path = _variant_dir / hub_filename
    if hub_path.exists():
        return load_quantile_normalizer(str(hub_path))
    local_path = Path(__file__).parent / "config" / local_filename
    return load_quantile_normalizer(str(local_path))

_normalizer = _load_normalizer(
    "uncertainty_quantiles_lora_laplace.json", "QUANTILE_CONFIG_PATH",
    "uncertainty_quantiles_lora_laplace.json",
)
_amb_normalizer = _load_normalizer(
    "ambiguity_quantiles_lora_laplace.json", "AMBIGUITY_QUANTILE_CONFIG_PATH",
    "ambiguity_quantiles_lora_laplace.json",
)
_con_normalizer = _load_normalizer(
    "consistency_quantiles_lora_laplace.json", "CONSISTENCY_QUANTILE_CONFIG_PATH",
    "consistency_quantiles_lora_laplace.json",
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
# ---------------------------------------------------------------------------

def _score_api(
    source: str,
    summary: str,
    sample_count: float,  # gr.Number returns float
    seed_raw: float,      # -1 sentinel means None
    compute_consistency: bool,
) -> dict:
    """Gradio-queue-compatible wrapper called by both UI and the /score endpoint."""
    seed = int(seed_raw) if seed_raw is not None and seed_raw >= 0 else None
    return _score(source, summary, int(sample_count), seed, compute_consistency)


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
        fn=lambda src, smr: _score_api(src, smr, 10, -1, True),
        inputs=[src_box, smr_box],
        outputs=out_box,
    )

    # Hidden full-parameter API function so gradio_client can call it with all
    # options.  api_name exposes it at /run/score_api on the Gradio server.
    with gr.Row(visible=False):
        _h_src = gr.Textbox()
        _h_smr = gr.Textbox()
        _h_sc  = gr.Number(value=10)
        _h_seed = gr.Number(value=-1)
        _h_cc  = gr.Checkbox(value=True)
        _h_out = gr.JSON()
        _h_btn = gr.Button()
    _h_btn.click(
        fn=_score_api,
        inputs=[_h_src, _h_smr, _h_sc, _h_seed, _h_cc],
        outputs=[_h_out],
        api_name="score_api",
    )

# launch() creates a brand-new App instance (overwriting demo.app from __exit__),
# so routes must be added to the app returned by launch(), not to demo.app above.
# prevent_thread_lock=True makes launch() return immediately so we can do that.

_api_token = os.environ.get("API_TOKEN") or None

demo.queue()
app, _, _ = demo.launch(prevent_thread_lock=True, ssr_mode=False)

# The /score FastAPI endpoint must NOT call @spaces.GPU directly from a thread —
# ZeroGPU only works through Gradio's event queue.  We proxy the call via a
# local gradio_client so it goes through the queue the same way the UI does.

import threading
from gradio_client import Client as GradioClient

_gradio_client: GradioClient | None = None
_gradio_client_lock = threading.Lock()


def _get_gradio_client() -> GradioClient:
    global _gradio_client
    if _gradio_client is None:
        with _gradio_client_lock:
            if _gradio_client is None:
                _gradio_client = GradioClient("http://localhost:7860", verbose=False)
    return _gradio_client


@app.post("/score")
def score_endpoint(
    request: ScoreRequest,
    x_api_token: str | None = Header(default=None),
) -> dict:
    if _api_token and x_api_token != _api_token.strip():
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")
    client = _get_gradio_client()
    return client.predict(
        request.source,
        request.summary,
        float(request.sample_count),
        float(request.seed) if request.seed is not None else -1.0,
        request.compute_consistency,
        api_name="/score_api",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/wake")
def wake() -> dict:
    return {"status": "awake"}


demo.block_thread()
