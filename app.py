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

# _score_inner contains the actual work.  It must be called from within a
# @spaces.GPU-decorated function so the GPU worker is already live when it runs.
def _score_inner(
    source: str,
    summary: str,
    sample_count: int,
    seed: int | None,
    compute_consistency: bool,
) -> dict:
    logger.info(
        "_score_inner ENTER: device=%s IS_ZERO_GPU=%s sample_count=%d seed=%s",
        backend._device, _IS_ZERO_GPU, sample_count, seed,
    )
    if _IS_ZERO_GPU:
        backend.to("cuda")
        logger.info("_score_inner: backend moved to %s", backend._device)
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
            logger.info("_score_inner: backend moved back to %s", backend._device)
    out = _serialize_summary_score(
        result, _normalizer, _amb_normalizer, _con_normalizer,
        compute_consistency=compute_consistency,
    )
    logger.info("_score_inner EXIT: %d sentence(s)", len(out.get("sentence_results", [])))
    return out

# ---------------------------------------------------------------------------
# ZeroGPU diagnostic probes — graduated complexity
# ---------------------------------------------------------------------------
# Each probe is a separate @spaces.GPU-decorated function registered as a
# Gradio API endpoint.  Hit /probe/N in sequence to isolate where things break:
#   1 — does @spaces.GPU itself work? (no model code at all)
#   2 — can we allocate a CUDA tensor?
#   3 — can we move the backend model to CUDA and back?
# A non-GPU /probe/env endpoint returns CUDA_VISIBLE_DEVICES and related vars.

@spaces.GPU
def _probe1_spaces_gpu() -> dict:
    """Probe 1: confirm the @spaces.GPU worker starts and CUDA is visible."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        ),
    }


@spaces.GPU
def _probe2_cuda_tensor() -> dict:
    """Probe 2: allocate a small tensor on CUDA."""
    t = torch.zeros(4, device="cuda")
    return {"ok": True, "device": str(t.device), "sum": float(t.sum())}


@spaces.GPU
def _probe3_model_move() -> dict:
    """Probe 3: move the backend model to CUDA, check a parameter device, move back."""
    backend.to("cuda")
    try:
        sample_param = next(iter(backend._lora_baseline.values()))
        return {
            "ok": True,
            "backend_device": backend._device,
            "baseline_param_device": str(sample_param.device),
        }
    finally:
        backend.to("cpu")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

# ZeroGPU requires @spaces.GPU to be on the exact function registered with
# Gradio as the event handler — nested calls don't get the GPU worker context.

@spaces.GPU
def _score_api(
    source: str,
    summary: str,
    sample_count: float,  # gr.Number returns float
    seed_raw: float,      # -1 sentinel means None
    compute_consistency: bool,
) -> dict:
    """Full-parameter handler — registered as api_name='score_api' and used by the FastAPI proxy."""
    logger.info(
        "_score_api ENTER: sample_count=%s seed_raw=%s compute_consistency=%s "
        "src_len=%d smr_len=%d",
        sample_count, seed_raw, compute_consistency, len(source), len(summary),
    )
    seed = int(seed_raw) if seed_raw is not None and seed_raw >= 0 else None
    result = _score_inner(source, summary, int(sample_count), seed, compute_consistency)
    logger.info("_score_api EXIT: OK")
    return result


@spaces.GPU
def _ui_score(source: str, summary: str) -> dict:
    """Simplified handler for the UI Score button (fixed defaults)."""
    return _score_inner(source, summary, 10, None, True)


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
        fn=_ui_score,
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

    # Probe endpoints — hidden, no inputs, JSON output each.
    with gr.Row(visible=False):
        _p1_out = gr.JSON()
        _p1_btn = gr.Button()
        _p2_out = gr.JSON()
        _p2_btn = gr.Button()
        _p3_out = gr.JSON()
        _p3_btn = gr.Button()
    _p1_btn.click(fn=_probe1_spaces_gpu, inputs=[], outputs=[_p1_out], api_name="probe1")
    _p2_btn.click(fn=_probe2_cuda_tensor, inputs=[], outputs=[_p2_out], api_name="probe2")
    _p3_btn.click(fn=_probe3_model_move,  inputs=[], outputs=[_p3_out], api_name="probe3")

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


@app.get("/probe/env")
def probe_env() -> dict:
    """No GPU — just return environment variables relevant to ZeroGPU."""
    return {
        "SPACES_ZERO_GPU": os.environ.get("SPACES_ZERO_GPU"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "is_zero_gpu_flag": _IS_ZERO_GPU,
        "backend_device": backend._device,
        "cuda_available_no_gpu_context": torch.cuda.is_available(),
    }


def _call_probe(api_name: str) -> dict:
    client = _get_gradio_client()
    return client.predict(api_name=f"/{api_name}")


@app.get("/probe/1")
def probe1() -> dict:
    """Probe 1: @spaces.GPU worker starts, CUDA visible (no model code)."""
    return _call_probe("probe1")


@app.get("/probe/2")
def probe2() -> dict:
    """Probe 2: allocate a CUDA tensor inside @spaces.GPU."""
    return _call_probe("probe2")


@app.get("/probe/3")
def probe3() -> dict:
    """Probe 3: move backend model to CUDA and back inside @spaces.GPU."""
    return _call_probe("probe3")


demo.block_thread()
