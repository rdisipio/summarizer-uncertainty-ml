FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIPENV_VENV_IN_PROJECT=0
ENV NLTK_DATA=/usr/local/share/nltk_data
ENV SCORING_BACKEND=lora_laplace
ENV LORA_BASE_MODEL=facebook/bart-large-xsum
ENV LORA_HUB_REPO=rdisipio/summarizer-uncertainty-models
ENV LORA_HUB_SUBFOLDER=bart-large-xsum-lora-laplace
ENV LORA_ADAPTER_PATH=/app/models/bart-large-xsum-lora-laplace
ENV LORA_SAMPLER_PATH=/app/models/bart-large-xsum-lora-laplace/laplace_sampler.npz
ENV QUANTILE_CONFIG_PATH=/app/models/bart-large-xsum-lora-laplace/uncertainty_quantiles_lora_laplace.json
ENV PORT=7860

WORKDIR /app

RUN pip install --no-cache-dir pipenv

COPY Pipfile Pipfile.lock /app/
RUN pipenv install --system --deploy

# PyPI ships the CUDA-enabled torch wheel by default.  HuggingFace Spaces
# CPU instances have no CUDA libraries, so we reinstall from the CPU-only
# index to avoid the libcudart / libcublasLt load failure at startup.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY src /app/src
COPY config /app/config
COPY README.md AGENTS.md /app/

# Download adapter, sampler and quantile config from the HuggingFace Hub at build time.
RUN python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="rdisipio/summarizer-uncertainty-models",
    allow_patterns="bart-large-xsum-lora-laplace/*",
    local_dir="/app/models",
)
EOF

RUN python -m src.nltk_setup

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "-m", "src.main"]
