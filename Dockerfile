FROM python:3.13-slim

# Set to "gpu" at build time to keep the CUDA-enabled torch wheel.
# Default is "cpu": after pipenv resolves packages, the CUDA wheel is replaced
# with the CPU-only build so the server imports cleanly on CPU-only HF Spaces.
ARG DEVICE=cpu

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIPENV_VENV_IN_PROJECT=0
ENV NLTK_DATA=/usr/local/share/nltk_data
ENV SCORING_BACKEND=lora_laplace
ENV LORA_BASE_MODEL=google/flan-t5-small
ENV LORA_HUB_REPO=rdisipio/summarizer-uncertainty-models
ENV LORA_HUB_SUBFOLDER=flan-t5-small-lora-laplace
ENV LORA_ADAPTER_PATH=/app/models/flan-t5-small-lora-laplace
ENV LORA_SAMPLER_PATH=/app/models/flan-t5-small-lora-laplace/laplace_sampler.npz
ENV QUANTILE_CONFIG_PATH=/app/config/uncertainty_quantiles_lora_laplace.json
ENV AMBIGUITY_QUANTILE_CONFIG_PATH=/app/config/ambiguity_quantiles_lora_laplace.json
ENV CONSISTENCY_QUANTILE_CONFIG_PATH=/app/config/consistency_quantiles_lora_laplace.json
ENV API_TOKEN=
ENV PORT=7860

WORKDIR /app

RUN pip install --no-cache-dir pipenv

COPY Pipfile Pipfile.lock /app/
RUN pipenv install --system --deploy

# The default PyPI torch wheel bundles CUDA and fails to import when CUDA
# runtime libraries are absent (CPU-only HF Spaces).  Replace it with the
# CPU-only wheel unless this is a GPU build.
RUN if [ "$DEVICE" = "cpu" ]; then \
    pip install --no-cache-dir --force-reinstall torch \
        --index-url https://download.pytorch.org/whl/cpu; \
fi

COPY src /app/src
COPY config /app/config
COPY README.md AGENTS.md /app/

# Download adapter, sampler and quantile config from the HuggingFace Hub at build time.
RUN python - <<EOF
import os
from huggingface_hub import snapshot_download
subfolder = os.environ.get("LORA_HUB_SUBFOLDER", "flan-t5-small-lora-laplace")
repo_id   = os.environ.get("LORA_HUB_REPO", "rdisipio/summarizer-uncertainty-models")
snapshot_download(
    repo_id=repo_id,
    allow_patterns=f"{subfolder}/*",
    local_dir="/app/models",
)
EOF

RUN python -m src.nltk_setup

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "-m", "src.main"]
