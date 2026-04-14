FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIPENV_VENV_IN_PROJECT=0
ENV NLTK_DATA=/usr/local/share/nltk_data
ENV SCORING_BACKEND=dummy
ENV QUANTILE_CONFIG_PATH=/app/config/uncertainty_quantiles_mc_dropout.json
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

RUN python -m src.nltk_setup

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "-m", "src.main"]
