FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIPENV_VENV_IN_PROJECT=0
ENV NLTK_DATA=/usr/local/share/nltk_data
ENV SCORING_BACKEND=dummy
ENV QUANTILE_CONFIG_PATH=/app/config/uncertainty_quantiles.json
ENV PORT=8000

WORKDIR /app

RUN pip install --no-cache-dir pipenv

COPY Pipfile Pipfile.lock /app/
RUN pipenv install --system --deploy

COPY src /app/src
COPY config /app/config
COPY README.md AGENTS.md /app/

RUN python -m src.nltk_setup

EXPOSE 8000

CMD ["python", "-m", "src.main"]
