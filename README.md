---
title: Sentence Uncertainty
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# summarizer-uncertainty-ml
ML backend for the text summarizer with uncertainty

## Environment setup

This project uses Pipenv and targets Python 3.13.

Install dependencies:

```bash
pipenv install --dev
```

Start the API locally:

```bash
pipenv run python -m src.main
```

The server listens on `http://127.0.0.1:8000` by default.

## Docker

Build the container image:

```bash
docker build -t summarizer-uncertainty-ml .
```

Run the container and expose the API on port `8000`:

```bash
docker run --rm -p 8000:8000 --name summarizer-uncertainty-ml summarizer-uncertainty-ml
```

The backend is selected via the `SCORING_BACKEND` environment variable.

## Scoring backends

### `dummy` (default) — implemented

A rule-based mock that produces deterministic pseudo-uncertainty scores without loading any model. Useful for smoke-testing the API and downstream consumers without GPU or network access.

### `mc_dropout` — implemented

Loads a pre-trained HuggingFace seq2seq model (default: `sshleifer/distilbart-cnn-12-6`) and scores the provided summary using Monte Carlo Dropout. The model is kept in training mode so dropout remains active, and `sample_count` independent forward passes are run with different dropout masks. Epistemic uncertainty per sentence is derived from the disagreement across passes (mutual information between predictions and model weights).

Configure with:
- `MC_DROPOUT_MODEL` — HuggingFace model identifier
- `MC_DROPOUT_DEVICE` — torch device (`cpu`, `cuda`; auto-detected if unset)

```bash
docker run --rm -p 8000:8000 \
  -e SCORING_BACKEND=mc_dropout \
  -e MC_DROPOUT_MODEL=sshleifer/distilbart-cnn-12-6 \
  summarizer-uncertainty-ml
```

### `laplace` — not yet implemented

Will fine-tune a LoRA adapter on `(source, summary)` pairs, fit a Laplace approximation over the adapter weights, and draw posterior weight samples for each scoring request. This is the most principled approach: uncertainty will reflect genuine knowledge gaps relative to the training data rather than dropout noise.

## Example request

Check that the service is up:

```bash
curl http://127.0.0.1:8000/health
```

Send a scoring request:

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "source": "OpenAI released a new summarization system for long technical documents.",
    "summary": "OpenAI released a new summarization system. It is designed for long technical documents.",
    "sample_count": 20,
    "seed": 7
  }'
```

Expected response shape:

```json
{
  "sentence_results": [
    {
      "sentence_index": 0,
      "sentence_text": "OpenAI released a new summarization system.",
      "uncertainty": 0.12,
      "uncertainty_raw": 0.12,
      "uncertainty_score": 12.0,
      "uncertainty_band": "low"
    }
  ],
  "normalization": {
    "boundaries": [0.1, 0.2, 0.4]
  }
}
```
