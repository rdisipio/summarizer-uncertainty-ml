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

Loads a pre-trained HuggingFace seq2seq model (default: `facebook/bart-large-xsum`) and scores the provided summary using Monte Carlo Dropout. The model is kept in training mode so dropout remains active, and `sample_count` independent forward passes are run with different dropout masks. Epistemic uncertainty per sentence is derived from the disagreement across passes (mutual information between predictions and model weights).

Configure with:
- `MC_DROPOUT_MODEL` — HuggingFace model identifier
- `MC_DROPOUT_DEVICE` — torch device (`cpu`, `cuda`; auto-detected if unset)

```bash
docker run --rm -p 8000:8000 \
  -e SCORING_BACKEND=mc_dropout \
  -e MC_DROPOUT_MODEL=facebook/bart-large-xsum \
  summarizer-uncertainty-ml
```

**Caveat: scoring model bias toward formal style.** `sshleifer/distilbart-cnn-12-6` (and its parent `facebook/bart-large-cnn`) were fine-tuned on CNN/DailyMail news articles, so they expect formal, extractive summaries close to the source wording. When the summary uses informal vocabulary or paraphrases the source loosely, the model assigns low probability to those tokens under teacher-forcing — not because the content is wrong, but because the phrasing is unexpected given its training distribution. This inflates uncertainty scores for informal summaries relative to formal ones covering the same content. The effect is reduced when using the Laplace backend fine-tuned on your own data.

### `lora_laplace` — implemented

Fine-tunes a LoRA adapter on `(source, summary)` pairs, fits a diagonal Laplace approximation over the adapter weights, and draws posterior weight samples for each scoring request. Uncertainty reflects genuine knowledge gaps relative to the training data rather than dropout noise.

Configure with:
- `LORA_BASE_MODEL` — HuggingFace base model identifier (default: `facebook/bart-large-xsum`)
- `LORA_ADAPTER_PATH` — path to the PEFT adapter checkpoint directory (required)
- `LORA_SAMPLER_PATH` — path to the pre-fitted `laplace_sampler.npz` (required)
- `LORA_DEVICE` — torch device (`cpu`, `cuda`; auto-detected if unset)

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
  "source": "OpenAI released a new summarization system for long technical documents.",
  "summary": "OpenAI released a new summarization system. It is designed for long technical documents.",
  "sample_count": 20,
  "sentence_results": [
    {
      "sentence_index": 0,
      "sentence_text": "OpenAI released a new summarization system.",
      "uncertainty": 0.12,
      "uncertainty_raw": 0.12,
      "uncertainty_score": 12.0,
      "uncertainty_band": "low",
      "mean_logprob": -2.45,
      "epistemic_mi": 0.12,
      "predictive_entropy": 0.74,
      "expected_entropy": 0.62,
      "ambiguity_score": 38.5,
      "ambiguity_band": "mid",
      "token_scores": [
        {
          "token_index": 0,
          "target_token_id": 1234,
          "mean_logprob": -1.80,
          "predictive_entropy": 0.71,
          "expected_entropy": 0.60,
          "epistemic_mi": 0.11
        }
      ]
    }
  ],
  "normalization": {
    "boundaries": [0.032, 0.073, 0.084, 0.096, 0.352]
  }
}
```

### Uncertainty signals

| Field | Derived from | Meaning |
|---|---|---|
| `uncertainty_score` | `epistemic_mi` (normalized 0–100) | Model's knowledge uncertainty — high when the input is out-of-distribution relative to training data. The primary "don't trust this" signal. |
| `ambiguity_score` | `expected_entropy` (normalized 0–100) | Linguistic vagueness intrinsic to the text — high when the summary sentence is inherently indeterminate, independent of model confidence. |
| `uncertainty_band` / `ambiguity_band` | | Coarse categorical label: `low`, `mid`, or `high`. |
