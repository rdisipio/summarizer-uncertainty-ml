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

The current container defaults to the dummy scoring backend via `SCORING_BACKEND=dummy`.

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
      "uncertainty_score": 0.21,
      "uncertainty_band": "low"
    }
  ],
  "normalization": {
    "boundaries": [0.1, 0.2, 0.4]
  }
}
```
