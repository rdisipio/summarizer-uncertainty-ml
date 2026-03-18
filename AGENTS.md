# AGENTS.md

## Purpose
This repository contains the ML backend for a text summarizer with uncertainty scoring.

The main product goal is to produce per-sentence epistemic uncertainty for an existing summary using a Laplace approximation over LoRA adapter weights on a seq2seq summarization model.

## Project Context
- The displayed summary may come from any external LLM or generator.
- Uncertainty must come only from the Laplace posterior over LoRA adapter parameters.
- Inference is a post-hoc scoring pass over an existing summary, not a generation workflow.
- Sentence scoring should preserve the preceding summary prefix as decoder context.

## Expected Architecture
- Keep importable Python code under [`src/`](/Users/disipio/development/summarizer-uncertainty-ml/src) as a package-oriented layout.
- Treat loose files at repository root as entrypoints or project metadata only.
- Group related code by feature or domain.

Suggested module responsibilities:
- `train.py`: LoRA training pipeline
- `laplace.py`: Laplace fitting utilities
- `sampler.py`: posterior sampling and adapter injection
- `scorer.py`: token and sentence scoring
- `api_server.py`: REST API for scoring requests
- `utils.py`: tokenization and model-loading helpers
- `config.py`: hyperparameters and constants

## Environment
- Use Pipenv for dependency and environment management.
- Target Python version is `>=3.13`.
- Do not introduce tooling that replaces Pipenv unless explicitly requested.

## Code Standards
- Follow PEP 8.
- Prefer clear, readable code over clever code.
- Keep functions and classes focused on a single responsibility.
- Avoid duplicated logic; extract shared reusable behavior when justified.
- Put imports at the top of each source file.
- Use type annotations for all function parameters and return values.
- Add a short docstring to every public function, method, and class.
- If a function has side effects, mention them in the docstring.

## Implementation Rules
- Make minimal, focused changes that solve the requested task.
- Do not refactor unrelated code unless explicitly asked.
- Do not add new dependencies unless they are necessary for the requested task.
- Preserve backward compatibility unless the user explicitly requests a breaking change.
- If a requirement is ambiguous, choose the simplest valid implementation.
- State assumptions clearly when they materially affect behavior.

## Testing
- Add or update tests for new or changed behavior when a test suite exists.
- Do not break existing tests.
- Prefer deterministic behavior; avoid hidden global state and uncontrolled time dependence.

## Security
- Never hardcode secrets, API keys, passwords, or tokens.
- Validate external input at boundaries.
- Prefer safe standard-library or well-maintained library solutions over custom insecure code.

## Domain-Specific Guidance
- Default modeling approach is a seq2seq summarizer with PEFT/LoRA adapters.
- Start with diagonal Laplace over adapter parameters only.
- Teacher-forced scoring should evaluate the exact sentence tokens of the provided summary.
- Sentence-level uncertainty should be derived from token-level predictive disagreement across posterior samples.
- Prefer returning metrics such as mean log-probability, predictive entropy, expected entropy, and epistemic mutual information.
- If calibration is implemented, keep it as a separate mapping layer from raw epistemic features to human-facing uncertainty scores.

## API Expectations
- A scoring endpoint should accept `source`, `summary`, and optional `sentences`.
- If `sentences` are omitted, the server may derive them.
- Load model artifacts at server startup rather than per request when practical.
- Treat inference as compute-heavy and avoid unbounded concurrency.

## Source Material
- [`README.md`](/Users/disipio/development/summarizer-uncertainty-ml/README.md): brief repository description
- [`WORKFLOW.md`](/Users/disipio/development/summarizer-uncertainty-ml/WORKFLOW.md): product and architecture guidance
- [`RULES.md`](/Users/disipio/development/summarizer-uncertainty-ml/RULES.md): coding and assistant behavior rules

## Maintenance
- Keep this file aligned with the intent of [`WORKFLOW.md`](/Users/disipio/development/summarizer-uncertainty-ml/WORKFLOW.md) and [`RULES.md`](/Users/disipio/development/summarizer-uncertainty-ml/RULES.md).
- Do not edit [`RULES.md`](/Users/disipio/development/summarizer-uncertainty-ml/RULES.md) unless explicitly requested.
