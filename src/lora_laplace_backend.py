"""LoRA + Laplace Approximation backend for epistemic uncertainty estimation.

After fine-tuning a LoRA adapter (MAP estimate), the posterior over LoRA
parameters is approximated as a Gaussian via the Laplace approximation:

    p(theta | D) ≈ N(theta_MAP, (F + lambda * I)^{-1})

where F is the diagonal empirical Fisher information matrix computed on a small
calibration corpus.  Each "posterior sample" is an additive perturbation

    delta_theta ~ N(0, Sigma),   Sigma = diag(1 / (F_ii + lambda))

applied to the LoRA weight matrices before the forward pass and removed
immediately after.  Diversity across samples arises from weight-space
uncertainty rather than from dropout.

The model is kept in eval() mode throughout so that batch-norm and dropout
layers are deterministic; stochasticity is confined to the weight perturbation.

Memory note: the per-sample perturbation dict is CPU-resident and moved to the
model device only during the forward pass.  LoRA baseline weights are stored
once at construction time and used to restore exact parameter values after
each sample, avoiding floating-point accumulation errors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as exc:
    raise ImportError(
        "The 'peft' package is required for the LoRA-Laplace backend. "
        "Install it with: pip install peft"
    ) from exc

from .scorer import (
    PreparedSummary,
    RuleBasedSentenceBackend,
    SampledSentenceDistributions,
    SummaryUncertaintyScorer,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "facebook/bart-large-xsum"


@dataclass
class LaplaceState:
    """A posterior sample expressed as per-LoRA-parameter additive perturbations.

    Each key in ``perturbations`` is the fully-qualified parameter name as
    returned by ``model.named_parameters()``.  Values are CPU tensors with the
    same shape as the corresponding model parameter.
    """

    perturbations: dict[str, torch.Tensor]


class DiagonalLaplacePosteriorSampler:
    """Samples LoRA weight perturbations from a diagonal Laplace posterior.

    The posterior is N(0, diag(posterior_variance)) where each element
    corresponds to one scalar LoRA parameter.  The MAP point (posterior mean)
    is baked into the model weights; only the zero-mean perturbation is
    returned.

    Args:
        param_names: Ordered list of LoRA parameter names.
        param_shapes: Shape of each parameter, in the same order.
        posterior_variance: Per-scalar posterior variance — a flat array whose
            length equals the total number of LoRA scalars, ordered by
            (param_name, flattened position within that parameter).
    """

    def __init__(
        self,
        param_names: list[str],
        param_shapes: list[tuple[int, ...]],
        posterior_variance: np.ndarray,
    ) -> None:
        if len(param_names) != len(param_shapes):
            raise ValueError("param_names and param_shapes must have the same length.")
        param_sizes = [int(np.prod(s)) for s in param_shapes]
        total_scalars = sum(param_sizes)
        if posterior_variance.shape != (total_scalars,):
            raise ValueError(
                f"posterior_variance must be 1-D with {total_scalars} elements; "
                f"got shape {posterior_variance.shape}."
            )
        self._param_names = param_names
        self._param_shapes = param_shapes
        self._param_sizes = param_sizes
        self._posterior_std = np.sqrt(np.clip(posterior_variance, 0.0, None))

    def sample(self, seed: int | None = None) -> LaplaceState:
        """Draw one perturbation sample from the diagonal Laplace posterior.

        Args:
            seed: Seed for numpy's default_rng.  Passing the same seed
                returns an identical sample, enabling reproducibility.

        Returns:
            A LaplaceState whose perturbations add to the MAP LoRA weights.
        """
        import random as _random
        _random.seed(seed)
        perturbations: dict[str, torch.Tensor] = {}
        offset = 0
        for name, shape, size in zip(
            self._param_names, self._param_shapes, self._param_sizes
        ):
            std_chunk = self._posterior_std[offset : offset + size]
            delta = torch.tensor(
                [_random.gauss(0.0, float(s)) for s in std_chunk],
                dtype=torch.float32,
            ).reshape(shape)
            perturbations[name] = delta
            offset += size

        return LaplaceState(perturbations=perturbations)


class LoraLaplaceBackend(RuleBasedSentenceBackend):
    """Teacher-forced scoring backend using LoRA + Laplace for epistemic uncertainty.

    The base model weights are frozen (requires_grad=False).  On each call to
    score_posterior_sample the LoRA adapter weights are temporarily replaced
    with MAP + delta (from the LaplaceState), a single teacher-forced forward
    pass is executed, and the original MAP weights are restored exactly from a
    saved copy.

    Args:
        peft_model: A loaded PeftModel with LoRA adapters.
        tokenizer: Matching tokenizer for the model.
        device: Torch device string.  Auto-detected when None.
    """

    def __init__(
        self,
        peft_model: PeftModel,
        tokenizer: Any,
        device: str | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device
        self._model = peft_model.to(device)
        self._model.eval()
        self._tokenizer = tokenizer

        # Save MAP LoRA weights once so we can restore them exactly after each
        # sample without floating-point accumulation errors.
        self._lora_baseline: dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in self._model.named_parameters()
            if param.requires_grad
        }

        lora_scalar_count = sum(t.numel() for t in self._lora_baseline.values())
        logger.info(
            "LoraLaplaceBackend ready on %r: %d total params, %d LoRA scalars",
            device,
            sum(p.numel() for p in self._model.parameters()),
            lora_scalar_count,
        )
        logger.info("Running warm-up forward pass")
        with torch.no_grad():
            _dummy = torch.tensor([[self._model.config.decoder_start_token_id]], device=self._device)
            self._model(input_ids=_dummy, decoder_input_ids=_dummy)
        logger.info("Warm-up complete")

    # ------------------------------------------------------------------
    # SummaryScoringBackend interface
    # ------------------------------------------------------------------

    def prepare_summary(
        self,
        source: str,
        summary: str,
        sentences: Sequence[str] | None = None,
    ) -> PreparedSummary:
        """Tokenize source and summary; map sentence specs to decoder token ranges.

        Character offsets from NLTK sentence splitting are used to align each
        sentence to a contiguous slice of summary BPE tokens.  The slices are
        stored in metadata so that score_posterior_sample can extract per-
        sentence logit rows without re-tokenizing.
        """
        logger.info(
            "Preparing summary: %d source chars, %d summary chars",
            len(source),
            len(summary),
        )
        prepared = super().prepare_summary(source, summary, sentences)
        logger.info("%d sentence(s) identified", len(prepared.sentences))

        max_length = min(self._tokenizer.model_max_length, 1024)

        encoder_encoding = self._tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoder_input_ids = encoder_encoding["input_ids"].to(self._device)
        encoder_attention_mask = encoder_encoding["attention_mask"].to(self._device)
        logger.info("Source tokenized: %d tokens", encoder_input_ids.shape[1])

        summary_encoding = self._tokenizer(
            summary,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        summary_token_ids_tensor = summary_encoding["input_ids"]
        offset_mapping: list[tuple[int, int]] = (
            summary_encoding["offset_mapping"][0].tolist()
        )
        summary_token_ids: list[int] = summary_token_ids_tensor.squeeze(0).tolist()
        logger.info("Summary tokenized: %d tokens", len(summary_token_ids))

        sentence_token_slices: dict[int, tuple[int, int]] = {}
        for sentence_spec in prepared.sentences:
            char_start = sentence_spec.char_start
            char_end = sentence_spec.char_end
            token_indices = [
                i
                for i, (tok_s, tok_e) in enumerate(offset_mapping)
                if tok_e > char_start and tok_s < char_end
            ]
            if not token_indices:
                raise ValueError(
                    f"Sentence {sentence_spec.sentence_index!r} could not be aligned "
                    "to any summary tokens.  The summary may have been truncated."
                )
            tok_start, tok_end = token_indices[0], token_indices[-1] + 1
            sentence_token_slices[sentence_spec.sentence_index] = (tok_start, tok_end)
            logger.info(
                "Sentence %d aligned: tokens [%d, %d) (%d tokens)",
                sentence_spec.sentence_index,
                tok_start,
                tok_end,
                tok_end - tok_start,
            )

        decoder_start = torch.tensor(
            [[self._model.config.decoder_start_token_id]],
            device=self._device,
        )
        decoder_input_ids = torch.cat(
            [decoder_start, summary_token_ids_tensor.to(self._device)],
            dim=1,
        )
        logger.info(
            "Decoder input ready: %d tokens (including BOS)", decoder_input_ids.shape[1]
        )

        metadata = dict(prepared.metadata)
        metadata["encoder_input_ids"] = encoder_input_ids
        metadata["encoder_attention_mask"] = encoder_attention_mask
        metadata["decoder_input_ids"] = decoder_input_ids
        metadata["summary_token_ids"] = summary_token_ids
        metadata["sentence_token_slices"] = sentence_token_slices

        return PreparedSummary(
            source=prepared.source,
            summary=prepared.summary,
            sentences=prepared.sentences,
            metadata=metadata,
        )

    def score_posterior_sample(
        self,
        prepared_summary: PreparedSummary,
        posterior_sample: LaplaceState,
    ) -> Sequence[SampledSentenceDistributions]:
        """Apply LoRA perturbation, run a teacher-forced forward pass, restore weights.

        The LoRA adapter weights are set to MAP + delta for the duration of the
        forward pass, then restored to the exact MAP values from the saved
        baseline.  Restoration happens in a finally block so partial failures
        cannot leave the model in a perturbed state.
        """
        encoder_input_ids: torch.Tensor = prepared_summary.metadata["encoder_input_ids"]
        encoder_attention_mask: torch.Tensor = prepared_summary.metadata[
            "encoder_attention_mask"
        ]
        decoder_input_ids: torch.Tensor = prepared_summary.metadata["decoder_input_ids"]
        summary_token_ids: list[int] = prepared_summary.metadata["summary_token_ids"]
        sentence_token_slices: dict[int, tuple[int, int]] = prepared_summary.metadata[
            "sentence_token_slices"
        ]

        self._apply_perturbation(posterior_sample.perturbations)
        try:
            with torch.no_grad():
                outputs = self._model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
        finally:
            self._restore_baseline()

        # logits: (1, decoder_seq_len, vocab_size)
        logits = outputs.logits.squeeze(0)
        probs = F.softmax(logits, dim=-1).cpu().float().numpy()

        sentence_distributions: list[SampledSentenceDistributions] = []
        for sentence_spec in prepared_summary.sentences:
            tok_start, tok_end = sentence_token_slices[sentence_spec.sentence_index]
            target_ids = np.array(
                summary_token_ids[tok_start:tok_end], dtype=np.int64
            )
            token_probs = probs[tok_start:tok_end, :]
            sentence_distributions.append(
                SampledSentenceDistributions(
                    sentence_index=sentence_spec.sentence_index,
                    target_token_ids=target_ids,
                    token_probabilities=token_probs,
                )
            )

        return sentence_distributions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_perturbation(self, perturbations: dict[str, torch.Tensor]) -> None:
        """Set each LoRA parameter to its MAP value plus the sampled delta."""
        for name, delta in perturbations.items():
            param = self._model.get_parameter(name)
            param.data.copy_(self._lora_baseline[name] + delta.to(param.device))

    def _restore_baseline(self) -> None:
        """Restore all LoRA parameters to their saved MAP values."""
        for name, baseline in self._lora_baseline.items():
            self._model.get_parameter(name).data.copy_(baseline)


# ---------------------------------------------------------------------------
# Calibration: fit the diagonal Laplace approximation
# ---------------------------------------------------------------------------


def fit_laplace_approximation(
    backend: LoraLaplaceBackend,
    calibration_data: Sequence[tuple[str, str]],
    prior_precision: float = 1.0,
) -> DiagonalLaplacePosteriorSampler:
    """Fit a diagonal Laplace approximation over the LoRA parameters.

    For each (source, summary) pair in calibration_data, a teacher-forced
    forward pass is run, the cross-entropy loss is back-propagated, and the
    squared gradient for each LoRA parameter scalar is accumulated into the
    empirical Fisher diagonal:

        F_ii  ≈  E[(d L / d theta_i)^2]

    The posterior variance is then:

        Sigma_ii  =  1 / (F_ii + prior_precision)

    Args:
        backend: An initialised LoraLaplaceBackend (model in MAP state).
        calibration_data: List of (source, summary) pairs to accumulate
            Fisher statistics.  More pairs → better Fisher estimate.
        prior_precision: Isotropic Gaussian prior precision added to the Fisher
            diagonal.  Acts as L2 regularisation; higher values reduce
            posterior variance (more conservative uncertainty estimates).

    Returns:
        A DiagonalLaplacePosteriorSampler ready to use with
        SummaryUncertaintyScorer.
    """
    if not calibration_data:
        raise ValueError("calibration_data must contain at least one (source, summary) pair.")
    if prior_precision <= 0.0:
        raise ValueError("prior_precision must be positive.")

    model = backend._model
    tokenizer = backend._tokenizer
    device = backend._device

    lora_params: list[tuple[str, torch.nn.Parameter]] = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    if not lora_params:
        raise ValueError(
            "No trainable parameters found.  Ensure the PeftModel has LoRA "
            "adapters with requires_grad=True."
        )

    param_names = [name for name, _ in lora_params]
    param_shapes = [tuple(param.shape) for _, param in lora_params]
    param_sizes = [int(np.prod(s)) for s in param_shapes]
    total_scalars = sum(param_sizes)

    logger.info(
        "Fitting diagonal Laplace approximation: %d LoRA params, %d scalars, "
        "%d calibration samples",
        len(lora_params),
        total_scalars,
        len(calibration_data),
    )

    fisher_diag = np.zeros(total_scalars, dtype=np.float64)
    max_length = min(tokenizer.model_max_length, 1024)

    # Switch to train mode only for the backward pass, to allow gradients.
    # We do NOT activate dropout — eval() semantics are retained for all
    # non-LoRA layers by temporarily setting only requires_grad on LoRA params.
    model.eval()

    for sample_index, (source, summary) in enumerate(calibration_data):
        logger.info(
            "Calibration sample %d / %d", sample_index + 1, len(calibration_data)
        )

        encoder_encoding = tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        summary_encoding = tokenizer(
            summary,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        encoder_input_ids = encoder_encoding["input_ids"].to(device)
        encoder_attention_mask = encoder_encoding["attention_mask"].to(device)
        # labels: the model internally applies teacher-forcing and computes
        # cross-entropy loss.  -100 positions are ignored; here we score all
        # summary tokens.
        labels = summary_encoding["input_ids"].to(device)

        model.zero_grad()
        outputs = model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        # Accumulate squared gradients into the empirical Fisher diagonal.
        offset = 0
        for i, (_, param) in enumerate(lora_params):
            size = param_sizes[i]
            if param.grad is not None:
                grad_sq = (
                    param.grad.detach().cpu().double().pow(2).flatten().numpy()
                )
                fisher_diag[offset : offset + size] += grad_sq
            offset += size

    # Average over calibration samples.
    fisher_diag /= len(calibration_data)

    posterior_variance = 1.0 / (fisher_diag + prior_precision)

    logger.info(
        "Laplace fit complete: posterior std range [%.3e, %.3e]",
        float(np.sqrt(posterior_variance.min())),
        float(np.sqrt(posterior_variance.max())),
    )

    # Restore model to inference state and clear gradients.
    model.zero_grad()
    # Restore exact MAP weights in case any in-place ops touched them.
    backend._restore_baseline()

    return DiagonalLaplacePosteriorSampler(
        param_names=param_names,
        param_shapes=param_shapes,
        posterior_variance=posterior_variance,
    )


# ---------------------------------------------------------------------------
# Serialisation helpers for the fitted sampler
# ---------------------------------------------------------------------------


def save_laplace_sampler(sampler: DiagonalLaplacePosteriorSampler, path: str) -> None:
    """Persist a fitted DiagonalLaplacePosteriorSampler to a .npz file.

    Args:
        sampler: A sampler produced by fit_laplace_approximation.
        path: Destination file path (a .npz extension is conventional).
    """
    np.savez(
        path,
        param_names=np.array(sampler._param_names),
        param_shapes=np.array(
            [list(s) for s in sampler._param_shapes], dtype=object
        ),
        posterior_std=sampler._posterior_std,
    )
    logger.info("Laplace sampler saved to %r", path)


def load_laplace_sampler(path: str) -> DiagonalLaplacePosteriorSampler:
    """Load a DiagonalLaplacePosteriorSampler previously saved by save_laplace_sampler.

    Args:
        path: Path to the .npz file written by save_laplace_sampler.

    Returns:
        A DiagonalLaplacePosteriorSampler ready for use.
    """
    data = np.load(path, allow_pickle=True)
    param_names: list[str] = data["param_names"].tolist()
    param_shapes: list[tuple[int, ...]] = [
        tuple(int(d) for d in shape) for shape in data["param_shapes"].tolist()
    ]
    posterior_std: np.ndarray = data["posterior_std"]
    posterior_variance = posterior_std ** 2
    logger.info("Laplace sampler loaded from %r (%d params)", path, len(param_names))
    return DiagonalLaplacePosteriorSampler(
        param_names=param_names,
        param_shapes=param_shapes,
        posterior_variance=posterior_variance,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_lora_laplace_scorer(
    base_model_name: str,
    peft_model_path: str,
    calibration_data: Sequence[tuple[str, str]],
    prior_precision: float = 1.0,
    device: str | None = None,
) -> SummaryUncertaintyScorer:
    """Load a LoRA-adapted model, fit a Laplace approximation, return a scorer.

    This is the primary entry-point for end-to-end use.  For workflows that
    re-use a pre-fitted sampler (e.g., the sampler was saved with
    save_laplace_sampler), construct LoraLaplaceBackend and
    DiagonalLaplacePosteriorSampler separately and pass them to
    SummaryUncertaintyScorer directly.

    Args:
        base_model_name: HuggingFace identifier for the seq2seq base model.
        peft_model_path: Local path or HuggingFace Hub ID of the PEFT/LoRA
            checkpoint directory.
        calibration_data: List of (source, summary) pairs used to estimate the
            Fisher information diagonal.  Aim for at least 50–200 pairs that
            represent the deployment distribution.
        prior_precision: Isotropic prior precision (lambda) added to the Fisher
            diagonal.  Defaults to 1.0; increase to shrink posterior variance.
        device: Torch device string.  Auto-detected when None.

    Returns:
        A SummaryUncertaintyScorer wired to the LoRA-Laplace backend and
        sampler.
    """
    logger.info("Loading base model %r", base_model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    logger.info("Loading LoRA adapter from %r", peft_model_path)
    peft_model = PeftModel.from_pretrained(base_model, peft_model_path, is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    backend = LoraLaplaceBackend(
        peft_model=peft_model,
        tokenizer=tokenizer,
        device=device,
    )

    logger.info(
        "Fitting Laplace approximation on %d calibration samples", len(calibration_data)
    )
    sampler = fit_laplace_approximation(
        backend=backend,
        calibration_data=calibration_data,
        prior_precision=prior_precision,
    )

    return SummaryUncertaintyScorer(backend=backend, posterior_sampler=sampler)
