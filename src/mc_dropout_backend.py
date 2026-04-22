"""MC Dropout inference backend for epistemic uncertainty estimation.

Each "posterior sample" is one teacher-forced forward pass through the model
with dropout active (model.train() mode) but no gradient computation.  A
different RNG seed is set before every pass so each pass uses an independent
dropout mask.  Aggregating across passes gives sample-based estimates of
predictive entropy and expected entropy, from which epistemic MI is derived
by the orchestration layer in scorer.py.

Memory note: the full vocabulary distribution is retained for each token
across all samples before aggregation.  For large models (vocab ~50 k) and
long summaries, memory usage scales as:

    sample_count × total_summary_tokens × vocab_size × 8 bytes

Consider reducing sample_count if you run into memory pressure.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .scorer import (
    PreparedSummary,
    RuleBasedSentenceBackend,
    SampledSentenceDistributions,
    SummaryUncertaintyScorer,
)


_DEFAULT_MODEL = "facebook/bart-large-xsum"


class MCDropoutPosteriorSampler:
    """Posterior sampler that drives diversity through independent dropout masks.

    The sampled value is the integer seed used to initialise the PyTorch RNG
    before each forward pass.  Different seeds → different dropout masks →
    different token distributions → estimated posterior disagreement.
    """

    def sample(self, seed: int | None = None) -> int | None:
        """Return the seed as the posterior sample identifier."""

        return seed


class MCDropoutBackend(RuleBasedSentenceBackend):
    """Teacher-forced scoring backend using MC Dropout for epistemic uncertainty.

    The model is kept in train() mode throughout so that dropout layers remain
    active.  Gradients are disabled at forward-pass time to avoid the memory
    and compute cost of the gradient tape.

    Args:
        model_name: Any HuggingFace seq2seq model identifier.
        device: Torch device string.  Auto-detected when None.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
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
        logger.info("Loading tokenizer and model %r on device %r", model_name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.to(self._device)
        # train() activates dropout; gradients are still disabled per forward pass
        self._model.train()
        logger.info("Model ready (%d parameters)", sum(p.numel() for p in self._model.parameters()))
        logger.info("Running warm-up forward pass")
        with torch.no_grad():
            _dummy = torch.tensor([[self._model.config.decoder_start_token_id]], device=self._device)
            self._model(input_ids=_dummy, decoder_input_ids=_dummy)
        logger.info("Warm-up complete")

    def prepare_summary(
        self,
        source: str,
        summary: str,
        sentences: Sequence[str] | None = None,
    ) -> PreparedSummary:
        """Tokenize source and summary; map sentence specs to decoder token ranges.

        Character offsets from the NLTK sentence splitter are used to align
        each sentence to the subset of summary BPE tokens it covers.  The
        aligned token ranges are stored in metadata so that score_posterior_sample
        can slice the logit tensor per sentence without re-tokenizing.
        """

        logger.info(
            "Preparing summary: %d source chars, %d summary chars",
            len(source),
            len(summary),
        )
        prepared = super().prepare_summary(source, summary, sentences)
        logger.info("%d sentence(s) identified", len(prepared.sentences))

        max_length = self._tokenizer.model_max_length

        # Encode the source text for the encoder stack.
        logger.info("Tokenizing source (max_length=%d)", max_length)
        encoder_encoding = self._tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoder_input_ids = encoder_encoding["input_ids"].to(self._device)
        encoder_attention_mask = encoder_encoding["attention_mask"].to(self._device)
        logger.info("Source tokenized: %d tokens", encoder_input_ids.shape[1])

        # Encode the full summary without special tokens so that
        # return_offsets_mapping gives character spans relative to summary.
        logger.info("Tokenizing summary")
        summary_encoding = self._tokenizer(
            summary,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        summary_token_ids_tensor = summary_encoding["input_ids"]  # (1, num_tokens)
        offset_mapping: list[tuple[int, int]] = (
            summary_encoding["offset_mapping"][0].tolist()
        )
        summary_token_ids: list[int] = summary_token_ids_tensor.squeeze(0).tolist()
        logger.info("Summary tokenized: %d tokens", len(summary_token_ids))

        # Map each sentence's character span to a contiguous slice of token indices.
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
                    "to any summary tokens.  The summary may have been truncated by "
                    "the tokenizer.  Try a shorter source or summary."
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

        # Construct decoder input: [decoder_start_token] + summary_tokens.
        # logits[:, i, :] predicts summary token at 0-based index i.
        decoder_start = torch.tensor(
            [[self._model.config.decoder_start_token_id]],
            device=self._device,
        )
        decoder_input_ids = torch.cat(
            [decoder_start, summary_token_ids_tensor.to(self._device)],
            dim=1,
        )
        logger.info("Decoder input ready: %d tokens (including BOS)", decoder_input_ids.shape[1])

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
        posterior_sample: int | None,
    ) -> Sequence[SampledSentenceDistributions]:
        """Run one teacher-forced forward pass with a fresh dropout mask.

        The posterior_sample value is used as the PyTorch RNG seed so that
        each sample index produces a deterministic but independent dropout mask.
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

        logger.debug("Forward pass: sample seed=%s", posterior_sample)
        if posterior_sample is not None:
            torch.manual_seed(posterior_sample)
            torch.cuda.manual_seed_all(posterior_sample)

        with torch.no_grad():
            outputs = self._model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
            )

        # logits: (1, decoder_seq_len, vocab_size)
        # logits[0, i, :] is the distribution over summary token at index i.
        logits = outputs.logits.squeeze(0)  # (decoder_seq_len, vocab_size)
        probs = F.softmax(logits, dim=-1).cpu().float().numpy()  # (seq_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1).cpu().float().numpy()  # (seq_len, vocab_size)

        sentence_distributions: list[SampledSentenceDistributions] = []
        for sentence_spec in prepared_summary.sentences:
            tok_start, tok_end = sentence_token_slices[sentence_spec.sentence_index]
            target_ids = np.array(
                summary_token_ids[tok_start:tok_end],
                dtype=np.int64,
            )
            token_probs = probs[tok_start:tok_end, :]  # (num_sentence_tokens, vocab_size)
            token_log_probs = log_probs[tok_start:tok_end, :]
            sentence_distributions.append(
                SampledSentenceDistributions(
                    sentence_index=sentence_spec.sentence_index,
                    target_token_ids=target_ids,
                    token_probabilities=token_probs,
                    token_log_probabilities=token_log_probs,
                )
            )

        return sentence_distributions


def build_mc_dropout_scorer(
    model_name: str = _DEFAULT_MODEL,
    device: str | None = None,
) -> SummaryUncertaintyScorer:
    """Create an MC Dropout scorer backed by the given HuggingFace model.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            ``facebook/bart-large-cnn``.  Any seq2seq model works; smaller
            models (e.g. ``sshleifer/distilbart-cnn-12-6``) are faster.
        device: Torch device string.  Detected automatically when None.
    """

    backend = MCDropoutBackend(model_name=model_name, device=device)
    sampler = MCDropoutPosteriorSampler()
    return SummaryUncertaintyScorer(backend=backend, posterior_sampler=sampler)
