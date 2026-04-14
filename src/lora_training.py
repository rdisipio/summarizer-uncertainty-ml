"""Shared LoRA fine-tuning and quantile-fitting utilities.

Used by both the CLI scripts (scripts/finetune_lora.py,
scripts/compute_uncertainty_scores_lora_laplace.py, scripts/fit_quantiles.py)
and the Jupyter notebooks so all training logic lives in one place.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pairs(path: str, n_max: int | None = None) -> list[dict[str, str]]:
    """Load (source, summary) pairs from a JSONL file produced by summarize_and_store.

    Returns a list of {"source": ..., "target": ...} dicts with empty rows dropped.
    """
    from .data_pipeline import read_jsonl

    pairs: list[dict[str, str]] = []
    for record in read_jsonl(path):
        if n_max is not None and len(pairs) >= n_max:
            break
        source = record.get("paragraph_text", "").strip()
        target = record.get("summary", "").strip()
        if source and target:
            pairs.append({"source": source, "target": target})
        else:
            logger.warning("Skipping record with missing source or summary: %r", record.get("id"))
    logger.info("Loaded %d pairs from %r", len(pairs), path)
    return pairs


def split_pairs(
    pairs: list[dict],
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split pairs into (train, val) partitions."""
    pairs = list(pairs)
    random.seed(seed)
    random.shuffle(pairs)
    val_size = max(1, int(len(pairs) * val_split))
    return pairs[val_size:], pairs[:val_size]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def make_preprocess_fn(tokenizer, max_source_length: int, max_target_length: int):
    """Return a batched HuggingFace Dataset map function that tokenizes pairs.

    Labels have pad token ids replaced with -100 so they are ignored by the loss.
    """

    def preprocess(batch):
        inputs = tokenizer(
            batch["source"],
            max_length=max_source_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in labels["input_ids"]
        ]
        return inputs

    return preprocess


# ---------------------------------------------------------------------------
# LoRA model building
# ---------------------------------------------------------------------------


def build_lora_model(
    model_name: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
    layers_to_transform: list[int] | None = None,
    layers_pattern: str | None = None,
):
    """Load a seq2seq base model and wrap it with a LoRA adapter.

    Args:
        layers_to_transform: If set, only inject LoRA into these layer indices
            (0-based). E.g. ``[10, 11]`` for the last two decoder layers of a
            12-layer BART model.
        layers_pattern: Component of the module path that scopes the layer
            filter, e.g. ``"decoder"`` to restrict to decoder layers only.
            Combined with ``layers_to_transform`` to build a regex for
            ``target_modules`` (avoids PEFT version incompatibilities with
            the native ``layers_to_transform`` / ``layers_pattern`` API).

    Returns (peft_model, tokenizer).
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    if layers_to_transform is None:
        layers_to_transform = [10, 11]
    if layers_pattern is None:
        layers_pattern = "decoder"

    # Build a regex that selects only the requested layers within the
    # requested component (e.g. decoder layers 10 and 11).
    # BART module paths look like:
    #   model.decoder.layers.10.self_attn.q_proj
    #   model.decoder.layers.11.encoder_attn.v_proj
    layer_alts = "|".join(str(l) for l in layers_to_transform)
    module_alts = "|".join(re.escape(m) for m in target_modules)
    computed_target = rf".*{re.escape(layers_pattern)}\.layers\.({layer_alts})\.\w+\.({module_alts})$"
    logger.info("LoRA target regex: %s", computed_target)

    logger.info("Loading base model %r", model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=computed_target,
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_training_args(
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 3e-4,
    val_split: float = 0.1,
    fp16: bool = False,
    bf16: bool = False,
    seed: int = 42,
):
    """Build a Seq2SeqTrainingArguments object from common hyperparameters."""
    from transformers import Seq2SeqTrainingArguments

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=False,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=0,
        seed=seed,
        logging_steps=10,
        report_to="none",
    )


def train_lora(
    model,
    tokenizer,
    train_pairs: list[dict],
    val_pairs: list[dict],
    output_dir: str,
    max_source_length: int = 512,
    max_target_length: int = 128,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 3e-4,
    fp16: bool = False,
    bf16: bool = False,
    seed: int = 42,
) -> None:
    """Fine-tune *model* with LoRA and save the adapter to *output_dir*.

    Saves only the PEFT adapter weights (not the full base model).
    """
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

    preprocess = make_preprocess_fn(tokenizer, max_source_length, max_target_length)

    train_dataset = Dataset.from_list(train_pairs).map(
        preprocess, batched=True, remove_columns=["source", "target"],
        desc="Tokenising train set",
    )
    val_dataset = Dataset.from_list(val_pairs).map(
        preprocess, batched=True, remove_columns=["source", "target"],
        desc="Tokenising validation set",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        label_pad_token_id=-100, pad_to_multiple_of=8,
    )

    training_args = make_training_args(
        output_dir=output_dir, epochs=epochs, batch_size=batch_size,
        grad_accum=grad_accum, learning_rate=learning_rate,
        fp16=fp16, bf16=bf16, seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving LoRA adapter to %r", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete")


# ---------------------------------------------------------------------------
# Quantile normalizer fitting
# ---------------------------------------------------------------------------


def fit_quantiles(
    scores: list[float],
    quantile_points: list[float] | None = None,
) -> list[float]:
    """Compute quantile boundary values from a list of raw uncertainty scores.

    Returns a list of boundary values (one per quantile point) suitable for
    writing to a uncertainty_quantiles_*.json config file.
    """
    if not scores:
        raise ValueError("scores must not be empty.")
    if quantile_points is None:
        quantile_points = [0.0, 0.25, 0.5, 0.75, 1.0]

    arr = np.array(scores, dtype=np.float64)
    logger.info(
        "Score summary — min: %.6f  median: %.6f  max: %.6f  mean: %.6f  std: %.6f",
        arr.min(), float(np.median(arr)), arr.max(), arr.mean(), arr.std(),
    )
    boundaries = [float(np.quantile(arr, q)) for q in sorted(set(quantile_points))]
    if boundaries[0] == boundaries[-1]:
        raise ValueError(
            f"All quantile boundaries are equal ({boundaries[0]:.6f}). "
            "The score distribution has no spread."
        )
    return boundaries


def save_quantiles(boundaries: list[float], path: str) -> None:
    """Write quantile boundaries to a JSON config file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"boundaries": boundaries}, f, indent=2)
        f.write("\n")
    logger.info("Quantile boundaries written to %r", path)


def collect_scores_from_jsonl(path: str) -> list[float]:
    """Extract all sentence-level uncertainty scores from a scored JSONL file."""
    from .data_pipeline import read_jsonl

    scores: list[float] = []
    for record in read_jsonl(path):
        for sentence in record.get("sentence_scores", []):
            value = sentence.get("uncertainty")
            if value is not None:
                scores.append(float(value))
    return scores


# ---------------------------------------------------------------------------
# HuggingFace Hub upload
# ---------------------------------------------------------------------------


def upload_to_hub(
    files: list[tuple[str, str]],
    repo_id: str,
    repo_type: str = "model",
) -> None:
    """Upload a list of (local_path, path_in_repo) pairs to the HuggingFace Hub.

    Creates the repo if it does not already exist.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=False)
    logger.info("Hub repo ready: %s", repo_id)

    for local_path, remote_path in files:
        logger.info("Uploading %s -> %s", local_path, remote_path)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
    logger.info("Upload complete: https://huggingface.co/%s", repo_id)
