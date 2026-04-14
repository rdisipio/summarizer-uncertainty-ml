#!/usr/bin/env python3
"""
finetune_lora.py

Fine-tune a seq2seq model (default: facebook/bart-large-xsum) with LoRA
adapters on (source, summary) pairs from a JSONL file produced by
summarize_and_store.py.  Only the LoRA parameters are updated; base model
weights are frozen.  The resulting PEFT adapter checkpoint is what
lora_laplace_backend.py loads for uncertainty estimation.

Usage:
    python scripts/finetune_lora.py \
        --infile data/summaries_v3.jsonl \
        --outdir models/bart-large-xsum-lora \
        --model facebook/bart-large-xsum \
        --epochs 3 \
        --batch-size 4 \
        --grad-accum 4 \
        --lr 3e-4

The adapter checkpoint (adapter_config.json + adapter_model.safetensors) is
written to --outdir and can be loaded with:

    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM
    base = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")
    model = PeftModel.from_pretrained(base, "<outdir>")

Requirements (all in Pipfile):
    peft, transformers, torch, datasets
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def read_jsonl(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    records = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_pairs(path: str) -> list[dict[str, str]]:
    """Return a list of {'source': ..., 'target': ...} dicts from the JSONL."""
    raw = read_jsonl(path)
    pairs = []
    for record in raw:
        source = record.get("paragraph_text", "").strip()
        target = record.get("summary", "").strip()
        if source and target:
            pairs.append({"source": source, "target": target})
        else:
            logger.warning("Skipping record with missing source or target: %r", record.get("id"))
    logger.info("Loaded %d (source, summary) pairs from %r", len(pairs), path)
    return pairs


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def make_preprocess_fn(tokenizer, max_source_length: int, max_target_length: int):
    """Return a batched tokenisation function for use with Dataset.map."""

    def preprocess(batch):
        model_inputs = tokenizer(
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
        # Replace padding token id in labels with -100 so the loss ignores them.
        label_ids = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    return preprocess


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------


def build_lora_model(
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
) -> tuple:
    """Load base model, apply LoRA config, return (peft_model, tokenizer)."""
    logger.info("Loading base model %r", model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    model, tokenizer = build_lora_model(
        model_name=args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    pairs = load_pairs(args.infile)
    if not pairs:
        logger.error("No valid (source, summary) pairs found. Aborting.")
        sys.exit(1)

    # Train / validation split
    random.shuffle(pairs)
    val_size = max(1, int(len(pairs) * args.val_split))
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    logger.info("Split: %d train, %d validation", len(train_pairs), len(val_pairs))

    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)

    preprocess = make_preprocess_fn(
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["source", "target"],
        desc="Tokenising train set",
    )
    val_dataset = val_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["source", "target"],
        desc="Tokenising validation set",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=False,   # we need logits, not decoded text
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=0,
        seed=args.seed,
        logging_steps=10,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting training")
    trainer.train()

    # Save only the LoRA adapter weights (not the full model).
    logger.info("Saving LoRA adapter to %r", args.outdir)
    model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fine-tune a seq2seq model with LoRA on (source, summary) pairs."
    )
    p.add_argument(
        "--infile",
        required=True,
        help="Input JSONL from summarize_and_store.py (gz ok)",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Directory to write the PEFT adapter checkpoint",
    )
    p.add_argument(
        "--model",
        default="facebook/bart-large-xsum",
        help="HuggingFace base model identifier (default: facebook/bart-large-xsum)",
    )
    # LoRA hyperparameters
    p.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank r — controls capacity of the adapter (default: 8)",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA scaling alpha — effective LR scale = alpha/r (default: 16)",
    )
    p.add_argument(
        "--lora-dropout",
        type=float,
        default=0.1,
        help="Dropout applied inside LoRA layers (default: 0.1)",
    )
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj",
        help=(
            "Comma-separated attention module names to adapt with LoRA "
            "(default: q_proj,v_proj)"
        ),
    )
    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps — effective batch = batch-size × grad-accum",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data held out for validation (default: 0.1)",
    )
    p.add_argument(
        "--max-source-length",
        type=int,
        default=512,
        help="Maximum encoder input tokens (default: 512)",
    )
    p.add_argument(
        "--max-target-length",
        type=int,
        default=128,
        help="Maximum decoder target tokens (default: 128)",
    )
    p.add_argument("--fp16", action="store_true", help="Enable fp16 mixed precision (CUDA)")
    p.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision (CUDA/MPS)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args)
