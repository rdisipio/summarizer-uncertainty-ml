#!/usr/bin/env python3
"""
finetune_lora.py

Fine-tune a seq2seq model with LoRA adapters on (source, summary) pairs from
a JSONL file produced by summarize_and_store.py.

Usage:
    python scripts/finetune_lora.py \\
        --infile data/summaries_v3.jsonl \\
        --outdir models/bart-large-xsum-lora \\
        --model facebook/bart-large-xsum \\
        --epochs 3 --batch-size 4 --grad-accum 4 --lr 3e-4 --bf16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.lora_training import build_lora_model, load_pairs, split_pairs, train_lora

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pairs = load_pairs(args.infile)
    if not pairs:
        logger.error("No valid pairs found in %r. Aborting.", args.infile)
        sys.exit(1)

    train_pairs, val_pairs = split_pairs(pairs, val_split=args.val_split, seed=args.seed)
    logger.info("Train: %d  Val: %d", len(train_pairs), len(val_pairs))

    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    model, tokenizer = build_lora_model(
        model_name=args.model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    train_lora(
        model=model,
        tokenizer=tokenizer,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        output_dir=args.outdir,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune a seq2seq model with LoRA.")
    p.add_argument("--infile", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", default="facebook/bart-large-xsum")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-target-modules", default="q_proj,v_proj")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-source-length", type=int, default=512)
    p.add_argument("--max-target-length", type=int, default=128)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
