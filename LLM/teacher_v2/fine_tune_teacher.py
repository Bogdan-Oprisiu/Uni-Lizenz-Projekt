#!/usr/bin/env python3
"""Domain‑specific fine‑tune for T5‑Large on robot‑command data.
Supports external override of all hyperparameters, input/output paths, model ID, tokenizer, and device.
Malformed lines in the input files are skipped with a warning. Use your custom tokenizer loader from utils.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from teacher_v2 import config, utils


# --------------------------------------------------------------------------- #
# 1 Arguments                                                                 #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine‑tune T5‑Large for robot commands with flexible CLI options"
    )
    # Data paths
    p.add_argument(
        "--input-files",
        nargs="+",
        type=Path,
        default=[
            config.BASIC_DATA_DIR / "synthetic_basic_labeled_robot_commands_json.txt",
            config.MULTI_PARAM_DATA_DIR / "synthetic_labeled_robot_commands_with_accel_json.txt",
        ],
        help="Space‑separated list of labelled input files",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "teacher_ft",
        help="Directory to save the fine‑tuned checkpoint",
    )
    # Model/tokenizer
    p.add_argument(
        "--model-id",
        type=str,
        default=config.TEACHER_MODEL_ID,
        help="HuggingFace model identifier or local path",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=config.TOKENIZER_JSON,
        help="Path to your trained tokenizer JSON",
    )
    p.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help="Device to run training on (cpu, cuda, etc.)",
    )
    # Training hyperparams
    p.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=8, help="Per‑device batch size")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate for optimizer")
    p.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for optimizer"
    )
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.06,
        help="Warm‑up steps as fraction of total steps (0–1)",
    )
    p.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Steps to accumulate gradients before backward pass",
    )
    p.add_argument("--fp16", action="store_true", help="Use mixed‑precision (fp16) training")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 2 Load and validate corpus                                                  #
# --------------------------------------------------------------------------- #
def iter_labeled(files: List[Path]) -> Iterator[Dict[str, str]]:
    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(f"Labelled input file not found: {fp}")
        with fp.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    if "\t" in raw:
                        src, tgt = raw.split("\t", 1)
                    else:
                        src, tgt = raw.split("|||", 1)
                except ValueError:
                    # skip any line that doesn't split into two parts
                    print(f"[warning] skipping invalid line in {fp.name}: {raw}")
                    continue
                yield {"input": src.strip(), "output": tgt.strip()}


# --------------------------------------------------------------------------- #
# 3 Tokenization                                                              #
# --------------------------------------------------------------------------- #
def tokenize_dataset(
        dataset: Dataset, tokenizer_path: Path, max_len: int
) -> Dataset:
    tok = utils.load_tokenizer(tokenizer_path)

    def _map(batch):
        enc = tok(batch["input"], max_length=max_len, truncation=True)
        with tok.as_target_tokenizer():
            lbl = tok(batch["output"], max_length=max_len, truncation=True)
        enc["labels"] = lbl["input_ids"]
        return enc

    return dataset.map(
        _map,
        batched=True,
        remove_columns=["input", "output"],
        num_proc=4,
    )


# --------------------------------------------------------------------------- #
# 4 Main routine                                                             #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    # 1. Prepare output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # 2. Load and preprocess data
    raw_ds = Dataset.from_generator(lambda: iter_labeled(args.input_files))
    ds = tokenize_dataset(raw_ds, args.tokenizer, config.MAX_SEQ_LEN)

    # 3. Load model and move to device
    print(f"[finetune] Loading model {args.model_id} on {args.device}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.fp16 else None,
    )
    model.to(args.device)

    # 4. Prepare tokenizer & data collator
    tok = utils.load_tokenizer(args.tokenizer)
    collator = DataCollatorForSeq2Seq(tok, model, label_pad_token_id=-100)

    # 5. Compute scheduling
    total_steps = (len(ds) // (args.batch_size * args.gradient_accumulation)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=args.fp16,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    # 6. Initialize Trainer & train
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tok,
        data_collator=collator,
    )
    trainer.train()

    print(f"[finetune] ✓ Model saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
