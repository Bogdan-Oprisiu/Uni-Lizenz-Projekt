#!/usr/bin/env python3
"""Domain‑specific fine‑tune for **google/t5‑large**.

The goal is **not** to train a brand‑new model but to nudge the public
checkpoint towards robot‑command phrasing, so that its soft targets carry
more domain knowledge for distillation.

Typical run (single GPU, fp16):

```bash
python -m teacher_v2.fine_tune_teacher \
       --epochs 6 \
       --batch-size 8 \
       --lr 5e-5 \
       --output artefacts/teacher_ft
```

The defaults below are conservative enough for an RTX 4090 (24 GB) but still
finish in < 2 hours on ~150 k examples.
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
# 1  Arguments                                                                 #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Fine‑tune T5‑Large for robot commands")
    p.add_argument("--epochs", type=int, default=6,                      # <<< default 6 epochs
                   help="Training epochs (default: 6 – enough for convergence)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Per‑device train batch size (default: 8)")
    p.add_argument("--lr", type=float, default=5e-5, dest="learning_rate",
                   help="Adam learning‑rate (default: 5e‑5)")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW weight decay (default: 0.01)")
    p.add_argument("--warmup-ratio", type=float, default=0.06,
                   help="Warm‑up steps ratio (default: 6 %% of total) ")
    p.add_argument("--gradient-accumulation", type=int, default=4,
                   help="Update steps to accumulate before backward() (default: 4)")
    p.add_argument("--output", type=Path, default=config.ARTIFACTS_DIR / "teacher_ft",
                   help="Where to save the fine‑tuned checkpoint")
    p.add_argument("--fp16", action="store_true", help="Enable mixed‑precision training")
    return p.parse_args()

args = parse_args()

# --------------------------------------------------------------------------- #
# 2  Load corpus                                                               #
# --------------------------------------------------------------------------- #

LABELLED_FILES: List[Path] = [
    config.BASIC_DATA_DIR / "synthetic_basic_labeled_robot_commands_json.txt",
    config.MULTI_PARAM_DATA_DIR / "synthetic_labeled_robot_commands_with_accel_json.txt",
]

if not all(p.exists() for p in LABELLED_FILES):
    missing = [str(p) for p in LABELLED_FILES if not p.exists()]
    raise FileNotFoundError("Some labelled corpora are missing:\n  • " + "\n  • ".join(missing))


def _iter_labeled() -> Iterator[Dict[str, str]]:
    """Yield dicts of {input, output}. Supports either TAB or `|||` separator."""
    for fp in LABELLED_FILES:
        with fp.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                if "\t" in raw:
                    src, tgt = raw.split("\t", 1)
                else:
                    src, tgt = raw.split("|||", 1)
                yield {"input": src.strip(), "output": tgt.strip()}


# --------------------------------------------------------------------------- #
# 3  Tokenisation                                                              #
# --------------------------------------------------------------------------- #

tokenizer = utils.load_tokenizer()

MAX_LEN = config.MAX_SEQ_LEN


def _tokenise(examples):
    model_inputs = tokenizer(examples["input"], max_length=MAX_LEN, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], max_length=MAX_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


ds = Dataset.from_generator(_iter_labeled)
proc_ds = ds.map(_tokenise, batched=True, remove_columns=["input", "output"], num_proc=4)

# --------------------------------------------------------------------------- #
# 4  Model & Trainer                                                           #
# --------------------------------------------------------------------------- #

print(f"[finetune] Loading base model {config.TEACHER_MODEL_ID}…")
model = AutoModelForSeq2SeqLM.from_pretrained(
    config.TEACHER_MODEL_ID,
    torch_dtype=(torch.float16 if args.fp16 else None),
)

collator = DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=-100)

total_steps = (len(proc_ds) // (args.batch_size * args.gradient_accumulation)) * args.epochs
warmup_steps = int(total_steps * args.warmup_ratio)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(args.output),
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    fp16=args.fp16,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=proc_ds,
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()

print("[finetune] ✓ saved fine‑tuned checkpoint to", args.output)
