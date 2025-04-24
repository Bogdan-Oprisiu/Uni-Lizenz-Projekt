#!/usr/bin/env python
"""
Fine‑tune a (seq2seq) teacher model that turns natural‑language
robot commands into JSON.

You can just run:

    python fine_tune_teacher.py

…or tweak things:

    python fine_tune_teacher.py --variant basic        # use the 1‑parameter dataset
    python fine_tune_teacher.py --epochs 5             # more training
    python fine_tune_teacher.py --text_file my.txt ... # fully custom paths
"""
# ---------------------------------------------------------------------------

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# ------------------------------- constants ----------------------------------

DATA_ROOT = Path("../training_data")

DEFAULTS = {
    "basic": {
        "text": DATA_ROOT / "basic_data" / "synthetic_labeled_robot_commands_with_accel.txt",
        "json": DATA_ROOT / "basic_data" / "synthetic_labeled_robot_commands_with_accel_json.txt",
    },
    "multi": {
        "text": DATA_ROOT / "multiple_parameter_data" / "synthetic_labeled_robot_commands_with_accel.txt",
        "json": DATA_ROOT / "multiple_parameter_data" / "synthetic_labeled_robot_commands_with_accel_json.txt",
    },
}


# ----------------------------- args & config --------------------------------


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finetune teacher LLM for robot‑command translation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--variant", choices=["basic", "multi"], default="multi",
                   help="Which built‑in dataset to use when you *don’t* pass "
                        "`--text_file/--json_file`.")
    p.add_argument("--text_file", type=str,
                   help="Natural‑language command file (paired, line‑aligned).")
    p.add_argument("--json_file", type=str,
                   help="JSON‑label file (paired, line‑aligned).")
    p.add_argument("--model_name", default="google/t5-v1_1-small",
                   help="🤗 model name or local checkpoint")
    p.add_argument("--output_dir", default="./finetuned_teacher",
                   help="Where to save checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


args = get_args()
set_seed(args.seed)

# fall back to variant defaults iff user didn’t set explicit paths
text_path = Path(args.text_file) if args.text_file else DEFAULTS[args.variant]["text"]
json_path = Path(args.json_file) if args.json_file else DEFAULTS[args.variant]["json"]

print(f"📂  Using dataset variant: {args.variant}")
print(f"    text : {text_path}")
print(f"    json : {json_path}\n")


# ------------------------- dataset construction -----------------------------


def iter_paired(text_file: Path, json_file: Path):
    """Yield dicts for perfectly line‑aligned text / JSON files."""
    if not text_file.exists():
        raise FileNotFoundError(text_file)
    if not json_file.exists():
        raise FileNotFoundError(json_file)

    with text_file.open(encoding="utf-8") as ft, json_file.open(encoding="utf-8") as fj:
        for line_no, (txt, js) in enumerate(zip(ft, fj), start=1):
            txt, js = txt.strip(), js.strip()
            if not txt or not js:
                continue  # keep alignment but skip blanks
            try:
                parsed = json.loads(js)  # early validation
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no} of {json_file}: {e}") from None
            yield {"text": txt, "label": json.dumps(parsed, separators=(",", ":"))}


features = Features({"text": Value("string"), "label": Value("string")})

print("📚  Building HuggingFace‑Datasets dataset …")
dataset = Dataset.from_generator(
    lambda: iter_paired(text_path, json_path),
    features=features,
    cache_dir=".cache",
)
print(f"✔  Loaded {len(dataset):,} examples\n")

# --------------------------- preprocessing ----------------------------------

tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)


def preprocess(examples):
    model_in = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    with tokenizer.as_target_tokenizer():
        lab = tokenizer(
            examples["label"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    model_in["labels"] = lab["input_ids"]
    return model_in


print("🔄  Tokenising …")
tokenised_ds = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

collator = DataCollatorForSeq2Seq(tokenizer, model=None, padding="longest")

# ------------------------------ model ---------------------------------------

model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

# --------------------------- training config --------------------------------

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.lr,
    logging_steps=100,
    save_strategy="epoch",
    seed=args.seed,
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenised_ds,
    tokenizer=tokenizer,
    data_collator=collator,
)

# -------------------------------- train -------------------------------------

print("🚂  Training …\n")
trainer.train()

print("\n💾  Saving final model …")
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print(f"\n✅  Done – model & tokenizer saved to: {args.output_dir}\n")
