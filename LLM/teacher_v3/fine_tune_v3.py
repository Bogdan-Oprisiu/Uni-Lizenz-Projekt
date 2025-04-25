#!/usr/bin/env python
# ----------------------------- teacher_v3/fine_tune_v3.py (CORRECTED AGAIN) -----------------------------
"""
Fine-tunes a base sequence-to-sequence model (e.g., T5) using LoRA (PEFT)
to translate natural language robot commands into abstract map JSON strings.
Combines both basic and multi-parameter labeled datasets for training.
"""

import argparse  # <<< RESTORED IMPORT
import json
import sys
import warnings
from pathlib import Path
from typing import List  # <<< RESTORED Dict

# Keep top-level imports used directly in this script
import datasets
import torch  # <<< RESTORED IMPORT
import transformers  # <<< RESTORED IMPORT

# --- Make sure local modules are importable ---
MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path: sys.path.insert(0, str(MODULE_DIR))
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

try:
    import config_v3
    import utils_v3
except ImportError:
    print("ERROR: Could not import config_v3.py or utils_v3.py.")
    print("Ensure they are in the same directory or sys.path is configured correctly.")
    sys.exit(1)

# --- Library Imports ---
# Specific imports from datasets and transformers
from datasets import Dataset, Features, Value, concatenate_datasets  # <<< RESTORED Value
from transformers import (  # <<< RESTORED IMPORT BLOCK
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
# Specific imports from peft
from peft import (  # <<< RESTORED IMPORT BLOCK
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# Filter out specific UserWarnings from huggingface/datasets regarding caching
warnings.filterwarnings("ignore", category=UserWarning, message=".*Using custom data configuration.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Loading a dataset cached.*")


# ---------------------------------------------------------------------------
# Argument Parsing <<< RESTORED FUNCTION >>>
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments, using config_v3 for defaults."""
    p = argparse.ArgumentParser(
        description="Fine-tune teacher LLM using LoRA for robot command -> abstract map translation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model Arguments
    p.add_argument(
        "--base_model_id", type=str, default=config_v3.BASE_MODEL_ID,
        help="Base model identifier (Hugging Face Hub or local path)."
    )
    p.add_argument(
        "--adapter_output_dir", type=str, default=str(config_v3.LORA_ADAPTER_DIR),
        help="Directory to save the trained LoRA adapter weights."
    )
    # LoRA Arguments
    p.add_argument("--lora_r", type=int, default=config_v3.LORA_R, help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=config_v3.LORA_ALPHA, help="LoRA alpha.")
    p.add_argument("--lora_dropout", type=float, default=config_v3.LORA_DROPOUT, help="LoRA dropout.")
    p.add_argument(
        "--lora_target_modules", nargs='+', default=config_v3.LORA_TARGET_MODULES,
        help="Modules to target with LoRA (e.g., 'q' 'v')."
    )
    # Training Arguments
    p.add_argument("--epochs", type=int, default=config_v3.NUM_TRAIN_EPOCHS, help="Number of training epochs.")
    p.add_argument("--batch_size", type=int, default=config_v3.PER_DEVICE_TRAIN_BATCH_SIZE,
                   help="Batch size per device.")
    p.add_argument("--lr", type=float, default=config_v3.LEARNING_RATE, help="Learning rate.")
    p.add_argument("--weight_decay", type=float, default=config_v3.WEIGHT_DECAY, help="Weight decay.")
    p.add_argument("--optimizer", type=str, default=config_v3.OPTIMIZER, help="Optimizer type.")
    p.add_argument("--lr_scheduler_type", type=str, default=config_v3.LR_SCHEDULER_TYPE,
                   help="Learning rate scheduler type.")
    p.add_argument("--warmup_ratio", type=float, default=config_v3.WARMUP_RATIO, help="Warmup ratio for LR scheduler.")
    p.add_argument("--logging_steps", type=int, default=config_v3.LOGGING_STEPS, help="Log metrics every N steps.")
    p.add_argument("--save_strategy", type=str, default=config_v3.SAVE_STRATEGY,
                   help="Checkpoint saving strategy ('steps' or 'epoch').")
    p.add_argument("--save_steps", type=int, default=config_v3.SAVE_STEPS,
                   help="Save checkpoint every N steps (if save_strategy='steps').")
    p.add_argument("--seed", type=int, default=config_v3.SEED,
                   help="Random seed for reproducibility.")  # <<< --seed IS DEFINED HERE
    p.add_argument("--max_input_length", type=int, default=config_v3.MAX_INPUT_LENGTH,
                   help="Max sequence length for inputs.")
    p.add_argument("--max_target_length", type=int, default=config_v3.MAX_TARGET_LENGTH,
                   help="Max sequence length for labels (map JSON).")
    p.add_argument("--use_bnb", action='store_true', default=config_v3.USE_BNB_INT8_BASE,
                   help="Enable 8-bit quantization for the base model via bitsandbytes.")
    p.add_argument("--dtype", type=str, default=config_v3.MODEL_DTYPE,
                   help="Model dtype for base model (e.g., float16, bfloat16, None for float32). Overrides config.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Number of steps to accumulate gradients.")
    p.add_argument("--gradient_checkpointing", action='store_true', default=False,
                   help="Enable gradient checkpointing to save memory.")
    p.add_argument("--report_to", type=str, default="tensorboard",
                   help="Integration for reporting metrics (e.g., 'tensorboard', 'wandb', 'none').")
    p.add_argument("--eval_split_ratio", type=float, default=0.1, help="Ratio of data to use for evaluation split.")

    parsed_args = p.parse_args()

    # --- Update config from args ---
    # This section might be redundant if config_v3 is the main source, but we'll keep it
    # It ensures the args passed override config defaults for this run.
    config_v3.BASE_MODEL_ID = parsed_args.base_model_id
    config_v3.LORA_ADAPTER_DIR = Path(parsed_args.adapter_output_dir)
    config_v3.LORA_R = parsed_args.lora_r
    config_v3.LORA_ALPHA = parsed_args.lora_alpha
    config_v3.LORA_DROPOUT = parsed_args.lora_dropout
    config_v3.LORA_TARGET_MODULES = parsed_args.lora_target_modules
    config_v3.USE_BNB_INT8_BASE = parsed_args.use_bnb
    config_v3.MODEL_DTYPE = parsed_args.dtype
    config_v3.SEED = parsed_args.seed  # Assign seed to config as well

    return parsed_args


# ---------------------------------------------------------------------------
# Helper Function for Data Loading (keep as is - check Value import)
# ---------------------------------------------------------------------------
def load_and_prepare_data(text_path: Path, map_json_path: Path) -> Dataset:
    """Loads text and map_jsonl files into a Hugging Face Dataset."""
    resolved_text_path = text_path.resolve()
    resolved_map_path = map_json_path.resolve()
    print(f"   Attempting to load text from: {resolved_text_path}")
    print(f"   Attempting to load map JSONL from: {resolved_map_path}")

    texts = utils_v3.load_text_file(resolved_text_path)
    maps = utils_v3.load_jsonl_file(resolved_map_path)

    if len(texts) != len(maps):
        raise ValueError(
            f"Line count mismatch for dataset in {resolved_text_path.parent.name}: "
            f"{len(texts)} text lines in '{resolved_text_path.name}' vs "
            f"{len(maps)} map JSON lines in '{resolved_map_path.name}'."
        )

    paired_data = [{"text": txt, "label": json.dumps(m, separators=(",", ":"), ensure_ascii=False)}
                   for txt, m in zip(texts, maps)]

    # Ensure Value is imported correctly from datasets
    features = Features({"text": Value("string"), "label": Value("string")})
    return Dataset.from_list(paired_data, features=features)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    """Main function to execute the fine-tuning process."""
    args = parse_args()  # Now calls the local parse_args function
    set_seed(args.seed)  # Should now work as args.seed is defined
    output_dir = Path(args.adapter_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- Starting LoRA Fine-tuning (v3) ---")
    print(f"Configuration:")
    print(f"  Base Model: {args.base_model_id}")
    print(f"  Adapter Output Dir: {output_dir.resolve()}")
    print(f"  LoRA R: {args.lora_r}, Alpha: {args.lora_alpha}, Dropout: {args.lora_dropout}")
    print(f"  Target Modules: {args.lora_target_modules}")
    print(f"  Use 8-bit Base: {args.use_bnb}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print("-" * 36)

    # Using the correct filenames provided by the user.
    basic_text_path = config_v3.TRAINING_DATA_DIR / "basic_data" / "synthetic_basic_labeled_commands.txt"
    basic_map_path = config_v3.TRAINING_DATA_DIR / "basic_data" / "synthetic_basic_labeled_commands_map.jsonl"
    multi_text_path = config_v3.TRAINING_DATA_DIR / "multiple_parameter_data" / "synthetic_accel_labeled_commands.txt"
    multi_map_path = config_v3.TRAINING_DATA_DIR / "multiple_parameter_data" / "synthetic_accel_labeled_command_smap.jsonl"

    print("\n📚 Loading and preparing datasets...")
    all_datasets: List[Dataset] = []
    try:
        print(f"DEBUG: Attempting to load BASIC dataset pair:")
        print(f"       Text: {basic_text_path.resolve()}")
        print(f"       Map:  {basic_map_path.resolve()}")
        basic_dataset = load_and_prepare_data(basic_text_path, basic_map_path)
        all_datasets.append(basic_dataset)
        print(f"    -> Loaded {len(basic_dataset)} 'basic' examples.")

        print(f"DEBUG: Attempting to load MULTI dataset pair:")
        print(f"       Text: {multi_text_path.resolve()}")
        print(f"       Map:  {multi_map_path.resolve()}")
        multi_dataset = load_and_prepare_data(multi_text_path, multi_map_path)
        all_datasets.append(multi_dataset)
        print(f"    -> Loaded {len(multi_dataset)} 'multi-parameter' examples.")

        if not all_datasets:
            raise ValueError("No datasets were loaded. Check file paths and ensure MAP JSONL files exist.")
        combined_dataset = concatenate_datasets(all_datasets)
        print(f"✅ Combined dataset size: {len(combined_dataset):,} examples.")

        combined_dataset = combined_dataset.shuffle(seed=args.seed)

        print(f"   Splitting dataset (Test Ratio: {args.eval_split_ratio})...")
        if args.eval_split_ratio <= 0 or args.eval_split_ratio >= 1:
            print("   Evaluation split ratio is invalid. Using full dataset for training.")
            train_dataset = combined_dataset
            eval_dataset = None
            print(f"   Train size: {len(train_dataset)}")
        else:
            dataset_dict = combined_dataset.train_test_split(test_size=args.eval_split_ratio, seed=args.seed)
            train_dataset = dataset_dict["train"]
            eval_dataset = dataset_dict["test"]
            print(f"   Train size: {len(train_dataset)}")
            print(f"   Eval size: {len(eval_dataset)}")

    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not load required data file: {e}")
        print("       Please verify the file exists at the specified path with the exact name.")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ ERROR: Problem processing data files: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred during dataset loading: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n🔄 Loading tokenizer...")
    tokenizer = utils_v3.load_tokenizer(use_fast=True)
    task_prefix = "translate English command to robot JSON map: "
    if task_prefix:
        print(f"   Using task prefix: '{task_prefix}'")

    def preprocess_function(examples):
        """Tokenizes inputs and labels for the seq2seq model."""
        inputs = [task_prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_input_length,
            padding="max_length",
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["label"],
                max_length=args.max_target_length,
                padding="max_length",
                truncation=True,
            )
        processed_labels = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = processed_labels
        return model_inputs

    print("🔄 Tokenizing datasets...")
    tokenized_train_dataset = None
    tokenized_eval_dataset = None
    try:
        tokenized_train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing Training Set",
            num_proc=4
        )
        if eval_dataset:
            tokenized_eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing Evaluation Set",
                num_proc=4
            )
        print("✅ Tokenization complete.")

    except Exception as e:
        print(f"❌ ERROR: Failed during dataset mapping/tokenization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )
    print("   Data collator initialized.")

    print(f"\n🧠 Loading base model: {args.base_model_id} for training...")
    model = utils_v3.load_teacher_model(for_training=True)

    if args.use_bnb:
        print("🔧 Preparing model for k-bit training (quantization)...")
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
        print("   Model prepared for k-bit training.")
        if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("   Gradient checkpointing enabled for k-bit training.")

    print("🔧 Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    print("🔧 Applying LoRA adapter to the model...")
    model = get_peft_model(model, lora_config)
    print("✅ LoRA adapter applied.")
    model.print_trainable_parameters()

    print("\n⚙️ Configuring training arguments...")
    do_eval = tokenized_eval_dataset is not None
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optim=args.optimizer,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and args.dtype == "float16",
        bf16=torch.cuda.is_bf16_supported() and args.dtype == "bfloat16",
        gradient_checkpointing=args.gradient_checkpointing and not args.use_bnb,
        evaluation_strategy="epoch" if do_eval else "no",
        predict_with_generate=True if do_eval else False,
        generation_max_length=args.max_target_length,
        seed=args.seed,
        report_to=args.report_to.split(','),
        load_best_model_at_end=True if do_eval else False,
        metric_for_best_model="eval_loss" if do_eval else None,
        greater_is_better=False,
        push_to_hub=False,
    )

    print("🚀 Initializing Trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    transformers.logging.set_verbosity_warning()
    datasets.logging.set_verbosity_warning()

    print("\n🚂 Starting training...")
    try:
        train_result = trainer.train()
        print("✅ Training completed.")

        print("\n💾 Saving final LoRA adapter and training state...")
        trainer.save_model()
        trainer.save_state()
        metrics = train_result.metrics

        if do_eval and training_args.load_best_model_at_end:
            print("   Evaluating best model...")
            eval_metrics = trainer.evaluate()
            metrics.update(eval_metrics)
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        print(f"   Adapter saved to: {output_dir}")
        tokenizer.save_pretrained(str(output_dir))
        print(f"   Tokenizer saved to: {output_dir}")

    except Exception as e:
        print(f"❌ ERROR: An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        print("Attempting to save state on error...")
        try:
            trainer.save_state()
            model.save_pretrained(str(output_dir / "error_checkpoint"))
        except Exception as save_e:
            print(f"   Failed to save state on error: {save_e}")
        sys.exit(1)

    print("\n🎉 Fine-tuning finished successfully!")
    print(f"   Best LoRA adapter weights saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
