#!/usr/bin/env python
# ----------------------------- teacher_v3/extract_soft_targets_v3.py -----------------------------
"""
Runs the LoRA-adapted teacher model over unlabeled command corpora to generate:
1. Predicted abstract map JSON strings.
2. Soft targets (top-k token probabilities) for distillation.

Example Usage:
    python teacher_v3/extract_soft_targets_v3.py
    python teacher_v3/extract_soft_targets_v3.py --batch_size 32
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from tqdm import tqdm

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

# Filter specific warnings if needed
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for inference."""
    p = argparse.ArgumentParser(
        description="Extract soft targets and predicted maps using the LoRA teacher model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--batch_size", type=int, default=config_v3.PER_DEVICE_EVAL_BATCH_SIZE,
        help="Batch size for inference."
    )
    p.add_argument(
        "--top_k", type=int, default=config_v3.TOP_K,
        help="Number of top probabilities to save per token."
    )
    p.add_argument(
        "--temperature", type=float, default=config_v3.TEMPERATURE,
        help="Softmax temperature for scaling logits before extracting probabilities."
    )
    p.add_argument(
        "--max_input_len", type=int, default=config_v3.MAX_INPUT_LENGTH,
        help="Maximum sequence length for input commands."
    )
    p.add_argument(
        "--max_target_len", type=int, default=config_v3.MAX_TARGET_LENGTH,
        help="Maximum sequence length for generated abstract map JSON strings."
    )
    p.add_argument(
        "--compress_output", action="store_true", default=False,  # Default to False
        help="Save the soft targets file with gzip compression (.json.gz)."
    )
    p.add_argument(
        "--device", type=str, default=config_v3.DEVICE,
        help="Device to run inference on (e.g., 'cuda', 'cpu', 'auto')."
    )
    # Arguments to specify model/adapter paths, overriding config if needed
    p.add_argument(
        "--adapter_dir", type=str, default=str(config_v3.LORA_ADAPTER_DIR),
        help="Path to the trained LoRA adapter directory."
    )
    p.add_argument(
        "--base_model_id", type=str, default=config_v3.BASE_MODEL_ID,
        help="Base model identifier used during LoRA training."
    )
    # Arguments to specify output file paths, overriding config if needed
    p.add_argument(
        "--output_maps_file", type=str, default=str(config_v3.PREDICTED_MAPS_FILE),
        help="Output file path for predicted abstract map JSONL."
    )
    p.add_argument(
        "--output_soft_targets_file", type=str, default=str(config_v3.SOFT_TARGETS_FILE),
        help="Output file path for soft targets JSON."
    )
    # Add argument for quantization of base model during inference
    p.add_argument(
        "--use_bnb_inf", action='store_true', default=config_v3.USE_BNB_INT8_BASE,
        help="Enable 8-bit quantization for the base model during inference."
    )
    p.add_argument(
        "--dtype_inf", type=str, default=config_v3.MODEL_DTYPE,
        help="Model dtype for base model during inference (e.g., float16, bfloat16, None)."
    )

    parsed_args = p.parse_args()

    # --- Update config from args for inference settings ---
    # This ensures utils_v3 uses the command-line specified values if provided
    config_v3.LORA_ADAPTER_DIR = Path(parsed_args.adapter_dir)
    config_v3.BASE_MODEL_ID = parsed_args.base_model_id
    config_v3.DEVICE = parsed_args.device
    config_v3.USE_BNB_INT8_BASE = parsed_args.use_bnb_inf  # Use inference-specific flag
    config_v3.MODEL_DTYPE = parsed_args.dtype_inf  # Use inference-specific flag

    return parsed_args


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_all_unlabeled_commands() -> List[str]:
    """Loads and deduplicates commands from all configured unlabeled corpora."""
    all_commands: List[str] = []
    seen_commands = set()
    print("📄 Loading unlabeled commands...")
    if not config_v3.UNLABELED_CORPORA:
        print("⚠️ WARNING: No unlabeled corpora specified in config_v3.UNLABELED_CORPORA.")
        return []

    for corpus_path in config_v3.UNLABELED_CORPORA:
        try:
            print(f"   - Reading from: {corpus_path.resolve()}")
            # Use utils_v3 to handle potential FileNotFoundError
            commands = utils_v3.load_text_file(corpus_path)
            new_count = 0
            for cmd in commands:
                cmd_strip = cmd.strip()  # Ensure stripped before adding
                if cmd_strip and cmd_strip not in seen_commands:
                    seen_commands.add(cmd_strip)
                    all_commands.append(cmd_strip)
                    new_count += 1
            print(f"     Added {new_count} new unique commands from this file.")
        except FileNotFoundError:
            print(f"⚠️ WARNING: Corpus file not found: {corpus_path.resolve()}. Skipping.")
        except Exception as e:
            print(f"⚠️ WARNING: Error reading {corpus_path.resolve()}: {e}. Skipping.")

    if not all_commands:
        print("❌ ERROR: No unlabeled commands were successfully loaded.")
        print("       Please check paths in config_v3.UNLABELED_CORPORA and file contents.")
        sys.exit(1)

    print(f"✅ Loaded {len(all_commands):,} total unique unlabeled commands.")
    return all_commands


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main() -> None:
    """Runs inference and saves predictions and soft targets."""
    args = parse_args()

    # Resolve output paths
    predicted_maps_output_path = Path(args.output_maps_file).resolve()
    soft_targets_output_path = Path(args.output_soft_targets_file).resolve()

    print("--- Starting Soft Target Extraction (v3) ---")
    print(f"Configuration:")
    print(f"  Base Model: {config_v3.BASE_MODEL_ID}")
    print(f"  Adapter Path: {config_v3.LORA_ADAPTER_DIR.resolve()}")
    print(f"  Use 8-bit Base (Inf): {config_v3.USE_BNB_INT8_BASE}")
    print(f"  Dtype (Inf): {config_v3.MODEL_DTYPE}")
    print(f"  Device: {config_v3.DEVICE}")
    print(f"  Batch Size: {args.batch_size}, Top K: {args.top_k}, Temp: {args.temperature}")
    print(f"  Output Maps: {predicted_maps_output_path}")
    print(f"  Output Soft Targets: {soft_targets_output_path}{'.gz' if args.compress_output else ''}")
    print("-" * 40)

    # --- Load Data ---
    commands_to_process = load_all_unlabeled_commands()

    # --- Load Tokenizer & Model ---
    print("\n🔄 Loading tokenizer...")
    # Use the utility function which handles loading priority
    tokenizer = utils_v3.load_tokenizer(use_fast=True)
    # Define task prefix - MUST match the one used during fine-tuning
    task_prefix = "translate English command to robot JSON map: "
    if task_prefix:
        print(f"   Using task prefix: '{task_prefix}'")

    print("\n🧠 Loading LoRA-adapted teacher model for inference...")
    # Use the utility function, ensuring for_training=False
    model = utils_v3.load_teacher_model(device=args.device, for_training=False)

    # --- Prepare for Inference ---
    all_soft_targets: List[Sequence[Tuple[int, float]]] = []
    predicted_maps_output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Running inference on {len(commands_to_process):,} commands...")
    # Use try-finally to ensure file is closed even on error
    pred_maps_fh = None
    invalid_json_count = 0
    try:
        # Open file for writing predicted maps
        pred_maps_fh = utils_v3.open_writable(predicted_maps_output_path,
                                              compress=False)  # JSONL usually not compressed

        with torch.no_grad():
            # Process commands in batches
            for batch_commands in tqdm(
                    utils_v3.batched(commands_to_process, args.batch_size),
                    total=(len(commands_to_process) + args.batch_size - 1) // args.batch_size,
                    desc="Generating Maps & Targets"
            ):
                # Prepare inputs
                inputs_with_prefix = [task_prefix + cmd for cmd in batch_commands]
                inputs = tokenizer(
                    inputs_with_prefix,
                    return_tensors="pt",
                    padding=True,  # Pad batch to longest sequence
                    truncation=True,
                    max_length=args.max_input_len
                )
                # Move inputs to the correct device
                try:
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                except Exception as e:
                    print(f"\n❌ ERROR: Failed to move batch to device '{model.device}'. Error: {e}")
                    print(f"       Model device: {model.device}")
                    print(f"       Input devices: { {k: v.device for k, v in inputs.items()} }")
                    continue  # Skip this batch

                # T5 doesn't use token_type_ids
                inputs.pop("token_type_ids", None)

                # Generate sequences and capture scores for soft targets
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_target_len,  # Max tokens to generate for the map
                        do_sample=False,  # Use greedy decoding for deterministic output
                        return_dict_in_generate=True,
                        output_scores=True,  # MUST be True to get logits for soft targets
                        # Add other generation params if needed (e.g., num_beams, temperature for sampling if do_sample=True)
                        # early_stopping=True, # Stop generation early if EOS token is produced
                        pad_token_id=tokenizer.pad_token_id,  # Ensure pad token is set
                        eos_token_id=tokenizer.eos_token_id  # Ensure EOS token is set
                    )
                except Exception as gen_e:
                    print(f"\n❌ ERROR during model.generate for batch: {gen_e}")
                    print(f"       Skipping batch: {batch_commands}")
                    continue  # Skip this batch if generation fails

                sequences = outputs.sequences
                scores = outputs.scores  # Tuple of logits for each generated token step

                # Process each sequence in the batch
                for i, seq_tensor in enumerate(sequences):
                    # Decode the generated sequence (predicted abstract map JSON string)
                    # Skip special tokens like <pad>, </s>
                    # Important: Ensure start token (like <pad> for T5) is handled correctly if present
                    predicted_map_str = tokenizer.decode(seq_tensor, skip_special_tokens=True)

                    # Write the predicted map string to the JSONL file
                    # Attempt to parse to ensure it's valid JSON before writing
                    try:
                        # Basic validation: try parsing the JSON
                        parsed_map = json.loads(predicted_map_str)
                        utils_v3.write_jsonl(pred_maps_fh, parsed_map)  # Write the parsed dict
                    except json.JSONDecodeError:
                        invalid_json_count += 1
                        print(
                            f"\n⚠️ WARNING: Generated output is not valid JSON (Count: {invalid_json_count}): {predicted_map_str}")
                        # Write raw string with error indication
                        utils_v3.write_jsonl(pred_maps_fh, {"error": "Invalid JSON", "raw_output": predicted_map_str})
                    except Exception as write_e:
                        print(f"\n❌ ERROR writing prediction to file: {write_e}")

                    # Extract top-k probabilities for soft targets
                    # scores is a tuple of tensors (one per timestep), shape: (batch_size, vocab_size)
                    num_generated_tokens = len(scores)
                    sequence_soft_targets = []
                    if num_generated_tokens > 0:
                        try:
                            # Get logits for the i-th sequence in the batch at each step
                            sequence_soft_targets = [
                                utils_v3.extract_topk(step_scores[i], args.top_k, args.temperature)
                                for step_scores in scores  # Iterate through scores tuple (timesteps)
                            ]
                        except IndexError:
                            print(
                                f"\n⚠️ WARNING: IndexError accessing scores for sequence {i} in batch. Skipping soft targets for this sequence.")
                            print(f"       Batch size: {len(batch_commands)}, scores length: {len(scores)}")
                            if scores: print(f"       Score tensor shape: {scores[0].shape}")
                            sequence_soft_targets = []  # Append empty list if error occurs
                        except Exception as soft_e:
                            print(f"\n❌ ERROR extracting soft targets for sequence {i}: {soft_e}")
                            sequence_soft_targets = []  # Append empty list on error
                    else:
                        # This might happen if generation immediately produces EOS or fails
                        print(
                            f"\n⚠️ WARNING: No scores generated for sequence {i}. Input: '{batch_commands[i]}'. Predicted: '{predicted_map_str}'")

                    all_soft_targets.append(sequence_soft_targets)

    except Exception as e:
        print(f"\n❌ ERROR: An unexpected error occurred during inference loop: {e}")
        import traceback
        traceback.print_exc()
        # Decide if you want to exit or try saving partial results
    finally:
        # Ensure the prediction file is closed
        if pred_maps_fh:
            pred_maps_fh.close()
            print(f"\n✅ Predicted maps saved to: {predicted_maps_output_path}")
            if invalid_json_count > 0:
                print(f"   ({invalid_json_count} predictions were not valid JSON)")

    # --- Save Soft Targets ---
    # Check if soft targets were actually generated
    if not all_soft_targets:
        print("\n❌ ERROR: No soft targets were generated. Cannot save soft targets file.")
    else:
        # Ensure the output path uses .gz if compression is enabled
        final_soft_target_path = soft_targets_output_path
        if args.compress_output and not final_soft_target_path.name.endswith(".gz"):
            final_soft_target_path = final_soft_target_path.with_name(final_soft_target_path.name + ".gz")
        elif not args.compress_output and final_soft_target_path.name.endswith(".gz"):
            final_soft_target_path = final_soft_target_path.with_suffix('')

        utils_v3.save_json(
            all_soft_targets,
            final_soft_target_path,  # Use the potentially adjusted path
            compress=args.compress_output
        )
        # Path printed within save_json

    print("\n🎉 Soft target extraction finished!")
    if invalid_json_count > 0:
        print(f"   Note: {invalid_json_count} generated outputs were not valid JSON.")


if __name__ == "__main__":
    main()
