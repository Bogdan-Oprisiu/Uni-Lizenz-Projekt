# ----------------------------- teacher_v3/config_v3.py -----------------------------
"""Central settings for the LoRA-based teacher pipeline (v3)."""
from pathlib import Path
from typing import List, Optional, Dict  # Added Any for assert_config check

# ---------------------------------------------------------------------------
# 0. Filesystem layout
# ---------------------------------------------------------------------------
# Assume this file is in teacher_v3/
BASE_DIR: Path = Path(__file__).resolve().parent  # .../teacher_v3
PROJECT_ROOT: Path = BASE_DIR.parent  # project root (e.g., LLM/)
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artefacts"  # Correctly points to LLM/artefacts

# Location of the shared tokenizer. Based on image_59527e.png, tokenizer_v2 is inside ARTIFACTS_DIR.
TOKENIZER_PATH: Path = ARTIFACTS_DIR / "tokenizer_v2" / "bpe_tokenizer_v2.json"  # Correct path inside artefacts

# ---------------------------------------------------------------------------
# 1. Teacher Model (Base + Adapter)
# ---------------------------------------------------------------------------
# a) Base model identifier on Hugging Face Hub or local path
BASE_MODEL_ID: str = "google/t5-v1_1-large"  # Using t5-v1_1-large as specified

# b) Path where the trained LoRA adapter weights are saved/loaded from
LORA_ADAPTER_DIR: Path = BASE_DIR / "lora_teacher_adapter"

# c) Model loading options
DEVICE: str = "cuda"
USE_BNB_INT8_BASE: bool = False
MODEL_DTYPE: Optional[str] = "float16"

# ---------------------------------------------------------------------------
# 2. LoRA Fine-tuning Configuration
# ---------------------------------------------------------------------------
LORA_R: int = 8
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.1
LORA_TARGET_MODULES: List[str] = ["q", "v"]

# ---------------------------------------------------------------------------
# 3. Training Hyperparameters
# ---------------------------------------------------------------------------
NUM_TRAIN_EPOCHS: int = 3
PER_DEVICE_TRAIN_BATCH_SIZE: int = 8
LEARNING_RATE: float = 2e-4
WEIGHT_DECAY: float = 0.01
OPTIMIZER: str = "adamw_torch"
LR_SCHEDULER_TYPE: str = "linear"
WARMUP_RATIO: float = 0.05
LOGGING_STEPS: int = 100
SAVE_STRATEGY: str = "epoch"
SAVE_STEPS: int = 500
SEED: int = 42

# ---------------------------------------------------------------------------
# 4. Inference Hyperparameters
# ---------------------------------------------------------------------------
PER_DEVICE_EVAL_BATCH_SIZE: int = 16
TOP_K: int = 5
TEMPERATURE: float = 1.0
MAX_INPUT_LENGTH: int = 128
MAX_TARGET_LENGTH: int = 128

# ---------------------------------------------------------------------------
# 5. Input Data (for Fine-tuning)
# ---------------------------------------------------------------------------
TRAINING_DATA_DIR: Path = PROJECT_ROOT / "training_data"  # Points to LLM/training_data

LABELED_DATA_CONFIG: Dict[str, Dict[str, Path]] = {
    "default": {
        # These paths should be relative to PROJECT_ROOT or absolute
        "text": TRAINING_DATA_DIR / "multiple_parameter_data" / "synthetic_labeled_robot_commands_with_accel.txt",
        "map_json": TRAINING_DATA_DIR / "multiple_parameter_data" / "synthetic_labeled_robot_commands_with_accel_MAP.jsonl",
    }
}

# ---------------------------------------------------------------------------
# 6. Input Data (for Soft Target Extraction)
# ---------------------------------------------------------------------------
UNLABELED_CORPORA: List[Path] = [
    # These paths should be relative to PROJECT_ROOT or absolute
    TRAINING_DATA_DIR / "basic_data" / "synthetic_basic_unlabeled_robot_commands.txt",
    TRAINING_DATA_DIR / "multiple_parameter_data" / "synthetic_unlabeled_robot_commands_with_accel.txt",
]

# ---------------------------------------------------------------------------
# 7. Output Artifacts (from Soft Target Extraction)
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = ARTIFACTS_DIR / "teacher_v3_outputs"  # Inside LLM/artefacts
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTED_MAPS_FILE: Path = OUTPUT_DIR / "teacher_v3_predicted_maps.jsonl"
SOFT_TARGETS_FILE: Path = OUTPUT_DIR / "teacher_v3_soft_targets_top5.json"


# ---------------------------------------------------------------------------
# 8. Sanity Check Helper
# ---------------------------------------------------------------------------
def assert_config(is_training: bool = True) -> None:
    """Basic checks for mandatory files/settings."""
    missing_files_details: List[Tuple[Path, Path]] = []  # Store (original_path, resolved_path)
    files_to_check: List[Path] = []

    # --- Resolve paths relative to PROJECT_ROOT for checking ---
    def resolve_path(p: Path) -> Path:
        if p.is_absolute():
            return p.resolve()
        # If relative, assume it's relative to PROJECT_ROOT (adjust if needed)
        # Ensure PROJECT_ROOT itself is resolved correctly
        return (PROJECT_ROOT.resolve() / p).resolve()

    # Check tokenizer path defined in config
    files_to_check.append(TOKENIZER_PATH)

    if is_training:
        # Check labeled data for training
        for config_name, paths in LABELED_DATA_CONFIG.items():
            files_to_check.append(paths["text"])
            files_to_check.append(paths["map_json"])
    else:
        # Check unlabeled data for inference
        files_to_check.extend(UNLABELED_CORPORA)
        # Check if adapter exists for inference (only warn if missing)
        adapter_config_file = LORA_ADAPTER_DIR / "adapter_config.json"
        resolved_adapter_config = resolve_path(adapter_config_file)
        if not resolved_adapter_config.exists():
            print(f"INFO: LoRA adapter config not found at '{resolved_adapter_config}'.")
            print("      (This is expected if fine-tuning hasn't run yet).")
            # Decide if this should be a fatal error for inference
            # missing_files_details.append((adapter_config_file, resolved_adapter_config)) # Uncomment to make it fatal

    # Check existence of collected paths
    print("\n--- Checking Required Paths ---")
    print(f"Project Root Resolved To: {PROJECT_ROOT.resolve()}")  # Print resolved project root
    for f_path in files_to_check:
        resolved_f_path = resolve_path(f_path)
        status = "Found"
        if not resolved_f_path.exists():
            status = "MISSING"
            missing_files_details.append((f_path, resolved_f_path))  # Add original and resolved path
        print(f"Checking: '{f_path}' -> Resolved: '{resolved_f_path}' -> Status: {status}")

    # Specific check for map_json existence during training with warning
    # (This might be redundant now with the loop above, but keeps the explicit warning)
    if is_training:
        for config_name, paths in LABELED_DATA_CONFIG.items():
            map_json_path = paths["map_json"]
            resolved_map_json_path = resolve_path(map_json_path)
            if not resolved_map_json_path.exists():
                # Check if it's already in the missing list to avoid duplicate warnings
                if not any(mf[1] == resolved_map_json_path for mf in missing_files_details):
                    print(
                        f"\nWARNING: Expected map JSON label file '{map_json_path}' for training config '{config_name}' does not exist.")
                    print(f"         (Resolved path: {resolved_map_json_path})")
                    print(f"         Ensure your data generation script creates this file.")
                    # missing_files_details already contains it if check above failed

    if missing_files_details:
        print("\n--- Configuration Error: Missing Required Files ---")
        for original_path, resolved_path in missing_files_details:
            print(f"  - Required: '{original_path}'")
            print(f"    Resolved to: '{resolved_path}' (Not Found)")
        print("-----------------------------------------------------")
        raise FileNotFoundError(
            "One or more required files specified in config_v3.py were not found at their resolved locations.")

    print("-----------------------------")
    print("✅ Config check passed.")


# Example of how to run the check (e.g., at the start of scripts)
if __name__ == "__main__":
    print("Running config sanity check (assuming training context)...")
    try:
        assert_config(is_training=True)
    except FileNotFoundError as e:
        print(f"\nConfig check failed (training): {e}")
    except Exception as e:
        print(f"\nUnexpected error during config check (training): {e}")

    print("\nRunning config sanity check (assuming inference context)...")
    try:
        assert_config(is_training=False)
    except FileNotFoundError as e:
        print(f"\nConfig check failed (inference): {e}")
    except Exception as e:
        print(f"\nUnexpected error during config check (inference): {e}")
