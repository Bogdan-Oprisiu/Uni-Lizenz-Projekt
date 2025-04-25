# ----------------------------- teacher_v3/config_v3.py (CORRECTED PATHS) -----------------------------
"""Central settings for the LoRA‑based teacher pipeline (v3).
This patched version relaxes the *assert_config* sanity check so missing
resources emit **warnings** instead of hard‑stopping with an exception. That
lets you iterate even while some artefacts (e.g. training files) are still
being generated or copied over.

Key changes
-----------
1. Added **STRICT_CHECK = False** flag. When *False* (default) the sanity
   checker just prints what is missing and returns, keeping the process
   alive. If you need the old behaviour simply set the flag to *True* or
   call ``assert_config(strict=True)``.
2. Re‑ordered *assert_config* signature – the first positional parameter is
   now *strict* followed by *is_training* (both keyword‑only for clarity).
3. Minor docstrings / log tweaks – no functional impact beyond readability.
4. **MODIFIED**: Updated `assert_config` to check for the specific labeled
   training files provided by the user.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# 0. Global behaviour flags
# ---------------------------------------------------------------------------
STRICT_CHECK: bool = False

# ---------------------------------------------------------------------------
# 1. Filesystem layout
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent  # …/teacher_v3
PROJECT_ROOT: Path = BASE_DIR.parent  # project root (e.g. LLM/)
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artefacts"  # → LLM/artefacts

# Tokenizer artefact (can be over‑ridden via env var TOKENIZER_PATH)
TOKENIZER_PATH: Path = Path(
    os.getenv("TOKENIZER_PATH", ARTIFACTS_DIR / "tokenizer_v2" / "bpe_tokenizer_v2.json")
)

# ---------------------------------------------------------------------------
# 2. Teacher Model (Base + Adapter)
# ---------------------------------------------------------------------------
BASE_MODEL_ID: str = "google/t5-v1_1-large"
LORA_ADAPTER_DIR: Path = BASE_DIR / "lora_teacher_adapter"

DEVICE: str = "cuda"
USE_BNB_INT8_BASE: bool = False
MODEL_DTYPE: Optional[str] = "float16"  # "float16", "bfloat16", None

# ---------------------------------------------------------------------------
# 3. LoRA fine‑tune hyper‑parameters
# ---------------------------------------------------------------------------
LORA_R: int = 8
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.1
LORA_TARGET_MODULES: List[str] = ["q", "v"]

# ---------------------------------------------------------------------------
# 4. Training hyper‑parameters
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
# 5. Inference hyper‑parameters
# ---------------------------------------------------------------------------
PER_DEVICE_EVAL_BATCH_SIZE: int = 16
TOP_K: int = 5
TEMPERATURE: float = 1.0
MAX_INPUT_LENGTH: int = 128
MAX_TARGET_LENGTH: int = 128

# ---------------------------------------------------------------------------
# 6. Dataset paths
# ---------------------------------------------------------------------------
TRAINING_DATA_DIR: Path = PROJECT_ROOT / "training_data"
# Note: UNLABELED_CORPORA is used for inference checks or other parts of the pipeline
UNLABELED_CORPORA: List[Path] = [
    TRAINING_DATA_DIR / "basic_data" / "synthetic_basic_unlabeled_robot_commands.txt",
    TRAINING_DATA_DIR / "multiple_parameter_data" / "synthetic_unlabeled_robot_commands_with_accel.txt",
]

# ---------------------------------------------------------------------------
# 7. Output artefacts
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = ARTIFACTS_DIR / "teacher_v3_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTED_MAPS_FILE: Path = OUTPUT_DIR / "teacher_v3_predicted_maps.jsonl"
SOFT_TARGETS_FILE: Path = OUTPUT_DIR / "teacher_v3_soft_targets_top5.json"


# ---------------------------------------------------------------------------
# 8. Sanity‑check helper
# ---------------------------------------------------------------------------

def _resolve(path: Path) -> Path:
    """Resolve *path* against PROJECT_ROOT unless already absolute."""
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


# <<< MODIFIED FUNCTION with CORRECTED PATHS >>>
def assert_config(*, strict: bool | None = None, is_training: bool = True) -> None:
    """Verifies that critical files & directories exist.

    Parameters
    ----------
    strict:
        If *True* raise ``FileNotFoundError`` for missing items – emulates the
        previous behaviour. If *False* merely *warns* and returns.  When set to
        ``None`` (default) the module‑level ``STRICT_CHECK`` value is used.
    is_training:
        When *True* (default) includes labelled dataset files in the check,
        otherwise only resources required for inference are inspected.
    """
    strict = STRICT_CHECK if strict is None else strict

    missing: List[Tuple[str, Path]] = []  # (logical name, resolved path)

    # --- always check: tokenizer
    to_check: List[Tuple[str, Path]] = [("TOKENIZER", TOKENIZER_PATH)]

    if is_training:
        # MODIFIED: Use the filenames provided by the user
        basic_text = TRAINING_DATA_DIR / "basic_data" / "synthetic_basic_labeled_commands.txt"
        basic_map = TRAINING_DATA_DIR / "basic_data" / "synthetic_basic_labeled_commands_map.jsonl"
        multi_text = TRAINING_DATA_DIR / "multiple_parameter_data" / "synthetic_accel_labeled_commands.txt"
        multi_map = TRAINING_DATA_DIR / "multiple_parameter_data" / "synthetic_accel_labeled_command_smap.jsonl"  # Using exact name provided

        to_check.extend([
            ("basic text", basic_text),
            ("basic map", basic_map),
            ("multi text", multi_text),
            ("multi map", multi_map),
        ])
    else:
        # Check unlabeled corpora and adapter for inference mode
        for i, p in enumerate(UNLABELED_CORPORA):
            to_check.append((f"unlabeled corpus [{i}]", p))
        # Check if adapter exists (can be optional depending on workflow)
        adapter_config_path = LORA_ADAPTER_DIR / "adapter_config.json"
        if adapter_config_path.exists():  # Only add check if adapter dir is present
            to_check.append(("LoRA adapter", adapter_config_path))
        elif LORA_ADAPTER_DIR.exists():
            print(f"      INFO: LoRA adapter dir exists ({LORA_ADAPTER_DIR}), but config is missing.")
        else:
            print(f"      INFO: LoRA adapter dir does not exist ({LORA_ADAPTER_DIR}).")

    print("\n--- Config sanity check (strict=" + str(strict) + ") ---")
    for logical, p in to_check:
        rp = _resolve(p)
        exists = rp.exists()
        status = "OK" if exists else "MISSING"
        print(f"{logical:>28}: {rp}  →  {status}")  # Adjusted padding
        if not exists:
            missing.append((logical, rp))

    if missing and strict:
        details = "\n".join(f" - {name}: {path}" for name, path in missing)
        raise FileNotFoundError(
            "Required resource(s) missing:\n" + details + "\nSet STRICT_CHECK=False to just warn."
        )

    if missing and not strict:
        # Modified warning message for clarity
        missing_names = ', '.join([f"'{name}'" for name, _ in missing])
        warnings.warn(
            f"Config check identified {len(missing)} missing item(s): {missing_names}. "
            f"Proceeding anyway because strict=False.",
            RuntimeWarning,
            stacklevel=2  # Point warning to caller
        )
    elif not missing:
        print("✅ All checked resources present.")
    # else: # Case: missing and not strict (already handled by warning)
    #    print(" Proceeding with missing resources (strict=False).")


# ---------------------------------------------------------------------------
# 9. Self‑test entry‑point (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: python config_v3.py --train/--infer
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", dest='is_training', action="store_true", default=True,
                    help="Check training-mode resources (default)")
    ap.add_argument("--infer", dest='is_training', action="store_false", help="Check inference-mode resources")
    ap.add_argument("--strict", action="store_true", help="Enable strict checking (override flag)")
    ns = ap.parse_args()
    try:
        # Pass is_training explicitly based on args
        assert_config(strict=ns.strict, is_training=ns.is_training)
        print("\nProcess finished with exit code 0")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        exit(1)
