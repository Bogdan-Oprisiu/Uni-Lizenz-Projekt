# ----------------------------- teacher_v2/config.py -----------------------------
"""Central settings for the *teacher* pipeline (T5 – fine‑tuned)"""
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# 0. Filesystem layout
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent  # …/teacher_v2
PROJECT_ROOT: Path = BASE_DIR.parent  # project root
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artefacts"

# location of the *shared* tokenizer (used when no fine‑tuned checkpoint)
TOKENIZER_JSON: Path = ARTIFACTS_DIR / "tokenizer_v2" / "bpe_tokenizer_v2.json"

# ---------------------------------------------------------------------------
# 1. Teacher model
# ---------------------------------------------------------------------------
# a) path to the **fine‑tuned** checkpoint (produced by fine_tune_teacher.py)
#    This path is checked first and preferred when it exists.
FINETUNED_CKPT: Optional[Path] = BASE_DIR / "finetuned_teacher"

# b) fallback model id on the Hugging Face Hub (when no fine‑tuned ckpt found)
TEACHER_MODEL_ID: str = "t5-large"  # public hub id

# Loading options -------------------------------------------------------------
DEVICE: str = "cuda"  # "cpu", "cuda", or e.g. "cuda:0"
USE_BNB_INT8: bool = False  # disable if bitsandbytes is unavailable
FP16: bool = True  # automatic mixed‑precision when possible

# ---------------------------------------------------------------------------
# 2. Inference hyper‑parameters
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 8
TOP_K: int = 5
TEMPERATURE: float = 2.0
MAX_SEQ_LEN: int = 128

# ---------------------------------------------------------------------------
# 3. Input corpora
# ---------------------------------------------------------------------------
TRAINING_DATA_DIR: Path = PROJECT_ROOT / "training_data"
BASIC_DATA_DIR: Path = TRAINING_DATA_DIR / "basic_data"
MULTI_PARAM_DATA_DIR: Path = TRAINING_DATA_DIR / "multiple_parameter_data"

INPUT_CORPORA: List[Path] = [
    BASIC_DATA_DIR / "synthetic_basic_unlabeled_robot_commands.txt",
    MULTI_PARAM_DATA_DIR / "synthetic_unlabeled_robot_commands_with_accel.txt",
]

# ---------------------------------------------------------------------------
# 4. Output artefacts
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = ARTIFACTS_DIR / "teacher_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOFT_TARGETS_FILE: Path = OUTPUT_DIR / "soft_targets_top5.json"
PRED_JSONL_FILE: Path = OUTPUT_DIR / "teacher_predictions.jsonl"


# ---------------------------------------------------------------------------
# 5. Sanity helpers
# ---------------------------------------------------------------------------

def assert_paths() -> None:
    """Fail fast if mandatory resources are missing."""
    missing = []

    # the tokenizer is mandatory **only** if we do *not* have a fine‑tuned ckpt
    if (not FINETUNED_CKPT or not FINETUNED_CKPT.exists()) and not TOKENIZER_JSON.exists():
        missing.append(TOKENIZER_JSON)

    # the input corpora are always required
    missing.extend(p for p in INPUT_CORPORA if not p.exists())

    if missing:
        joined = "\n  • " + "\n  • ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Required file(s) not found:{joined}")


# run on import (comment out for unit tests)
assert_paths()