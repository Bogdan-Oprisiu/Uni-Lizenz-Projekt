from __future__ import annotations

"""Central settings for the *teacher* pipeline (T5‑Large distillation).

All paths are computed relative to this file so the project can be cloned
anywhere and still work out of the box.
"""

from pathlib import Path
from typing import List

# --------------------------------------------------------------------------- #
# 0. Filesystem layout                                                         #
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BASE_DIR.parent
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artefacts"

# Tokenizer trained by `tokenizer_v2/train.py` --------------------------------
TOKENIZER_JSON: Path = ARTIFACTS_DIR / "tokenizer_v2" / "bpe_tokenizer_v2.json"

# Directory where this teacher run will drop its outputs ----------------------
OUTPUT_DIR: Path = ARTIFACTS_DIR / "teacher_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOFT_TARGETS_FILE: Path = OUTPUT_DIR / "soft_targets_top5.json"  # large – consider .gz
PRED_JSONL_FILE: Path = OUTPUT_DIR / "teacher_predictions.jsonl"  # one JSON command per line

# --------------------------------------------------------------------------- #
# 1. Teacher model                                                             #
# --------------------------------------------------------------------------- #
TEACHER_MODEL_ID: str = "google/t5-large"  # HF hub id or local path
# If you fine‑tune T5‑Large via teacher_v2/fine_tune_teacher.py, set this to the ckpt dir
FINETUNED_CKPT: Path | None = ARTIFACTS_DIR / "teacher_ft"  # may not exist yet

# Loading options -------------------------------------------------------------
DEVICE: str = "cuda"  # "cpu", "cuda", or e.g. "cuda:0"
USE_BNB_INT8: bool = True  # load with bitsandbytes 8‑bit quantisation
FP16: bool = True  # cast weights/inputs to fp16 (amp)

# --------------------------------------------------------------------------- #
# 2. Inference hyper‑parameters                                                #
# --------------------------------------------------------------------------- #
BATCH_SIZE: int = 8  # adjust to your GPU vram
TOP_K: int = 5  # number of probabilities to save per token
TEMPERATURE: float = 2.0  # soften logits for distillation
MAX_SEQ_LEN: int = 128  # truncation/padding length

# --------------------------------------------------------------------------- #
# 3. Input corpora                                                             #
# --------------------------------------------------------------------------- #
TRAINING_DATA_DIR: Path = PROJECT_ROOT / "training_data"
BASIC_DATA_DIR: Path = TRAINING_DATA_DIR / "basic_data"
MULTI_PARAM_DATA_DIR: Path = TRAINING_DATA_DIR / "multiple_parameter_data"

# Command‑only (no JSON) files – same ones you fed to the tokenizer -----------
INPUT_CORPORA: List[Path] = [
    BASIC_DATA_DIR / "synthetic_basic_unlabeled_robot_commands.txt",
    MULTI_PARAM_DATA_DIR / "synthetic_unlabeled_robot_commands_with_accel.txt",
]


# --------------------------------------------------------------------------- #
# 4. Sanity helpers                                                            #
# --------------------------------------------------------------------------- #

def assert_paths() -> None:
    """Fail fast if mandatory resources are missing."""
    missing = [p for p in [TOKENIZER_JSON, *INPUT_CORPORA] if not p.exists()]
    if missing:
        joined = "\n  • " + "\n  • ".join(str(m) for m in missing)
        raise FileNotFoundError(f"Required file(s) not found:{joined}")


# Auto‑run the check at import time (comment out for unit tests)
assert_paths()
