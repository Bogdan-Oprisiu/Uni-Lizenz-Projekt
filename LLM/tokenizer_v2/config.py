from dataclasses import dataclass, field
from pathlib import Path
from typing import List

"""Central configuration for tokenizer training & related utilities.

Everything path‑related is resolved relative to this file, so cloning the
project elsewhere will not break import paths.
"""

# --------------------------------------------------------------------------- #
# 0. Filesystem layout                                                         #
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BASE_DIR.parent
TRAINING_DATA_DIR: Path = PROJECT_ROOT / "training_data"
TOKENIZER_DATA_DIR: Path = TRAINING_DATA_DIR / "tokenizer"

# Raw corpora -----------------------------------------------------------------
BASIC_DATA_DIR: Path = TRAINING_DATA_DIR / "basic_data"
MULTI_PARAM_DATA_DIR: Path = TRAINING_DATA_DIR / "multiple_parameter_data"

DICTIONARY_FILE: Path = TOKENIZER_DATA_DIR / "google-10000-english-no-swears.txt"
UNITS_NUMBERS_FILE: Path = TOKENIZER_DATA_DIR / "units_and_numbers.txt"

BASIC_CORPORA: List[Path] = [
    BASIC_DATA_DIR / "synthetic_basic_labeled_robot_commands_json.txt",
    BASIC_DATA_DIR / "synthetic_basic_unlabeled_robot_commands.txt",
]

MULTI_PARAM_CORPORA: List[Path] = [
    MULTI_PARAM_DATA_DIR / "synthetic_labeled_robot_commands_with_accel_json.txt",
    MULTI_PARAM_DATA_DIR / "synthetic_unlabeled_robot_commands_with_accel.txt",
]

# Convenience union
ALL_COMMAND_CORPORA: List[Path] = BASIC_CORPORA + MULTI_PARAM_CORPORA

# --------------------------------------------------------------------------- #
# 1. Tokenizer hyper‑parameters                                                #
# --------------------------------------------------------------------------- #
VOCAB_SIZE: int = 12_000          # target size *including* specials
MIN_FREQUENCY: int = 3            # BPE merge cut‑off
REPLICATE_DICT_FACTOR: int = 2    # weighting of the generic dictionary

# BPE model & Token conventions ----------------------------------------------
UNK_TOKEN: str = "<unk>"
PAD_TOKEN: str = "<pad>"
BOS_TOKEN: str = "<s>"
EOS_TOKEN: str = "</s>"
SEP_TOKEN: str = "<sep>"
CONTINUING_SUBWORD_PREFIX: str = "▁"   # SentencePiece style

# --------------------------------------------------------------------------- #
# 2. Special‑token list                                                        #
# --------------------------------------------------------------------------- #

# Static core specials
SPECIAL_TOKENS: List[str] = [
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN,
    # JSON punctuation (single‑character)
    "{", "}", "[", "]", ":", ",",
    # Frequent digrams that appear verbatim in JSON
    ", ", ": ", "]}",
    # Domain keys – **bare, without quotes** so the model does not have to copy quotes
    "commandLanguage", "errors", "commands", "parameters",
    "name", "description", "distance", "acceleration", "angle", "direction",
    "INVALID_COMMAND", "MISSING_PARAMETER", "INVALID_PARAMETER_TYPE",
    "OUT_OF_RANGE", "SYNTAX_ERROR", "UNSUPPORTED_UNIT",
]

# Units & numeric literals ----------------------------------------------------
# We extend SPECIAL_TOKENS with the contents of units_and_numbers.txt if it
# exists; otherwise, we fall back to a small baked‑in subset.
try:
    _extra = (Path(UNITS_NUMBERS_FILE).read_text(encoding="utf-8").splitlines())
except FileNotFoundError:
    _extra = [
        # minimal units fallback
        "cm", "deg", "rad", "mm", "m/s", "cm/s2", "deg/s2",
    ] + [str(n) for n in range(0, 361)]

SPECIAL_TOKENS.extend(_extra)

# Ensure uniqueness while preserving order (Python 3.7+ insertion order)
_seen = set()
SPECIAL_TOKENS = [tok for tok in SPECIAL_TOKENS if not (tok in _seen or _seen.add(tok))]

# --------------------------------------------------------------------------- #
# 3. Output artefacts                                                          #
# --------------------------------------------------------------------------- #
OUTPUT_DIR: Path = PROJECT_ROOT / "artefacts" / "tokenizer_v2"
TOKENIZER_JSON: Path = OUTPUT_DIR / "bpe_tokenizer_v2.json"
TOKENIZER_CONFIG: Path = OUTPUT_DIR / "tokenizer_config_v2.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 4. Helper dataclass for reproducibility                                      #
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    """Lightweight snapshot of the training recipe."""

    vocab_size: int = VOCAB_SIZE
    min_frequency: int = MIN_FREQUENCY
    replicate_dict_factor: int = REPLICATE_DICT_FACTOR
    special_tokens: List[str] = field(default_factory=lambda: SPECIAL_TOKENS.copy())
    continuing_subword_prefix: str = CONTINUING_SUBWORD_PREFIX

    def dump(self, destination: Path) -> None:
        import json
        destination.write_text(json.dumps(self.__dict__, indent=2))
