from dataclasses import dataclass, field
from pathlib import Path
from typing import List

"""Central configuration for tokenizer training & related utilities.

All paths are resolved relative to this file, so the project can be cloned
anywhere without breaking hard‑coded locations.
"""

# —— Filesystem layout ———————————————————————————————————————————
BASE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BASE_DIR.parent
TRAINING_DATA_DIR: Path = PROJECT_ROOT / "training_data"
TOKENIZER_DIR: Path = TRAINING_DATA_DIR / "tokenizer"

# — Raw corpora —
BASIC_DATA_DIR: Path = TRAINING_DATA_DIR / "basic_data"
MULTI_PARAM_DATA_DIR: Path = TRAINING_DATA_DIR / "multiple_parameter_data"

DICTIONARY_FILE: Path = TOKENIZER_DIR / "google-10000-english-no-swears.txt"
UNITS_NUMBERS_FILE: Path = TOKENIZER_DIR / "units_and_numbers.txt"

BASIC_CORPORA: List[Path] = [
    BASIC_DATA_DIR / "synthetic_basic_labeled_robot_commands_json.txt",
    BASIC_DATA_DIR / "synthetic_basic_unlabeled_robot_commands.txt",
]

MULTI_PARAM_CORPORA: List[Path] = [
    MULTI_PARAM_DATA_DIR / "synthetic_labeled_robot_commands_with_accel_json.txt",
    MULTI_PARAM_DATA_DIR / "synthetic_unlabeled_robot_commands_with_accel.txt",
]

# Combine for convenience; the training script can still pick subsets.
ALL_COMMAND_CORPORA: List[Path] = BASIC_CORPORA + MULTI_PARAM_CORPORA

# —— Tokenizer hyper‑parameters ———————————————————————————————
VOCAB_SIZE: int = 12_000  # target size after special tokens
MIN_FREQUENCY: int = 3  # min token frequency for merges
REPLICATE_DICT_FACTOR: int = 2  # how many times to repeat dictionary words

# Byte‑Pair Encoding model tweaks
UNK_TOKEN: str = "<unk>"
PAD_TOKEN: str = "<pad>"
BOS_TOKEN: str = "<s>"
EOS_TOKEN: str = "</s>"
SEP_TOKEN: str = "<sep>"

SPECIAL_TOKENS: List[str] = [
                                PAD_TOKEN,
                                UNK_TOKEN,
                                BOS_TOKEN,
                                EOS_TOKEN,
                                SEP_TOKEN,
                                # JSON punctuation
                                "{", "}", "[", "]", ":", ",", ", ", ": ", "]}",
                                # Domain keys (bare)</s>
                                "commandLanguage", "errors", "commands", "parameters",
                                "name", "description", "distance", "acceleration", "angle", "direction",
                                "INVALID_COMMAND", "MISSING_PARAMETER", "INVALID_PARAMETER_TYPE",
                                "OUT_OF_RANGE", "SYNTAX_ERROR", "UNSUPPORTED_UNIT",
                                # Common units & magnitudes
                                "cm", "deg", "rad", "cm/s^2", "deg/s^2",
                            ] + [str(n) for n in range(10, 361, 10)]  # 10,20,…,360

# ——Output artefacts———————————————————————————————————————————
OUTPUT_DIR: Path = PROJECT_ROOT / "artefacts" / "tokenizer_v2"
TOKENIZER_JSON: Path = OUTPUT_DIR / "bpe_tokenizer_v2.json"
TOKENIZER_CONFIG: Path = OUTPUT_DIR / "tokenizer_config_v2.json"

# Ensure directories exist when a module is imported (safe‑guard for scripts).
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ——Utility dataclass (optional) ————————————————————————————
@dataclass
class TrainingConfig:
    """Snapshot of the hyper‑parameters so they can be pickled / logged."""

    vocab_size: int = VOCAB_SIZE
    min_frequency: int = MIN_FREQUENCY
    replicate_dict_factor: int = REPLICATE_DICT_FACTOR
    special_tokens: List[str] = field(default_factory=lambda: SPECIAL_TOKENS.copy())
    continuing_subword_prefix: str = "▁"  # SentencePiece style

    def dump(self, destination: Path) -> None:
        import json
        destination.write_text(json.dumps(self.__dict__, indent=2))
