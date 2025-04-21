from __future__ import annotations

"""Utility helpers shared by tokenizer training / inference code.

All functions are deliberately lightweight to avoid pulling heavy deps
outside the actual training script.
"""

import json
from pathlib import Path
from typing import Iterable, List, Iterator

from tokenizers import Tokenizer

from . import config

# --------------------------------------------------------------------------- #
# 1. Filesystem helpers                                                        #
# --------------------------------------------------------------------------- #

def ensure_exists(paths: Iterable[Path]) -> None:
    """Raise FileNotFoundError if any of *paths* is missing."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        joined = "\n  • ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Required file(s) not found:\n  • {joined}")


def read_text_lines(path: Path) -> List[str]:
    """Read a UTF‑8 text file and strip CR/LF."""
    return path.read_text(encoding="utf-8").splitlines()


# --------------------------------------------------------------------------- #
# 2. Corpus iterators                                                          #
# --------------------------------------------------------------------------- #

def iterate_corpora(files: Iterable[Path]) -> Iterator[str]:
    """Yield each line from a list of text files (utf‑8).

    Empty lines are skipped; whitespace is preserved because the tokenizer's
    pre‑tokeniser may rely on prefix spaces.
    """
    for fp in files:
        for line in read_text_lines(fp):
            if line:
                yield line


def replicate_iterator(it: Iterable[str], times: int) -> Iterator[str]:
    """Repeat the whole iterator *times* times."""
    for _ in range(times):
        for item in it:
            yield item


def build_training_stream() -> Iterator[str]:
    """Create a generator that yields all training lines with weighting.

    * Dictionary words are repeated *REPLICATE_DICT_FACTOR* times.
    * Units & numbers file is included once (if present).
    * Command corpora are streamed once each (labeled + unlabeled).
    """
    ensure_exists(config.ALL_COMMAND_CORPORA)

    # 1) dictionary (weighted)
    if config.DICTIONARY_FILE.exists():
        dict_iter = read_text_lines(config.DICTIONARY_FILE)
        yield from replicate_iterator(dict_iter, config.REPLICATE_DICT_FACTOR)

    # 2) units & numbers (if any)
    if config.UNITS_NUMBERS_FILE.exists():
        yield from read_text_lines(config.UNITS_NUMBERS_FILE)

    # 3) command corpora
    for line in iterate_corpora(config.ALL_COMMAND_CORPORA):
        yield line


# --------------------------------------------------------------------------- #
# 3. Tokenizer helpers                                                         #
# --------------------------------------------------------------------------- #

def load_tokenizer(tokenizer_path: Path | None = None) -> Tokenizer:
    """Load a *tokenizers* Tokenizer and register special tokens.

    If *tokenizer_path* is None, fall back to config.TOKENIZER_JSON.
    """
    path = tokenizer_path or config.TOKENIZER_JSON
    tok = Tokenizer.from_file(str(path))

    # Register pad/bos/eos ids so HuggingFace transformers can auto‑detect.
    tok.add_special_tokens([])  # ensure internal data structures exist

    model_kwargs = {
        "unk_token": config.UNK_TOKEN,
        "pad_token": config.PAD_TOKEN,
        "bos_token": config.BOS_TOKEN,
        "eos_token": config.EOS_TOKEN,
    }
    # 'tokenizers' core doesn't store these attr names; but we can store them
    # in the JSON config used by transformers save_pretrained later.
    tok.model.unk_token = config.UNK_TOKEN  # type: ignore[attr-defined]

    return tok


# --------------------------------------------------------------------------- #
# 4. JSON serialisation helpers                                                #
# --------------------------------------------------------------------------- #

def dump_json(obj, path: Path) -> None:
    """Pretty‑print *obj* to *path* (UTF‑8, 2‑space indent)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
