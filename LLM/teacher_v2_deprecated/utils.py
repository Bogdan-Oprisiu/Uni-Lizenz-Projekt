"""utils.py – helper utilities for the teacher_v2 pipeline.
Ensures both teacher_v2 and tokenizer_v2 folders are on sys.path and provides
wrappers for tokenizer/model loading that work in Colab or normal Python.
"""
from __future__ import annotations

import sys, json, gzip
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

# ── make sure local folders are importable ────────────────────────────────
MODULE_DIR = Path(__file__).resolve().parent       # …/teacher_v2
ROOT_DIR   = MODULE_DIR.parent                     # …/LLM (also holds tokenizer_v2)
for p in (MODULE_DIR, ROOT_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import config  # noqa: E402  (local sibling)

# ---------------------------------------------------------------------------
# 1. Tokenizer loader
# ---------------------------------------------------------------------------

def load_tokenizer():
    """Return a *callable* HuggingFace tokenizer.

    • If a fine‑tuned checkpoint exists – load its tokenizer.
    • Else – load the shared tokenizer_v2 model **and wrap** the raw
      tokenizers.Tokenizer in a PreTrainedTokenizerFast so it’s usable with
      HF Trainer APIs (padding/truncation etc.).
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    # 1) prefer tokenizer saved in the finetuned ckpt dir
    if config.FINETUNED_CKPT and config.FINETUNED_CKPT.exists():
        try:
            return AutoTokenizer.from_pretrained(str(config.FINETUNED_CKPT), use_fast=True)
        except Exception:
            return AutoTokenizer.from_pretrained(str(config.FINETUNED_CKPT), use_fast=False)

    # 2) fallback: shared BPE tokenizer built by tokenizer_v2
    from tokenizer_v2.utils import load_tokenizer as _load
    raw_tok = _load(config.TOKENIZER_JSON)  # -> tokenizers.Tokenizer (not callable)

    # Wrap into a PreTrainedTokenizerFast so `.()` calls work
    return PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="[PAD]",
    )

# ---------------------------------------------------------------------------
# 2. Teacher model loader
# ---------------------------------------------------------------------------

def load_teacher(device: str | None = None):
    """Load fine‑tuned teacher (preferred) or hub model."""
    from transformers import AutoModelForSeq2SeqLM

    device     = device or config.DEVICE
    model_path = (str(config.FINETUNED_CKPT)
                  if config.FINETUNED_CKPT and config.FINETUNED_CKPT.exists()
                  else config.TEACHER_MODEL_ID)

    kwargs = {"device_map": "auto" if device.startswith("cuda") else None}
    if config.USE_BNB_INT8:
        try:
            import bitsandbytes  # noqa: F401
            kwargs.update({"load_in_8bit": True,
                           "bnb_8bit_compute_dtype": "float16" if config.FP16 else "float32"})
        except ModuleNotFoundError:
            print("⚠️ bitsandbytes not installed – loading full‑precision model.")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **kwargs)
    if config.FP16 and not config.USE_BNB_INT8:
        model = model.half()  # type: ignore
    model.eval()
    return model

# ---------------------------------------------------------------------------
# 3. Batching helper
# ---------------------------------------------------------------------------

def batched(iterable: Iterable[str], n: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

# ---------------------------------------------------------------------------
# 4. Soft‑probabilities helper
# ---------------------------------------------------------------------------

def extract_topk(logits, k: int, temperature: float) -> List[Tuple[int, float]]:
    import torch
    logits = logits.float() / temperature
    probs  = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k)
    return [(int(i), float(v.cpu().half())) for i, v in zip(idx, vals)]

# ---------------------------------------------------------------------------
# 5. Lightweight I/O helpers
# ---------------------------------------------------------------------------

def open_append(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")

def write_predictions(fh, text: str):
    fh.write(json.dumps(text, ensure_ascii=False) + "\n")

def save_soft_targets(all_topk: Sequence[Sequence[Tuple[int, float]]], path: Path, compress: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress or path.suffix.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(all_topk, fh, ensure_ascii=False)
    else:
        path.write_text(json.dumps(all_topk, ensure_ascii=False), encoding="utf-8")
