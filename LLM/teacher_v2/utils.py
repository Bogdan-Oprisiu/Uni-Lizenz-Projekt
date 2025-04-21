# ----------------------------- teacher_v2/utils.py -----------------------------
"""Utility helpers used throughout *teacher_v2* (tokenizer/model I/O, etc.)"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

from . import config


# ---------------------------------------------------------------------------
# 1. Tokenizer loader
# ---------------------------------------------------------------------------

def load_tokenizer():
    """Return tokenizer from fine‑tuned checkpoint (slow) or shared BPE (fast)."""
    from transformers import AutoTokenizer
    if config.FINETUNED_CKPT and config.FINETUNED_CKPT.exists():
        try:
            return AutoTokenizer.from_pretrained(
                str(config.FINETUNED_CKPT), use_fast=True)
        except Exception:
            # fast format may be incompatible; fall back to slow tokenizer
            return AutoTokenizer.from_pretrained(
                str(config.FINETUNED_CKPT), use_fast=False)

    # no fine‑tuned tokenizer: load shared BPE
    from tokenizer_v2.utils import load_tokenizer as _load
    return _load(config.TOKENIZER_JSON)


# ---------------------------------------------------------------------------
# 2. Teacher model loader
# ---------------------------------------------------------------------------

def load_teacher(device: str | None = None):
    """Load the fine‑tuned teacher (preferred) or the base model."""
    from transformers import AutoModelForSeq2SeqLM

    device = device or config.DEVICE

    model_name = (
        str(config.FINETUNED_CKPT)
        if config.FINETUNED_CKPT and config.FINETUNED_CKPT.exists()
        else config.TEACHER_MODEL_ID
    )

    kwargs = {"device_map": "auto" if device.startswith("cuda") else None}

    if config.USE_BNB_INT8:
        try:
            import bitsandbytes as _  # noqa: F401
            kwargs.update({
                "load_in_8bit": True,
                "bnb_8bit_compute_dtype": "float16" if config.FP16 else "float32",
            })
        except ModuleNotFoundError:
            print("⚠️  bitsandbytes not available – falling back to full precision.")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)

    if config.FP16 and not config.USE_BNB_INT8:
        model = model.half()  # type: ignore

    model.eval()
    return model


# ---------------------------------------------------------------------------
# 3. Simple batching helper
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
# 4. Top‑k extraction helper
# ---------------------------------------------------------------------------

def extract_topk(logits, k: int, temperature: float) -> List[Tuple[int, float]]:
    import torch

    logits = logits.float() / temperature
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k)
    return [(int(i), float(v.cpu().half())) for i, v in zip(idx, vals)]


# ---------------------------------------------------------------------------
# 5. Simple I/O convenience wrappers
# ---------------------------------------------------------------------------

def _open_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def write_jsonl_line(fh, obj):
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_soft_targets(all_topk: Sequence[Sequence[Tuple[int, float]]], path: Path, compress: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress or path.suffix.endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(all_topk, fh, ensure_ascii=False)
    else:
        path.write_text(json.dumps(all_topk, ensure_ascii=False), encoding="utf-8")
