from __future__ import annotations

"""Utility helpers for the **teacher_v2** pipeline.

All heavy imports (`torch`, `transformers`) are local to the functions that
need them so that simply importing this module in a CPU‑only environment does
not explode.
"""

from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple
import json
import gzip

from . import config


# --------------------------------------------------------------------------- #
# 1. Tokenizer (shared with tokenizer_v2)                                      #
# --------------------------------------------------------------------------- #

def load_tokenizer():
    """Load the *student/teacher‑shared* tokenizer built in tokenizer_v2."""
    from tokenizer_v2.utils import load_tokenizer as _load

    return _load(config.TOKENIZER_JSON)


# --------------------------------------------------------------------------- #
# 2. Teacher model loader                                                      #
# --------------------------------------------------------------------------- #

def load_teacher(device: str | None = None):
    """Load T5‑Large (or fine‑tuned ckpt) with optional 8‑bit quantisation.

    Parameters
    ----------
    device : str | None
        "cpu", "cuda", "cuda:0", etc.  If None, falls back to config.DEVICE.
    """
    device = device or config.DEVICE

    from transformers import AutoModelForSeq2SeqLM

    model_name = (
        str(config.FINETUNED_CKPT) if config.FINETUNED_CKPT and config.FINETUNED_CKPT.exists()
        else config.TEACHER_MODEL_ID
    )

    kwargs = {
        "device_map": "auto" if device.startswith("cuda") else None,
    }

    if config.USE_BNB_INT8:
        try:
            import bitsandbytes as _  # noqa: F401  (only to ensure pkg present)
            kwargs.update({
                "load_in_8bit": True,
                "bnb_8bit_compute_dtype": "float16" if config.FP16 else "float32",
            })
        except ModuleNotFoundError:
            raise RuntimeError("bitsandbytes not installed but USE_BNB_INT8=True in config.")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)

    # FP16 autocast (weights are already fp16 when load_in_8bit False & FP16 True)
    if config.FP16 and not config.USE_BNB_INT8:
        model = model.half()  # type: ignore[call-arg]

    model.eval()
    return model


# --------------------------------------------------------------------------- #
# 3. Small iterator helpers                                                    #
# --------------------------------------------------------------------------- #

def batched(iterable: Iterable[str], n: int) -> Iterator[List[str]]:
    """Yield *n*-sized lists from *iterable* (last batch may be smaller)."""
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------------------------------------------------------- #
# 4. Soft‑probabilities extraction                                             #
# --------------------------------------------------------------------------- #

def extract_topk(logits, k: int, temperature: float) -> List[Tuple[int, float]]:  # noqa: D401
    """Return *k* (token_id, prob) pairs from a 1‑D tensor of logits.

    Converts to probabilities with temperature‐scaled softmax **on CPU** and
    casts to *float16* for storage efficiency.
    """
    import torch

    logits = logits.float() / temperature
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k)
    return [(int(i), float(v.cpu().half())) for i, v in zip(idx, vals)]


# --------------------------------------------------------------------------- #
# 5. Streaming writers                                                         #
# --------------------------------------------------------------------------- #

def _open_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def write_jsonl_line(fh, obj) -> None:  # fh is an open text file handle
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_soft_targets(all_topk: Sequence[Sequence[Tuple[int, float]]], path: Path, compress: bool = False):
    """Persist the list‐of‐lists structure to JSON (optionally gzip)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress or path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(all_topk, fh, ensure_ascii=False)
    else:
        path.write_text(json.dumps(all_topk, ensure_ascii=False), encoding="utf-8")
