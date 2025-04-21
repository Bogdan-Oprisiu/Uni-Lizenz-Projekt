#!/usr/bin/env python
"""
extract_soft_targets.py – run the *teacher* model over the unlabeled
command corpora and persist *top‑k* soft probabilities for every
generated token.

The resulting list‑of‑lists structure is stored to
``teacher_v2/artefacts/teacher_outputs/soft_targets_top5.json`` (or *.gz*)
as configured in ``teacher_v2/config.py``.  Optionally, the raw JSON
predictions are also saved – one per line – to the companion
``PRED_JSONL_FILE`` for inspection and debugging.

Typical usage
-------------
Simply run (defaults picked up from ``teacher_v2/config.py``)::

    python extract_soft_targets.py

You may override the *temperature*, *top‑k*, etc. via command‑line
flags – see ``--help`` for details.
"""

from __future__ import annotations

import argparse
from typing import List, Sequence, Tuple

import torch
from tqdm import tqdm

from teacher_v2 import config, utils


# ---------------------------------------------------------------------------
# 1. CLI – allow quick overrides without touching the config -----------------
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser(
        description="Extract top‑k soft targets from the teacher model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                   help="Batch size (overrides config.BATCH_SIZE)")
    p.add_argument("--top_k", type=int, default=config.TOP_K,
                   help="Number of probabilities to keep per token")
    p.add_argument("--temperature", type=float, default=config.TEMPERATURE,
                   help="Softmax temperature for distillation")
    p.add_argument("--max_len", type=int, default=config.MAX_SEQ_LEN,
                   help="Maximum length for generated sequences")
    p.add_argument("--compress", action="store_true",
                   help="Gzip‑compress the soft‑target file (overrides path suffix)")

    return p.parse_args()


args = parse_args()

# ---------------------------------------------------------------------------
# 2. Early sanity check – ensure all required artefacts exist ----------------
# ---------------------------------------------------------------------------
config.assert_paths()


# ---------------------------------------------------------------------------
# 3. Data helpers ------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_all_commands() -> List[str]:
    """Return **deduplicated** list of commands from all corpora."""
    commands: List[str] = []
    for path in config.INPUT_CORPORA:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                cmd = line.strip()
                if cmd:
                    commands.append(cmd)
    # keep order‑stable uniqueness
    seen = set()
    deduped = []
    for c in commands:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# 4. Main extraction loop ----------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901  (single entry point – fine)
    cmds = load_all_commands()
    print(f"📄  Loaded {len(cmds):,} distinct commands from corpora\n")

    print("🔧  Loading tokenizer …")
    tokenizer = utils.load_tokenizer()

    print("🧠  Loading teacher model …")
    model = utils.load_teacher()

    soft_target_batches: List[Sequence[Sequence[Tuple[int, float]]]] = []

    # Prepare output file handles -------------------------------------------
    pred_fh = utils._open_text(config.PRED_JSONL_FILE)

    # Model is in eval() mode already; no gradients required -----------------
    print("🚀  Running inference …\n")
    with torch.no_grad():
        for batch in tqdm(utils.batched(cmds, args.batch_size),
                          total=(len(cmds) + args.batch_size - 1) // args.batch_size):
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Greedy decode *with* score traces
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_len,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

            sequences = outputs.sequences  # (batch, seq_len)
            scores = outputs.scores  # Tuple[length] of (batch, vocab)

            for seq_idx, seq in enumerate(sequences):
                token_scores: List[Tuple[int, float]] = []
                per_token_topk: List[Tuple[int, float]] | List[List[Tuple[int, float]]] = []

                for step, score_t in enumerate(scores):  # one score tensor per generated token
                    logits = score_t[seq_idx]
                    per_token_topk.append(utils.extract_topk(logits, k=args.top_k, temperature=args.temperature))

                soft_target_batches.append(per_token_topk)  # type: ignore[arg-type]

                # Decode the sequence (teacher prediction) ------------------
                pred_text = tokenizer.decode(seq, skip_special_tokens=True)
                utils.write_jsonl_line(pred_fh, pred_text)

    pred_fh.close()

    print("💾  Saving soft targets …")
    utils.save_soft_targets(
        soft_target_batches,  # type: ignore[arg-type]
        config.SOFT_TARGETS_FILE.with_suffix(config.SOFT_TARGETS_FILE.suffix + (".gz" if args.compress else "")),
        compress=args.compress,
    )

    print("✅  Done!  🎉\n")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
