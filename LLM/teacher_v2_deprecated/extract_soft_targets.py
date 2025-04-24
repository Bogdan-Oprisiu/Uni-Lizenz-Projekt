#!/usr/bin/env python
"""
extract_soft_targets.py – run the *teacher* model over the unlabeled
command corpora and save top‑k soft‑probs for every generated token.

Outputs:
    artefacts/teacher_outputs/soft_targets_top5.json[.gz]
    artefacts/teacher_outputs/teacher_predictions.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from tqdm import tqdm

# ── ensure local imports work even when not installed as a package ────────────
MODULE_DIR = Path(__file__).resolve().parent  # …/teacher_v2
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import config, utils  # noqa: E402


# ─────────────────────────── CLI overrides ────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--top_k", type=int, default=config.TOP_K)
    p.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    p.add_argument("--max_len", type=int, default=config.MAX_SEQ_LEN)
    p.add_argument("--compress", action="store_true",
                   help="Save *.json.gz instead of plain *.json")
    return p.parse_args()


args = parse_args()
config.assert_paths()


# ───────────────────── data helper ────────────────────────────────────────────
def load_all_commands() -> List[str]:
    """Return a deduplicated command list from all corpora."""
    cmds, seen = [], set()
    for path in config.INPUT_CORPORA:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                cmd = line.strip()
                if cmd and cmd not in seen:
                    seen.add(cmd)
                    cmds.append(cmd)
    return cmds


# ───────────────────── main logic ─────────────────────────────────────────────
def main() -> None:
    cmds = load_all_commands()
    print(f"📄  Loaded {len(cmds):,} distinct commands\n")

    print("🔧  Loading tokenizer …")
    tokenizer = utils.load_tokenizer()

    print("🧠  Loading teacher model …")
    model = utils.load_teacher()

    soft_batches: List[Sequence[Sequence[Tuple[int, float]]]] = []
    pred_fh = utils.open_append(config.PRED_JSONL_FILE)

    print("🚀  Running inference …\n")
    with torch.no_grad():
        for batch in tqdm(utils.batched(cmds, args.batch_size),
                          total=(len(cmds) + args.batch_size - 1) // args.batch_size):
            # tokenise
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=args.max_len)
            # T5 does not use segment embeddings
            inputs.pop("token_type_ids", None)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # generate with score traces
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_len,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

            seqs, scores = outputs.sequences, outputs.scores
            for i, seq in enumerate(seqs):
                per_tok = [utils.extract_topk(sc[i], args.top_k, args.temperature)
                           for sc in scores]
                soft_batches.append(per_tok)
                utils.write_predictions(pred_fh,
                                        tokenizer.decode(seq, skip_special_tokens=True))

    pred_fh.close()

    out_path = config.SOFT_TARGETS_FILE.with_suffix(
        config.SOFT_TARGETS_FILE.suffix + (".gz" if args.compress else ""))
    print("💾  Saving soft targets …")
    utils.save_soft_targets(soft_batches, out_path, compress=args.compress)
    print("✅  Done!  🎉")


if __name__ == "__main__":
    main()
