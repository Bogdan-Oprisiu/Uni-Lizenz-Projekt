#!/usr/bin/env python3
"""Train the v2 BPE tokenizer.

Can be invoked **either** as a module

    python -m tokenizer_v2.train

or directly

    python tokenizer_v2/train.py

The second style works because the script adds the projectroot to
`sys.path` if it detects that it is being executed outside a package
context (when `__package__` is `None`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, List

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors

# --------------------------------------------------------------------------- #
# 0. Handle import context (module vs. script)                                 #
# --------------------------------------------------------------------------- #

if __package__ is None:
    # Script executed directly: append project root so that
    # `import tokenizer_v2` works regardless of cwd.
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

from tokenizer_v2 import config, utils  # noqa: E402  (after path fix)

# --------------------------------------------------------------------------- #
# 1. Custom normaliser                                                         #
# --------------------------------------------------------------------------- #

try:
    # Optional user‑supplied normaliser factory
    from tokenizer_v2.custom_normalizer import get_normalizer  # type: ignore
except ModuleNotFoundError:  # fallback to thesis version
    def get_normalizer() -> normalizers.Normalizer:  # type: ignore
        return normalizers.Sequence([
            normalizers.Lowercase(),
            normalizers.Replace(pattern=r"[^\x00-\x7F]+", content=""),
        ])


# --------------------------------------------------------------------------- #
# 2. CLI                                                                       #
# --------------------------------------------------------------------------- #

def parse_args(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Train BPE tokenizer v2")
    p.add_argument("--out", type=Path, default=config.TOKENIZER_JSON,
                   help="Destination tokenizer JSON (default: artefacts dir)")
    p.add_argument("--vocab", type=int, default=config.VOCAB_SIZE,
                   help="Target vocabulary size (specials included)")
    p.add_argument("--min-freq", type=int, default=config.MIN_FREQUENCY,
                   help="Minimum token frequency included in vocab")
    p.add_argument("--prefix", default=config.CONTINUING_SUBWORD_PREFIX,
                   help="Continuing‑subword prefix (default: ▁)")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm progress bar even if available")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# 3. Helper: progress wrapper                                                  #
# --------------------------------------------------------------------------- #

def _with_progress(it: Iterator[str], length: int | None, disable: bool):
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(it, total=length, unit="lines", disable=disable)
    except ModuleNotFoundError:
        # tqdm not installed: silently continue without bar
        return it


# --------------------------------------------------------------------------- #
# 4. Training routine                                                          #
# --------------------------------------------------------------------------- #

def train_tokenizer(args) -> None:
    # Build training data stream ------------------------------------------------
    stream = list(utils.build_training_stream())  # materialise for length stats
    iterator = iter(stream)
    length = len(stream)

    # Initialise BPE model ------------------------------------------------------
    model = models.BPE(
        unk_token=config.UNK_TOKEN,
        continuing_subword_prefix=args.prefix,
    )
    tokenizer = Tokenizer(model)

    # Components ---------------------------------------------------------------
    tokenizer.normalizer = get_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=True)

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{config.BOS_TOKEN} $A {config.EOS_TOKEN}",
        pair=f"{config.BOS_TOKEN} $A {config.EOS_TOKEN} {config.SEP_TOKEN} $B {config.EOS_TOKEN}",
        special_tokens=[
            (config.BOS_TOKEN, 0),
            (config.EOS_TOKEN, 0),
            (config.SEP_TOKEN, 0),
        ],
    )

    # Trainer ------------------------------------------------------------------
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        min_frequency=args.min_freq,
        special_tokens=config.SPECIAL_TOKENS,
        continuing_subword_prefix=args.prefix,
    )

    # Train --------------------------------------------------------------------
    print(f"[tokenizer] vocab={args.vocab}  min_freq={args.min_freq}  samples={length:,}")
    tokenizer.train_from_iterator(
        _with_progress(iterator, length, args.no_progress),
        trainer=trainer,
        length=length,
    )

    # Save artefacts -----------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.out))

    cfg_out = args.out.with_name(args.out.stem + "_config.json")
    hf_cfg = {
        "unk_token": config.UNK_TOKEN,
        "pad_token": config.PAD_TOKEN,
        "bos_token": config.BOS_TOKEN,
        "eos_token": config.EOS_TOKEN,
        "sep_token": config.SEP_TOKEN,
    }
    utils.dump_json(hf_cfg, cfg_out)

    snap_out = args.out.with_name(args.out.stem + "_training_params.json")
    utils.dump_json(config.TrainingConfig().__dict__, snap_out)

    try:
        rel_path = args.out.relative_to(Path.cwd())
    except ValueError:
        rel_path = args.out  # fallback to absolute if not a subpath
    print(f"[tokenizer] ✓ saved {rel_path}")


# --------------------------------------------------------------------------- #
# 5. Entrypoint                                                                #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    train_tokenizer(parse_args(sys.argv[1:]))
