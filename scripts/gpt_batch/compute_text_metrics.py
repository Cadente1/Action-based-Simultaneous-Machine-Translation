#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute text-level MT metrics against a reference file.

This script is intentionally lightweight and independent from model training.
It reads one hypothesis file and one reference file, both line-aligned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sacrebleu


def read_lines(path: Path, keep_empty_lines: bool = False) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if keep_empty_lines:
        return [line.rstrip("\n") for line in lines]
    return [line.strip() for line in lines if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute BLEU, chrF, and TER.")
    parser.add_argument("--hypothesis_file", required=True)
    parser.add_argument("--reference_file", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument(
        "--tokenize",
        default="13a",
        help="SacreBLEU tokenizer, e.g., 13a, zh, ja-mecab, none.",
    )
    parser.add_argument("--keep_empty_lines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hypotheses = read_lines(Path(args.hypothesis_file), args.keep_empty_lines)
    references = read_lines(Path(args.reference_file), args.keep_empty_lines)

    if len(hypotheses) != len(references):
        raise ValueError(
            f"Line count mismatch: hypothesis={len(hypotheses)}, reference={len(references)}"
        )

    bleu = sacrebleu.metrics.BLEU(tokenize=args.tokenize)
    chrf = sacrebleu.metrics.CHRF()
    ter = sacrebleu.metrics.TER()

    results = {
        "num_segments": len(hypotheses),
        "bleu": bleu.corpus_score(hypotheses, [references]).score,
        "chrf": chrf.corpus_score(hypotheses, [references]).score,
        "ter": ter.corpus_score(hypotheses, [references]).score,
        "tokenize": args.tokenize,
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
