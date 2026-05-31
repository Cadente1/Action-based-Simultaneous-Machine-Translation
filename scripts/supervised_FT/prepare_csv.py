#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create a seq2seq training CSV from aligned source and target text files.

Each input file should contain one example per line. Lines are paired by index.
The output CSV contains train/eval split labels plus source and target text
columns. By default, the last ``dev_size`` examples are assigned to the eval
split and the remaining examples are assigned to the train split.
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a train/eval CSV from line-aligned source and target files."
    )
    parser.add_argument(
        "--source_file",
        type=Path,
        required=True,
        help="Path to the source text file, with one example per line.",
    )
    parser.add_argument(
        "--target_file",
        type=Path,
        required=True,
        help="Path to the target text file, with one example per line.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path where the output CSV will be written.",
    )
    parser.add_argument(
        "--total_lines",
        type=int,
        default=None,
        help=(
            "Number of aligned examples to use from the beginning of the files. "
            "If omitted, all available aligned examples are used."
        ),
    )
    parser.add_argument(
        "--dev_size",
        type=int,
        default=2000,
        help="Number of examples reserved for the eval split.",
    )
    parser.add_argument(
        "--split_column",
        type=str,
        default="split",
        help="Name of the split column in the output CSV.",
    )
    parser.add_argument(
        "--source_column",
        type=str,
        default="src",
        help="Name of the source-text column in the output CSV.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="tgt",
        help="Name of the target-text column in the output CSV.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Label used for training examples.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="dev",
        help="Label used for evaluation examples.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Input and output file encoding.",
    )
    parser.add_argument(
        "--allow_shorter",
        action="store_true",
        help=(
            "If --total_lines exceeds the number of aligned examples, use all "
            "available aligned examples instead of raising an error."
        ),
    )
    return parser.parse_args()


def read_lines(path: Path, encoding: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding=encoding) as f:
        return [line.rstrip("\n") for line in f]


def resolve_example_count(
    requested_total: int | None,
    available: int,
    allow_shorter: bool,
) -> int:
    if available <= 0:
        raise ValueError("No aligned examples are available.")

    if requested_total is None:
        return available

    if requested_total <= 0:
        raise ValueError("--total_lines must be a positive integer.")

    if requested_total > available:
        if allow_shorter:
            print(
                (
                    f"[WARN] Requested {requested_total} examples, but only "
                    f"{available} aligned examples are available. Using {available}."
                ),
                file=sys.stderr,
            )
            return available

        raise ValueError(
            f"Requested {requested_total} examples, but only {available} aligned "
            "examples are available. Use --allow_shorter to continue with the "
            "available examples."
        )

    return requested_total


def write_csv(
    output_file: Path,
    source_lines: list[str],
    target_lines: list[str],
    dev_size: int,
    split_column: str,
    source_column: str,
    target_column: str,
    train_split: str,
    eval_split: str,
    encoding: str,
) -> None:
    total = len(source_lines)

    if total != len(target_lines):
        raise ValueError(
            f"Source and target lengths differ after truncation: "
            f"{total} vs {len(target_lines)}"
        )

    if dev_size <= 0:
        raise ValueError("--dev_size must be a positive integer.")

    if total <= dev_size:
        raise ValueError(
            f"Not enough examples ({total}) for dev_size={dev_size}. "
            "Reduce --dev_size or provide more data."
        )

    train_end = total - dev_size
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [split_column, source_column, target_column]

    with output_file.open("w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for src, tgt in zip(source_lines[:train_end], target_lines[:train_end]):
            writer.writerow(
                {
                    split_column: train_split,
                    source_column: src,
                    target_column: tgt,
                }
            )

        for src, tgt in zip(source_lines[train_end:], target_lines[train_end:]):
            writer.writerow(
                {
                    split_column: eval_split,
                    source_column: src,
                    target_column: tgt,
                }
            )


def main() -> None:
    args = parse_args()

    source_lines_all = read_lines(args.source_file, args.encoding)
    target_lines_all = read_lines(args.target_file, args.encoding)

    n_source = len(source_lines_all)
    n_target = len(target_lines_all)
    n_aligned = min(n_source, n_target)

    if n_source != n_target:
        print(
            (
                f"[WARN] Source and target files have different lengths "
                f"({n_source} vs {n_target}). Using the first {n_aligned} "
                "line-aligned examples."
            ),
            file=sys.stderr,
        )

    total = resolve_example_count(
        requested_total=args.total_lines,
        available=n_aligned,
        allow_shorter=args.allow_shorter,
    )

    source_lines = source_lines_all[:total]
    target_lines = target_lines_all[:total]

    write_csv(
        output_file=args.output_file,
        source_lines=source_lines,
        target_lines=target_lines,
        dev_size=args.dev_size,
        split_column=args.split_column,
        source_column=args.source_column,
        target_column=args.target_column,
        train_split=args.train_split,
        eval_split=args.eval_split,
        encoding=args.encoding,
    )

    print(f"[INFO] Source lines: {n_source}")
    print(f"[INFO] Target lines: {n_target}")
    print(f"[INFO] Used examples: {total}")
    print(f"[INFO] Train examples: {total - args.dev_size}")
    print(f"[INFO] Eval examples: {args.dev_size}")
    print(f"[INFO] Output CSV: {args.output_file}")


if __name__ == "__main__":
    main()
