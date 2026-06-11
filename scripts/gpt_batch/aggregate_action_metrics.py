#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute per-action BLEU and merge precomputed average LAAL values.

This script builds the action-level statistics block used by full-sentence
action-prompt inference.

It directly computes BLEU / chrF / TER from:
  - one reference translation file
  - one hypothesis translation file per action

It does NOT recompute LAAL. LAAL should be computed separately by the
TTS-based LAAL pipeline. This script only reads already-computed average LAAL
values.

Example:
    python scripts/gpt_batch/aggregate_action_metrics.py \
        --language_pair en-zh \
        --reference_file data/dev.zh.txt \
        --hypothesis DROP=outputs/drop.zh.txt \
        --hypothesis PARTIAL_SUMMARIZATION=outputs/partial_summarization.zh.txt \
        --hypothesis CUT=outputs/cut.zh.txt \
        --hypothesis PRONOUN=outputs/pronoun.zh.txt \
        --laal 0.851 0.847 0.824 0.858 \
        --tokenize zh \
        --output_json outputs/action_statistics.en_zh.json \
        --output_prompt_txt outputs/action_statistics.en_zh.prompt.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import sacrebleu


ACTION_ALIASES = {
    "SENTENCE_CUT": "CUT",
    "PRONOMINALIZATION": "PRONOUN",
}


def normalize_action(action: str) -> str:
    action = action.strip().upper()
    return ACTION_ALIASES.get(action, action)


def parse_hypothesis_args(items: List[str]) -> Tuple[List[str], Dict[str, str]]:
    action_order: List[str] = []
    mapping: Dict[str, str] = {}

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --hypothesis value: {item!r}. "
                f"Expected ACTION=path/to/hypothesis.txt."
            )
        action, path = item.split("=", 1)
        action = normalize_action(action)
        path = path.strip()

        if not action or not path:
            raise ValueError(
                f"Invalid --hypothesis value: {item!r}. "
                f"Expected ACTION=path/to/hypothesis.txt."
            )
        if action in mapping:
            raise ValueError(f"Duplicated hypothesis for action: {action}")

        action_order.append(action)
        mapping[action] = path

    return action_order, mapping


def is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def parse_laal_values(items: Optional[List[str]], action_order: List[str]) -> Dict[str, float]:
    if not items:
        return {}

    has_mapping = any("=" in item for item in items)

    if has_mapping:
        mapping: Dict[str, float] = {}
        for item in items:
            if "=" not in item:
                raise ValueError(
                    "Do not mix ordered numeric LAAL values and ACTION=VALUE values. "
                    f"Problematic value: {item!r}"
                )
            action, value = item.split("=", 1)
            action = normalize_action(action)
            value = value.strip()
            if not is_number(value):
                raise ValueError(f"Invalid LAAL value for {action}: {value!r}")
            mapping[action] = float(value)

        missing = [action for action in action_order if action not in mapping]
        if missing:
            raise ValueError(f"Missing LAAL values for actions: {missing}")
        return {action: mapping[action] for action in action_order}

    if len(items) != len(action_order):
        raise ValueError(
            f"Ordered --laal values must match the number of hypotheses. "
            f"Got {len(items)} LAAL values for {len(action_order)} actions: {action_order}"
        )

    result: Dict[str, float] = {}
    for action, value in zip(action_order, items):
        if not is_number(value):
            raise ValueError(f"Invalid ordered LAAL value for {action}: {value!r}")
        result[action] = float(value)
    return result


def read_lines(path: Path, keep_empty_lines: bool = True) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if keep_empty_lines:
        return [line.rstrip("\n") for line in lines]
    return [line.strip() for line in lines if line.strip()]


def compute_text_metrics(
    *,
    action: str,
    hypothesis_file: Path,
    reference_lines: List[str],
    tokenize: str,
    lowercase: bool,
    keep_empty_lines: bool,
) -> Dict[str, float]:
    if not hypothesis_file.exists():
        raise FileNotFoundError(f"Missing hypothesis file for {action}: {hypothesis_file}")

    hypothesis_lines = read_lines(hypothesis_file, keep_empty_lines=keep_empty_lines)

    if len(hypothesis_lines) != len(reference_lines):
        raise ValueError(
            f"Line count mismatch for {action}: "
            f"hypothesis has {len(hypothesis_lines)} lines, "
            f"reference has {len(reference_lines)} lines."
        )

    bleu = sacrebleu.corpus_bleu(
        hypothesis_lines,
        [reference_lines],
        tokenize=tokenize,
        lowercase=lowercase,
    )
    chrf = sacrebleu.corpus_chrf(hypothesis_lines, [reference_lines])
    ter = sacrebleu.corpus_ter(
        hypothesis_lines,
        [reference_lines],
        normalized=True,
        no_punct=False,
        asian_support=True,
        case_sensitive=not lowercase,
    )

    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "ter": float(ter.score),
        "num_segments": len(hypothesis_lines),
    }


def infer_action_column(columns: Iterable[str]) -> Optional[str]:
    lower_to_original = {col.lower(): col for col in columns}
    for name in ["action", "actions", "strategy", "method", "condition"]:
        if name in lower_to_original:
            return lower_to_original[name]
    return None


def infer_laal_column(columns: Iterable[str]) -> Optional[str]:
    lower_to_original = {col.lower(): col for col in columns}
    for name in ["laal", "mean_laal", "avg_laal", "average_laal", "tts_laal", "tts_based_laal"]:
        if name in lower_to_original:
            return lower_to_original[name]
    candidates = [col for col in columns if "laal" in col.lower()]
    return candidates[0] if candidates else None


def to_float(value: Any, context: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Cannot parse a numeric LAAL value from {context}: {value!r}") from None


def read_laal_from_table(
    path: Path,
    *,
    action_order: List[str],
    action_column: Optional[str],
    value_column: Optional[str],
) -> Dict[str, float]:
    suffix = path.suffix.lower()
    sep = "\t" if suffix == ".tsv" else ","

    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path, sep=sep)

    if df.empty:
        raise ValueError(f"LAAL table is empty: {path}")

    action_col = action_column or infer_action_column(df.columns)
    laal_col = value_column or infer_laal_column(df.columns)

    if action_col and laal_col:
        result: Dict[str, float] = {}
        for _, row in df.iterrows():
            action = normalize_action(str(row[action_col]))
            if action in action_order:
                result[action] = to_float(row[laal_col], f"{path}:{action}")
        missing = [action for action in action_order if action not in result]
        if missing:
            raise ValueError(
                f"Missing LAAL values for actions {missing} in {path}. "
                f"Detected action column={action_col!r}, value column={laal_col!r}."
            )
        return {action: result[action] for action in action_order}

    if len(df) == 1:
        row = df.iloc[0]
        normalized_columns = {normalize_action(col): col for col in df.columns}
        result = {}
        for action in action_order:
            if action not in normalized_columns:
                raise ValueError(
                    f"Could not find a column for action {action} in {path}. "
                    f"Columns: {list(df.columns)}"
                )
            col = normalized_columns[action]
            result[action] = to_float(row[col], f"{path}:{col}")
        return result

    raise ValueError(
        f"Could not parse LAAL table {path}. Use long format with columns "
        f"'action,laal', or wide format with one row and action-named columns."
    )


def read_laal_from_json(
    path: Path,
    *,
    action_order: List[str],
    value_key: Optional[str],
) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        normalized = {normalize_action(str(k)): v for k, v in data.items()}

        if "ACTIONS" in normalized and isinstance(normalized["ACTIONS"], dict):
            normalized = {normalize_action(str(k)): v for k, v in normalized["ACTIONS"].items()}

        result = {}
        for action in action_order:
            if action not in normalized:
                raise ValueError(f"Missing LAAL value for action {action} in {path}")
            value = normalized[action]
            if isinstance(value, dict):
                key = value_key or infer_laal_column(value.keys()) or "laal"
                if key not in value:
                    raise ValueError(
                        f"Could not find LAAL key for action {action} in {path}. "
                        f"Available keys: {list(value.keys())}"
                    )
                value = value[key]
            result[action] = to_float(value, f"{path}:{action}")
        return result

    if isinstance(data, list):
        df = pd.DataFrame(data)
        action_col = infer_action_column(df.columns)
        laal_col = value_key or infer_laal_column(df.columns)
        if not action_col or not laal_col:
            raise ValueError(f"Could not parse JSON list in {path}. Need action and laal fields.")
        result = {}
        for _, row in df.iterrows():
            action = normalize_action(str(row[action_col]))
            if action in action_order:
                result[action] = to_float(row[laal_col], f"{path}:{action}")
        missing = [action for action in action_order if action not in result]
        if missing:
            raise ValueError(f"Missing LAAL values for actions {missing} in {path}")
        return {action: result[action] for action in action_order}

    raise ValueError(f"Unsupported LAAL JSON structure in {path}")


def read_laal_file(
    path: Path,
    *,
    action_order: List[str],
    action_column: Optional[str],
    value_column: Optional[str],
) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"LAAL file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv", ".jsonl"}:
        return read_laal_from_table(
            path,
            action_order=action_order,
            action_column=action_column,
            value_column=value_column,
        )

    if suffix == ".json":
        return read_laal_from_json(
            path,
            action_order=action_order,
            value_key=value_column,
        )

    text = path.read_text(encoding="utf-8").strip()
    tokens = re.split(r"[\s,]+", text)
    tokens = [tok for tok in tokens if tok]
    if not tokens:
        raise ValueError(f"LAAL text file is empty: {path}")

    if any("=" in tok for tok in tokens):
        return parse_laal_values(tokens, action_order)

    return parse_laal_values(tokens, action_order)


def format_num(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def build_prompt_snippet(language_pair: str, stats: Dict[str, Any]) -> str:
    lines = [f"Based on development-set evaluation for {language_pair}:"]

    for action, values in stats["actions"].items():
        lines.append(
            f"- {action} -> LAAL ≈ {format_num(values.get('laal'), 3)}s, "
            f"BLEU ≈ {format_num(values.get('bleu'), 2)}"
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute BLEU from per-action hypothesis files and merge "
            "precomputed average LAAL values."
        )
    )

    parser.add_argument("--language_pair", required=True, help="Example: en-zh, en-ja, en-de")
    parser.add_argument("--reference_file", required=True)

    parser.add_argument(
        "--hypothesis",
        action="append",
        required=True,
        metavar="ACTION=PATH",
        help=(
            "Per-action hypothesis file. Repeat this argument in the same order "
            "as ordered --laal values. Example: DROP=outputs/drop.zh.txt"
        ),
    )

    parser.add_argument(
        "--laal",
        nargs="+",
        default=None,
        help=(
            "Precomputed average LAAL values. Either provide ordered numeric values "
            "matching the order of --hypothesis, e.g. --laal 0.851 1.204 0.932, "
            "or provide ACTION=VALUE items, e.g. --laal DROP=0.851 CUT=1.204."
        ),
    )
    parser.add_argument(
        "--laal_file",
        default=None,
        help=(
            "Optional file containing one average LAAL value per action. "
            "Supported formats: .csv, .tsv, .json, .jsonl, .txt."
        ),
    )
    parser.add_argument("--laal_action_column", default=None)
    parser.add_argument("--laal_value_column", default=None)

    parser.add_argument("--tokenize", default="13a", help="sacreBLEU tokenizer, e.g. zh, ja-mecab, 13a")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--keep_empty_lines", action="store_true")

    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_prompt_txt", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.laal is None and args.laal_file is None:
        raise ValueError("Provide either --laal values or --laal_file.")
    if args.laal is not None and args.laal_file is not None:
        raise ValueError("Use only one of --laal or --laal_file, not both.")

    reference_path = Path(args.reference_file)
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    action_order, hypotheses = parse_hypothesis_args(args.hypothesis)

    if args.laal is not None:
        laal_values = parse_laal_values(args.laal, action_order)
        laal_source = "command_line"
    else:
        laal_values = read_laal_file(
            Path(args.laal_file),
            action_order=action_order,
            action_column=args.laal_action_column,
            value_column=args.laal_value_column,
        )
        laal_source = str(args.laal_file)

    reference_lines = read_lines(reference_path, keep_empty_lines=args.keep_empty_lines)

    result: Dict[str, Any] = {
        "language_pair": args.language_pair,
        "reference_file": str(reference_path),
        "laal_source": laal_source,
        "tokenize": args.tokenize,
        "lowercase": args.lowercase,
        "note": (
            "Per-action BLEU/chrF/TER are computed from action-specific hypothesis "
            "files against the shared reference file. LAAL values are precomputed "
            "average values from the TTS-based LAAL pipeline. These statistics are "
            "language-specific and should be recomputed when the language pair, "
            "dataset split, TTS system, source timestamps, or LAAL pipeline changes."
        ),
        "actions": {},
    }

    for action in action_order:
        hyp_path = Path(hypotheses[action])
        text_metrics = compute_text_metrics(
            action=action,
            hypothesis_file=hyp_path,
            reference_lines=reference_lines,
            tokenize=args.tokenize,
            lowercase=args.lowercase,
            keep_empty_lines=args.keep_empty_lines,
        )

        result["actions"][action] = {
            **text_metrics,
            "laal": float(laal_values[action]),
            "hypothesis_file": str(hyp_path),
        }

    output_json = Path(args.output_json)
    output_prompt_txt = Path(args.output_prompt_txt)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_prompt_txt.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output_prompt_txt.write_text(build_prompt_snippet(args.language_pair, result), encoding="utf-8")

    print(f"[done] Wrote JSON statistics to: {output_json}")
    print(f"[done] Wrote prompt snippet to: {output_prompt_txt}")


if __name__ == "__main__":
    main()
