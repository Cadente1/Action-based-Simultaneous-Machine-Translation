#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Upload JSONL files and create OpenAI Batch API jobs.

Authentication:
    export OPENAI_API_KEY="..."

The script does not store API keys. It reads the key from the environment by
default, or from --api_key if explicitly provided.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from openai import OpenAI
from tqdm import tqdm


def collect_jsonl_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix != ".jsonl":
            raise ValueError(f"Expected a .jsonl file, got: {input_path}")
        return [input_path]

    if input_path.is_dir():
        return sorted(input_path.glob(pattern))

    raise FileNotFoundError(f"Input path not found: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload JSONL files and create OpenAI Batch API jobs."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="A .jsonl file or a directory containing .jsonl files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern used when --input is a directory.",
    )
    parser.add_argument(
        "--endpoint",
        default="/v1/chat/completions",
        help="Batch endpoint. Must match the url field inside the JSONL requests.",
    )
    parser.add_argument(
        "--completion_window",
        default="24h",
        help="Batch completion window.",
    )
    parser.add_argument(
        "--description",
        default="batch translation job",
        help="Description stored in batch metadata.",
    )
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="Optional path to save upload and batch IDs as JSON.",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="Optional API key. Prefer using the OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print files that would be uploaded without creating API jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    jsonl_files = collect_jsonl_files(input_path, args.pattern)

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under {input_path}")

    print(f"[info] Found {len(jsonl_files)} JSONL file(s).")

    if args.dry_run:
        for path in jsonl_files:
            print(f"[dry-run] {path}")
        return

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your-key'"
        )

    client = OpenAI(api_key=api_key)
    records = []

    for path in tqdm(jsonl_files, desc="Creating batches"):
        print(f"[upload] {path}")

        with path.open("rb") as fin:
            batch_input_file = client.files.create(file=fin, purpose="batch")

        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=args.endpoint,
            completion_window=args.completion_window,
            metadata={
                "description": args.description,
                "source_file": path.name,
            },
        )

        record = {
            "jsonl_file": str(path),
            "input_file_id": batch_input_file.id,
            "batch_id": batch.id,
            "endpoint": args.endpoint,
            "completion_window": args.completion_window,
            "status": getattr(batch, "status", None),
        }
        records.append(record)
        print(f"[created] file_id={batch_input_file.id} batch_id={batch.id}")

    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else input_path / "batch_upload_manifest.json" if input_path.is_dir()
        else input_path.with_suffix(".batch_manifest.json")
    )
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
