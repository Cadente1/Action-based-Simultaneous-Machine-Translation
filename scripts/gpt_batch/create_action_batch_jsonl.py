#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create OpenAI Batch API JSONL files for action-based translation.

This script reads one or more plain-text source files and creates JSONL batch
request files. Prompt content is loaded from external template files, so prompts
can be reused by both offline API batch generation and online step-wise inference.

Example:
    python scripts/gpt_batch/create_action_batch_jsonl.py \
      --input_files data/dev.en.txt \
      --output_dir data/batch_jsonl/drop \
      --user_prompt_file prompts/action_policy/read_write_drop_batch_prompt.txt \
      --system_prompt_file prompts/action_policy/system_prompt.txt \
      --source_language English \
      --target_language Japanese
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from string import Formatter
from typing import Dict, Iterable, List, Sequence

from tqdm import tqdm


DEFAULT_MODEL = "gpt-4o-2024-05-13"


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_lines(path: Path, keep_empty_lines: bool = False) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if keep_empty_lines:
        return [line.rstrip("\n") for line in lines]
    return [line.strip() for line in lines if line.strip()]


def deduplicate_preserve_order(lines: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            output.append(line)
    return output


def short_hash(text: str, length: int = 12) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def validate_template(template: str, name: str, allowed_fields: set[str]) -> None:
    fields = {field for _, field, _, _ in Formatter().parse(template) if field}
    unknown_fields = fields - allowed_fields
    if unknown_fields:
        raise ValueError(
            f"{name} contains unsupported placeholder(s): {sorted(unknown_fields)}. "
            f"Allowed placeholders: {sorted(allowed_fields)}"
        )


def render_template(template: str, values: Dict[str, str]) -> str:
    return template.format_map(SafeFormatDict(values))


def batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_request(
    *,
    custom_id: str,
    text: str,
    model: str,
    system_prompt_template: str | None,
    user_prompt_template: str,
    source_language: str,
    target_language: str,
    temperature: float,
    top_p: float,
    seed: int | None,
    max_tokens: int,
    response_format: str,
) -> Dict:
    values = {
        "text": text,
        "source_language": source_language,
        "target_language": target_language,
    }

    messages = []
    if system_prompt_template:
        messages.append(
            {
                "role": "system",
                "content": render_template(system_prompt_template, values),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": render_template(user_prompt_template, values),
        }
    )

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "max_tokens": max_tokens,
    }

    if seed is not None:
        body["seed"] = seed

    if response_format == "json_object":
        body["response_format"] = {"type": "json_object"}

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create JSONL request files for OpenAI Batch API action-based translation."
    )
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--user_prompt_file", required=True)
    parser.add_argument("--system_prompt_file", default=None)

    parser.add_argument("--source_language", default="English")
    parser.add_argument("--target_language", required=True)

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument(
        "--response_format",
        choices=["json_object", "none"],
        default="json_object",
    )

    parser.add_argument("--max_requests_per_file", type=int, default=50000)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--keep_empty_lines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = [Path(path) for path in args.input_files]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    user_prompt_template = read_text(Path(args.user_prompt_file))
    system_prompt_template = (
        read_text(Path(args.system_prompt_file)) if args.system_prompt_file else None
    )

    batch_allowed_fields = {"text", "source_language", "target_language"}
    validate_template(user_prompt_template, "user prompt template", batch_allowed_fields)
    if system_prompt_template:
        validate_template(system_prompt_template, "system prompt template", batch_allowed_fields)

    manifest = []

    for input_path in input_paths:
        lines = read_lines(input_path, keep_empty_lines=args.keep_empty_lines)
        if args.deduplicate:
            lines = deduplicate_preserve_order(lines)

        if not lines:
            print(f"[skip] No usable lines in {input_path}")
            continue

        total_written = 0
        stem = input_path.stem

        for part_idx, chunk in enumerate(
            batched(lines, args.max_requests_per_file), start=1
        ):
            output_path = output_dir / f"{stem}_part{part_idx}.jsonl"

            with output_path.open("w", encoding="utf-8") as fout:
                for offset, text in enumerate(tqdm(chunk, desc=f"{input_path.name} part {part_idx}")):
                    global_idx = total_written + offset + 1
                    custom_id = f"{stem}-{global_idx:06d}-{short_hash(text)}"

                    request = build_request(
                        custom_id=custom_id,
                        text=text,
                        model=args.model,
                        system_prompt_template=system_prompt_template,
                        user_prompt_template=user_prompt_template,
                        source_language=args.source_language,
                        target_language=args.target_language,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                        max_tokens=args.max_tokens,
                        response_format=args.response_format,
                    )
                    fout.write(json.dumps(request, ensure_ascii=False) + "\n")

            total_written += len(chunk)
            manifest.append(
                {
                    "input_file": str(input_path),
                    "jsonl_file": str(output_path),
                    "num_requests": len(chunk),
                    "prompt_file": args.user_prompt_file,
                    "system_prompt_file": args.system_prompt_file,
                    "model": args.model,
                    "source_language": args.source_language,
                    "target_language": args.target_language,
                }
            )
            print(f"[write] {output_path} ({len(chunk)} requests)")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
