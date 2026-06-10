#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create JSONL request files for OpenAI Batch API translation experiments.

The script converts one or more plain-text source files into JSONL files. Each
line is a batch request for the chat completions endpoint. Prompt content can be
provided through external template files, so action-specific prompts can be kept
separate from the code.

Template placeholders:
    {text}
    {source_language}
    {target_language}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from string import Formatter
from typing import Dict, Iterable, List, Sequence

from tqdm import tqdm


LANGUAGE_NAMES = {
    "zh": "Chinese",
    "zh_CN": "Chinese",
    "ja": "Japanese",
    "ja_XX": "Japanese",
    "de": "German",
    "de_DE": "German",
    "en": "English",
    "en_XX": "English",
}


DEFAULT_SYSTEM_TEMPLATE = (
    "You will be provided with a sentence in {source_language}. "
    "Your task is to interpret it into {target_language}. "
    "Return a valid JSON object with two fields: "
    "\"segmented_pairs\" as a list of source-target segment pairs, and "
    "\"output\" as the final {target_language} translation."
)


DEFAULT_SALAMI_USER_TEMPLATE = """Instructions:
The Salami Technique in simultaneous interpretation means breaking the source
input into smaller, manageable segments that each contain enough information to
be accurately interpreted.

1. Break down the following sentence into smaller segments for easier simultaneous interpretation.
2. Translate each segment into {target_language}.
3. Connect the translated segments into a fluent final translation.

Input:
{text}
"""


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


def normalize_language(value: str) -> str:
    return LANGUAGE_NAMES.get(value, value)


def format_template(template: str, values: Dict[str, str]) -> str:
    return template.format_map(SafeFormatDict(values))


def validate_template(template: str, template_name: str) -> None:
    allowed = {"text", "source_language", "target_language"}
    fields = {field for _, field, _, _ in Formatter().parse(template) if field}
    unknown = fields - allowed
    if unknown:
        raise ValueError(
            f"{template_name} contains unsupported placeholder(s): {sorted(unknown)}. "
            f"Allowed placeholders: {sorted(allowed)}"
        )


def batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_request(
    *,
    custom_id: str,
    text: str,
    model: str,
    source_language: str,
    target_language: str,
    system_template: str,
    user_template: str,
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

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": format_template(system_template, values)},
            {"role": "user", "content": format_template(user_template, values)},
        ],
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
        description="Create JSONL files for OpenAI Batch API translation requests."
    )
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="One or more plain-text source files, one sentence per line.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where JSONL batch input files will be written.",
    )
    parser.add_argument("--source_language", default="English")
    parser.add_argument("--target_language", required=True)
    parser.add_argument(
        "--strategy",
        choices=["salami", "custom"],
        default="salami",
        help="Use the built-in Salami prompt or external custom templates.",
    )
    parser.add_argument(
        "--system_prompt_file",
        default=None,
        help="Optional system prompt template file.",
    )
    parser.add_argument(
        "--user_prompt_file",
        default=None,
        help="Optional user prompt template file. Required for --strategy custom.",
    )
    parser.add_argument("--model", default="gpt-4o-2024-05-13")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument(
        "--response_format",
        choices=["json_object", "none"],
        default="json_object",
    )
    parser.add_argument(
        "--max_requests_per_file",
        type=int,
        default=50000,
        help="Split output into multiple JSONL files after this many requests.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Remove duplicate source lines while preserving their first occurrence.",
    )
    parser.add_argument(
        "--keep_empty_lines",
        action="store_true",
        help="Keep empty input lines. Usually not recommended for translation requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = [Path(path) for path in args.input_files]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_language = normalize_language(args.source_language)
    target_language = normalize_language(args.target_language)

    system_template = (
        read_text(Path(args.system_prompt_file))
        if args.system_prompt_file
        else DEFAULT_SYSTEM_TEMPLATE
    )

    if args.user_prompt_file:
        user_template = read_text(Path(args.user_prompt_file))
    elif args.strategy == "salami":
        user_template = DEFAULT_SALAMI_USER_TEMPLATE
    else:
        raise ValueError("--user_prompt_file is required when --strategy custom.")

    validate_template(system_template, "system prompt template")
    validate_template(user_template, "user prompt template")

    manifest = []

    for input_path in input_paths:
        lines = read_lines(input_path, keep_empty_lines=args.keep_empty_lines)
        if args.deduplicate:
            lines = deduplicate_preserve_order(lines)

        if not lines:
            print(f"[skip] No usable lines in {input_path}")
            continue

        stem = input_path.stem
        total_written = 0

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
                        source_language=source_language,
                        target_language=target_language,
                        system_template=system_template,
                        user_template=user_template,
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
                    "model": args.model,
                    "source_language": source_language,
                    "target_language": target_language,
                    "strategy": args.strategy,
                }
            )

            print(f"[write] {output_path} ({len(chunk)} requests)")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
