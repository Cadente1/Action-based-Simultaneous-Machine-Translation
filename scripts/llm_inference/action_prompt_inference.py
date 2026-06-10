#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run action-prompt inference with a Qwen-style causal language model.

This script is intended for decoder-only LLM adaptation experiments where a
local model such as Qwen3-8B is prompted with interpreter-inspired actions.

It performs line-by-line inference, supports checkpoint-style resume from an
existing output file, and writes results incrementally to avoid losing progress
during long runs.

Example:
    python scripts/few_shot/action_prompt_inference.py \
      --model_name_or_path Qwen/Qwen3-8B \
      --source_file examples/source.sample.txt \
      --output_file outputs/sample.qwen3.action.txt \
      --source_language English \
      --target_language Japanese \
      --trust_remote_code

Optional prompt files can be provided with:
    --system_prompt_file prompts/action_prompts/system_prompt.txt
    --user_prompt_file prompts/action_prompts/final_translation_action_prompt.txt

Supported placeholders in prompt files:
    {sentence}
    {text}
    {source_language}
    {target_language}
    {allowed_actions}
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_ALLOWED_ACTIONS = (
    "READ, WRITE, DROP, CUT, PRONOUN, PARTIAL_SUMMARIZATION"
)

DEFAULT_SYSTEM_PROMPT = """You are simulating a simultaneous interpreter.
Your task is to translate from {source_language} into {target_language} using incremental, step-by-step processing, similar to real-time simultaneous interpreting.

At each step, you may use the following actions:

1. READ — Read the next source word.
2. WRITE — Output a target-language translation fragment based on the words read so far.
3. DROP — Remove previous word(s) only when they are meaningless fillers such as "um" or "uh", thinking words, repetitions, or self-corrections. Do not use DROP unless these conditions are met.
4. CUT — Intentionally split a long or syntactically complex sentence into smaller independently translatable units.
5. PRONOUN — Replace repeated or already mentioned noun phrases with pronouns only if the referents are unambiguous.
6. PARTIAL_SUMMARIZATION — Combine or simplify semantically equivalent or repetitive expressions while preserving the original meaning and tone.

"""

DEFAULT_USER_PROMPT = """Translate the following {source_language} sentence into {target_language} under the simultaneous-interpreting constraints above.

Allowed actions:
{allowed_actions}

Output only the final {target_language} translation. Do not include quotes, explanations, action labels, or reasoning.

Sentence:
{sentence}
"""


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>.*", flags=re.DOTALL | re.IGNORECASE)
COMMON_PREFIX_RE = re.compile(
    r"^\s*(translation|final translation|output|answer|target)\s*[:：]\s*",
    flags=re.IGNORECASE,
)


def read_prompt_file(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def read_source_lines(path: str, keep_empty_lines: bool) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if keep_empty_lines:
        return [line.rstrip("\n") for line in lines]
    return [line.strip() for line in lines if line.strip()]


def count_lines(path: str) -> int:
    output_path = Path(path)
    if not output_path.exists():
        return 0
    with output_path.open("r", encoding="utf-8") as fin:
        return sum(1 for _ in fin)


def format_prompt(template: str, values: Dict[str, str]) -> str:
    class SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(SafeDict(values))


def strip_think(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text)
    text = THINK_OPEN_RE.sub("", text)
    return text.strip()


def clean_generation(text: str, keep_multiline_output: bool = False) -> str:
    text = strip_think(text)
    text = text.strip().strip("`").strip()

    if not keep_multiline_output:
        nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if nonempty_lines:
            text = nonempty_lines[0]

    text = COMMON_PREFIX_RE.sub("", text)
    text = text.strip().strip("“”\"'")
    return text


def parse_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def get_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(args):
    print(f"[info] Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"[info] Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_reply(
    messages: List[Dict[str, str]],
    tokenizer,
    model,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    input_device = get_input_device(model)
    input_ids = input_ids.to(input_device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if do_sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    output_ids = model.generate(**generation_kwargs)
    new_tokens = output_ids[:, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()


def translate_line(
    sentence: str,
    tokenizer,
    model,
    system_template: str,
    user_template: str,
    args,
) -> str:
    values = {
        "sentence": sentence.strip(),
        "text": sentence.strip(),
        "source_language": args.source_language,
        "target_language": args.target_language,
        "allowed_actions": args.allowed_actions,
    }

    system_prompt = format_prompt(system_template, values)
    user_prompt = format_prompt(user_template, values)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    output = generate_reply(
        messages=messages,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    return clean_generation(output, keep_multiline_output=args.keep_multiline_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen-style action-prompt inference for simultaneous interpretation."
    )

    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--source_file", required=True)
    parser.add_argument("--output_file", required=True)

    parser.add_argument("--source_language", default="English")
    parser.add_argument("--target_language", required=True)
    parser.add_argument("--allowed_actions", default=DEFAULT_ALLOWED_ACTIONS)

    parser.add_argument("--system_prompt_file", default=None)
    parser.add_argument("--user_prompt_file", default=None)

    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")

    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--flush_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file line count.")
    parser.add_argument("--keep_empty_lines", action="store_true")
    parser.add_argument("--keep_multiline_output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_path = Path(args.source_file)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    system_template = read_prompt_file(args.system_prompt_file) or DEFAULT_SYSTEM_PROMPT
    user_template = read_prompt_file(args.user_prompt_file) or DEFAULT_USER_PROMPT

    source_lines = read_source_lines(args.source_file, keep_empty_lines=args.keep_empty_lines)

    done = count_lines(args.output_file) if args.resume else 0
    if done > len(source_lines):
        raise ValueError(
            f"Output file already has {done} lines, but source file has only {len(source_lines)} lines."
        )

    print(f"[info] Total input lines: {len(source_lines)}")
    if args.resume:
        print(f"[info] Resume enabled. Existing output lines: {done}. Start from line {done + 1}.")

    tokenizer, model = load_model_and_tokenizer(args)

    mode = "a" if args.resume else "w"
    buffer: List[str] = []
    newly_generated = 0

    with output_path.open(mode, encoding="utf-8") as fout:
        for line_idx, sentence in enumerate(source_lines[done:], start=done + 1):
            if not sentence.strip():
                translation = ""
            else:
                translation = translate_line(
                    sentence=sentence,
                    tokenizer=tokenizer,
                    model=model,
                    system_template=system_template,
                    user_template=user_template,
                    args=args,
                )

            buffer.append(translation + "\n")
            newly_generated += 1

            print(f"[{line_idx:06d}] SRC: {sentence}")
            print(f"[{line_idx:06d}] OUT: {translation}")
            print("-" * 60)

            if len(buffer) >= args.flush_every:
                fout.writelines(buffer)
                fout.flush()
                os.fsync(fout.fileno())
                buffer.clear()
                print(f"[info] Wrote {newly_generated} new line(s). Latest line: {line_idx}")

        if buffer:
            fout.writelines(buffer)
            fout.flush()
            os.fsync(fout.fileno())

    print("=" * 60)
    print(f"[done] Newly generated lines: {newly_generated}")
    print(f"[done] Output file: {output_path}")


if __name__ == "__main__":
    main()
