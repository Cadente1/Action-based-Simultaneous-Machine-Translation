#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sentence-level translation with a chat-based causal language model."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--source_file",
        type=Path,
        required=True,
        help="Input text file, one source sentence per line.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Output text file, one translation per line.",
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default="English",
        help="Source language name used in the prompt.",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="Japanese",
        help="Target language name used in the prompt.",
    )
    parser.add_argument(
        "--few_shot_file",
        type=Path,
        default=None,
        help="Optional file containing reference examples to include in the prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=160,
        help="Maximum number of generated tokens per sentence.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map passed to from_pretrained. Use 'none' to disable.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom model code from the model repository.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print source, raw completion, and cleaned translation.",
    )
    return parser.parse_args()


def resolve_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def read_optional_text(path: Optional[Path]) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8").strip()


def build_prompt(
    source: str,
    source_lang: str,
    target_lang: str,
    few_shot_examples: str = "",
) -> str:
    parts = [
        f"Translate the following text from {source_lang} to {target_lang}.",
        "Return only the translation, with no explanation or extra text.",
    ]

    if few_shot_examples:
        parts.append(f"Reference examples:\n{few_shot_examples}")

    parts.append(f"Text: {source}")
    return "\n\n".join(parts)


def strip_internal_tags(text: str) -> str:
    if not text:
        return ""

    cleaned = text
    for tag in ["think", "analysis", "thinking"]:
        cleaned = re.sub(rf"(?is)<{tag}>.*?</{tag}>", "", cleaned)
        cleaned = re.sub(rf"(?is)<{tag}>.*$", "", cleaned)
    return cleaned.strip()


def clean_translation(text: str) -> str:
    cleaned = strip_internal_tags(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return ""

    output = lines[0]
    output = re.sub(
        r"(?i)^\s*(translation|target|answer|output)\s*:\s*",
        "",
        output,
    ).strip()
    output = output.strip().strip('"').strip("'").strip()
    return re.sub(r"\s+", " ", output).strip()


def build_bad_words_ids(tokenizer) -> list[list[int]]:
    blocked_strings = [
        "<think>",
        "</think>",
        "<analysis>",
        "</analysis>",
        "<thinking>",
        "</thinking>",
    ]
    bad_words_ids = []
    for text in blocked_strings:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids:
            bad_words_ids.append(token_ids)
    return bad_words_ids


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    device_map = None if args.device_map.lower() == "none" else args.device_map
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_completion(
    prompt: str,
    tokenizer,
    model,
    bad_words_ids: list[list[int]],
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a translation engine. Output only the translated text.",
        },
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    input_ids = input_ids.to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        bad_words_ids=bad_words_ids or None,
    )

    completion_ids = output_ids[0][input_ids.shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model_and_tokenizer(args)
    bad_words_ids = build_bad_words_ids(tokenizer)
    few_shot_examples = read_optional_text(args.few_shot_file)

    sentences = [
        line.strip()
        for line in args.source_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.open("w", encoding="utf-8") as output_handle:
        for index, source in enumerate(tqdm(sentences, desc="Translating"), start=1):
            prompt = build_prompt(
                source=source,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                few_shot_examples=few_shot_examples,
            )
            raw_completion = generate_completion(
                prompt=prompt,
                tokenizer=tokenizer,
                model=model,
                bad_words_ids=bad_words_ids,
                max_new_tokens=args.max_new_tokens,
            )
            translation = clean_translation(raw_completion)

            if args.verbose:
                print(f"\n[{index:04d}] Source: {source}")
                print(f"[{index:04d}] Raw completion: {raw_completion}")
                print(f"[{index:04d}] Translation: {translation}")

            output_handle.write(translation + "\n")

    print(f"Saved translations to {args.output_file}")


if __name__ == "__main__":
    main()
