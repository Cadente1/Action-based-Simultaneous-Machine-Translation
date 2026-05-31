#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run batched inference with a fine-tuned sequence-to-sequence model.

The script is intended for line-by-line translation or text generation. It works
with mBART/mBART50 checkpoints and other Hugging Face seq2seq models supported
by AutoModelForSeq2SeqLM.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate line-by-line outputs with a fine-tuned seq2seq model."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model name on Hugging Face Hub or local checkpoint path.",
    )
    parser.add_argument(
        "--source_file",
        type=Path,
        required=True,
        help="Input text file. Each line is treated as one example.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Output file. One generated line is written for each input line.",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default=None,
        help="Optional source language code, e.g. en_XX for mBART50.",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default=None,
        help="Optional target language code, e.g. de_DE, zh_CN, ja_XX for mBART50.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size used during generation.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help="Maximum input sequence length after tokenization.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens generated for each example.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search. Use 1 for greedy decoding.",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="Length penalty used by beam search.",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="If greater than 0, repeated n-grams of this size are blocked.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for inference, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model dtype used when loading the checkpoint.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom model/tokenizer code from the model repository.",
    )
    parser.add_argument(
        "--skip_empty_lines",
        action="store_true",
        help="Skip empty input lines instead of preserving them in the output.",
    )
    parser.add_argument(
        "--print_examples",
        type=int,
        default=3,
        help="Number of input/output examples to print after generation.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def read_lines(path: Path, skip_empty_lines: bool) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if skip_empty_lines:
        return [line.strip() for line in lines if line.strip()]

    return [line.strip() for line in lines]


def get_forced_bos_token_id(tokenizer, tgt_lang: Optional[str]) -> Optional[int]:
    if tgt_lang is None:
        return None

    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if lang_code_to_id is None:
        raise ValueError(
            "The tokenizer does not expose lang_code_to_id. "
            "Remove --tgt_lang or use a tokenizer that supports language codes."
        )

    if tgt_lang not in lang_code_to_id:
        supported = ", ".join(sorted(lang_code_to_id.keys())[:20])
        raise ValueError(
            f"Unsupported target language code: {tgt_lang}. "
            f"Examples of supported codes: {supported} ..."
        )

    return lang_code_to_id[tgt_lang]


def configure_tokenizer(tokenizer, src_lang: Optional[str]) -> None:
    if src_lang is None:
        return

    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if lang_code_to_id is not None and src_lang not in lang_code_to_id:
        supported = ", ".join(sorted(lang_code_to_id.keys())[:20])
        raise ValueError(
            f"Unsupported source language code: {src_lang}. "
            f"Examples of supported codes: {supported} ..."
        )

    tokenizer.src_lang = src_lang


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    batch: List[str],
    device: str,
    forced_bos_token_id: Optional[int],
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
) -> List[str]:
    non_empty_positions = [idx for idx, text in enumerate(batch) if text]
    outputs = [""] * len(batch)

    if not non_empty_positions:
        return outputs

    non_empty_inputs = [batch[idx] for idx in non_empty_positions]
    encoded = tokenizer(
        non_empty_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    ).to(device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
    }

    if forced_bos_token_id is not None:
        generation_kwargs["forced_bos_token_id"] = forced_bos_token_id

    if no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    generated = model.generate(**encoded, **generation_kwargs)
    decoded = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    for position, text in zip(non_empty_positions, decoded):
        outputs[position] = text.strip()

    return outputs


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0")

    dtype = resolve_dtype(args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    configure_tokenizer(tokenizer, args.src_lang)
    forced_bos_token_id = get_forced_bos_token_id(tokenizer, args.tgt_lang)

    source_lines = read_lines(args.source_file, args.skip_empty_lines)
    print(f"Loaded {len(source_lines)} input lines from {args.source_file}")

    outputs: List[str] = []
    for start in tqdm(range(0, len(source_lines), args.batch_size), desc="Generating"):
        batch = source_lines[start : start + args.batch_size]
        batch_outputs = generate_batch(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            device=args.device,
            forced_bos_token_id=forced_bos_token_id,
            max_source_length=args.max_source_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        outputs.extend(batch_outputs)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        for output in outputs:
            f.write(output + "\n")

    print(f"Saved {len(outputs)} outputs to {args.output_file}")

    if args.print_examples > 0 and source_lines:
        print("\nExamples:")
        for idx in range(min(args.print_examples, len(source_lines))):
            print(f"Input  {idx + 1}: {source_lines[idx]}")
            print(f"Output {idx + 1}: {outputs[idx]}")
            print()


if __name__ == "__main__":
    main()
