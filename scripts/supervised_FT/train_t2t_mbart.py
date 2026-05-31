#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune a sequence-to-sequence translation model on a CSV dataset.

The default configuration uses mBART-50 many-to-many. The input CSV is expected
to contain one split column and two text columns, for example:

split,src,tgt
train,This is a sentence.,これは文です。
dev,This is another sentence.,これは別の文です。
"""

import argparse
import inspect
import os
from typing import Iterable

import evaluate
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a seq2seq translation model such as mBART-50."
    )

    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to a CSV file containing train/eval examples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for checkpoints and the final model.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/mbart-large-50-many-to-many-mmt",
        help="Hugging Face model name or local model path.",
    )

    parser.add_argument(
        "--source_column",
        type=str,
        default="src",
        help="Column name for source-side text.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="tgt",
        help="Column name for target-side text.",
    )
    parser.add_argument(
        "--split_column",
        type=str,
        default="split",
        help="Column name that identifies train/eval examples.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Value in the split column used for training.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="dev",
        help="Value in the split column used for evaluation.",
    )

    parser.add_argument(
        "--src_lang",
        type=str,
        default=None,
        help="Source language code for multilingual models, e.g. en_XX.",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default=None,
        help="Target language code for multilingual models, e.g. ja_XX or zh_CN.",
    )

    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_train_epochs", type=float, default=10)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--generation_num_beams", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--fp16", action="store_true", help="Use FP16 training.")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 training.")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite output_dir if it already exists.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help='Experiment tracking target passed to Trainer, e.g. "none", "wandb".',
    )

    return parser.parse_args()


def validate_columns(dataset: Dataset, required_columns: Iterable[str]) -> None:
    missing = [col for col in required_columns if col not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Missing column(s): {missing}. Available columns: {dataset.column_names}"
        )


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def configure_tokenizer_and_model(
    tokenizer,
    model,
    src_lang: str | None,
    tgt_lang: str | None,
) -> None:
    """Set language codes for multilingual seq2seq models when supported."""
    if src_lang is not None and hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang

    if tgt_lang is not None and hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = tgt_lang

    if tgt_lang is None:
        return

    lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lang_code_to_id, dict) and tgt_lang in lang_code_to_id:
        model.config.forced_bos_token_id = lang_code_to_id[tgt_lang]
        return

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if callable(convert_tokens_to_ids):
        token_id = convert_tokens_to_ids(tgt_lang)
        if token_id is not None and token_id != unk_token_id:
            model.config.forced_bos_token_id = token_id


def build_training_args(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    save_steps = args.save_steps if args.save_steps is not None else args.eval_steps

    kwargs = {
        "output_dir": args.output_dir,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "logging_steps": args.logging_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_steps": args.warmup_steps,
        "predict_with_generate": True,
        "generation_max_length": args.max_target_length,
        "generation_num_beams": args.generation_num_beams,
        "load_best_model_at_end": True,
        "metric_for_best_model": "bleu",
        "greater_is_better": True,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "overwrite_output_dir": args.overwrite_output_dir,
        "report_to": args.report_to,
    }

    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return Seq2SeqTrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    configure_tokenizer_and_model(tokenizer, model, args.src_lang, args.tgt_lang)

    raw_dataset = load_dataset("csv", data_files=args.data_file)["train"]
    validate_columns(
        raw_dataset,
        [args.split_column, args.source_column, args.target_column],
    )

    train_dataset = raw_dataset.filter(
        lambda example: example[args.split_column] == args.train_split
    )
    eval_dataset = raw_dataset.filter(
        lambda example: example[args.split_column] == args.eval_split
    )

    if len(train_dataset) == 0:
        raise ValueError(f"No training examples found for split={args.train_split!r}.")
    if len(eval_dataset) == 0:
        raise ValueError(f"No evaluation examples found for split={args.eval_split!r}.")

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")

    def preprocess_function(batch):
        source_texts = [normalize_text(text) for text in batch[args.source_column]]
        target_texts = [normalize_text(text) for text in batch[args.target_column]]

        model_inputs = tokenizer(
            source_texts,
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=target_texts,
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_columns = train_dataset.column_names

    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,
    )
    tokenized_eval = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=remove_columns,
    )

    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.asarray(preds)
        labels = np.asarray(labels)

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = sacrebleu.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )
        return {"bleu": result["score"]}

    training_args = build_training_args(args)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
