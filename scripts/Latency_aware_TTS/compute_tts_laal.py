#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compute TTS-based AL and LAAL from source timestamps and target speech.

Inputs:
    1. target translation text, one sentence per line
    2. target speech directory, typically 0.wav, 1.wav, ...
    3. source-side word timestamps JSONL
    4. reference translation text, one sentence per line

The script uses WhisperX to transcribe and align target speech, then computes
seconds-based AL/LAAL variants.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import librosa
import pandas as pd
import torch
import whisperx


def read_lines(path: Path, keep_empty_lines: bool = False) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if keep_empty_lines:
        return [line.rstrip("\n") for line in lines]
    return [line.strip() for line in lines if line.strip()]


def ref_text_len_units(text: str) -> int:
    text = "" if text is None else str(text).strip()
    if not text:
        return 0
    if " " in text:
        return len(text.split())
    return len(text)


def time_at_token_index(end_times: List[float], x: float) -> float:
    n = len(end_times)
    if n == 0:
        raise ValueError("empty end_times")

    if x <= 1:
        return end_times[0]
    if x >= n:
        return end_times[-1]

    lo = math.floor(x)
    hi = lo + 1
    frac = x - lo
    t_lo = end_times[lo - 1]
    t_hi = end_times[hi - 1]
    return t_lo + frac * (t_hi - t_lo)


def load_source_timestamps(path: Path) -> Dict[int, List[dict]]:
    """Load JSONL timestamps.

    Supported formats per line:
        {"audio_index": 1, "words": [{"word": "...", "end": 0.42}, ...]}
        {"words": [{"word": "...", "end": 0.42}, ...]}  # line number is used
    """
    timestamps = {}
    with path.open("r", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            audio_index = int(obj.get("audio_index", line_idx))
            words = obj.get("words", [])
            timestamps[audio_index] = words
    return timestamps


def source_end_times(words: Iterable[dict]) -> List[float]:
    return [float(word["end"]) for word in words if "end" in word]


def compute_available_source_counts(
    source_end_times_: List[float],
    target_start_times: List[float],
    fallback_duration: Optional[float] = None,
) -> List[int]:
    counts = []
    j = 0
    x_len = len(source_end_times_)

    for target_time in target_start_times:
        t = fallback_duration if target_time is None else target_time
        while j < x_len and source_end_times_[j] <= t:
            j += 1
        counts.append(j)

    return counts


def compute_lagging_from_policy(
    source_end_times_: List[float],
    policy_indices: List[float],
    y_len: int,
    use_laal: bool = False,
    ref_len: Optional[int] = None,
) -> Optional[float]:
    x_len = len(source_end_times_)
    if x_len == 0 or y_len == 0 or not policy_indices:
        return None

    tau = y_len
    for t in range(1, y_len + 1):
        if policy_indices[t - 1] >= x_len:
            tau = t
            break

    if use_laal and ref_len is not None and ref_len > 0:
        gamma = float(max(y_len, int(ref_len))) / float(x_len)
    else:
        gamma = float(y_len) / float(x_len)

    total = 0.0
    for t in range(1, tau + 1):
        policy_time = time_at_token_index(source_end_times_, policy_indices[t - 1])
        diag_idx = min(max((t - 1) / gamma, 1.0), float(x_len))
        diag_time = time_at_token_index(source_end_times_, diag_idx)
        total += policy_time - diag_time

    return total / tau


def build_g_based_policy(g: List[int], x_len: int) -> List[float]:
    return [float(min(max(value, 1), x_len)) for value in g]


def build_waitk_policy(
    y_len: int,
    x_len: int,
    wait_k: int,
    mode: str,
    g: Optional[List[int]] = None,
) -> List[float]:
    if mode == "classic":
        start_src = wait_k
    elif mode == "align_first_or_k":
        if not g:
            start_src = wait_k
        else:
            start_src = max(wait_k, max(1, g[0]))
    else:
        raise ValueError("mode must be 'classic' or 'align_first_or_k'")

    return [float(min(start_src + (t - 1), x_len)) for t in range(1, y_len + 1)]


def build_alignment_constrained_policy(
    source_end_times_: List[float],
    target_start_times: List[float],
    required_source_counts: List[int],
    wait_k: Optional[int] = None,
    mode: str = "classic",
) -> List[float]:
    x_len = len(source_end_times_)
    y_len = len(target_start_times)

    if x_len == 0 or y_len == 0 or len(required_source_counts) != y_len:
        raise ValueError("Invalid source/target lengths for alignment-constrained policy.")

    required = [max(1, min(int(value), x_len)) for value in required_source_counts]
    for i in range(1, y_len):
        if required[i] < required[i - 1]:
            required[i] = required[i - 1]

    first_target_time = target_start_times[0]
    g0 = 0
    for j, end_time in enumerate(source_end_times_, start=1):
        if end_time <= first_target_time:
            g0 = j
        else:
            break

    policy = [0] * y_len

    if wait_k is None:
        start_src = 1
        policy[0] = max(1, required[0])
    else:
        k = int(wait_k)
        if mode == "classic":
            start_src = k
        elif mode == "align_first_or_k":
            start_src = max(k, max(1, g0))
        else:
            raise ValueError("mode must be 'classic' or 'align_first_or_k'")
        policy[0] = max(start_src, required[0])

    for t in range(2, y_len + 1):
        candidate = max(policy[t - 2], required[t - 1])
        if wait_k is not None:
            if mode == "classic":
                candidate = max(candidate, start_src + (t - 1) - 1)
            else:
                candidate = max(candidate, start_src + (t - 1))
        policy[t - 1] = min(candidate, x_len)

    return [float(value) for value in policy]


def transcribe_target_audio(
    audio_path: Path,
    model,
    align_model,
    metadata,
    device: str,
    batch_size: int,
) -> Tuple[List[float], float]:
    audio = whisperx.load_audio(str(audio_path))
    duration = librosa.get_duration(path=str(audio_path))

    asr_result = model.transcribe(audio, batch_size=batch_size)
    aligned = whisperx.align(
        asr_result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    units = aligned.get("word_segments", [])
    target_start_times = [max(0.0, unit.get("start", 0.0)) for unit in units]
    return target_start_times, float(duration)


def safe_fmt(value: Optional[float]) -> str:
    return "nan" if value is None else f"{value:.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute TTS-based AL/LAAL.")
    parser.add_argument("--translation_file", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--source_timestamps_jsonl", required=True)
    parser.add_argument("--reference_file", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_jsonl", default=None)

    parser.add_argument("--asr_language", required=True, help="WhisperX language code for target audio.")
    parser.add_argument("--asr_model_name", default="medium")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute_type", default="auto")
    parser.add_argument("--asr_batch_size", type=int, default=8)

    parser.add_argument("--wait_k", type=int, default=3)
    parser.add_argument(
        "--waitk_mode",
        choices=["classic", "align_first_or_k"],
        default="classic",
    )
    parser.add_argument(
        "--audio_index_base",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use 0 if audio files are 0.wav, 1.wav, ...; use 1 if they are 1.wav, 2.wav, ...",
    )
    parser.add_argument(
        "--timestamp_index_base",
        type=int,
        choices=[0, 1],
        default=1,
        help="Index base used by source_timestamps_jsonl audio_index.",
    )
    parser.add_argument("--audio_extension", default=".wav")
    parser.add_argument("--keep_empty_lines", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    translation_file = Path(args.translation_file)
    audio_dir = Path(args.audio_dir)
    timestamps_file = Path(args.source_timestamps_jsonl)
    reference_file = Path(args.reference_file)

    translations = read_lines(translation_file, args.keep_empty_lines)
    references = read_lines(reference_file, args.keep_empty_lines)

    if len(translations) != len(references):
        raise ValueError(
            f"Line count mismatch: translation={len(translations)}, reference={len(references)}"
        )

    source_timestamp_dict = load_source_timestamps(timestamps_file)
    ref_lens = [max(1, ref_text_len_units(text)) for text in references]

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    compute_type = args.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "float32"

    print(
        f"[WhisperX] device={device}, model={args.asr_model_name}, "
        f"language={args.asr_language}, compute_type={compute_type}"
    )

    model = whisperx.load_model(
        args.asr_model_name,
        device=device,
        language=args.asr_language,
        compute_type=compute_type,
    )
    align_model, metadata = whisperx.load_align_model(
        language_code=args.asr_language,
        device=device,
    )

    rows = []

    for line_idx in range(1, len(translations) + 1):
        audio_index = line_idx - 1 if args.audio_index_base == 0 else line_idx
        timestamp_index = line_idx - 1 if args.timestamp_index_base == 0 else line_idx

        audio_path = audio_dir / f"{audio_index}{args.audio_extension}"
        source_words = source_timestamp_dict.get(timestamp_index)

        if not source_words:
            print(f"[skip] Missing source timestamps for index {timestamp_index}")
            continue

        if not audio_path.exists():
            print(f"[skip] Missing target audio: {audio_path}")
            continue

        try:
            target_start_times, duration = transcribe_target_audio(
                audio_path=audio_path,
                model=model,
                align_model=align_model,
                metadata=metadata,
                device=device,
                batch_size=args.asr_batch_size,
            )
        except Exception as exc:
            print(f"[skip] ASR/alignment failed for {audio_path}: {type(exc).__name__}: {exc}")
            continue

        source_ends = source_end_times(source_words)
        x_len = len(source_ends)
        y_len = len(target_start_times)

        if x_len == 0 or y_len == 0:
            print(f"[skip] Empty source/target units at line {line_idx}: X={x_len}, Y={y_len}")
            continue

        g = compute_available_source_counts(source_ends, target_start_times, duration)
        required = [max(1, min(value, x_len)) for value in g]
        ref_len = ref_lens[line_idx - 1]

        policy_base = build_g_based_policy(g, x_len)
        policy_waitk = build_waitk_policy(
            y_len=y_len,
            x_len=x_len,
            wait_k=args.wait_k,
            mode=args.waitk_mode,
            g=g,
        )
        policy_align_base = build_alignment_constrained_policy(
            source_end_times_=source_ends,
            target_start_times=target_start_times,
            required_source_counts=required,
            wait_k=None,
            mode=args.waitk_mode,
        )
        policy_align_waitk = build_alignment_constrained_policy(
            source_end_times_=source_ends,
            target_start_times=target_start_times,
            required_source_counts=required,
            wait_k=args.wait_k,
            mode=args.waitk_mode,
        )

        result = {
            "index": line_idx,
            "audio_path": str(audio_path),
            "X_len": x_len,
            "Y_len": y_len,
            "ref_len": ref_len,
            "AL_base": compute_lagging_from_policy(source_ends, policy_base, y_len, False, None),
            "AL_waitk": compute_lagging_from_policy(source_ends, policy_waitk, y_len, False, None),
            "AL_align_base": compute_lagging_from_policy(source_ends, policy_align_base, y_len, False, None),
            "AL_align_waitk": compute_lagging_from_policy(source_ends, policy_align_waitk, y_len, False, None),
            "LAAL_base": compute_lagging_from_policy(source_ends, policy_base, y_len, True, ref_len),
            "LAAL_waitk": compute_lagging_from_policy(source_ends, policy_waitk, y_len, True, ref_len),
            "LAAL_align_base": compute_lagging_from_policy(source_ends, policy_align_base, y_len, True, ref_len),
            "LAAL_align_waitk": compute_lagging_from_policy(source_ends, policy_align_waitk, y_len, True, ref_len),
        }
        rows.append(result)

        print(
            f"Index {line_idx}: "
            f"AL(base)={safe_fmt(result['AL_base'])}s "
            f"AL(wait-{args.wait_k})={safe_fmt(result['AL_waitk'])}s "
            f"LAAL(base)={safe_fmt(result['LAAL_base'])}s "
            f"LAAL(wait-{args.wait_k})={safe_fmt(result['LAAL_waitk'])}s"
        )

    if not rows:
        print("[done] No valid rows.")
        return

    df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[write] {output_csv}")

    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[write] {output_jsonl}")

    metric_cols = [col for col in df.columns if col.startswith("AL") or col.startswith("LAAL")]
    summary = df[metric_cols].mean(numeric_only=True).to_dict()
    print("[summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
