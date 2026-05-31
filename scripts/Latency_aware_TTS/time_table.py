import json

# === Paths ===
ALIGNMENT_PATH = "/path/to/alignment.jsonl"
TIMESTAMP_PATH = "/path/to/en_word_timestamps.jsonl"
OUTPUT_PATH = "/path/to/zh_segment_timetable.jsonl"

# === Load data ===
with open(ALIGNMENT_PATH, "r", encoding="utf-8") as f:
    alignments = [json.loads(line) for line in f]

with open(TIMESTAMP_PATH, "r", encoding="utf-8") as f:
    timestamps = [json.loads(line) for line in f]

assert len(alignments) == len(timestamps), "Mismatched data length!"

# === Build alignment mapping ===
def build_zh_to_en_map(alignment):
    zh_to_en = {}
    for en_idx, zh_idx in alignment:
        zh_to_en.setdefault(zh_idx, []).append(en_idx)
    return zh_to_en

# === Post-processing merge function ===
def merge_early_or_invalid_segments(time_table):
    merged = []
    for seg in time_table:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]

        # Condition: The time is not aligned or the time is rolled back
        if seg["start"] <= last["start"]:
            last["text"] += seg["text"]
        else:
            merged.append(seg)
    return merged

# === Main processing logic ===
all_results = []

for align_info, time_info in zip(alignments, timestamps):
    zh_tokens = align_info["zh_tokens"]
    en_word_times = time_info["words"]
    zh_to_en = build_zh_to_en_map(align_info["alignment"])

    # 1. Split the Chinese by <WAIT>
    segments = []
    buffer = []
    base_zh_idx = 0  # offset of segment start

    for i, tok in enumerate(zh_tokens):
        if tok == "<WAIT>":
            if buffer:
                segments.append((base_zh_idx, buffer))
                buffer = []
            base_zh_idx = i + 1
        else:
            buffer.append(tok)
    if buffer:
        segments.append((base_zh_idx, buffer))

    # 2. Get the start time of each paragraph (try to get the time of the earliest aligned English word)
    zh_segment_with_time = []

    for zh_start_idx, zh_words in segments:
        en_indices = zh_to_en.get(zh_start_idx, [])

        # fallback：Find the nearest alignment backward
        if not en_indices:
            for j in range(zh_start_idx + 1, len(zh_tokens)):
                if zh_tokens[j] == "<WAIT>":
                    continue
                en_indices = zh_to_en.get(j, [])
                if en_indices:
                    break

        # Get the earliest start time
        valid_starts = [
            en_word_times[i]["start"]
            for i in en_indices
            if 0 <= i < len(en_word_times)
        ]

        if valid_starts:
            start_time = min(valid_starts)
            zh_segment_with_time.append({
                "text": ''.join(zh_words),
                "start": round(start_time, 2)
            })
        else:
            # Temporarily add a segment with start=0.0 and handle the merge later
            zh_segment_with_time.append({
                "text": ''.join(zh_words),
                "start": 0.0
            })

    # 3. Merge fragments without time or time rollback
    merged_segments = merge_early_or_invalid_segments(zh_segment_with_time)

    all_results.append(merged_segments)

# === Save the output ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
    for item in all_results:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

