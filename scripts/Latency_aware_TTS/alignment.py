import os
import jieba
from simalign import SentenceAligner
from tqdm import tqdm
import json

# === Paths ===
en_path = "/path/to/source/sentences.txt"
zh_path = "/path/to/target/language/translations.txt"
output_path = "/path/to/alignment.jsonl"

# === Load data ===
with open(en_path, 'r', encoding='utf-8') as f:
    en_lines = [line.strip() for line in f]

with open(zh_path, 'r', encoding='utf-8') as f:
    zh_lines = [line.strip() for line in f]

assert len(en_lines) == len(zh_lines), "The number of lines in English and Chinese is inconsistent."

# === Initialize the alignment model ===
aligner = SentenceAligner(model="", token_type="", matching_methods="")

# === Limiting condition ===
PRONOUNS = {"it", "he", "she", "they", "we", "i", "you"}

# === 对齐处理 ===
with open(output_path, "w", encoding="utf-8") as fout:
    for en, zh in tqdm(zip(en_lines, zh_lines), total=len(en_lines), desc="Aligning"):
        en_tokens = en.strip().split()
        zh_tokens = list(jieba.cut(zh.strip()))

        raw_alignment = aligner.get_word_aligns(en_tokens, zh_tokens)["itermax"]

        # Build the mapping of zh_idx → en_idx
        zh2en = {}
        for en_idx, zh_idx in raw_alignment:
            if en_idx < len(en_tokens):
                en_word = en_tokens[en_idx].lower()

                # Skip the pronouns with span alignment
                if en_word in PRONOUNS and zh_idx < en_idx - 5:
                    continue

                # Only retain the earliest English token that appears for each Chinese token
                if zh_idx not in zh2en or en_idx < zh2en[zh_idx]:
                    zh2en[zh_idx] = en_idx

        # Construct a reverse mapping (for easier lookup of Chinese in English later)
        en2zh = {}
        for zh_idx, en_idx in zh2en.items():
            en2zh.setdefault(en_idx, []).append(zh_idx)

        # Sort alignment
        alignment_pairs = sorted([[e, z] for z, e in zh2en.items()])

        fout.write(json.dumps({
            "en_tokens": en_tokens,
            "zh_tokens": zh_tokens,
            "alignment": alignment_pairs,
            "zh2en": zh2en,  # Used for subsequent time mapping
        }, ensure_ascii=False) + "\n")
