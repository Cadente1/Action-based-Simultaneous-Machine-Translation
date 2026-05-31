import json

# Input/output path
input_path = "/path/to/alignment.jsonl"
output_path = "/path/to/alignment_with_wait_tokens.jsonl"

def apply_wait_tokens(en_tokens, zh_tokens, alignment):
    zh_tokens_out = zh_tokens[:]
    offset = 0
    wait_positions = []

    # Build the alignment mapping of the English index -> Chinese index
    en_to_zh = {}
    for en_idx, zh_idx in alignment:
        if en_idx not in en_to_zh:
            en_to_zh[en_idx] = []
        en_to_zh[en_idx].append(zh_idx)

    # Record whether the Chinese position is earlier than the corresponding English position
    for en_idx, zh_idx in alignment:
        adjusted_zh_idx = zh_idx + offset
        if adjusted_zh_idx < en_idx:
            wait_positions.append(adjusted_zh_idx)
            zh_tokens_out.insert(adjusted_zh_idx, "<WAIT>")
            offset += 1

    # Update the Chinese position in alignment
    updated_alignment = []
    for en_idx, zh_idx in alignment:
        shift = sum(1 for pos in wait_positions if pos <= zh_idx)
        updated_alignment.append([en_idx, zh_idx + shift])

    return en_tokens, zh_tokens_out, updated_alignment

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        en_tokens = data["en_tokens"]
        zh_tokens = data["zh_tokens"]
        alignment = data["alignment"]

        en_out, zh_out, new_alignment = apply_wait_tokens(en_tokens, zh_tokens, alignment)

        fout.write(json.dumps({
            "en_tokens": en_out,
            "zh_tokens": zh_out,
            "alignment": new_alignment
        }, ensure_ascii=False) + "\n")
