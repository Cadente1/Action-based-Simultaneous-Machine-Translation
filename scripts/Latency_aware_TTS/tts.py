import os
import json
import sys
import time
from tqdm import tqdm
from pydub import AudioSegment
import torchaudio

# === Add Cosyvoice paths ===
COSYVOICE_ROOT = "/path/to/Cosyvoice"
sys.path.append(COSYVOICE_ROOT)
sys.path.append(os.path.join(COSYVOICE_ROOT, "third_party/Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice

# === Paths ===
SEGMENT_JSONL = "/path/to/zh_segment_timetable.jsonl"
MODEL_PATH = "/path/to/Cosyvoice/model"
OUTPUT_DIR = "/path/to/tts/output/sentences"
TMP_DIR = "/path/to/tts/output/tmp"         # 'tmp' will store segments of speech, and the speech of whole sentences will be stored in 'sentences'.

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# === Initialize the CosyVoice model ===
cosyvoice = CosyVoice(MODEL_PATH, load_jit=False, load_trt=False, fp16=True)
print("List of available speakers: ", cosyvoice.list_available_spks())
# === Load time-labeled data ===
with open(SEGMENT_JSONL, "r", encoding="utf-8") as fin:
    all_segments = [json.loads(line) for line in fin]

# === Calculate from which index to start (the first ungenerated idx) ===
def find_start_idx(output_dir: str, total: int) -> int:
    existing = {
        int(fn[:-4]) for fn in os.listdir(output_dir)
        if fn.endswith(".wav") and fn[:-4].isdigit()
    }
    for i in range(total):
        if i not in existing:
            return i
    return total  # All have been generated

n_total = len(all_segments)
start_idx = find_start_idx(OUTPUT_DIR, n_total)

if start_idx >= n_total:
    print("✅ All generated. No need to continue.")
    sys.exit(0)

print(f"▶️ Start from the first ungenerated index: {start_idx}/{n_total-1}")


# === Synthetic main cycle ===
for idx in tqdm(range(start_idx, len(all_segments))):
    zh_segments = all_segments[idx]
    final_path = os.path.join(OUTPUT_DIR, f"{idx}.wav")
    if os.path.exists(final_path):
        continue  # If it already exists, skip it (double insurance)

    tmp_sentence_dir = os.path.join(TMP_DIR, str(idx))
    os.makedirs(tmp_sentence_dir, exist_ok=True)

    full_audio = AudioSegment.silent(duration=500)  # Mute for 0.5 seconds at the beginning

    for seg_id, seg in enumerate(zh_segments):
        text = seg["text"]
        seg_start = seg["start"]  # Unit: second (s)

        # Synthesize the speech
        for i, j in enumerate(cosyvoice.inference_sft(text, "中文女", stream=False)):       # Choose the speaker you'd like to use listed in 'available speakers'
            tmp_path = os.path.join(tmp_sentence_dir, f"{seg_id}_{i}.wav")
            torchaudio.save(tmp_path, j["tts_speech"], cosyvoice.sample_rate)

            seg_audio = AudioSegment.from_wav(tmp_path)

            # Ensure insertion at the start time 
            current_duration = full_audio.duration_seconds
            delay_ms = max(0, (seg_start - current_duration) * 1000)
            full_audio += AudioSegment.silent(duration=int(delay_ms))
            full_audio += seg_audio

    # Export the final merged sentence audio
    full_audio.export(final_path, format="wav")

