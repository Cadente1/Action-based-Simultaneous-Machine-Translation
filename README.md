# SiMT Translation and Latency-aware TTS Toolkit

This repository contains lightweight research code for simultaneous machine translation / interpretation experiments and latency-aware text-to-speech synthesis.

The codebase is organized into three script groups:

1. **Supervised fine-tuning**: prepare text-to-text data, fine-tune seq2seq models, and run seq2seq inference.
2. **Few-shot LLM translation**: run prompt-based translation with causal language models.
3. **Latency-aware TTS**: generate aligned and latency-aware speech outputs through alignment, WAIT insertion, timestamp construction, and TTS synthesis.

Full datasets, model checkpoints, generated translations, and synthesized audio files are not included.

---

## Repository structure

```text
.
├── README.md
├── requirement.txt
├── .gitignore
├── examples/
│   ├── source.sample.txt
│   ├── target.sample.txt
│   ├── few_shot.sample.txt
│   └── t2t_data.sample.csv
└── scripts/
    ├── few_shot/
    │   └── infer_llm_translation.py
    ├── supervised_FT/
    │   ├── prepare_t2t_csv.py
    │   ├── train_seq2seq.py
    │   └── infer_seq2seq.py
    └── Latency_aware_TTS/
        ├── README.md
        ├── alignment.py
        ├── wait_insertion.py
        ├── time_table.py
        ├── tts.py
        └── assets/
```

If your local filenames are different, adjust the commands below accordingly.

> Note: `requirements.txt` is the more common filename convention. This repository currently uses `requirement.txt` following the shown structure. If you rename it to `requirements.txt`, update the installation command accordingly.

---

## Installation

Python 3.10 or newer is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -U pip
pip install -r requirement.txt
```

If you need a CUDA-specific PyTorch build, install PyTorch according to your CUDA version first, then install the remaining packages.

The latency-aware TTS scripts may require additional local setup, including a timestamp generation tool such as Whisper and a local CosyVoice installation.

---

## Data format

### Plain text files

Source and target text files should contain one sentence or segment per line.

Example source file:

```text
This is a test sentence.
The model translates source text into target text.
We evaluate translation quality and latency.
```

Example target file:

```text
これはテスト文です。
モデルは原文を目標言語に翻訳します。
翻訳品質と遅延を評価します。
```

### Text-to-text CSV format

The supervised fine-tuning script expects a CSV file with the following default columns:

```csv
split,src,tgt
train,This is a test sentence.,これはテスト文です。
dev,We evaluate translation quality and latency.,翻訳品質と遅延を評価します。
```

The default split names are `train` and `dev`. Column names and split names can be changed through command-line arguments.

---

## Examples

The `examples/` folder contains small toy files for checking the expected input formats:

```text
examples/source.sample.txt
examples/target.sample.txt
examples/few_shot.sample.txt
examples/t2t_data.sample.csv
```

These files are only format examples and are not intended to reproduce experimental results.

---

## Supervised fine-tuning

The scripts for supervised seq2seq fine-tuning are stored in:

```text
scripts/supervised_FT/
```

### 1. Prepare text-to-text CSV data

Use `prepare_t2t_csv.py` to merge line-aligned source and target files into a CSV file.

```bash
python scripts/supervised_FT/prepare_t2t_csv.py \
  --source_file examples/source.sample.txt \
  --target_file examples/target.sample.txt \
  --output_file examples/t2t_data.generated.csv \
  --dev_size 2
```

Optional arguments include:

```bash
--total_lines 10000
--allow_shorter
--source_column src
--target_column tgt
--split_column split
--train_split train
--eval_split dev
```

### 2. Fine-tune a seq2seq model

Use `train_seq2seq.py` to fine-tune an mBART50-style or other Hugging Face seq2seq model.

```bash
python scripts/supervised_FT/train_seq2seq.py \
  --data_file examples/t2t_data.sample.csv \
  --output_dir checkpoints/mbart50-sample \
  --model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
  --src_lang en_XX \
  --tgt_lang ja_XX \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --eval_steps 50 \
  --logging_steps 10
```

For real experiments, replace the sample CSV with your own data and adjust the training configuration.

### 3. Run seq2seq inference

Use `infer_seq2seq.py` to generate translations with a fine-tuned seq2seq model.

```bash
python scripts/supervised_FT/infer_seq2seq.py \
  --model_name_or_path checkpoints/mbart50-sample \
  --source_file examples/source.sample.txt \
  --output_file outputs/sample.seq2seq.pred.txt \
  --src_lang en_XX \
  --tgt_lang ja_XX \
  --batch_size 8 \
  --num_beams 4
```

For models that do not require language codes, omit `--src_lang` and `--tgt_lang`.

---

## Few-shot LLM translation

The few-shot translation script is stored in:

```text
scripts/few_shot/
```

Use `infer_llm_translation.py` to run prompt-based translation with a causal language model.

```bash
python scripts/few_shot/infer_llm_translation.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --source_file examples/source.sample.txt \
  --output_file outputs/sample.llm.pred.txt \
  --source_lang English \
  --target_lang Japanese \
  --few_shot_file examples/few_shot.sample.txt \
  --trust_remote_code
```

For local models, replace `Qwen/Qwen3-8B` with the local model path.

---

## Latency-aware TTS

The latency-aware TTS pipeline is stored in:

```text
scripts/Latency_aware_TTS/
```

It implements a multi-step pipeline for latency-aware speech synthesis:

1. Text alignment from parallel source and target text.
2. WAIT insertion into the target token stream.
3. Segment time table construction using source-side word-level timestamps.
4. TTS synthesis with silence insertion according to segment start times.

Run the scripts in order:

```text
alignment.py → wait_insertion.py → time_table.py → tts.py
```

Example workflow:

```bash
cd scripts/Latency_aware_TTS

python alignment.py
python wait_insertion.py
python time_table.py
python tts.py
```

This pipeline currently relies on script-level path configuration. Before running each script, edit the input/output paths and local model paths inside the corresponding file.

Expected intermediate files include:

```text
alignment.jsonl
alignment_with_wait.jsonl
segment_timetable.jsonl
```

The final output is synthesized audio, usually saved as `.wav` files under the configured output directory.

For detailed input formats, timestamp requirements, and CosyVoice setup notes, see:

```text
scripts/Latency_aware_TTS/README.md
```

---

## Suggested workflow

A typical supervised translation experiment can be run as follows:

```bash
# 1. Prepare training CSV
python scripts/supervised_FT/prepare_t2t_csv.py \
  --source_file data/train.source.txt \
  --target_file data/train.target.txt \
  --output_file data/t2t_train.csv \
  --dev_size 2000

# 2. Fine-tune seq2seq model
python scripts/supervised_FT/train_seq2seq.py \
  --data_file data/t2t_train.csv \
  --output_dir checkpoints/mbart50-exp \
  --model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
  --src_lang en_XX \
  --tgt_lang ja_XX \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 10

# 3. Run inference
python scripts/supervised_FT/infer_seq2seq.py \
  --model_name_or_path checkpoints/mbart50-exp \
  --source_file data/test.source.txt \
  --output_file outputs/test.pred.txt \
  --src_lang en_XX \
  --tgt_lang ja_XX
```

The latency-aware TTS pipeline can then be used to synthesize speech from translated or segmented target text, depending on the experiment setup.

---

## Files not included

This repository does not include:

```text
full training datasets
model checkpoints
generated translations
generated audio files
large experiment outputs
private evaluation files
```

Recommended local directories:

```text
data/
outputs/
checkpoints/
models/
logs/
```

These directories should generally be excluded from Git tracking.

---

## Notes

- Do not commit large datasets, checkpoints, generated audio, or private evaluation files.
- For mBART50, use valid language codes such as `en_XX`, `zh_CN`, `ja_XX`, or `de_DE`.
- For causal LLM inference, make sure the model supports chat templates if the script uses `tokenizer.apply_chat_template`.
- For TTS synthesis, install and configure CosyVoice separately.
- For source-side timestamps, prepare word-level timestamp JSONL files using Whisper or another compatible tool.

---

## Citation

If you use this repository or find it useful for your research, please cite our arXiv preprint:

```bibtex
@misc{zhang2026redefining,
  title         = {Redefining Machine Simultaneous Interpretation: From Incremental Translation to Human-Like Strategies},
  author        = {Zhang, Qianen and Yang, Zeyu and Nakamura, Satoshi},
  year          = {2026},
  eprint        = {2601.11002},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2601.11002}
}
```
