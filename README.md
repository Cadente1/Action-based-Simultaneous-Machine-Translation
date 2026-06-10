# Action-based Simultaneous Machine Interpretation

This repository contains public code, prompts, and examples for the project **Action-based Simultaneous Machine Interpretation**.

The project studies how simultaneous machine translation / interpretation can move beyond the conventional **READ/WRITE** policy by introducing interpreter-inspired actions. In addition to READ and WRITE, we consider actions such as:

```text
DROP
SENTENCE_CUT / CUT
PARTIAL_SUMMARIZATION
PRONOMINALIZATION / PRONOUN
```

The repository follows the main project workflow:

1. define action and Salami prompt templates;
2. generate action-specific translations through GPT batch requests or local Qwen action-prompt inference;
3. fine-tune or run translation models under different supervision settings;
4. evaluate translation quality and latency;
5. synthesize latency-aware speech outputs for TTS-based LAAL analysis.

Full datasets, private evaluation files, model checkpoints, GPT batch outputs, generated translations, and synthesized audio files are not included.

Project page:

```text
https://cadente1.github.io/Action-based-Simultaneous-Machine-Translation/
```

---

## Project motivation

Most simultaneous MT systems model simultaneous translation with two basic actions:

```text
READ  — wait for more source context
WRITE — generate target output
```

However, human interpreters use richer strategies under latency pressure. They may omit fillers, split long sentences, compress repetitive expressions, or replace repeated noun phrases with pronouns when the context is clear. This project explores whether such interpreter-inspired actions can improve the quality-latency trade-off in machine simultaneous interpretation.

---

## Proposed action space

| Action | Description |
|---|---|
| `READ` | Read the next source word or segment. |
| `WRITE` | Generate a target-language fragment based on the observed source context. |
| `DROP` | Remove fillers, repetitions, self-corrections, or semantically weak material. |
| `SENTENCE_CUT` / `CUT` | Split a long or syntactically complex sentence into shorter independently translatable units. |
| `PARTIAL_SUMMARIZATION` | Compress redundant or repetitive expressions while preserving meaning and tone. |
| `PRONOMINALIZATION` / `PRONOUN` | Replace repeated noun phrases with pronouns when the referent is unambiguous. |

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── index.html
│   └── assets/
├── examples/
│   ├── few_shot.sample.txt
│   ├── source.sample.txt
│   ├── target.sample.txt
│   └── t2t_data.sample.csv
├── prompts/
│   ├── action_prompts/
│   │   ├── README.md
│   │   ├── full_actions_batch_prompt.txt
│   │   ├── read_write_drop_batch_prompt.txt
│   │   ├── stepwise_action_prompt.txt
│   │   └── system_prompt.txt
│   └── salami_prompt/
│       ├── README.md
│       └── salami_batch_prompt.txt
└── scripts/
    ├── gpt_batch/
    │   ├── create_action_batch_jsonl.py
    │   ├── create_batch_jsonl.py
    │   ├── upload_batch_jsonl.py
    │   └── compute_text_metrics.py
    ├── latency_aware_TTS/
    │   ├── alignment.py
    │   ├── wait_insertion.py
    │   ├── time_table.py
    │   ├── tts.py
    │   └── compute_tts_laal.py
    ├── llm_inference/
    │   ├── README.md
    │   ├── action_prompt_inference.py
    │   └── few_shot_prompt.py
    └── supervised_FT/
        ├── prepare_csv.py
        ├── train_t2t_mbart.py
        └── inference.py
```

If your local filenames differ, adjust the commands below accordingly.

---

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

If you need a CUDA-specific PyTorch build, install PyTorch according to your CUDA version first, then install the remaining packages.

Some components require additional setup:

```text
OpenAI Batch API      required for GPT-based batch generation
Whisper / WhisperX   required for source/target timestamp extraction and TTS-based LAAL
CosyVoice            required for latency-aware TTS synthesis
TransLLaMA           use the official TransLLaMA implementation for that fine-tuning track
```

---

## Data format

### Plain text format

Source and target files should be line-aligned, with one sentence or segment per line.

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

The supervised fine-tuning scripts use a CSV format such as:

```csv
split,src,tgt
train,This is a test sentence.,これはテスト文です。
dev,We evaluate translation quality and latency.,翻訳品質と遅延を評価します。
```

The default split names are usually `train` and `dev`. If your scripts use different column names, adjust the command-line arguments or the script configuration accordingly.

### Source timestamp JSONL format

TTS-based LAAL computation expects source-side word timestamps in JSONL format. A typical line is:

```json
{"audio_index": 1, "words": [{"word": "This", "start": 0.00, "end": 0.32}, {"word": "is", "start": 0.32, "end": 0.48}]}
```

`audio_index` should align with the sentence index used by the translation and audio files.

---

## Prompt resources

Prompt templates are stored in:

```text
prompts/
├── action_prompts/
└── salami_prompt/
```

The prompts are kept outside Python scripts so that the same definitions can be reused in both offline GPT batch generation and online/local action-prompt inference.

### Action prompts

`prompts/action_prompts/` contains prompt templates for action-based simultaneous interpretation. These prompts define the action space and guide the model to produce action-aware translations or step-wise action decisions.

A full action prompt typically includes:

```text
READ
WRITE
DROP
CUT
PRONOUN
PARTIAL_SUMMARIZATION
```

For single-action or action-subset experiments, copy the full prompt and remove the definitions of actions that are not allowed in that setting. For example, a CUT-only setting can retain READ, WRITE, and CUT while removing DROP, PRONOUN, and PARTIAL_SUMMARIZATION.

### Salami prompt

`prompts/salami_prompt/` contains the prompt template for Salami-style segmented translation. This is used as a segmentation-oriented baseline. If you use this prompt, please cite the original work that introduced or used the Salami-style strategy.

### Batch prompt vs. step-wise prompt

The batch-generation setting and the online inference setting use related but not identical prompts:

```text
Batch generation:
input:  complete source sentence
output: action-aware translation or action trajectory

Step-wise inference:
input:  current source prefix + previous target output
output: next action + incremental translation fragment
```

---

## GPT batch generation

The GPT batch scripts are stored in:

```text
scripts/gpt_batch/
```

They are used to generate action-specific translations or Salami-style translations through the OpenAI Batch API.

### 1. Create batch JSONL files

Use `create_action_batch_jsonl.py` for action prompts:

```bash
python scripts/gpt_batch/create_action_batch_jsonl.py \
  --input_files data/dev.en.txt \
  --output_dir data/batch_jsonl/full_actions_zh \
  --user_prompt_file prompts/action_prompts/full_actions_batch_prompt.txt \
  --source_language English \
  --target_language Chinese
```

Use the corresponding Salami prompt file for Salami-style generation:

```bash
python scripts/gpt_batch/create_action_batch_jsonl.py \
  --input_files data/dev.en.txt \
  --output_dir data/batch_jsonl/salami_zh \
  --user_prompt_file prompts/salami_prompt/salami_batch_prompt.txt \
  --source_language English \
  --target_language Chinese
```

The output is a set of `.jsonl` files. Each line is one chat-completion request with a unique `custom_id`.

### 2. Upload batch files

Set the API key through an environment variable:

```bash
export OPENAI_API_KEY="your_api_key"
```

Then upload the generated JSONL files:

```bash
python scripts/gpt_batch/upload_batch_jsonl.py \
  --input data/batch_jsonl/full_actions_zh \
  --manifest_path data/batch_jsonl/full_actions_zh/batch_upload_manifest.json
```

After the batch job finishes, download the output file from the API platform. Use `custom_id` rather than output order to align outputs with source sentences.

---

## Track A: mBART50-style supervised fine-tuning

The supervised fine-tuning scripts are stored in:

```text
scripts/supervised_FT/
```

This track compares different target-side supervision signals, such as:

```text
offline gold translations
Salami-segmented translations
action-based translations
```

### 1. Prepare text-to-text CSV data

```bash
python scripts/supervised_FT/prepare_csv.py \
  --source_file examples/source.sample.txt \
  --target_file examples/target.sample.txt \
  --output_file examples/t2t_data.generated.csv \
  --dev_size 2
```

### 2. Fine-tune mBART50-style seq2seq model

```bash
python scripts/supervised_FT/train_t2t_mbart.py \
  --data_file examples/t2t_data.sample.csv \
  --output_dir checkpoints/mbart50-sample \
  --model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
  --src_lang en_XX \
  --tgt_lang ja_XX
```

For real experiments, replace the sample CSV with your own training file and adjust batch size, learning rate, epoch number, and language codes.

### 3. Run inference

```bash
python scripts/supervised_FT/inference.py \
  --model_name_or_path checkpoints/mbart50-sample \
  --source_file examples/source.sample.txt \
  --output_file outputs/sample.seq2seq.pred.txt \
  --src_lang en_XX \
  --tgt_lang ja_XX
```

For models that do not require language codes, omit `--src_lang` and `--tgt_lang` if the script supports this setting.

---

## Track B: decoder-only LLM inference

Decoder-only LLM scripts are stored in:

```text
scripts/llm_inference/
```

This track studies how decoder-only LLMs can be prompted or adapted to make action-aware translation decisions.

### Action-prompt inference

`action_prompt_inference.py` is the main local inference script for Qwen-style causal language models, such as Qwen3-8B. It uses the action policy prompt, performs line-by-line inference, supports incremental writing, and can resume from an existing output file.

Example usage:

```bash
python scripts/llm_inference/action_prompt_inference.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --source_file examples/source.sample.txt \
  --output_file outputs/sample.qwen3.action.ja.txt \
  --source_language English \
  --target_language Japanese \
  --trust_remote_code \
  --resume
```

For local models, replace `Qwen/Qwen3-8B` with your local model path.

To use an external prompt template:

```bash
python scripts/llm_inference/action_prompt_inference.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --source_file data/dev.en.txt \
  --output_file outputs/dev.action.zh.txt \
  --source_language English \
  --target_language Chinese \
  --system_prompt_file prompts/action_prompts/system_prompt.txt \
  --user_prompt_file prompts/action_prompts/full_actions_batch_prompt.txt \
  --trust_remote_code \
  --resume
```

If the prompt file is originally designed for batch generation and asks for JSON/action trajectories, create a separate prompt template for final-translation inference before using it here.

### Few-shot baseline

`few_shot_prompt.py` is kept as a simpler few-shot prompting baseline.

```bash
python scripts/llm_inference/few_shot_prompt.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --source_file examples/source.sample.txt \
  --output_file outputs/sample.llm.pred.txt \
  --source_lang English \
  --target_lang Japanese \
  --few_shot_file examples/few_shot.sample.txt
```

### TransLLaMA fine-tuning

The TransLLaMA fine-tuning track is not re-implemented in this repository. It can be reproduced by using the official TransLLaMA GitHub implementation and adapting the language settings, data paths, and prompt/action definitions according to the target language and experiment setting.

This repository provides the surrounding resources for that track:

```text
action prompt templates
Salami prompt templates
batch generation scripts
local Qwen action-prompt inference
example data formats
quality and latency evaluation scripts
```

---

## Evaluation

### Text-level MT metrics

`compute_text_metrics.py` is stored in:

```text
scripts/gpt_batch/compute_text_metrics.py
```

It computes BLEU, chrF, and TER against a line-aligned reference file.

```bash
python scripts/gpt_batch/compute_text_metrics.py \
  --hypothesis_file outputs/drop.zh.txt \
  --reference_file data/dev.zh.txt \
  --output_json outputs/drop_text_metrics.json \
  --tokenize zh
```

For Japanese or German, change the tokenizer setting as needed:

```bash
--tokenize ja-mecab
--tokenize 13a
```

### TTS-based AL / LAAL

`compute_tts_laal.py` is stored in:

```text
scripts/latency_aware_TTS/compute_tts_laal.py
```

It computes seconds-based latency metrics from target speech and source-side word timestamps.

```bash
python scripts/latency_aware_TTS/compute_tts_laal.py \
  --translation_file outputs/drop.zh.txt \
  --audio_dir outputs/drop_tts/sentences \
  --source_timestamps_jsonl data/en_word_timestamps_dev.jsonl \
  --reference_file data/dev.zh.txt \
  --output_csv outputs/drop_tts_laal.csv \
  --output_jsonl outputs/drop_tts_laal.jsonl \
  --asr_language zh \
  --wait_k 3
```

By default, the script expects audio files named:

```text
0.wav, 1.wav, 2.wav, ...
```

If your audio files are named:

```text
1.wav, 2.wav, 3.wav, ...
```

add:

```bash
--audio_index_base 1
```

---

## Latency-aware TTS

The latency-aware TTS pipeline is stored in:

```text
scripts/latency_aware_TTS/
```

It converts segmented or action-based translations into latency-aware speech through four steps:

1. text alignment from parallel source and target text;
2. WAIT insertion into the target token stream;
3. segment time-table construction using source-side word-level timestamps;
4. TTS synthesis with silence insertion according to segment start times.

Run the scripts in order:

```text
alignment.py → wait_insertion.py → time_table.py → tts.py
```

Example workflow:

```bash
cd scripts/latency_aware_TTS

python alignment.py
python wait_insertion.py
python time_table.py
python tts.py
```

This part currently relies on script-level path configuration. Before running each script, edit the input/output paths and local model paths inside the corresponding file.

Expected intermediate files include:

```text
alignment.jsonl
alignment_with_wait.jsonl
segment_timetable.jsonl
```

The final output is synthesized audio, usually saved as `.wav` files under the configured output directory.

---

## Suggested workflows

### Action-based GPT batch generation

```bash
# 1. Create action-aware JSONL requests
python scripts/gpt_batch/create_action_batch_jsonl.py \
  --input_files data/dev.en.txt \
  --output_dir data/batch_jsonl/full_actions_zh \
  --user_prompt_file prompts/action_prompts/full_actions_batch_prompt.txt \
  --source_language English \
  --target_language Chinese

# 2. Upload batch requests
export OPENAI_API_KEY="your_api_key"

python scripts/gpt_batch/upload_batch_jsonl.py \
  --input data/batch_jsonl/full_actions_zh \
  --manifest_path data/batch_jsonl/full_actions_zh/batch_upload_manifest.json

# 3. Download and parse batch outputs
# Use custom_id to align generated translations with source sentences.

# 4. Evaluate text quality
python scripts/gpt_batch/compute_text_metrics.py \
  --hypothesis_file outputs/full_actions.zh.txt \
  --reference_file data/dev.zh.txt \
  --output_json outputs/full_actions_text_metrics.json \
  --tokenize zh
```

### Local Qwen action-prompt inference

```bash
python scripts/llm_inference/action_prompt_inference.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --source_file data/dev.en.txt \
  --output_file outputs/dev.qwen3.action.zh.txt \
  --source_language English \
  --target_language Chinese \
  --trust_remote_code \
  --resume
```

### Supervised fine-tuning with generated targets

```bash
# 1. Prepare training CSV
python scripts/supervised_FT/prepare_csv.py \
  --source_file data/train.en.txt \
  --target_file data/train.action_zh.txt \
  --output_file data/t2t_train_action_zh.csv \
  --dev_size 2000

# 2. Fine-tune seq2seq model
python scripts/supervised_FT/train_t2t_mbart.py \
  --data_file data/t2t_train_action_zh.csv \
  --output_dir checkpoints/mbart50-action-zh \
  --model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
  --src_lang en_XX \
  --tgt_lang zh_CN

# 3. Run inference
python scripts/supervised_FT/inference.py \
  --model_name_or_path checkpoints/mbart50-action-zh \
  --source_file data/test.en.txt \
  --output_file outputs/test.action_zh.pred.txt \
  --src_lang en_XX \
  --tgt_lang zh_CN
```

### TTS-based latency evaluation

```bash
python scripts/latency_aware_TTS/compute_tts_laal.py \
  --translation_file outputs/test.action_zh.pred.txt \
  --audio_dir outputs/test_action_zh_tts/sentences \
  --source_timestamps_jsonl data/en_word_timestamps_test.jsonl \
  --reference_file data/test.zh.txt \
  --output_csv outputs/test_action_zh_laal.csv \
  --asr_language zh
```

---

## Examples

The `examples/` folder contains small toy files for checking expected input formats:

```text
examples/source.sample.txt
examples/target.sample.txt
examples/few_shot.sample.txt
examples/t2t_data.sample.csv
```

These files are only format examples and are not intended to reproduce experimental results.

---

## Files not included

This repository does not include:

```text
full training datasets
private evaluation files
API keys
OpenAI batch outputs
generated translations
source or target speech files
model checkpoints
large experiment logs
synthesized audio files
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

- Do not commit API keys, large datasets, checkpoints, generated audio, or private evaluation files.
- For mBART50, use valid language codes such as `en_XX`, `zh_CN`, `ja_XX`, or `de_DE`.
- For OpenAI batch outputs, align results using `custom_id`; do not assume the output order is identical to the input order.
- For Qwen-style local inference, make sure the model supports chat templates through `tokenizer.apply_chat_template`.
- For source-side timestamps, prepare word-level timestamp JSONL files using Whisper, WhisperX, or another compatible tool.
- For TTS synthesis, install and configure CosyVoice separately.
- For TransLLaMA fine-tuning, use the official implementation and adapt the language and data configuration for the target experiment.
- On case-sensitive systems, make sure the directory name `scripts/latency_aware_TTS/` matches the path used in the commands.

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
