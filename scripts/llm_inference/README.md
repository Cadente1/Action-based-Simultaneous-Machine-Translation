## Track B: decoder-only LLM action-prompt inference

Decoder-only LLM inference scripts are stored in:

```text
scripts/llm_inference/
```

This track studies how decoder-only LLMs can be prompted or adapted to perform action-aware simultaneous machine interpretation. The main script is:

```text
scripts/llm_inference/action_prompt_inference.py
```

This script runs step-wise inference with a Qwen-style causal language model, such as Qwen3-8B. For each input sentence, the model receives the complete source sentence, the allowed action set, and optional language-specific action statistics. It then outputs the final target-language translation. The script supports incremental writing and resume from an existing output file.

The action statistics are generated from development-set evaluation. For each action condition, GPT-4o or another LLM first generates translations under a restricted action set. For example, the DROP condition may use only:

```text
READ
WRITE
DROP
```

The generated translations are compared with the reference translations to compute BLEU / chrF / TER. TTS-based LAAL should be computed separately using the latency-aware TTS pipeline. The average LAAL values are then passed to `aggregate_action_metrics.py`, which produces a prompt-ready statistics block.

### 1. Build per-action statistics

Use `aggregate_action_metrics.py` to compute text metrics and merge precomputed average LAAL values:

```bash
python scripts/gpt_batch/aggregate_action_metrics.py \
  --language_pair en-zh \
  --reference_file data/dev.zh.txt \
  --hypothesis DROP=outputs/drop.zh.txt \
  --hypothesis PARTIAL_SUMMARIZATION=outputs/partial_summarization.zh.txt \
  --hypothesis CUT=outputs/cut.zh.txt \
  --hypothesis PRONOUN=outputs/pronoun.zh.txt \
  --laal 0.851 0.847 0.824 0.858 \
  --tokenize zh \
  --output_json outputs/action_statistics.en_zh.json \
  --output_prompt_txt outputs/action_statistics.en_zh.prompt.txt
```

The order of the values passed to `--laal` follows the order of `--hypothesis`. In the example above:

```text
DROP                  -> 0.851
PARTIAL_SUMMARIZATION -> 0.847
CUT                   -> 0.824
PRONOUN               -> 0.858
```

You may also provide explicit action-value pairs:

```bash
python scripts/gpt_batch/aggregate_action_metrics.py \
  --language_pair en-zh \
  --reference_file data/dev.zh.txt \
  --hypothesis DROP=outputs/drop.zh.txt \
  --hypothesis PARTIAL_SUMMARIZATION=outputs/partial_summarization.zh.txt \
  --hypothesis CUT=outputs/cut.zh.txt \
  --hypothesis PRONOUN=outputs/pronoun.zh.txt \
  --laal DROP=0.851 PARTIAL_SUMMARIZATION=0.847 CUT=0.824 PRONOUN=0.858 \
  --tokenize zh \
  --output_json outputs/action_statistics.en_zh.json \
  --output_prompt_txt outputs/action_statistics.en_zh.prompt.txt
```

The generated prompt snippet looks like:

```text
Based on development-set evaluation for en-zh:
- DROP -> LAAL ≈ 0.851s, BLEU ≈ 58.94
- PARTIAL_SUMMARIZATION -> LAAL ≈ 0.847s, BLEU ≈ 60.33
- CUT -> LAAL ≈ 0.824s, BLEU ≈ 60.28
- PRONOUN -> LAAL ≈ 0.858s, BLEU ≈ 60.85
```

These statistics are language-specific. If the target language changes, the action-specific translations and the corresponding BLEU / LAAL statistics should be regenerated.

### 2. Run action-prompt inference with Qwen3-8B

Use `action_prompt_inference.py` to run full-sentence action-aware inference:

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

To inject per-action development statistics into the prompt:

```bash
python scripts/llm_inference/action_prompt_inference.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --source_file data/dev.en.txt \
  --output_file outputs/dev.action.zh.txt \
  --source_language English \
  --target_language Chinese \
  --action_statistics_file outputs/action_statistics.en_zh.prompt.txt \
  --trust_remote_code \
  --resume
```

In this setting, the prompt contains:

```text
Allowed actions:
{allowed_actions}

Action-level development-set statistics:
{action_statistics}

Use the statistics above as empirical guidance when deciding whether to apply DROP, CUT, PRONOUN, or PARTIAL_SUMMARIZATION. Prefer actions that improve latency without hurting translation quality, but use an action only when its linguistic condition is satisfied.
```

The default prompt templates support the following placeholders:

```text
{sentence}
{text}
{source_language}
{target_language}
{allowed_actions}
{action_statistics}
```

External prompt templates can also be provided:

```bash
python scripts/llm_inference/action_prompt_inference.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --source_file data/dev.en.txt \
  --output_file outputs/dev.action.zh.txt \
  --source_language English \
  --target_language Chinese \
  --system_prompt_file prompts/action_prompts/system_prompt.txt \
  --user_prompt_file prompts/action_prompts/final_translation_action_prompt.txt \
  --action_statistics_file outputs/action_statistics.en_zh.prompt.txt \
  --trust_remote_code \
  --resume
```

`few_shot_prompt.py` is kept as a simpler few-shot prompting baseline, while `action_prompt_inference.py` corresponds to the full-sentence action-prompt inference setting used in the main method.

### 3. Few-shot baseline

The few-shot baseline script is:

```text
scripts/llm_inference/few_shot_prompt.py
```

Example:

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
GPT batch generation scripts
per-action BLEU / LAAL statistics aggregation
Qwen-style action-prompt inference
example data formats
quality and latency evaluation scripts
```
