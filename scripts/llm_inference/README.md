## Track B: decoder-only LLM action-prompt inference

The main local inference script for decoder-only LLM adaptation is:

```text
scripts/few_shot/action_prompt_inference.py
```

This script runs line-by-line inference with a Qwen-style causal language model, such as Qwen3-8B, using the action policy prompt. It supports incremental writing and resume from an existing output file.

Example:

```bash
python scripts/few_shot/action_prompt_inference.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --source_file examples/source.sample.txt \
  --output_file outputs/sample.qwen3.action.ja.txt \
  --source_language English \
  --target_language Japanese \
  --trust_remote_code \
  --resume
```

To use an external prompt template:

```bash
python scripts/few_shot/action_prompt_inference.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --source_file data/dev.en.txt \
  --output_file outputs/dev.action.zh.txt \
  --source_language English \
  --target_language Chinese \
  --system_prompt_file prompts/action_prompts/system_prompt.txt \
  --user_prompt_file prompts/action_prompts/final_translation_action_prompt.txt \
  --trust_remote_code \
  --resume
```

The prompt template may use the following placeholders:

```text
{sentence}
{text}
{source_language}
{target_language}
{allowed_actions}
```

`few_shot_prompt.py` can be kept as a simpler baseline script, while `action_prompt_inference.py` corresponds to the action-prompt inference setting used in the main method.
