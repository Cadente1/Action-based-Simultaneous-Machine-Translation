# Action Policy Prompts

This folder contains prompt templates for action-based simultaneous interpretation experiments.

Suggested files:

- `full_actions_batch_prompt.txt`  
  Used for generating full-sentence action-based translations with the OpenAI Batch API.
- `read_write_drop_batch_prompt.txt`  
  A smaller action subset example with READ, WRITE, and DROP only.
- `stepwise_action_prompt.txt`  
  Used for online step-wise action selection and inference. This is not directly uploaded to the Batch API.
- `system_prompt.txt`  
  Optional system message used by batch or online inference scripts.

## Placeholders

Batch prompt templates support:

```text
{text}
{source_language}
{target_language}
```

Step-wise prompt templates additionally use:

```text
{source_prefix}
{previous_output}
{allowed_actions}
```

## Creating action-specific prompts

To create a prompt for a single action or an action subset, copy
`full_actions_batch_prompt.txt` and remove the definitions of actions that are
not allowed in that experiment. For example, a DROP-only or CUT-only prompt can
be created by retaining READ, WRITE, and the target action while removing the
other proposed actions.

Keep prompt files separate from Python scripts so that the same prompt can be
used by both API batch generation and online step-wise inference.
