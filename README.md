# DS-LoRA

Code for **Depth-Selective LoRA: Augmenting Mixture-of-Recursions with Sparse Low-Rank Adaptation**.

This `main` branch is a refactored implementation produced with Codex. It exposes only the paper-facing paths:

- native **Mixture-of-Recursions (MoR)**
- **MoR + DS-LoRA**, where top-1 routed low-rank adapters are attached at explicitly selected recursion depths

The earlier experimental source, including the abandoned RMoE direction that routed over finer-grained recursive blocks, has been archived on the `archive/original-implementation` branch. The public `main` branch is intentionally lighter and no longer depends on `rmoe_core.py`.

## Repository Layout

```text
MoR_LanguageModel.py    # LM wrapper: embeddings, MoR/DS-LoRA backbone, final norm, lm head
mor_ds_lora_core.py     # Native MoR router/backbone and DS-LoRA gate/adapters
train_llm.py            # Causal LM training script
config.json             # Example WikiText-style configuration
```

## Method Summary

MoR assigns token-wise recursion depth with a continue/exit router while preserving dense self-attention. DS-LoRA keeps that routing behavior intact and adds a sparse low-rank branch only at selected recursion depths. In the default configuration, adapters are placed at the final recursion:

```json
"adapters": {
  "lora_depths": [2],
  "lora_num": 1,
  "lora_rank": 2,
  "lora_usage_target": 0.10
}
```

Set `"lora_num": 0` to train native MoR.

`lora_depths` is zero-based and can contain one or more recursion depths, for example `[0]`, `[1]`, `[2]`, or `[1, 2]`. Negative indices are also accepted, so `[-1]` means the final recursion. If `lora_depths` is omitted, the older `lora_depth_start` fallback is still supported and enables adapters from that depth through the final recursion.

## Quick Start

Install the usual PyTorch/Hugging Face stack, then point `config.json` at your tokenizer and processed dataset:

```bash
python train_llm.py --config config.json
```

For a short smoke run:

```bash
python train_llm.py --config config.json --max_micro_steps 10
```

The training log reports per-depth MoR keep rates and DS-LoRA adapter usage rates so native MoR and DS-LoRA runs can be compared under matched settings.

## Notes

- This branch removes legacy RMoE code from the active implementation surface.
- DS-LoRA is trained jointly with the MoR backbone rather than used as a post-hoc finetuning adapter.
- The default config follows the paper's final-depth insertion principle; ablations can be run by changing `lora_depths`, `lora_num`, and `lora_rank`.
