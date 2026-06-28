"""Language-model wrapper for native MoR / MoR + DS-LoRA."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from mor_ds_lora_core import MoRBackbone, RMSNorm


class MoR_Config:
    """Small attribute container used by the training script."""

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MoR_LanguageModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any] | MoR_Config):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, dict) else vars(cfg)

        def get(key: str, default: Any = None) -> Any:
            if key in self.cfg:
                return self.cfg[key]
            adapters = self.cfg.get("adapters", {})
            if isinstance(adapters, dict) and key in adapters:
                return adapters[key]
            return default

        vocab_size = int(get("vocab_size"))
        hidden_size = int(get("hidden_size"))
        causal = bool(get("causal", True))
        norm_eps = float(get("norm_eps", 1e-6))

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.backbone = MoRBackbone(self.cfg, causal=causal)
        self.final_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.embed_tokens(input_ids)
        out = self.backbone(hidden, attention_mask=attention_mask)
        hidden = self.final_norm(out["y"])
        logits = self.lm_head(hidden)
        return {
            "logits": logits,
            "loss": None,
            "_aux_for_train": out.get("_aux_for_train", {}),
            "loss_dict": out.get("loss_dict", {}),
            "stats": out.get("stats", {}),
        }
