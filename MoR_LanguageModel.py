# MoR_LanguageModel.py
import math
from typing import Dict, Any
import torch
import torch.nn as nn

from rmoe_core import UnifiedBackbone

# ---- Lightweight RMSNorm（内部用fp32做rms，输出保持输入dtype）----
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_in = x.dtype
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = (x32 * rms).to(dtype_in)
        return y * self.weight.to(dtype_in)

def _make_rmsnorm(hidden_size: int, eps: float = 1e-6) -> nn.Module:
    return RMSNorm(hidden_size, eps)

class MoR_Config:
    """轻量配置容器：把 dict/nested dict 拍平到属性"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class MoR_LanguageModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, dict) else vars(cfg)

        def get(k, default=None):
            if k in self.cfg: return self.cfg[k]
            ad = self.cfg.get("adapters", {})
            if isinstance(ad, dict) and k in ad: return ad[k]
            return default

        vocab_size = int(get("vocab_size"))
        C          = int(get("hidden_size"))
        causal     = bool(get("causal", True))
        eps        = float(get("norm_eps", 1e-6))

        # --- Embedding ---
        self.embed = nn.Embedding(vocab_size, C)
        self.embed_tokens = self.embed  # 兼容旧名

        # --- 输出头：不做 tie-weights，避免早期 logits 过大 ---
        self.lm_head = nn.Linear(C, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # --- Final RMSNorm（保持稳定尺度；无额外缩放）---
        self.final_norm = _make_rmsnorm(C, eps)

        # --- 组装 Backbone（把需要的key传下去）---
        backbone_keys = [
            "hidden_size","num_hidden_layers","layers_per_expert_block",
            "num_attention_heads","num_key_value_heads",
            "mla_dv","mla_dpe","dropout_p","norm_eps","use_te_fp8",
            "moe_hidden_expansion","moe_num_experts","moe_top_k","moe_num_shared","moe_z_loss",
            "router_temperature","router_aux_loss_coeff","router_lb_coeff","router_entropy_coeff",
            "router_keep_rate_target","router_keep_policy","router_keep_penalty_coeff",
            "router_eps_explore","router_stochastic_quota",
            "pad_align","router_min_pack_tokens","max_pack_tokens","pack_num_chunks",
            "use_triton_glue","use_cudagraphs","compile_attn","ffn_pad_mode","timing_enable",
            "lora_depth_start","lora_num","lora_rank","lora_alpha",
            "lora_temperature","lora_gate_hidden",
            "lora_usage_target","lora_usage_coeff",
            "lora_gate_bias_nonzero_init","lora_gate_jitter_amp","lora_min_usage_warmup",
            "lora_apply_every_n_blocks",
        ]
        bcfg = {k: get(k) for k in backbone_keys if get(k) is not None}
        self.backbone = UnifiedBackbone(bcfg, causal=causal)

    # 便捷：方案A需要的冻结/解冻
    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.embed.parameters():    p.requires_grad = False
        for p in self.lm_head.parameters():  p.requires_grad = False

    def unfreeze_last_mor_layer(self):
        M   = getattr(self.backbone, "num_hidden_layers")
        LPB = getattr(self.backbone, "layers_per_mor")
        l0  = (M - 1) * LPB
        blocks = getattr(self.backbone, "blocks")
        for l in range(l0, l0 + LPB):
            for p in blocks[l].parameters():
                p.requires_grad = True
        if hasattr(self, "final_norm"):
            for p in self.final_norm.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)                       # 保持 bf16
        out = self.backbone(x, attention_mask=attention_mask)
        h  = out["y"]
        h  = self.final_norm(h)                         # 稳定尺度
        logits = self.lm_head(h)                        # 无额外缩放/退火

        return {
            "logits": logits,
            "loss": None,
            "_aux_for_train": out.get("_aux_for_train", {}),
            "loss_dict": out.get("loss_dict", {}),
            "stats": out.get("stats", {}),
        }
