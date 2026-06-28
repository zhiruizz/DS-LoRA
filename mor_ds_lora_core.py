"""Native MoR and DS-LoRA backbone.

This module intentionally contains only the paper-facing implementation:
Mixture-of-Recursions (MoR) plus optional Depth-Selective LoRA at selected
recursion depths, defaulting to the final recursion.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x32.to(dtype) * self.weight.to(dtype)


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dropout_p: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout_p = dropout_p
        self.causal = causal

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_mask = attention_mask[:, None, None, :].to(torch.bool)
            elif attention_mask.dim() == 4:
                attn_mask = attention_mask
            else:
                raise ValueError("attention_mask must be [B,S] or [B,1,S,S]")

        if self.causal:
            causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
            causal_mask = causal_mask[None, None, :, :]
            if attn_mask is None:
                attn_mask = causal_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask & causal_mask
            else:
                causal_bias = torch.zeros_like(attn_mask)
                causal_bias = causal_bias.masked_fill(~causal_mask, float("-inf"))
                attn_mask = attn_mask + causal_bias

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, expansion: float = 4.0, dropout_p: float = 0.0):
        super().__init__()
        inner = int(math.ceil(hidden_size * expansion / 16.0) * 16)
        self.gate_proj = nn.Linear(hidden_size, inner, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        ffn_expansion: float = 4.0,
        dropout_p: float = 0.0,
        norm_eps: float = 1e-6,
        causal: bool = True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, norm_eps)
        self.attn = SelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            dropout_p=dropout_p,
            causal=causal,
        )
        self.norm2 = RMSNorm(hidden_size, norm_eps)
        self.ffn = FeedForward(hidden_size, ffn_expansion, dropout_p)

    def forward_attn(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x + self.attn(self.norm1(x), attention_mask=attention_mask)

    def ffn_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(x)

    def ffn_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ffn_input(x))


class ContinueRouter(nn.Module):
    """MoR continue/exit router with quota routing."""

    def __init__(
        self,
        hidden_size: int,
        keep_rate_target: float = 0.55,
        temperature: float = 1.0,
        keep_policy: str = "quota",
        keep_penalty_coeff: float = 0.1,
        lb_coeff: float = 0.0,
        entropy_coeff: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)
        keep_rate_target = float(min(max(keep_rate_target, 1e-4), 1.0 - 1e-4))
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, math.log(keep_rate_target / (1.0 - keep_rate_target)))

        self.keep_rate_target = keep_rate_target
        self.temperature = float(max(temperature, 1e-4))
        self.keep_policy = keep_policy
        self.keep_penalty_coeff = float(keep_penalty_coeff)
        self.lb_coeff = float(lb_coeff)
        self.entropy_coeff = float(entropy_coeff)

    def forward(self, h: torch.Tensor, active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        logits = self.proj(h).squeeze(-1) / self.temperature
        p_keep = torch.sigmoid(logits)

        if self.keep_policy == "threshold":
            keep_mask = (p_keep >= 0.5) & active_mask
        else:
            keep_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            active_counts = active_mask.sum(dim=1)
            quotas = torch.round(active_counts.float() * self.keep_rate_target).long()
            for row in range(active_mask.size(0)):
                quota = int(quotas[row].item())
                if quota <= 0:
                    continue
                scores = p_keep[row].masked_fill(~active_mask[row], float("-inf"))
                quota = min(quota, int(active_counts[row].item()))
                if quota > 0:
                    keep_mask[row].scatter_(0, torch.topk(scores, quota).indices, True)

        if active_mask.any():
            active_probs = p_keep[active_mask]
            keep_rate = active_probs.mean()
            target = h.new_tensor(self.keep_rate_target)
            keep_loss = (keep_rate - target).pow(2) * self.keep_penalty_coeff
            per_row = (p_keep * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp_min(1)
            lb_loss = per_row.float().var(unbiased=False) * self.lb_coeff
            ent = -(active_probs * active_probs.clamp_min(1e-6).log()
                    + (1 - active_probs) * (1 - active_probs).clamp_min(1e-6).log()).mean()
            aux = keep_loss + lb_loss - self.entropy_coeff * ent
        else:
            aux = h.new_tensor(0.0)

        monitor = {
            "active_tokens": active_mask.sum(),
            "kept_tokens": keep_mask.sum(),
            "keep_rate": keep_mask.float().sum() / active_mask.float().sum().clamp_min(1.0),
        }
        return p_keep, keep_mask, {"router_aux_loss": aux, "monitor": monitor}


class LoRAAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 2, alpha: float = 1.0, dropout_p: float = 0.0):
        super().__init__()
        self.rank = int(rank)
        self.scaling = float(alpha) / max(1, self.rank)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.A = nn.Linear(hidden_size, self.rank, bias=False)
        self.B = nn.Linear(self.rank, hidden_size, bias=False)
        nn.init.normal_(self.A.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(self.dropout(x))) * self.scaling


class LoRAGate(nn.Module):
    """Top-1 gate over K LoRA adapters plus route 0 = no adapter."""

    def __init__(self, hidden_size: int, num_lora: int, hidden: int = 0, temperature: float = 1.0):
        super().__init__()
        gate_hidden = int(hidden) if int(hidden or 0) > 0 else max(32, hidden_size // 2)
        self.temperature = float(max(temperature, 1e-4))
        self.net = nn.Sequential(
            RMSNorm(hidden_size),
            nn.Linear(hidden_size, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, int(num_lora) + 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) / self.temperature


def _parse_lora_depths(raw_depths: Any, num_recursions: int, fallback_start: int) -> tuple[int, ...]:
    if raw_depths is None:
        start = max(0, min(int(fallback_start), num_recursions - 1))
        return tuple(range(start, num_recursions))

    if isinstance(raw_depths, int):
        values = [raw_depths]
    elif isinstance(raw_depths, str):
        text = raw_depths.strip()
        if not text:
            values = []
        else:
            values = [int(part.strip()) for part in text.split(",") if part.strip()]
    else:
        values = list(raw_depths)

    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        depth = int(value)
        if depth < 0:
            depth = num_recursions + depth
        if depth < 0 or depth >= num_recursions:
            raise ValueError(
                f"LoRA recursion depth {value} is out of range for num_hidden_layers={num_recursions}."
            )
        if depth not in seen:
            normalized.append(depth)
            seen.add(depth)
    return tuple(normalized)


class MoRBackbone(nn.Module):
    def __init__(self, cfg: Dict[str, Any], causal: bool = True):
        super().__init__()

        def get(key: str, default: Any = None) -> Any:
            if isinstance(cfg, dict):
                if key in cfg:
                    return cfg[key]
                for section in ("model_architecture", "router", "adapters", "training", "data_and_paths"):
                    nested = cfg.get(section, {})
                    if isinstance(nested, dict) and key in nested:
                        return nested[key]
                return default
            return getattr(cfg, key, default)

        hidden_size = int(get("hidden_size"))
        self.num_recursions = int(get("num_hidden_layers"))
        self.layers_per_recursion = int(get("layers_per_expert_block", 1))
        self.num_layers_phys = self.num_recursions * self.layers_per_recursion
        self.router_aux_scale = float(get("router_aux_loss_coeff", 0.05))

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=int(get("num_attention_heads")),
                num_key_value_heads=int(get("num_key_value_heads", get("num_attention_heads"))),
                ffn_expansion=float(get("ffn_hidden_expansion", 4.0)),
                dropout_p=float(get("dropout_p", 0.0)),
                norm_eps=float(get("norm_eps", 1e-6)),
                causal=causal,
            )
            for _ in range(self.num_layers_phys)
        ])

        self.routers = nn.ModuleList([
            ContinueRouter(
                hidden_size=hidden_size,
                keep_rate_target=float(get("router_keep_rate_target", 0.55)),
                temperature=float(get("router_temperature", 1.0)),
                keep_policy=str(get("router_keep_policy", "quota")),
                keep_penalty_coeff=float(get("router_keep_penalty_coeff", 0.1)),
                lb_coeff=float(get("router_lb_coeff", 0.0)),
                entropy_coeff=float(get("router_entropy_coeff", 0.0)),
            )
            for _ in range(self.num_recursions)
        ])

        self.lora_num = int(get("lora_num", 0))
        self.lora_rank = int(get("lora_rank", 2))
        self.lora_alpha = float(get("lora_alpha", 1.0))
        self.lora_usage_target = float(get("lora_usage_target", 0.10))
        self.lora_usage_coeff = float(get("lora_usage_coeff", 0.01))
        fallback_start = int(get("lora_depth_start", self.num_recursions - 1))
        self.lora_depths = _parse_lora_depths(get("lora_depths", None), self.num_recursions, fallback_start)
        self.lora_depth_set = set(self.lora_depths)

        self.lora_adapters = nn.ModuleList()
        self.lora_gates = nn.ModuleList()
        for depth in range(self.num_recursions):
            if self.lora_num > 0 and depth in self.lora_depth_set:
                self.lora_adapters.append(nn.ModuleList([
                    LoRAAdapter(
                        hidden_size,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout_p=float(get("dropout_p", 0.0)),
                    )
                    for _ in range(self.lora_num)
                ]))
                self.lora_gates.append(
                    LoRAGate(
                        hidden_size,
                        self.lora_num,
                        hidden=int(get("lora_gate_hidden", 0)),
                        temperature=float(get("lora_temperature", 1.0)),
                    )
                )
            else:
                self.lora_adapters.append(nn.ModuleList())
                self.lora_gates.append(nn.Identity())

    def _select_active(self, tensor: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden = tensor.shape
        idx = torch.nonzero(mask, as_tuple=False)
        if idx.numel() == 0:
            return tensor.new_empty((0, hidden)), tensor.new_empty((0,), dtype=torch.long)
        flat_idx = idx[:, 0] * seq_len + idx[:, 1]
        return tensor.reshape(bsz * seq_len, hidden).index_select(0, flat_idx), flat_idx

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        h = x
        bsz, seq_len, hidden = h.shape
        if attention_mask is None:
            active = torch.ones(bsz, seq_len, dtype=torch.bool, device=h.device)
        else:
            active = attention_mask.to(torch.bool)

        router_aux = h.new_tensor(0.0)
        lora_aux = h.new_tensor(0.0)
        active_token_counts: list[int] = []
        active_keep_rates: list[float] = []
        lora_usage_rates: list[float] = []

        for depth in range(self.num_recursions):
            keep_mask = active
            hard_lora: Optional[torch.Tensor] = None
            flat_keep_idx: Optional[torch.Tensor] = None

            for offset in range(self.layers_per_recursion):
                block = self.blocks[depth * self.layers_per_recursion + offset]
                h_attn = block.forward_attn(h, attention_mask=attention_mask)

                if offset == 0:
                    _, keep_mask, aux = self.routers[depth](h_attn, active)
                    router_aux = router_aux + aux["router_aux_loss"] * self.router_aux_scale
                    active_count = int(active.sum().item())
                    kept_count = int(keep_mask.sum().item())
                    active_token_counts.append(active_count)
                    active_keep_rates.append(0.0 if active_count == 0 else kept_count / float(active_count))

                    selected, flat_keep_idx = self._select_active(h_attn, keep_mask)
                    if selected.numel() > 0 and self.lora_num > 0 and len(self.lora_adapters[depth]) > 0:
                        logits = self.lora_gates[depth](selected)
                        hard_lora = torch.argmax(torch.softmax(logits.float(), dim=-1), dim=-1)
                        usage = (hard_lora != 0).float().mean()
                        lora_aux = lora_aux + (usage - self.lora_usage_target).pow(2) * self.lora_usage_coeff
                        lora_usage_rates.append(float(usage.detach().cpu()))
                    else:
                        hard_lora = None
                        lora_usage_rates.append(0.0)

                if flat_keep_idx is None or flat_keep_idx.numel() == 0:
                    h = h_attn
                    continue

                selected = h_attn.reshape(bsz * seq_len, hidden).index_select(0, flat_keep_idx)
                ffn_in = block.ffn_input(selected)
                delta = block.ffn(ffn_in)

                if hard_lora is not None and hard_lora.numel() == selected.size(0):
                    delta = delta.clone()
                    for adapter_idx, adapter in enumerate(self.lora_adapters[depth], start=1):
                        pos = torch.nonzero(hard_lora == adapter_idx, as_tuple=False).squeeze(-1)
                        if pos.numel() > 0:
                            delta[pos] = delta[pos] + adapter(ffn_in.index_select(0, pos))

                flat = h_attn.reshape(bsz * seq_len, hidden).clone()
                flat.index_add_(0, flat_keep_idx, delta)
                h = flat.view(bsz, seq_len, hidden)

            active = keep_mask

        aux_total = router_aux + lora_aux
        loss_dict = {
            "router_aux_loss_weighted": aux_total.detach(),
            "router_aux_loss_scalar": aux_total.detach(),
            "active_token_counts": torch.tensor(active_token_counts, device=h.device),
            "active_keep_rates": torch.tensor(active_keep_rates, device=h.device, dtype=torch.float32),
            "lora_adapter_usage_rates": torch.tensor(lora_usage_rates, device=h.device, dtype=torch.float32),
        }
        stats = {
            "active_token_counts": active_token_counts,
            "active_keep_rates": active_keep_rates,
            "lora_adapter_usage_rates": lora_usage_rates,
            "lora_depths": list(self.lora_depths),
            "depth_used": self.num_recursions,
        }
        return {
            "y": h,
            "_aux_for_train": {"router_aux": aux_total},
            "loss_dict": loss_dict,
            "stats": stats,
        }
