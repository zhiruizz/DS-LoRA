# file: mla_moe_block.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fp8_utils import fp8_autocast_if_available, make_linear, has_te
from torch.utils.checkpoint import checkpoint

# 可选：安装并导入 flash-mla（仅用于解码快路径）
try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    _HAS_FLASH_MLA = True
except Exception:
    _HAS_FLASH_MLA = False

try:
    from torch._dynamo import disable as dynamo_disable
except Exception:
    def dynamo_disable(fn):  # 安全兜底
        return fn

# ---------------------------
# 1) 基础组件：RMSNorm
# ---------------------------
class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm（无均值中心化，无 bias）。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, C]
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        x = x.to(orig_dtype)
        return x * self.weight


# ---------------------------
# 2) DeepSeekMoE（复刻+工程化）
# ---------------------------
def _round_up(x: int, multiple: int) -> int:
    r = x % multiple
    return x if r == 0 else x + (multiple - r)

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer=nn.GELU,
                 use_te_fp8: bool = False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = _round_up(int(hidden_features), 16)  # 末维 16 对齐

        self.act = act_layer()
        # 用 TE Linear，但“**不启 FP8**”，让它走 cuBLASLt 的 BF16 快路径
        self.fc1 = make_linear(in_features, hidden_features, bias=True, use_te=True)
        self.fc2 = make_linear(hidden_features, out_features, bias=True, use_te=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DeepSeekMoE(nn.Module):
    """
    - 路由专家 + 共享专家
    - top-k 路由（确定性）
    - 负载均衡 + z-loss（可调系数）
    输入支持 [B,S,C] 或 [T,C]；返回 (y, aux_loss)
    """

    def __init__(
            self,
            dim: int,
            num_experts: int,
            top_k: int,
            num_shared_experts: int = 1,
            hidden_expansion: float = 2.0,
            z_loss_coeff: float = 1e-3,
            use_te_fp8: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.num_shared_experts = num_shared_experts
        self.z_loss_coeff = z_loss_coeff
        self.use_te_fp8 = use_te_fp8

        # gate 也很大，吃 TE
        self.gate = make_linear(dim, num_experts, bias=True, use_te=self.use_te_fp8)

        # 隐藏维对齐到 16，满足 TE 末维约束
        hidden = ((int(dim * hidden_expansion) + 15) // 16) * 16

        # 用 TE 线性（支持 FP8）
        self.routed_experts = nn.ModuleList(
            [Mlp(dim, hidden, use_te_fp8=self.use_te_fp8) for _ in range(num_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [Mlp(dim, hidden, use_te_fp8=self.use_te_fp8) for _ in range(num_shared_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() in (2, 3), f"DeepSeekMoE expects 2D or 3D, got {tuple(x.shape)}"
        dtype = x.dtype
        device = x.device

        if x.dim() == 3:
            B, S, C = x.shape
            X = x.reshape(-1, C)  # [T, C]
            restore_3d = True
            T = B * S
        else:
            T, C = x.shape
            X = x
            restore_3d = False

        E = self.num_experts
        k = self.top_k

        # 全局 FP8 条件（对 gate / shared_experts 生效；线性层已用 nn.Linear 并不依赖 TE）
        fp8_ok_global = self.use_te_fp8 and (T % 8 == 0)

        # 共享专家（整批）
        shared = 0
        with fp8_autocast_if_available(enabled=fp8_ok_global):
            for expert in self.shared_experts:
                shared = shared + expert.fc2(expert.act(expert.fc1(X)))

        # 门控
        logits = self.gate(X)
        logits = torch.clamp(logits, -10.0, 10.0)
        probs = torch.softmax(logits.float(), dim=-1).to(dtype)  # [T,E]

        # top-k 选择
        topk_val, topk_idx = torch.topk(probs, k, dim=-1, largest=True, sorted=False)  # [T,k],[T,k]

        # 负载均衡 + z-loss
        with torch.no_grad():
            counts = torch.bincount(topk_idx.reshape(-1), minlength=E).to(probs.dtype)
            f_i = counts / max(1, (T * k))  # [E]
        P_i = probs.mean(dim=0)  # [E]
        load_balance = E * (f_i * P_i).sum()

        log_z = torch.logsumexp(logits.float(), dim=-1)  # [T]
        z_loss = (log_z ** 2).mean()
        aux_loss = load_balance + self.z_loss_coeff * z_loss

        # --------- 聚合向量化路径（k == 1，开启 TE FP8 时做行数对齐）---------
        if k == 1:
            exp_ids = topk_idx.view(-1)  # [T]
            gates = topk_val.view(-1, 1)  # [T,1]

            order = torch.argsort(exp_ids, stable=True)
            exp_sorted = exp_ids[order]
            X_sorted = X[order]
            gates_sorted = gates[order]

            unique_exp, counts = torch.unique_consecutive(exp_sorted, return_counts=True)
            starts = torch.cumsum(torch.cat([torch.tensor([0], device=device, dtype=torch.long), counts[:-1]]), dim=0)
            ends = starts + counts

            out_sorted = torch.zeros_like(X_sorted)

            for e, st, ed in zip(unique_exp.tolist(), starts.tolist(), ends.tolist()):
                if ed <= st:
                    continue
                xs = X_sorted[st:ed]  # [Me, C]
                gs = gates_sorted[st:ed]  # [Me, 1]
                Me = xs.shape[0]

                # —— 行数补齐到 8 的倍数（只在 TE FP8 打开的情况下做；否则直接算）——
                use_te = self.use_te_fp8
                if use_te:
                    pad = (8 - (Me % 8)) % 8
                else:
                    pad = 0

                if pad:
                    pad_x = torch.zeros(pad, xs.size(1), dtype=xs.dtype, device=xs.device)
                    xs_pad = torch.cat([xs, pad_x], dim=0)  # [Me+pad, C]
                else:
                    xs_pad = xs

                # TE FP8 上下文（输入行数已补齐，末维已在 __init__ 对齐到 16）
                with fp8_autocast_if_available(enabled=use_te):
                    mlp: Mlp = self.routed_experts[e]
                    y_pad = mlp.fc2(mlp.act(mlp.fc1(xs_pad)))  # [Me+pad, C]

                y = y_pad[:Me]  # 去掉 padding
                out_sorted[st:ed] = (gs * y).to(out_sorted.dtype)

            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(T, device=device)
            out = out_sorted[inv_order]

            Y = (shared + out).to(dtype)
            if restore_3d:
                Y = Y.view(B, S, C)
            return Y, aux_loss
        # --------- 通用路径（k > 1）---------
        # 保持原有、稳定的实现（逐专家段落 index_add_）
        out = torch.zeros_like(X)

        flat_exp = topk_idx.reshape(-1)  # [T*k]
        flat_tok = torch.arange(T, device=device).repeat_interleave(k)
        flat_gate = topk_val.reshape(-1, 1)  # [T*k,1]

        sort_key = torch.stack([flat_exp, flat_tok], dim=1)
        order = torch.argsort(sort_key[:, 0], stable=True)
        flat_exp, flat_tok, flat_gate = flat_exp[order], flat_tok[order], flat_gate[order]

        ptr = 0
        for e in range(E):
            while ptr < flat_exp.numel() and flat_exp[ptr] < e:
                ptr += 1
            if ptr >= flat_exp.numel() or flat_exp[ptr] != e:
                continue
            start = ptr
            while ptr < flat_exp.numel() and flat_exp[ptr] == e:
                ptr += 1
            end = ptr

            tok = flat_tok[start:end].long()
            gate = flat_gate[start:end]
            if tok.numel() == 0:
                continue

            y_e = self.routed_experts[e].fc2(self.routed_experts[e].act(self.routed_experts[e].fc1(X[tok])))
            out.index_add_(0, tok, (gate * y_e).to(out.dtype))

        Y = (shared + out).to(dtype)
        if restore_3d:
            Y = Y.view(B, S, C)
        return Y, aux_loss


# ---------------------------
# 3) MLA 注意力（稠密训练 + 可选解码快路径）
# ---------------------------
class MLAAttention(nn.Module):
    """
    稠密 MLA 训练路径（支持反向）+ 可选 flash-mla 解码快路径（仅解码、无反传）。
    - 输入 hidden_states: [B, S, C]
    - 输出 same shape
    - 头设定：num_heads=h_q, num_kv_heads=h_kv
    - MLA 维度：d = dv + d_pe（q/k 拆分为 [dv]内容 + [d_pe]位置部件）
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        dv: int,
        d_pe: int,
        bias: bool = False,
        causal: bool = True,
        use_te_fp8: bool = False,
    ):
        super().__init__()
        assert dv > 0 and d_pe > 0, "MLA requires dv > 0 and d_pe > 0"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dv = dv
        self.d_pe = d_pe
        self.d = dv + d_pe
        self.head_dim = self.d
        self.causal = causal
        self.use_te_fp8 = use_te_fp8

        # 线性映射（使用 TE 线性以支持 FP8）
        self.q_proj = make_linear(hidden_size, num_heads * self.d, bias=bias, use_te=self.use_te_fp8)
        self.k_proj = make_linear(hidden_size, num_kv_heads * self.d, bias=bias, use_te=self.use_te_fp8)
        self.v_proj = make_linear(hidden_size, num_kv_heads * self.dv, bias=bias, use_te=self.use_te_fp8)
        self.o_proj = make_linear(num_heads * self.dv, hidden_size, bias=bias, use_te=self.use_te_fp8)

        self.scale = 1.0 / math.sqrt(self.d)

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return self.q_proj(x).view(B, S, self.num_heads, self.d)

    def _shape_k(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return self.k_proj(x).view(B, S, self.num_kv_heads, self.d)

    def _shape_v(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        return self.v_proj(x).view(B, S, self.num_kv_heads, self.dv)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B,S,C]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, C = hidden_states.shape
        dtype = hidden_states.dtype

        # —— 当 (B*S % 8 == 0) 时才启用 FP8 —— #
        fp8_ok = self.use_te_fp8 and ((B * S) % 8 == 0)

        # QKV 线性投影在 FP8 autocast 内完成（GEMM），随后 softmax 等在 fp32
        with fp8_autocast_if_available(enabled=fp8_ok):
            q_lin = self.q_proj(hidden_states)  # [B,S,Hq*d]
            k_lin = self.k_proj(hidden_states)  # [B,S,Hkv*d]
            v_lin = self.v_proj(hidden_states)  # [B,S,Hkv*dv]
        q = q_lin.view(B, S, self.num_heads, self.d)
        k = k_lin.view(B, S, self.num_kv_heads, self.d)
        v = v_lin.view(B, S, self.num_kv_heads, self.dv)

        # GQA/MQA 展开
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for GQA")
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=2)  # [B,S,Hq,d]
        v = v.repeat_interleave(repeat, dim=2)  # [B,S,Hq,dv]

        # 拆 dv/d_pe
        q_nope, q_pe = q[..., :self.dv], q[..., self.dv:]  # [B,S,Hq,dv], [B,S,Hq,dpe]
        k_c, k_pe = k[..., :self.dv], k[..., self.dv:]    # [B,S,Hq,dv], [B,S,Hq,dpe]

        # logits = (q_nope@k_c^T + q_pe@k_pe^T) * scale
        q_nope_f = q_nope.float().transpose(1, 2)  # [B,H,S,dv]
        q_pe_f   = q_pe.float().transpose(1, 2)    # [B,H,S,dpe]
        k_c_f    = k_c.float().transpose(1, 2).transpose(-2, -1)  # [B,H,dv,S]
        k_pe_f   = k_pe.float().transpose(1, 2).transpose(-2, -1) # [B,H,dpe,S]

        logits = (q_nope_f @ k_c_f) + (q_pe_f @ k_pe_f)  # [B,H,S,S]
        logits = logits * self.scale

        # mask
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                logits = logits + attention_mask.to(logits.dtype)
            elif attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(torch.bool)  # [B,1,1,S]
                add = torch.full_like(logits, float("-inf"))
                logits = torch.where(mask, logits, add)
            else:
                raise ValueError("Unsupported attention_mask shape")

        if self.causal:
            causal = torch.ones(S, S, device=hidden_states.device, dtype=torch.bool).tril()
            add = torch.full_like(logits, float("-inf"))
            logits = torch.where(causal, logits, add)

        attn = torch.softmax(logits, dim=-1, dtype=torch.float32)  # [B,H,S,S]
        v_f = v.float().transpose(1, 2)  # [B,H,S,dv]
        out = attn @ v_f                 # [B,H,S,dv]
        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.dv)
        out = out.to(dtype)

        # 输出投影
        with fp8_autocast_if_available(enabled=fp8_ok):
            return self.o_proj(out)

    @torch.inference_mode()
    def decode_with_flash_mla(
        self,
        q: torch.Tensor,  # [B, S_q(=1 for decode), Hq, d]
        blocked_k: torch.Tensor,  # [num_blocks, block_size, Hkv, d] (paged KV)
        block_table: torch.Tensor,  # [B, max_seqlen_pad // block_size] (int32)
        cache_seqlens: torch.Tensor,  # [B] (int32)
        block_size: int,  # e.g., 64
    ) -> torch.Tensor:
        if not _HAS_FLASH_MLA:
            raise RuntimeError("flash_mla not available. Please install flash-mla to use this path.")

        B, S_q, Hq, d = q.shape
        assert d == (self.dv + self.d_pe), "q head dim must match dv + d_pe"
        Hkv = blocked_k.shape[-2]
        assert Hq % Hkv == 0, "Hq must be divisible by Hkv"

        tile_meta, num_splits = get_mla_metadata(cache_seqlens, S_q * Hq // Hkv, Hkv)
        out, _lse = flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, self.dv,
            tile_meta, num_splits, causal=True,
        )
        out = out.reshape(B, S_q, Hq * self.dv)
        return self.o_proj(out)


# ---------------------------
# 4) 完整 Transformer Block
# ---------------------------
class MLAMoEBlock(nn.Module):
    """
    RMSNorm → MLA → 残差 → RMSNorm → DeepSeekMoE → 残差
    训练与推理：forward() 走稠密 MLA；解码时可选手动调用 attn.decode_with_flash_mla()
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        dv: int,
        d_pe: int,
        moe_num_experts: int,
        moe_top_k: int,
        moe_num_shared: int = 1,
        attn_bias: bool = False,
        causal: bool = True,
        norm_eps: float = 1e-6,
        moe_z_loss: float = 1e-3,
        moe_hidden_expansion: float = 2.0,   # 和你的配置一致
        dropout_p: float = 0.0,
        use_te_fp8: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MLAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dv=dv,
            d_pe=d_pe,
            bias=attn_bias,
            causal=causal,
            use_te_fp8=use_te_fp8,
        )
        self.attn_dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.moe = DeepSeekMoE(
            dim=hidden_size,
            num_experts=moe_num_experts,
            top_k=moe_top_k,
            num_shared_experts=moe_num_shared,
            hidden_expansion=moe_hidden_expansion,
            z_loss_coeff=moe_z_loss,
            use_te_fp8=use_te_fp8,
        )
        self.ffn_dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def set_fp8(self, attn: bool, ffn: bool):
        """
        独立开关：注意力用 FP8 与否、FFN 用 FP8 与否。
        不改对象构造签名；默认沿用 self.attn.use_te_fp8 / self.moe.use_te_fp8。
        """
        attn_fp8 = bool(attn)
        ffn_fp8 = bool(ffn)
        # 注意力子模块（QKV / OProj 的 TE 线性会读取这个开关）
        if hasattr(self, "attn"):
            self.attn.use_te_fp8 = attn_fp8
        # FFN / MoE 子模块（其内部会根据 use_te_fp8 决定是否启 FP8 量化路径）
        if hasattr(self, "moe"):
            self.moe.use_te_fp8 = ffn_fp8

    def forward(
            self,
            hidden_states: torch.Tensor,  # [B, S, C]
            attention_mask: Optional[torch.Tensor] = None,
            return_aux: bool = True,
    ):
        # Block 1: RMSNorm → MLA → 残差（保留 checkpoint）
        x = hidden_states
        x_norm = self.norm1(x)

        def _attn_fn(a: torch.Tensor) -> torch.Tensor:
            return self.attn(a, attention_mask=attention_mask)

        attn_out = checkpoint(_attn_fn, x_norm, use_reentrant=False)
        x = x + self.attn_dropout(attn_out)

        # Block 2: RMSNorm → MoE → 残差（去掉 checkpoint，直接前向，避免重算时路由形变）
        x_norm = self.norm2(x)
        moe_out, aux_loss = self.moe(x_norm)
        x = x + self.ffn_dropout(moe_out)

        if return_aux:
            return x, {"moe_aux_loss": aux_loss}
        return x

    @dynamo_disable
    def forward_attn(self, x, attention_mask=None):
        """
        Block 的注意力半段：RMSNorm → MLA → Dropout → Residual
        返回值：加完残差后的张量，形状 [B,S,C]
        """
        h = x
        h_norm = self.norm1(h)  # [B,S,C]
        attn_out = self.attn(h_norm, attention_mask=attention_mask)  # [B,S,C]
        attn_out = self.attn_dropout(attn_out)
        y = h + attn_out  # 残差相加
        return y

    @dynamo_disable
    def forward_ffn(self, x, return_aux: bool = False):
        """
        Block 的 FFN 半段：RMSNorm → MoE(或MLP) → Dropout
        返回值：**增量**（delta），形状与 x 一致；不在这里做残差，便于上层 pack/scatter 写回。
        """
        h = x
        h_norm = self.norm2(h)  # [*,*,C] 或 [M_pad,1,C]
        # DeepSeekMoE.forward 返回 (y, aux_loss)
        y, aux = self.moe(h_norm)  # y 为 FFN 输出
        y = self.ffn_dropout(y)  # 仍不加残差
        if return_aux:
            return y, {"moe_aux_loss": aux}
        return y
