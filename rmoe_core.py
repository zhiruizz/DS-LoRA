# rmoe_core.py
# Attn 全局 + FFN 分流 的 RMoE 骨干；内嵌 (K+1)-way softmax 路由
from typing import Any, Callable, Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
from types import SimpleNamespace
from mla_moe_block import MLAMoEBlock  # 用作“共享注意力块”（只调 forward_attn）

def _make_event():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# ====== Triton kernels (autotune) ======
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

# -------- autotune configs --------
_gather_confs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_C': 128}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_C': 256}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_C': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_C': 256}, num_warps=8, num_stages=4),
]
_scatter_confs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_C': 128}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_C': 256}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_C': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 256, 'BLOCK_C': 256}, num_warps=8, num_stages=4),
]

@triton.autotune(configs=_gather_confs, key=['M', 'C'])
@triton.jit
def _ker_gather_pack(
    x_ptr, idx_ptr, out_ptr,
    T, C, M, M_PAD,
    STRIDE_XR, STRIDE_XC, STRIDE_OUTR, STRIDE_OUTC,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_c = tl.arange(0, BLOCK_C)                      # [BLOCK_C]

    mask_m = offs_m < M
    idx_m  = tl.load(idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)  # [BLOCK_M]

    # base row pointer: [BLOCK_M, 1]
    row_ptr = x_ptr + idx_m[:, None] * STRIDE_XR
    out_row = out_ptr + offs_m[:, None] * STRIDE_OUTR

    for c0 in range(0, C, BLOCK_C):
        c = c0 + offs_c                               # [BLOCK_C]
        mask_c = c < C
        # 2D pointers
        src = row_ptr + c[None, :] * STRIDE_XC        # [BLOCK_M, BLOCK_C]
        dst = out_row + c[None, :] * STRIDE_OUTC      # [BLOCK_M, BLOCK_C]
        m   = (mask_m[:, None] & mask_c[None, :])     # [BLOCK_M, BLOCK_C]
        vals = tl.load(src, mask=m, other=0.0)
        tl.store(dst, vals, mask=m)

@triton.autotune(configs=_scatter_confs, key=['M', 'C'])
@triton.jit
def _ker_scatter_add(
    base_ptr, idx_ptr, src_ptr,
    T, C, M,
    STRIDE_BR, STRIDE_BC, STRIDE_SR, STRIDE_SC,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_c = tl.arange(0, BLOCK_C)                      # [BLOCK_C]

    mask_m = offs_m < M
    idx_m  = tl.load(idx_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)

    base_row = base_ptr + idx_m[:, None] * STRIDE_BR    # [BLOCK_M, 1]
    src_row  =  src_ptr + offs_m[:, None] * STRIDE_SR   # [BLOCK_M, 1]

    for c0 in range(0, C, BLOCK_C):
        c = c0 + offs_c
        mask_c = c < C
        bptr = base_row + c[None, :] * STRIDE_BC        # [BLOCK_M, BLOCK_C]
        sptr =  src_row  + c[None, :] * STRIDE_SC       # [BLOCK_M, BLOCK_C]
        m    = (mask_m[:, None] & mask_c[None, :])
        vals = tl.load(sptr, mask=m, other=0.0)
        tl.atomic_add(bptr, vals, mask=m)

def _triton_gather_pack(x_flat: torch.Tensor, idx_flat: torch.Tensor, M_pad: int) -> torch.Tensor:
    if (not _TRITON_AVAILABLE) or (x_flat.numel() == 0):
        M = idx_flat.numel()
        y = x_flat.index_select(0, idx_flat.to(x_flat.device))
        if M_pad > M:
            pad = torch.zeros((M_pad - M, x_flat.size(1)), dtype=x_flat.dtype, device=x_flat.device)
            y = torch.cat([y, pad], dim=0)
        return y
    T, C = x_flat.shape
    M    = int(idx_flat.numel())
    out  = torch.zeros((M_pad, C), dtype=x_flat.dtype, device=x_flat.device)
    grid = (triton.cdiv(M, 128),)
    _ker_gather_pack[grid](
        x_flat, idx_flat.to(torch.int32), out,
        T, C, M, M_pad,
        x_flat.stride(0), x_flat.stride(1), out.stride(0), out.stride(1),
    )
    return out

def _triton_scatter_add(base_flat: torch.Tensor, idx_flat: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if (not _TRITON_AVAILABLE) or (src.numel() == 0):
        base_flat.index_add_(0, idx_flat.to(base_flat.device), src)
        return base_flat
    T, C = base_flat.shape
    M    = int(idx_flat.numel())
    grid = (triton.cdiv(M, 128),)
    _ker_scatter_add[grid](
        base_flat, idx_flat.to(torch.int32), src,
        T, C, M,
        base_flat.stride(0), base_flat.stride(1), src.stride(0), src.stride(1),
    )
    return base_flat

# -------- RMSNorm 兼容层（老版 PyTorch 没有 nn.RMSNorm 时使用） --------
class _RMSNormFallback(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def _make_rmsnorm(dim: int, eps: float):
    return nn.RMSNorm(dim, eps=eps) if hasattr(nn, "RMSNorm") else _RMSNormFallback(dim, eps)

class ContinueRouter(nn.Module):
    def __init__(self, hidden_size,
                 temperature=1.0, keep_rate_target=0.8,
                 lb_coeff=1.0, entropy_coeff=0.0, keep_penalty_coeff=0.10,
                 keep_policy: str = "quota",
                 eps_explore: float = 0.0, stochastic_quota: bool = False):
        """
        MoR 风格：只决定继续/退出（p_keep），不做专家选择。
        """
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True)
        keep_tgt = float(keep_rate_target)
        bias_init = math.log(keep_tgt / max(1e-6, 1.0 - keep_tgt))
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.bias.fill_(bias_init)

        self.temperature = float(max(temperature, 1e-4))
        self.keep_tgt = keep_tgt
        self.lb_coeff = float(lb_coeff)
        self.entropy_coeff = float(entropy_coeff)
        self.keep_penalty_coeff = float(keep_penalty_coeff)
        self.keep_policy = keep_policy            # "quota" / "threshold"
        self.eps_explore = float(max(0.0, min(1.0, eps_explore)))
        self.stochastic_quota = bool(stochastic_quota)

    def forward(self, h_attn: torch.Tensor, active_mask: torch.Tensor, depth: int, max_depth: int):
        B, S, C = h_attn.shape
        device = h_attn.device
        dtype = h_attn.dtype

        logits = self.proj(h_attn).squeeze(-1) / self.temperature  # [B,S]
        p_keep = torch.sigmoid(logits)

        # --- keep mask ---
        if self.keep_policy == "quota":
            act = active_mask
            A_b = act.sum(dim=1)  # [B]
            k_b = torch.clamp((self.keep_tgt * A_b.float()).round().to(torch.int64), min=0)  # [B]
            k_max = int(k_b.max().item()) if B > 0 else 0
            if k_max > 0:
                scores = p_keep.masked_fill(~act, float("-inf"))

                # ↓↓↓ 修复点：避免 in-place 随机；捕获期退化为确定性抖动 ↓↓↓
                if self.stochastic_quota and self.training:
                    is_capturing = torch.cuda.is_current_stream_capturing()
                    if not is_capturing:
                        # capture-safe Gumbel: u ~ U(0,1); g = -log(-log(u))
                        u = torch.rand_like(scores)
                        g = -torch.log(-torch.log(u.clamp_min(1e-6)))
                        scores = scores + g
                    else:
                        # deterministic tiny jitter（纯算子，无 RNG，无梯度）
                        with torch.no_grad():
                            B_, S_ = scores.shape
                            i = torch.arange(B_, device=scores.device).view(B_, 1).float()
                            j = torch.arange(S_, device=scores.device).view(1, S_).float()
                            jitter = torch.frac(torch.sin(i * 12.9898 + j * 78.233 + (depth + 1) * 37.719))
                            jitter = (jitter - 0.5) * 1e-3  # 极小扰动，避免并列
                        scores = scores + jitter

                topv, topi = torch.topk(scores, k=k_max, dim=1, largest=True, sorted=False)
                BIG = S + 1
                rank = torch.full((B, S), BIG, device=scores.device, dtype=torch.int32)
                r = torch.arange(k_max, device=scores.device, dtype=torch.int32)[None, :].expand(B, -1)
                rank.scatter_(1, topi, r)
                keep_mask = (rank < k_b.view(-1, 1).to(rank.dtype)) & act
            else:
                keep_mask = torch.zeros_like(active_mask, dtype=torch.bool)
        else:
            keep_mask = (p_keep >= 0.5) & active_mask
            if self.eps_explore > 0.0:
                rows = (torch.rand(B, device=device) < self.eps_explore) & active_mask.any(dim=1)
                if rows.any():
                    ixs = rows.nonzero(as_tuple=False).squeeze(1)
                    for i in ixs.tolist():
                        act_i = active_mask[i].nonzero(as_tuple=False).squeeze(1)
                        add = max(1, int(0.1 * act_i.numel()))
                        if add > 0 and act_i.numel() > 0:
                            perm = torch.randperm(act_i.numel(), device=device)
                            keep_mask[i, act_i[perm[:add]]] = True

        # --- aux loss（keep 目标 / 方差均衡 / 熵正则）---
        with torch.amp.autocast("cuda", enabled=False):
            if active_mask.any():
                keep_rate = (p_keep[active_mask.bool()].mean().float())
                delta = 0.02
                diff = keep_rate - self.keep_tgt
                huber = torch.where(diff.abs() < delta, 0.5 * diff * diff / delta, diff.abs() - 0.5 * delta)

                per_b_keep = (p_keep * active_mask).sum(dim=1) / active_mask.sum(dim=1).clamp_min(1)
                var_keep = torch.var(per_b_keep.float(), unbiased=False)

                ent = -(p_keep * torch.log(p_keep.clamp_min(1e-6)) + (1 - p_keep) * torch.log((1 - p_keep).clamp_min(1e-6)))
                ent = (ent * active_mask).sum() / active_mask.sum().clamp_min(1)

                aux = ( self.keep_penalty_coeff * huber
                        + self.lb_coeff * var_keep
                        - self.entropy_coeff * ent ).to(dtype)
            else:
                aux = h_attn.new_tensor(0.0, dtype=dtype)

        # --- 监控 ---
        with torch.no_grad():
            if active_mask.any():
                hist = torch.histc(p_keep[active_mask.bool()].float(), bins=10, min=0.0, max=1.0)
                hist = (hist / hist.sum().clamp_min(1)).to(dtype)
            else:
                hist = torch.zeros(10, device=device, dtype=dtype)

        monitor = {
            "active_tokens": active_mask.sum(),
            "kept_tokens": keep_mask.sum(),
            "keep_entropy": (- (p_keep.clamp_min(1e-6) * torch.log(p_keep.clamp_min(1e-6))
                              + (1 - p_keep).clamp_min(1e-6) * torch.log((1 - p_keep).clamp_min(1e-6)))).mean(),
            "pkeep_hist_10": hist
        }

        return p_keep, {"router_aux_loss": aux, "monitor": monitor}, keep_mask

class LoRAFFNAdapter(nn.Module):
    def __init__(self, hidden_size, rank=4, eps=1e-6, dropout_p=0.0, alpha=1.0):
        super().__init__()
        self.r = rank
        self.alpha = alpha
        self.scaling = alpha / max(1, rank)

        self.A = nn.Linear(hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, hidden_size, bias=False)
        # ★ 关键：B 零初始化，A 用小 std
        nn.init.normal_(self.A.weight, mean=0.0, std=1e-3)  # 小 std
        nn.init.zeros_(self.B.weight)  # ★ 零初始化，确保初始增量=0
        self.scaling = self.alpha / max(1, self.r)  # 仍然有个小缩放

        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.norm = nn.Identity()  # 或者 RMSNorm/LayerNorm，按你的实现

    def forward(self, x):
        # [M_pad, 1, C] or [M_pad, C] 都行，保持和主 FFN 的接口一致
        if x.dim() == 3: x = x.squeeze(1)
        delta = self.B(self.A(self.drop(x))) * self.scaling
        return delta.unsqueeze(1)  # 保持和主 FFN delta 的 shape 对齐

class LoRAGate(nn.Module):
    """
    K+1 路门控：第 0 路=不用 LoRA，1..K=各 LoRA 适配器
    """
    def __init__(self, hidden_size: int, num_lora: int, hidden: int = 0, temperature: float = 1.0,
                 bias_nonzero_init: float = 0.0):
        super().__init__()
        self.K = int(num_lora)
        self.temperature = float(max(temperature, 1e-4))
        h = int(hidden) if (isinstance(hidden, int) and hidden > 0) else max(32, hidden_size // 2)
        self.net = nn.Sequential(
            _make_rmsnorm(hidden_size, 1e-6),
            nn.Linear(hidden_size, h, bias=True),
            nn.SiLU(),
            nn.Linear(h, self.K + 1, bias=True),
        )
        if bias_nonzero_init != 0.0:
            with torch.no_grad():
                # 适度压低 0 路、抬高 1..K，帮助早期摆脱“全 0 路”
                self.net[-1].bias[0].add_(-float(bias_nonzero_init))
                self.net[-1].bias[1:].add_( float(bias_nonzero_init))
                # 也可根据需要把权重做个很小的正初始化：
                # nn.init.normal_(self.net[-1].weight[1:], mean=0.0, std=1e-3)

    def forward(self, x_tok: torch.Tensor) -> torch.Tensor:
        return self.net(x_tok)  # logits（不在这里除温度）

# ------------------------ RMoE Backbone: Attn 全局 + FFN 分流 ------------------------
class UnifiedBackbone(nn.Module):
    def __init__(self, cfg: Dict[str, Any], causal: bool = True):
        super().__init__()
        import os
        get = (lambda k, d=None: (cfg.get(k, d) if isinstance(cfg, dict) else getattr(cfg, k, d)))
        self.cfg = SimpleNamespace(**cfg) if isinstance(cfg, dict) else cfg

        # -------- 基本超参（M=递归层数，LPB=每层 blocks，L=M*LPB）--------
        C = int(get("hidden_size"))
        M = int(get("num_hidden_layers"))  # MoR 递归层数（深度）
        LPB = int(get("layers_per_expert_block", 1))  # 每个递归层的物理 blocks 数
        assert M >= 1, "num_hidden_layers (MoR depth) must be >= 1"
        assert LPB >= 1, "layers_per_expert_block must be >= 1"
        L = M * LPB  # 物理 Transformer blocks 总数

        Hq = int(get("num_attention_heads"))
        Hkv = int(get("num_key_value_heads"))
        dv = int(get("mla_dv"))
        dpe = int(get("mla_dpe"))
        drop_p = float(get("dropout_p", 0.0))
        eps = float(get("norm_eps", 1e-6))
        use_te_fp8 = bool(get("use_te_fp8", False))

        # MoE/FFN（与 MLAMoEBlock 兼容）
        E_ffn = int(get("moe_num_experts", 1))
        topk_ffn = int(get("moe_top_k", 1))
        shared_ffn = int(get("moe_num_shared", 0))
        moe_z = float(get("moe_z_loss", 1e-3))
        moe_exp = float(get("moe_hidden_expansion", 2.0))

        # -------- Router（每个递归层一个）--------
        r_temp = float(get("router_temperature", 1.0))
        r_aux = float(get("router_aux_loss_coeff", get("aux_loss_coeff", 0.01)))
        r_lb = float(get("router_lb_coeff", 1.0))
        r_ent = float(get("router_entropy_coeff", 0.0))
        r_keep = float(get("router_keep_rate_target", get("router_keep_rate", 1.0)))
        r_keep_pen = float(get("router_keep_penalty_coeff", 0.0))
        r_policy = str(get("router_keep_policy", "quota"))
        r_eps_exp = float(get("router_eps_explore", 0.0))
        r_stoch_q = bool(get("router_stochastic_quota", False))

        # -------- pack / CUDAGraphs / Triton 粘合 --------
        self._pad_align = int(get("pad_align", 128))
        self._min_pack_tokens = int(get("router_min_pack_tokens", 8192))
        self.max_pack_tokens = int(get("max_pack_tokens", 0))
        self._num_chunks = int(get("pack_num_chunks", 1) or 1)
        self._use_cudagraphs = bool(get("use_cudagraphs", False))
        self._use_triton_glue = bool(get("use_triton_glue", False)) and _TRITON_AVAILABLE

        if not hasattr(self.cfg, "timing_enable"):
            setattr(self.cfg, "timing_enable",
                    bool(get("timing_enable", False)) or (os.environ.get("TIMING_ENABLE", "0") == "1"))

        # -------- forward 关键属性 --------
        self.hidden_size = C
        self.causal = bool(causal)
        self.num_layers_phys = L  # 物理 blocks 总数
        self.layers_per_mor = LPB  # 每个 MoR 层包含的 blocks 数
        self.num_layers = M  # MoR 层数

        # -------- 构建物理 Transformer blocks（长度 L）--------
        self.blocks = nn.ModuleList([
            MLAMoEBlock(
                hidden_size=C, num_heads=Hq, num_kv_heads=Hkv,
                dv=dv, d_pe=dpe,
                moe_num_experts=E_ffn, moe_top_k=topk_ffn, moe_num_shared=shared_ffn,
                attn_bias=False, causal=self.causal, norm_eps=eps,
                moe_z_loss=moe_z, moe_hidden_expansion=moe_exp,
                dropout_p=drop_p, use_te_fp8=use_te_fp8,
            ) for _ in range(L)
        ])
        attn_fp8 = bool(get("attn_use_fp8", get("use_te_fp8", False)))
        ffn_fp8 = bool(get("ffn_use_fp8", get("use_te_fp8", False)))
        for blk in self.blocks:
            if hasattr(blk, "set_fp8"):
                blk.set_fp8(attn=attn_fp8, ffn=ffn_fp8)

        # -------- MoR Router（每个递归层一个）--------
        self.routers = nn.ModuleList([
            ContinueRouter(
                hidden_size=C, temperature=r_temp,
                keep_rate_target=r_keep, lb_coeff=r_lb, entropy_coeff=r_ent,
                keep_penalty_coeff=r_keep_pen, keep_policy=r_policy,
                eps_explore=r_eps_exp, stochastic_quota=r_stoch_q,
            ) for _ in range(M)
        ])

        # -------- LoRA（按 MoR 层：每层 K 个适配器 + 1 个门控）--------
        self.lora_depth_start = int(get("lora_depth_start", max(1, M - 2)))  # 基于 MoR 层索引
        self.lora_num = int(get("lora_num", 0))  # K
        self.lora_rank = int(get("lora_rank", 8))
        self.lora_alpha = float(get("lora_alpha", 1.0))
        self.lora_temp = float(get("lora_temperature", 1.0))
        self.lora_gate_hidden = int(get("lora_gate_hidden", 0))
        self.lora_usage_target = float(get("lora_usage_target", 0.2))
        self.lora_usage_coeff = float(get("lora_usage_coeff", 0.1))
        self.lora_gate_bias_nonzero_init = float(get("lora_gate_bias_nonzero_init", 0.0))
        self.lora_gate_jitter_amp = float(get("lora_gate_jitter_amp", 0.006))
        self.lora_min_usage_warmup = float(get("lora_min_usage_warmup", 0.05))
        self.lora_apply_every_n_blocks = int(get("lora_apply_every_n_blocks", LPB))

        self.lora_adapters = nn.ModuleList([
            nn.ModuleList([
                LoRAFFNAdapter(C, rank=self.lora_rank, eps=eps, dropout_p=drop_p, alpha=self.lora_alpha)
                for _ in range(self.lora_num)
            ]) if (d >= self.lora_depth_start and self.lora_num > 0) else nn.ModuleList([])
            for d in range(M)
        ])

        self.lora_gates = nn.ModuleList([
            (LoRAGate(C, num_lora=self.lora_num, hidden=self.lora_gate_hidden, temperature=self.lora_temp,
                      bias_nonzero_init=self.lora_gate_bias_nonzero_init)
             if (d >= self.lora_depth_start and self.lora_num > 0) else nn.Identity())
            for d in range(M)
        ])

        # -------- 统计缓冲 --------
        self.register_buffer("_depth_keep", torch.zeros(M, dtype=torch.long), persistent=False)
        self.register_buffer("_depth_total", torch.zeros(M, dtype=torch.long), persistent=False)
        self.router_aux_scale = r_aux

        # 可选：编译注意力
        if bool(getattr(self.cfg, "compile_attn", False)):
            for blk in self.blocks:
                blk.forward_attn = torch.compile(blk.forward_attn, dynamic=True, mode="reduce-overhead")

        # CUDAGraphs 缓存
        if self._use_cudagraphs and self._num_chunks >= 1:
            self._ffn_graphs = [[None for _ in range(self._num_chunks)] for _ in range(L)]
            self._ffn_static_in = [[None for _ in range(self._num_chunks)] for _ in range(L)]
            self._ffn_static_out = [[None for _ in range(self._num_chunks)] for _ in range(L)]
        else:
            self._ffn_graphs = self._ffn_static_in = self._ffn_static_out = None

        if bool(get("use_triton_glue", False)) and (not _TRITON_AVAILABLE):
            print("[UnifiedBackbone] use_triton_glue=True but Triton not available; falling back to torch ops.")

        # wiring 自检（rank0 打印一次）
        self._wiring_logged = False

        def _rank0():
            try:
                import torch.distributed as dist
                return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            except Exception:
                return True

        self._wiring_logged = False  # 供 forward 里兜底打印使用

        if _rank0():
            try:
                adapters_per_depth = [
                    (len(m) if hasattr(m, "__len__") else 0) for m in self.lora_adapters
                ]
                gates_per_depth = [type(g).__name__ for g in self.lora_gates]
                print("[LoRA/WIRING]init  M(num_layers) =", self.num_layers,
                      "| depth_start =", getattr(self, "lora_depth_start", None),
                      "| K =", getattr(self, "lora_num", None),
                      "| rank =", getattr(self, "lora_rank", None))
                print("[LoRA/WIRING]init  adapters per depth:", adapters_per_depth)
                print("[LoRA/WIRING]init  gates    per depth:", gates_per_depth)
            except Exception as e:
                print("[LoRA/WIRING]init  print failed:", repr(e))
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        def _rank0():
            try:
                import torch.distributed as dist
                return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            except Exception:
                return True

        if (not getattr(self, "_wiring_logged", False)) and _rank0():
            try:
                adapters_per_depth = [
                    (len(m) if hasattr(m, "__len__") else 0) for m in self.lora_adapters
                ]
                gates_per_depth = [type(g).__name__ for g in self.lora_gates]
                print("[LoRA/WIRING]fwd   M(num_layers) =", self.num_layers,
                      "| depth_start =", getattr(self, "lora_depth_start", None),
                      "| K =", getattr(self, "lora_num", None),
                      "| rank =", getattr(self, "lora_rank", None))
                print("[LoRA/WIRING]fwd   adapters per depth:", adapters_per_depth)
                print("[LoRA/WIRING]fwd   gates    per depth:", gates_per_depth)
            except Exception as e:
                print("[LoRA/WIRING]fwd   print failed:", repr(e))
            self._wiring_logged = True
        h = x
        B, S, C = h.shape
        device = h.device
        dtype = h.dtype

        timing_on = bool(getattr(self, "cfg", None) and getattr(self.cfg, "timing_enable", False))
        tms = {"attn": 0.0, "router": 0.0, "pack": 0.0, "ffn": 0.0, "scatter": 0.0}

        def _add_ms(key, ms):
            if timing_on: tms[key] += float(ms)

        # 统计容器
        active_token_counts: list[int] = []
        active_keep_rates: list[float] = []
        lora_adapter_usage_rates: list[float] = []  # ★ 统一用这个名字
        router_keep_entropy: list[torch.Tensor] = []
        router_kept_tokens: list[torch.Tensor] = []
        router_pkeep_hist: list[torch.Tensor] = []

        self._depth_keep.zero_()
        self._depth_total.zero_()

        ffn_aux_train = h.new_tensor(0.0, dtype=dtype)
        router_aux_train = h.new_tensor(0.0, dtype=dtype)
        adapter_aux_train = h.new_tensor(0.0, dtype=dtype)  # LoRA 使用率正则

        # pack/对齐配置
        pad_align = int(getattr(self, "_pad_align", 128))
        min_pack_tokens = int(getattr(self, "_min_pack_tokens", 8192))
        max_pack_tokens = int(self.max_pack_tokens)
        ffn_pad_mode = str(getattr(self.cfg, "ffn_pad_mode", "align")).lower()
        use_triton = bool(self._use_triton_glue)

        def _align(xv: int, a: int) -> int:
            return ((xv + a - 1) // a) * a

        # MoR 维度
        LPB = int(self.layers_per_mor)  # 每个 MoR 层包含的物理 blocks 数
        M = int(self.num_layers)  # MoR 层数
        L = int(self.num_layers_phys)  # 物理 blocks 总数

        # LoRA 配置
        K = int(getattr(self, "lora_num", 0))
        lora_temp = float(getattr(self, "lora_temp", getattr(self, "lora_temperature", 1.0)))
        lora_u_tgt = float(getattr(self, "lora_usage_target", 0.2))
        lora_u_coeff = float(getattr(self, "lora_usage_coeff", 0.1))

        # 递归激活掩码（按 token 是否仍继续递归）
        active = torch.ones(B, S, dtype=torch.bool, device=device)

        for d_mor in range(M):
            l0 = d_mor * LPB
            if l0 >= L:
                break

            # --- 该 MoR 层的第一个 block：跑注意力，作为 router/gate 的输入特征 ---
            if timing_on: e0, e1 = _make_event(); e0.record()
            h_attn = self.blocks[l0].forward_attn(h, attention_mask=attention_mask)
            if timing_on: e1.record(); torch.cuda.synchronize(); _add_ms("attn", e0.elapsed_time(e1))

            # --- Router（keep-only） ---
            if timing_on: r0, r1 = _make_event(); r0.record()
            p_keep, aux_d, keep_mask = self.routers[d_mor](h_attn, active, d_mor, M)
            router_aux_train = router_aux_train + aux_d.get("router_aux_loss", h.new_tensor(0.0, dtype=dtype))
            mon = aux_d.get("monitor", {})
            if isinstance(mon.get("keep_entropy", None), torch.Tensor):
                router_keep_entropy.append(mon["keep_entropy"].detach())
            if isinstance(mon.get("kept_tokens", None), torch.Tensor):
                router_kept_tokens.append(mon["kept_tokens"].detach())
            if isinstance(mon.get("pkeep_hist_10", None), torch.Tensor):
                router_pkeep_hist.append(mon["pkeep_hist_10"].detach())
            if timing_on: r1.record(); torch.cuda.synchronize(); _add_ms("router", r0.elapsed_time(r1))

            total_act = int(active.sum().item())
            keep_num = int(keep_mask.sum().item())
            active_token_counts.append(total_act)
            active_keep_rates.append(0.0 if total_act == 0 else keep_num / float(total_act))
            self._depth_total[d_mor] = total_act
            self._depth_keep[d_mor] = keep_num

            # 预备 keep 子批索引（在本 MoR 层内复用）
            if keep_num > 0:
                idx = torch.nonzero(keep_mask, as_tuple=False)  # [M_keep, 2]
                bidx, sidx = idx[:, 0], idx[:, 1]
                idx_flat = (bidx * S + sidx).contiguous()
                M_keep = int(idx_flat.numel())
            else:
                idx = bidx = sidx = idx_flat = None
                M_keep = 0

            # --- LoRA 门控（层级一次；在该层内复用） ---
            use_lora = (d_mor >= getattr(self, "lora_depth_start", max(1, M - 2))) \
                       and (K > 0) and (len(self.lora_adapters[d_mor]) > 0) and (M_keep > 0)

            if use_lora:
                x_flat = h_attn.view(B * S, C).contiguous()
                h_tok = x_flat.index_select(0, idx_flat)  # [M_keep, C]

                # ① logits + 温度缩放
                logits = self.lora_gates[d_mor](h_tok)  # [M_keep, K+1]
                logits = logits / max(lora_temp, 1e-6)

                # ② capture-safe 抖动（确定性，无 RNG）
                with torch.no_grad():
                    mk = logits.size(0)
                    if mk > 0:
                        i = torch.arange(mk, device=logits.device, dtype=torch.float32).view(-1, 1)
                        j = torch.arange(logits.size(1), device=logits.device, dtype=torch.float32).view(1, -1)
                        amp = float(getattr(self, "lora_gate_jitter_amp", 0.012))  # ↑默认 1.2%
                        jitter = torch.frac(torch.sin(i * 12.9898 + j * 78.233 + (d_mor + 1) * 37.719))
                        logits = logits + (jitter - 0.5) * amp

                # ③ 初步选择
                probs = torch.softmax(logits, dim=-1)
                hard = torch.argmax(probs, dim=-1)  # [M_keep]

                # ④ warm-up 配额（ceil 且至少 1）
                min_u = float(getattr(self, "lora_min_usage_warmup", 0.05))
                if self.training and (min_u > 0.0):
                    with torch.no_grad():
                        import math
                        target = max(1, math.ceil(min_u * M_keep))  # ★ 至少 1
                        cur = int((hard != 0).sum().item())
                        need = max(0, target - cur)
                        if need > 0:
                            gains, best_non0 = logits[:, 1:].max(dim=1)  # [M_keep]
                            margin = gains - logits[:, 0]  # 越大越该切到 LoRA
                            k = min(need, margin.numel())
                            if k > 0:
                                sel = torch.topk(margin, k=k, largest=True).indices
                                hard[sel] = (best_non0[sel] + 1)  # 指到各自最佳非 0 路

                # ⑤ 兜底：仍全 0 就强制开一点（确定性、很小比例）
                with torch.no_grad():
                    if (hard != 0).sum() == 0 and M_keep > 0:
                        gains, best_non0 = logits[:, 1:].max(dim=1)
                        margin = gains - logits[:, 0]
                        k = max(1, min(4, margin.numel() // 50))  # ~2% 或最多 4 个
                        sel = torch.topk(margin, k=k, largest=True).indices
                        hard[sel] = (best_non0[sel] + 1)

                # ⑥ 统计使用率 + 正则
                with torch.no_grad():
                    use_ratio = (hard != 0).float().mean()
                    lora_adapter_usage_rates.append(use_ratio.detach().item())
                enabled_layers = sum(1 for d in range(M) if (d >= self.lora_depth_start and self.lora_num > 0))
                if enabled_layers > 0:
                    adapter_aux_train = adapter_aux_train + ((use_ratio - lora_u_tgt).pow(2)).to(dtype) * (
                                lora_u_coeff / enabled_layers)
            else:
                hard = None
                lora_adapter_usage_rates.append(0.0)

            # --- 在该 MoR 层内依次跑 LPB 个物理 blocks；复用 keep 子批与 LoRA 指派 ---
            for j in range(LPB):
                l = l0 + j
                if l >= L:
                    break

                # 其余 blocks 的注意力
                if j > 0:
                    if timing_on: e0, e1 = _make_event(); e0.record()
                    h_attn = self.blocks[l].forward_attn(h, attention_mask=attention_mask)
                    if timing_on: e1.record(); torch.cuda.synchronize(); _add_ms("attn", e0.elapsed_time(e1))

                if M_keep == 0:
                    h = h_attn
                    continue

                # ===== pack keep 子批（主 FFN；对齐策略保持与全局一致，命中 CUDAGraphs） =====
                # 这里 pack 的是“主 FFN”的输入；对主干仍按 align + min/max_pack_tokens 来保证图稳定
                if timing_on: p0, p1 = _make_event(); p0.record()
                x_flat = h_attn.view(B * S, C).contiguous()

                mk_main = M_keep  # 当前层被保留的 token 数
                if ffn_pad_mode == "exact":
                    base_align_main = 16 if (C % 16 == 0) else 8
                    M_pad = _align(max(mk_main, 1), base_align_main)
                else:
                    M_align = _align(mk_main, pad_align)
                    M_pad = max(min_pack_tokens, M_align)
                    if max_pack_tokens > 0:
                        M_pad = min(M_pad, max_pack_tokens)

                if use_triton:
                    x_keep_2d = _triton_gather_pack(x_flat, idx_flat, M_pad)  # [M_pad, C]
                else:
                    x_sel = x_flat.index_select(0, idx_flat)
                    if M_pad > mk_main:
                        pad = torch.zeros((M_pad - mk_main, C), dtype=x_sel.dtype, device=x_sel.device)
                        x_keep_2d = torch.cat([x_sel, pad], dim=0)
                    else:
                        x_keep_2d = x_sel
                if timing_on: p1.record(); torch.cuda.synchronize(); _add_ms("pack", p0.elapsed_time(p1))

                # ===== 主 FFN =====
                x_keep = x_keep_2d.unsqueeze(1)  # [M_pad,1,C]
                if timing_on: f0, f1 = _make_event(); f0.record()
                delta_main, aux_ffn = self._ffn_run(l, 0, x_keep, self.blocks[l])
                if timing_on: f1.record(); torch.cuda.synchronize(); _add_ms("ffn", f0.elapsed_time(f1))
                ffn_aux_train = ffn_aux_train + aux_ffn.get("moe_aux_loss", h.new_tensor(0.0, dtype=dtype))

                y2d = delta_main.squeeze(1)  # [M_pad, C]
                if timing_on: s0, s1 = _make_event(); s0.record()
                flat = h_attn.view(B * S, C)
                if use_triton:
                    _triton_scatter_add(flat, idx_flat[:mk_main], y2d[:mk_main, :])
                else:
                    flat.index_add_(0, idx_flat[:mk_main], y2d[:mk_main, :])
                if timing_on: s1.record(); torch.cuda.synchronize(); _add_ms("scatter", s0.elapsed_time(s1))
                apply_lora_now = use_lora and (((j + 1) % max(1, self.lora_apply_every_n_blocks)) == 0)
                if apply_lora_now:
                    # fast-path: K == 1 时无需循环
                    if K == 1:
                        pos = torch.nonzero(hard != 0, as_tuple=False).squeeze(-1)
                        if pos.numel() > 0:
                            idx_k = idx_flat.index_select(0, pos)
                            mk = int(pos.numel())

                            # LoRA 分支：强制 exact 小对齐（16 或 8）
                            base_align = 16 if (C % 16 == 0) else 8
                            M_pad_k = _align(max(mk, 1), base_align)

                            if use_triton:
                                x_k_2d = _triton_gather_pack(x_flat, idx_k, M_pad_k)
                            else:
                                x_k = x_flat.index_select(0, idx_k)
                                if M_pad_k > mk:
                                    pad = torch.zeros((M_pad_k - mk, C), dtype=x_k.dtype, device=x_k.device)
                                    x_k_2d = torch.cat([x_k, pad], dim=0)
                                else:
                                    x_k_2d = x_k

                            # 单路 LoRA：adapters[d_mor][0]
                            yk = self.lora_adapters[d_mor][0](x_k_2d.unsqueeze(1)).squeeze(1)
                            if use_triton:
                                _triton_scatter_add(flat, idx_k[:mk], yk[:mk, :])
                            else:
                                flat.index_add_(0, idx_k[:mk], yk[:mk, :])
                    else:
                        # K >= 2 的原有多路逻辑保留
                        for k in range(1, K + 1):
                            pos = torch.nonzero(hard == k, as_tuple=False).squeeze(-1)
                            if pos.numel() == 0: continue
                            idx_k = idx_flat.index_select(0, pos)
                            mk = int(pos.numel())

                            base_align = 16 if (C % 16 == 0) else 8
                            M_pad_k = _align(max(mk, 1), base_align)

                            if use_triton:
                                x_k_2d = _triton_gather_pack(x_flat, idx_k, M_pad_k)
                            else:
                                x_k = x_flat.index_select(0, idx_k)
                                if M_pad_k > mk:
                                    pad = torch.zeros((M_pad_k - mk, C), dtype=x_k.dtype, device=x_k.device)
                                    x_k_2d = torch.cat([x_k, pad], dim=0)
                                else:
                                    x_k_2d = x_k

                            yk = self.lora_adapters[d_mor][k - 1](x_k_2d.unsqueeze(1)).squeeze(1)
                            if use_triton:
                                _triton_scatter_add(flat, idx_k[:mk], yk[:mk, :])
                            else:
                                flat.index_add_(0, idx_k[:mk], yk[:mk, :])

                # 写回到 h，进入本 MoR 层的下一个 block
                h = h_attn

            # 下一递归层的活跃掩码
            active = keep_mask

        # —— 汇总 / 返回 —— #
        loss_dict_log = {
            "ffn_moe_aux_loss_scalar": ffn_aux_train.detach(),
            "router_aux_loss_scalar": (router_aux_train + adapter_aux_train).detach(),
            "ffn_moe_aux_loss_weighted": ffn_aux_train.detach(),
            "router_aux_loss_weighted": (router_aux_train + adapter_aux_train).detach(),
            "active_token_counts": torch.tensor(active_token_counts, device=device),
            "active_keep_rates": torch.tensor(active_keep_rates, device=device, dtype=torch.float32),
            "router_keep_entropy": (torch.stack(router_keep_entropy, dim=0)
                                    if len(router_keep_entropy) > 0 else torch.zeros(0, device=device)),
            "router_kept_tokens": (torch.stack(router_kept_tokens, dim=0)
                                   if len(router_kept_tokens) > 0 else torch.zeros(0, device=device)),
            "router_pkeep_hist_10": (torch.stack(router_pkeep_hist, dim=0)
                                     if len(router_pkeep_hist) > 0 else torch.zeros((self.num_layers, 10),
                                                                                    device=device,
                                                                                    dtype=torch.float32)),
            "lora_adapter_usage_rates": torch.tensor(lora_adapter_usage_rates, device=device, dtype=torch.float32),
        }
        stats = {
            "active_token_counts": active_token_counts,
            "active_keep_rates": active_keep_rates,
            "lora_adapter_usage_rates": lora_adapter_usage_rates,  # ★ 同步到 stats
            "depth_used": self.num_layers,
        }
        if timing_on:
            total = sum(tms.values()) + 1e-6
            tms["total"] = total
            stats["timing_ms"] = tms

        return {
            "y": h,
            "_aux_for_train": {
                "ffn_moe_aux": ffn_aux_train,
                "router_aux": router_aux_train + adapter_aux_train
            },
            "loss_dict": loss_dict_log,
            "stats": stats,
        }

    def _ensure_ffn_graph(self, depth: int, ci: int, x: torch.Tensor, blk) -> bool:
        """
        尝试为第 depth 层，第 ci 个 chunk 捕获一张 CUDA 图。
        要求：x.is_cuda 且形状、dtype 稳定。捕获失败则返回 False。
        成功时，self._ffn_graphs[depth][ci] 非空，且 static_in/out 已分配。
        """
        if not x.is_cuda:
            return False
        # 分配静态缓冲并判断是否需要重建图
        need_rebuild = False
        si = self._ffn_static_in[depth][ci]
        so = self._ffn_static_out[depth][ci]
        if (si is None) or (si.shape != x.shape) or (si.dtype != x.dtype) or (si.device != x.device):
            self._ffn_static_in[depth][ci] = torch.empty_like(x)
            self._ffn_static_out[depth][ci] = torch.empty_like(x)
            si = self._ffn_static_in[depth][ci]
            so = self._ffn_static_out[depth][ci]
            need_rebuild = True

        g = self._ffn_graphs[depth][ci]
        if g is None:
            need_rebuild = True

        if not need_rebuild:
            return True  # 现有图可复用

        # —— 捕获前准备：warmup + 真正捕获 —— #
        try:
            torch.cuda.synchronize()
            # 先把本次输入拷到静态输入，做一次 eager 运行，确保 workspace/参数已初始化
            si.copy_(x)
            y_warm, _ = blk.forward_ffn(si, return_aux=True)
            self._ffn_static_out[depth][ci].copy_(y_warm)
            torch.cuda.synchronize()

            # 真正捕获
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y_cap, _ = blk.forward_ffn(self._ffn_static_in[depth][ci], return_aux=True)
                self._ffn_static_out[depth][ci].copy_(y_cap)

            self._ffn_graphs[depth][ci] = g
            torch.cuda.synchronize()
            return True
        except Exception as e:
            # 捕获失败：清除图，回退 eager
            self._ffn_graphs[depth][ci] = None
            # 只提醒一次
            if not hasattr(self, "_graph_warned") or not self._graph_warned:
                if getattr(self.cfg, "graph_verbose", False):
                    print(
                        f"[UnifiedBackbone] CUDAGraph capture failed at layer {depth}, chunk {ci}: {e}. Falling back to eager.")
                self._graph_warned = True
            return False

    def _ffn_run(self, depth: int, ci: int, x: torch.Tensor, blk):
        """
        如果可用，就复用 CUDAGraph；否则直接 eager 运行。
        返回：(y_eff, aux_dict)
        """
        use_graphs = bool(self._use_cudagraphs and self._ffn_graphs is not None)
        if use_graphs and x.is_cuda:
            ok = self._ensure_ffn_graph(depth, ci, x, blk)
            if ok:
                # 填充输入并 replay
                self._ffn_static_in[depth][ci].copy_(x)
                self._ffn_graphs[depth][ci].replay()
                return self._ffn_static_out[depth][ci], {}
        # 回退：eager 前向
        return blk.forward_ffn(x, return_aux=True)

    def _get_workspace(self, shape: Tuple[int, int, int], dtype, device) -> torch.Tensor:
        if self._workspace is None or self._ws_shape != shape or self._workspace.dtype != dtype or self._workspace.device != device:
            self._workspace = torch.zeros(shape, dtype=dtype, device=device)
            self._ws_shape = shape
        else:
            self._workspace.zero_()
        return self._workspace

    @torch.no_grad()
    def _index_put_delta(self, shape, b_idx, s_idx, delta, dtype, device):
        """
        在一个零张量上把 delta 写到 (b_idx, s_idx) 位置，返回与 h 同形的“增量”张量。
        这样就能用 h = h + delta_full 的方式避免任何 inplace。
        """
        delta_full = torch.zeros(shape, dtype=dtype, device=device)
        if delta.dim() == 3:
            delta = delta.squeeze(1)  # [M,C]
        delta_full[b_idx, s_idx] = delta.to(dtype)
        return delta_full

