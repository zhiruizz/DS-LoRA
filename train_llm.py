# train_llm.py
# 读取 config.json（嵌套字段），仅训练“稠密 MLA + DeepSeekMoE”的语言模型。
# - 不加载/使用 DSA/Indexer
# - 仅记录 main / ffn_moe / router 三类损失

import os
import json
import math
import time
import argparse
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch._dynamo import config as dynamo_config
import warnings
dynamo_config.suppress_errors = True
warnings.filterwarnings("ignore", "Dynamo does not know how to trace*", module="torch._dynamo")
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import bitsandbytes as bnb
from MoR_LanguageModel import MoR_LanguageModel, MoR_Config

copy_stream = torch.cuda.Stream()
compute_stream = torch.cuda.current_stream()

def _to_device_async(batch, device):
    with torch.cuda.stream(copy_stream):
        return {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

class Prefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if str(device).startswith("cuda") else None

    def _prefetch(self):
        try:
            self.next = next(self.loader)
        except StopIteration:
            self.next = None
            return
        if self.stream is None:
            return
        with torch.cuda.stream(self.stream):
            for k, v in list(self.next.items()):
                if torch.is_tensor(v):
                    self.next[k] = v.to(self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        if not hasattr(self, "next"):
            self._prefetch()
        if self.next is None:
            raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next
        self._prefetch()
        return batch

def _make_amp_tools(precision: str):
    """
    返回 (autocast_ctx, GradScalerCls, amp_dtype)
    - 新版: torch.amp.autocast("cuda", ...)
    - 旧版: torch.cuda.amp.autocast(dtype=..., enabled=...)
    """
    prec = (precision or "bf16").lower()
    amp_dtype = torch.bfloat16 if prec == "bf16" else torch.float16

    try:
        # 新 API（PyTorch>=2.1）
        from torch.amp import autocast as _autocast_new, GradScaler as _GradScalerNew
        def _ctx(enabled: bool):
            return _autocast_new("cuda", dtype=amp_dtype, enabled=enabled)
        def _scaler(enabled: bool):
            return _GradScalerNew("cuda", enabled=enabled)
        return _ctx, _scaler, amp_dtype
    except Exception:
        # 旧 API（兼容当前环境）
        from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScalerOld
        def _ctx(enabled: bool):
            return _autocast_old(dtype=amp_dtype, enabled=enabled)
        def _scaler(enabled: bool):
            return _GradScalerOld(enabled=enabled)
        return _ctx, _scaler, amp_dtype

def _flatten_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把多层 dict 拍平成一层（子键覆盖父键同名键）"""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update(v)
        else:
            out[k] = v
    return out

def move_batch_to_device(batch, device, non_blocking: bool = True):
    """
    将 DataCollator 产出的 batch（含 input_ids / attention_mask / labels 等）
    安全搬到 device。仅对张量做 .to(device, non_blocking=...)，其余原样返回。
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out

def load_config(path: str) -> MoR_Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    flat = _flatten_cfg(raw)
    # 给模型必要字段起别名/默认
    # vocab_size 需要在实例化 tokenizer 后再覆盖，这里先返回
    return MoR_Config(**flat)


def build_dataloaders(cfg, tokenizer):
    """
    统一构建 DataLoader：
      优先级：processed_dataset_path -> (dataset_path_train / dataset_path_val) -> dataset_name
    若数据已含 input_ids/attention_mask：
      - 使用轻量 collator（直接堆叠张量），避免 HF 的通用 collator 额外开销。
    否则：
      - 仅首次映射 tokenize 到磁盘（强烈建议离线 preprocess），运行时不再重复 tokenize。
    """
    import os
    from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    def _cfg_get(o, k, d=None):
        if isinstance(o, dict): return o.get(k, d)
        return getattr(o, k, d)

    bs          = int(_cfg_get(cfg, "batch_size", 8))
    num_workers = int(_cfg_get(cfg, "num_workers", 8))
    max_len     = int(_cfg_get(cfg, "max_seq_len", 512))
    shuffle     = bool(_cfg_get(cfg, "shuffle", True))
    drop_last   = True

    proc_path   = _cfg_get(cfg, "processed_dataset_path", None)
    path_train  = _cfg_get(cfg, "dataset_path_train", None)
    path_val    = _cfg_get(cfg, "dataset_path_val",   None)
    ds_name     = _cfg_get(cfg, "dataset_name",       None)
    split_tr    = _cfg_get(cfg, "dataset_split_train", "train")
    split_va    = _cfg_get(cfg, "dataset_split_val",   "validation")

    # ------------- load -------------
    def _pick(ds, split):
        if isinstance(ds, DatasetDict):
            if split in ds: return ds[split]
            for k in ("train", "validation", "valid"):
                if k in ds: return ds[k]
            return list(ds.values())[0]
        return ds

    if proc_path and os.path.exists(proc_path):
        ds = load_from_disk(proc_path)
        train_ds = _pick(ds, split_tr)
        val_ds   = _pick(ds, split_va)
        if val_ds is None:
            n_val = min(2048, len(train_ds))
            val_ds = train_ds.select(range(n_val))
    elif path_train and os.path.exists(path_train):
        ds_tr = load_from_disk(path_train); train_ds = _pick(ds_tr, split_tr)
        if path_val and os.path.exists(path_val):
            ds_va = load_from_disk(path_val); val_ds = _pick(ds_va, split_va)
        else:
            n_val = min(2048, len(train_ds))
            val_ds = train_ds.select(range(n_val))
    elif ds_name:
        ds_all = load_dataset(ds_name)
        train_ds = _pick(ds_all, split_tr)
        val_ds   = _pick(ds_all, split_va)
        if val_ds is None:
            n_val = min(2048, len(train_ds))
            val_ds = train_ds.select(range(n_val))
    else:
        raise ValueError("No dataset source found. Please set 'processed_dataset_path' or 'dataset_path_*' or 'dataset_name'.")

    # ------------- ensure tokenized -------------
    def _ensure_tokenized(ds):
        cols = set(ds.column_names)
        if "input_ids" in cols:
            # 补齐缺失字段
            if "attention_mask" not in cols:
                ds = ds.map(lambda ex: {"attention_mask": [1] * len(ex["input_ids"])}, batched=False)
            if "labels" not in cols:
                # 不要在这里把 labels 复制出来（保留=ids），屏蔽逻辑放在 loss 内做即可
                ds = ds.map(lambda ex: {"labels": ex["input_ids"]}, batched=False)
            return ds

        if "text" not in cols:
            raise ValueError("Dataset has neither 'input_ids' nor 'text' columns.")

        # 首次即时 tokenize（如可能，建议提前用 preprocess 脚本离线完成）
        def _tok(batch):
            enc = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_len,
                padding=False,
                return_attention_mask=True,
            )
            enc["labels"] = enc["input_ids"].copy()
            return enc

        return ds.map(_tok, batched=True, remove_columns=[c for c in ds.column_names if c != "text"])

    train_ds = _ensure_tokenized(train_ds)
    val_ds   = _ensure_tokenized(val_ds)

    # ------------- 选择 collator -------------
    pretokenized = all(k in train_ds.column_names for k in ("input_ids", "attention_mask", "labels"))

    if pretokenized:
        # 轻量 collator：直接把张量堆叠（Dataset 内部应为 python list，需要转换为 tensor）
        import torch
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        def _pad_to_max_len(seqs, pad_value):
            maxl = max(len(s) for s in seqs)
            out = torch.full((len(seqs), maxl), pad_value, dtype=torch.long)
            for i, s in enumerate(seqs):
                out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            return out

        def collate_tokenized(examples):
            input_ids      = _pad_to_max_len([e["input_ids"]      for e in examples], pad_id)
            attention_mask = _pad_to_max_len([e["attention_mask"] for e in examples], 0)
            labels         = _pad_to_max_len([e["labels"]         for e in examples], -100)  # 这里可直接置 -100；双保险
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        collator = collate_tokenized
    else:
        # 兜底：用 HF 的通用 collator
        from transformers import DataCollatorForLanguageModeling
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ------------- DataLoader -------------
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=drop_last,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
        collate_fn=collator,
    )
    return train_loader, val_loader

@torch.no_grad()
def evaluate(model, dataloader, device, precision="bf16", max_batches=50, pad_token_id=None):
    """验证：严格屏蔽 padding；返回 (avg_loss, ppl, excess_loss)"""
    model.eval()
    use_cuda = str(device).startswith("cuda")
    autocast_ctx, _, _ = _make_amp_tools(precision)

    total_loss = 0.0
    total_tokens = 0
    iters = 0
    vocab_size = None  # 用首个 batch 的 logits 推断 V

    for batch in dataloader:
        if iters >= max_batches:
            break
        iters += 1
        batch = move_batch_to_device(batch, device)

        with autocast_ctx(enabled=use_cuda):
            out = model(**batch)
            logits = out["logits"].float()   # 数值更稳
            labels = batch["labels"]
            attn   = batch.get("attention_mask", None)

            # 记录词表大小（取首个 batch）
            if vocab_size is None:
                vocab_size = int(logits.size(-1))

            # 统一掩码：labels!=-100 为准，同时 AND attention_mask (若提供)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid = (shift_labels != -100)
            if attn is not None:
                shift_attn = attn[:, 1:].contiguous().to(torch.bool)
                valid = valid & shift_attn

            # 数据一致性检查（只记一次）
            if iters == 1:
                ntok_eff = int(valid.sum().item())
                total = int(valid.numel())
                over_max = int((shift_labels >= logits.size(-1)).sum().item())
                neg_bad  = int(((shift_labels < -1) & (shift_labels != -100)).sum().item())
                print(f"[eval-check] valid_tokens={ntok_eff}/{total} | out_of_vocab={over_max} | bad_neg={neg_bad}")

            # 将无效位置的 label 设为 -100 再算 CE（sum 再除有效 token）
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.masked_fill(~valid, -100).view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            ntok = int(valid.sum().item())

        total_loss += float(ce)
        total_tokens += ntok

    model.train()
    if total_tokens == 0:
        return float("nan"), float("nan"), float("nan")

    avg_loss = total_loss / total_tokens
    avg_loss_f = float(avg_loss)  # 强制 Python float64
    ppl = math.exp(avg_loss_f) if avg_loss_f < 80.0 else float("inf")

    lnV = math.log(vocab_size) if vocab_size and vocab_size > 0 else float("nan")
    excess = avg_loss_f - lnV if math.isfinite(lnV) else float("nan")
    excess_ppl = math.exp(excess) if (math.isfinite(excess) and abs(excess) < 80.0) else float("inf")
    # 如果你也在打印 val_excess（相对 ln(V) 的超额损失），可以顺手给一个更稳的指标：
    if vocab_size is None or vocab_size <= 0:
        excess = float("nan")
    else:
        excess = avg_loss_f - math.log(vocab_size)

    return avg_loss_f, ppl, excess

def create_scheduler(optimizer, max_train_steps: int, cfg):
    """
    线性 warmup + cosine 衰减到 0。
    支持两种指定方式（择一即可）：
      - warmup_steps（优先级更高）
      - warmup_ratio（当 warmup_steps 未提供时生效）
    cfg 可为 dict 或任意带属性的对象（如 MoR_Config）。
    """
    import math

    def _cfg_get(c, key, default=None):
        if isinstance(c, dict):
            return c.get(key, default)
        return getattr(c, key, default)

    ws = _cfg_get(cfg, "warmup_steps", None)
    if ws is None:
        wr = float(_cfg_get(cfg, "warmup_ratio", 0.03))
        ws = int(max_train_steps * wr)
    ws = int(ws)

    def lr_lambda(step: int):
        if step < ws:
            # 线性 warmup
            return float(step) / float(max(1, ws))
        # cosine from 1 -> 0
        progress = (step - ws) / float(max(1, max_train_steps - ws))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def compute_main_loss(out: dict, batch: dict, cfg) -> torch.Tensor:
    """
    语言模型主损失：next-token cross-entropy（正确屏蔽 padding）。
    需要：
      - out["logits"]: Float[B,S,V]
      - batch["labels"]: Long[B,S]
      - 可选 batch["attention_mask"]: Long/Bool[B,S]，0 表示 pad
    """
    import torch.nn.functional as F

    logits = out.get("logits", None)
    if logits is None:
        raise KeyError("compute_main_loss: missing 'logits' in model output.")
    labels = batch["labels"]
    attn   = batch.get("attention_mask", None)

    # shift one token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # 用 attention_mask 屏蔽 pad（把 pad 位置置为 -100）
    if attn is not None:
        shift_attn = attn[:, 1:].contiguous().to(torch.bool)
        shift_labels = shift_labels.masked_fill(~shift_attn, -100)

    # label smoothing（可选）
    def _cfg_get(c, key, default=None):
        if isinstance(c, dict):
            return c.get(key, default)
        return getattr(c, key, default)
    ls = float(_cfg_get(cfg, "label_smoothing", 0.0) or 0.0)

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
        label_smoothing=ls if ls > 0.0 else 0.0,
    )
    return loss

# ------------------------ optimizer builder (with router/expert LR multipliers) ------------------------
def build_optimizer(cfg, model):
    import math
    import torch
    from torch.optim import AdamW as TorchAdamW

    try:
        import bitsandbytes as bnb
        HAS_BNB = True
    except Exception:
        HAS_BNB = False

    # 基础超参
    base_lr  = float(getattr(cfg, "lr", 3e-5))
    wd       = float(getattr(cfg, "weight_decay", 0.01))
    beta1    = float(getattr(cfg, "beta1", 0.9))
    beta2    = float(getattr(cfg, "beta2", 0.95))
    eps      = float(getattr(cfg, "adam_eps", 1e-8))
    opt_name = str(getattr(cfg, "optimizer", "adamw8bit")).lower()

    # 学习率乘子
    router_lr_mult       = float(getattr(cfg, "router_lr_mult", 1.0))
    expert_ffn_lr_mult   = float(getattr(cfg, "expert_ffn_lr_mult", 1.0))  # ★ 新增：专家块（FFN/Attn/Norm）组
    # 可选：限定专家匹配关键字（默认覆盖 expert_blocks 下的所有可训练参数）
    expert_name_scopes   = list(getattr(cfg, "expert_name_scopes", ["backbone.expert_blocks"]))
    expert_include_keys  = list(getattr(cfg, "expert_include_keys", ["ffn", "mlp", "attn", "attention", "norm", "rmsnorm", "ln", "gate", "up", "down"]))
    expert_exclusive     = bool(getattr(cfg, "expert_scope_strict", False))  # True 则只包含含上述关键词的参数；False 则 expert_blocks 下全收

    # 简单的“是否衰减”判定
    def use_decay(name, param):
        if param.ndim < 2:  # bias / 标量
            return False
        low = name.lower()
        if ("norm" in low) or ("rms" in low) or ("layernorm" in low) or low.endswith(".bias"):
            return False
        return True

    # 先按 name 分类
    base_decay, base_nodecay = [], []
    router_decay, router_nodecay = [], []
    expert_decay, expert_nodecay = [], []

    def is_router_name(n: str) -> bool:
        return "backbone.routers" in n

    def is_expert_name(n: str) -> bool:
        # 命中 expert scope
        hit_scope = any(scope in n for scope in expert_name_scopes)
        if not hit_scope:
            return False
        if not expert_exclusive:
            return True
        # 严格模式：还需包含 include 关键字之一
        low = n.lower()
        return any(k in low for k in expert_include_keys)

    # 遍历参数
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_router_name(name):
            if use_decay(name, p): router_decay.append(p)
            else:                  router_nodecay.append(p)
        elif is_expert_name(name):
            if use_decay(name, p): expert_decay.append(p)
            else:                  expert_nodecay.append(p)
        else:
            if use_decay(name, p): base_decay.append(p)
            else:                  base_nodecay.append(p)

    # 组装 param groups
    param_groups = []
    # base
    if base_decay:
        param_groups.append({"params": base_decay, "lr": base_lr, "weight_decay": wd})
    if base_nodecay:
        param_groups.append({"params": base_nodecay, "lr": base_lr, "weight_decay": 0.0})
    # router（乘子）
    if router_decay:
        param_groups.append({"params": router_decay, "lr": base_lr * router_lr_mult, "weight_decay": wd})
    if router_nodecay:
        param_groups.append({"params": router_nodecay, "lr": base_lr * router_lr_mult, "weight_decay": 0.0})
    # expert（乘子）★ 新增组
    if expert_decay:
        param_groups.append({"params": expert_decay, "lr": base_lr * expert_ffn_lr_mult, "weight_decay": wd})
    if expert_nodecay:
        param_groups.append({"params": expert_nodecay, "lr": base_lr * expert_ffn_lr_mult, "weight_decay": 0.0})

    # 构建优化器
    if opt_name in ("adamw8bit", "adamw_8bit", "adamw-bnb", "bnb_adamw") and HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(
            param_groups, lr=base_lr, betas=(beta1, beta2), eps=eps
        )
        opt_tag = "adamw8bit"
    else:
        optimizer = TorchAdamW(
            param_groups, lr=base_lr, betas=(beta1, beta2), eps=eps
        )
        opt_tag = "adamw"

    # 打印分组摘要，便于确认是否命中
    def _cnt(tensors):
        return sum(p.numel() for p in tensors)

    print("[optimizer] type=%s | lr=%.3e | wd=%.2e | router_lr_mult=%.2f | expert_ffn_lr_mult=%.2f"
          % (opt_tag, base_lr, wd, router_lr_mult, expert_ffn_lr_mult))
    print("  base:   decay=%d  nodecay=%d"   % (_cnt(base_decay),   _cnt(base_nodecay)))
    print("  router: decay=%d  nodecay=%d"   % (_cnt(router_decay), _cnt(router_nodecay)))
    print("  expert: decay=%d  nodecay=%d"   % (_cnt(expert_decay), _cnt(expert_nodecay)))

    return optimizer
# ------------------------------------------------------------------------------------------------------
def _apply_epoch_schedules(backbone, epoch_idx: int, cfg):
    """
    epoch_idx 从 0 开始；
    根据 config 中的退火设置，计算当前 epoch 对应的 LoRA 温度，
    并将其写入 backbone.lora_temp 属性。
    (cfg 应为 MoR_Config 对象或兼容 getattr 的对象)
    """

    def _cfg_get(obj, key, default=None):
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    # 1. 读取退火配置 (从 cfg 对象读取)
    anneal_enabled = bool(_cfg_get(cfg, "lora_temp_anneal_enabled", False))
    anneal_start_ep = int(_cfg_get(cfg, "lora_temp_anneal_start_epoch", 3))
    anneal_end_ep = int(_cfg_get(cfg, "lora_temp_anneal_end_epoch", 6))
    anneal_from = float(_cfg_get(cfg, "lora_temp_anneal_from", 1.2))
    anneal_to = float(_cfg_get(cfg, "lora_temp_anneal_to", 0.8))

    # 2. 计算当前温度
    # 获取模型当前的温度（或 config 中的初始值）
    cur_temp = float(getattr(backbone, "lora_temp", _cfg_get(cfg, "lora_temperature", 1.0)))

    if anneal_enabled and (anneal_end_ep > anneal_start_ep):
        current_epoch_num = epoch_idx + 1  # epoch_idx 从 0 开始，config 中的设置从 1 开始
        if current_epoch_num < anneal_start_ep:
            # 保持在起始温度（如果配置了 from）或初始温度
            cur_temp = anneal_from
        elif current_epoch_num >= anneal_end_ep:
            # 保持在目标温度
            cur_temp = anneal_to
        else:
            # 线性插值
            progress = (current_epoch_num - anneal_start_ep) / float(anneal_end_ep - anneal_start_ep)
            cur_temp = anneal_from + progress * (anneal_to - anneal_from)

    # 3. 写回到 backbone
    # UnifiedBackbone.forward (L598) 会读取这个属性
    backbone.lora_temp = float(cur_temp)

    # (我们不修改 lora_gates[...].temperature，因为 LoRAGate.forward 已被修改)

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--max_micro_steps", type=int, default=None)  # 以 micro-step 为单位的提前停止（可选）
    parser.add_argument("--router_freeze_steps", type=int, default=0,
                        help="(可选) 冻结 Router 的优化步数（解冻后开始学习）。")
    args = parser.parse_args()

    cfg = load_config(args.config)

    def _cfg_get(obj, key, default=None):
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    # ===== 设备与精度 =====
    device      = _cfg_get(cfg, "device", "cuda:0")
    precision   = _cfg_get(cfg, "precision", "bf16").lower()
    amp_enabled = (precision in ("bf16", "fp16"))
    torch.set_float32_matmul_precision("high")

    # ===== tokenizer / vocab =====
    tokenizer_path = _cfg_get(cfg, "tokenizer_path", "./tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg.vocab_size = len(tokenizer)

    # ===== 模型 =====
    model = MoR_LanguageModel(cfg).to(device)
    if precision == "bf16":
        model.to(dtype=torch.bfloat16)
    elif precision == "fp16":
        model.to(dtype=torch.float16)
    else:
        model.to(dtype=torch.float32)

    # ===== Data / Optim / Sched =====
    train_loader, val_loader = build_dataloaders(cfg, tokenizer)
    optimizer = build_optimizer(cfg, model)

    grad_accum   = int(_cfg_get(cfg, "gradient_accumulation_steps", _cfg_get(cfg, "grad_accum", 1)))
    epochs       = int(_cfg_get(cfg, "train_epochs", 1))
    log_interval = int(_cfg_get(cfg, "log_interval", 100))
    save_dir     = _cfg_get(cfg, "save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Router 冻结控制
    router_freeze_steps = int(args.router_freeze_steps or 0)
    if router_freeze_steps > 0:
        frozen = 0
        for n, p in model.named_parameters():
            if "backbone.routers" in n:
                p.requires_grad = False
                frozen += 1
        print(f"[router-freeze] freeze {frozen} tensors for first {router_freeze_steps} optimizer steps")
    else:
        print("[router-freeze] disabled")

    # epoch 粒度：完整遍历 dataloader = 1 个 epoch
    micro_steps_per_epoch = len(train_loader)                              # 每 epoch 的 micro-step 数
    opt_steps_per_epoch   = math.ceil(micro_steps_per_epoch / max(1, grad_accum))
    max_opt_steps_total   = epochs * opt_steps_per_epoch                   # lr 调度以“优化步”为刻度
    max_micro_steps_total = args.max_micro_steps

    scheduler = create_scheduler(optimizer, max_opt_steps_total, cfg)
    autocast_ctx, GradScalerCls, _ = _make_amp_tools(precision)
    scaler      = GradScalerCls(enabled=(precision == "fp16"))
    use_scaler  = bool(precision == "fp16")

    print(
        f"[train_llm] device={device} | precision={precision} | "
        f"workers(train)={getattr(train_loader, 'num_workers', 'NA')} | "
        f"workers(val)={getattr(val_loader, 'num_workers', 'NA')} | "
        f"optimizer={_cfg_get(cfg, 'optimizer', 'adamw')} | "
        f"grad_accum={grad_accum} | save_dir={save_dir}"
    )

    # ===== 窗口统计（关键修正：tok 与 tok_in 都按 micro-step 真实累计；逐层活跃只在“完成优化步”时入账）=====
    num_layers = int(_cfg_get(cfg, "num_hidden_layers", 12))
    window = {
        "loss": 0.0, "main": 0.0, "ffn": 0.0, "router": 0.0,
        "tok": 0,         # 有效 token（labels!=-100 且通过 attention_mask）
        "tok_in": 0,      # 输入 token：每个 micro-step 的 batch*(S-1)，用于 per-depth 分母
        "correct": 0, "steps": 0
    }
    window_active = [0.0 for _ in range(num_layers)]  # 已入账（按优化步）的逐层活跃 token
    window_lora_usage = [0.0 for _ in range(num_layers)]  # [新增] 已入账的 LoRA 使用 token 数
    pending_active = [0.0 for _ in range(num_layers)]  # 当前优化步内累计，等到 step 后再入账
    pending_lora_usage = [0.0 for _ in range(num_layers)]  # [新增] 当前优化步内累计

    global_micro_step = 0
    global_opt_step   = 0

    t0 = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # 早期评测配置（只在第 1 个 epoch）
    early_eval             = bool(_cfg_get(cfg, "early_eval", True))
    early_eval_steps       = int(_cfg_get(cfg, "early_eval_steps", 1000))
    early_eval_interval    = int(_cfg_get(cfg, "early_eval_interval", 100))
    early_eval_max_batches = int(_cfg_get(cfg, "early_eval_max_batches", 16))

    for ep in range(epochs):
        _apply_epoch_schedules(model.backbone, ep, cfg)
        for it, batch in enumerate(Prefetcher(train_loader, device), start=1):
            if (max_micro_steps_total is not None) and (global_micro_step >= max_micro_steps_total):
                break

            # 标记一个新的编译步开始（若可用）
            try:
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
            except Exception:
                pass

            # ===== 前向 =====
            with autocast_ctx(enabled=amp_enabled and str(device).startswith("cuda")):
                out = model(**batch)  # dict: logits + aux + stats

                # --- 主 CE（脚本统一计算，保障屏蔽一致） ---
                logits = out["logits"]
                labels = batch["labels"]
                attn   = batch.get("attention_mask", None)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                if attn is not None:
                    shift_attn   = attn[:, 1:].contiguous().to(torch.bool)
                    shift_labels = shift_labels.masked_fill(~shift_attn, -100)

                main_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )

                # --- 辅助损失（router/ffn_moe 已在 backbone 内带权） ---
                ld_aux     = out.get("_aux_for_train", {})
                router_w   = float(out.get("loss_dict", {}).get("router_aux_loss_weighted", 0.0))
                ffn_w      = float(out.get("loss_dict", {}).get("ffn_moe_aux_loss_weighted", 0.0))
                ffn_coeff  = float(_cfg_get(cfg, "ffn_moe_aux_coeff", 0.0))
                router_aux = ld_aux.get("router_aux", torch.zeros((), device=logits.device, dtype=logits.dtype))
                ffn_aux    = ld_aux.get("ffn_moe_aux", torch.zeros((), device=logits.device, dtype=logits.dtype))

                total_loss = main_loss + router_aux + ffn_coeff * ffn_aux
                micro_loss = total_loss / max(1, grad_accum)

            # ===== 统计（acc / 有效 token / 输入 token）=====
            with torch.no_grad():
                preds = shift_logits.argmax(dim=-1)
                valid = (shift_labels != -100)
                correct = int((preds[valid] == shift_labels[valid]).sum().item())
                ntok_scalar = int(valid.sum().item())
                bsz = batch["input_ids"].size(0)
                seql = batch["input_ids"].size(1)
                tok_in_step = int(bsz * max(seql - 1, 1))  # 分母：输入 token

                # 逐层 keep（仅做“待入账”累计，完成一次优化步后统一入账）
                keep_rates = out.get("stats", {}).get("active_keep_rates", None) \
                             or out.get("loss_dict", {}).get("active_keep_rates", None)

                # [新增] 读取 lora usage
                lora_rates = out.get("stats", {}).get("lora_adapter_usage_rates", None)
                if (lora_rates is None) or (len(lora_rates) != len(keep_rates)):
                    lora_rates = out.get("loss_dict", {}).get("lora_adapter_usage_rates", None)
                if keep_rates is not None:
                    a = float(tok_in_step)  # a0 (tokens entering layer 0)

                    kr_list = list(keep_rates)
                    # [修改] 确保 lora_rates 列表长度匹配
                    lr_list = list(lora_rates) if (lora_rates is not None and len(lora_rates) == len(keep_rates)) \
                        else [0.0] * len(keep_rates)

                    # [修改] 循环遍历 num_layers
                    for d in range(num_layers):
                        if d >= len(kr_list): break  # 安全退出

                        kr = float(kr_list[d])  # keep_rate at layer d
                        lr = float(lr_list[d])  # lora_usage_rate at layer d (among kept)

                        tokens_entering_d = a
                        tokens_kept_d = tokens_entering_d * kr
                        lora_using_tokens_d = tokens_kept_d * lr  # (Tokens @ d) * (KeepRate @ d) * (LoRAUsageRate @ d)

                        pending_active[d] += tokens_entering_d  #
                        pending_lora_usage[d] += lora_using_tokens_d  # [新增] 累加使用 lora 的 token

                        a = tokens_kept_d  # tokens entering next layer

                # timing（如存在）
                ts = out.get("stats", {}).get("timing_ms", None)
                if ts and bool(getattr(cfg, "timing_enable", False)):
                    tot = ts.get("total", 1e-6)
                    print(
                        f"[timing] attn={ts['attn']:.2f}ms({100 * ts['attn'] / tot:.1f}%) | "
                        f"pack={ts['pack']:.2f}ms({100 * ts['pack'] / tot:.1f}%) | "
                        f"ffn={ts['ffn']:.2f}ms({100 * ts['ffn'] / tot:.1f}%) | "
                        f"scatter={ts['scatter']:.2f}ms({100 * ts['scatter'] / tot:.1f}%) | "
                        f"router={ts['router']:.2f}ms({100 * ts['router'] / tot:.1f}%) | total={ts['total']:.2f}ms"
                    )

            # ===== 反传 =====
            if use_scaler:
                scaler.scale(micro_loss).backward()
            else:
                micro_loss.backward()

                # ===== 窗口即时累计（micro-step 级）=====
                # ★ 修复点：按 1/grad_accum 把每个 micro-step 的 loss/aux 累到窗口
            window["loss"] += float(total_loss.detach().cpu()) / max(1, grad_accum)
            window["main"] += float(main_loss.detach().cpu()) / max(1, grad_accum)
            window["router"] += float(router_w) / max(1, grad_accum)
            window["ffn"] += float(ffn_w) / max(1, grad_accum)

            window["tok"] += ntok_scalar
            window["tok_in"] += tok_in_step
            window["correct"] += correct

            global_micro_step += 1

            # ===== 完成一个优化步 =====
            finished_opt_step = (global_micro_step % grad_accum) == 0
            if finished_opt_step:
                if use_scaler:
                    scaler.step(optimizer);
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_opt_step += 1

                if router_freeze_steps > 0 and global_opt_step == router_freeze_steps:
                    unfrozen = 0
                    for n, p in model.named_parameters():
                        if "backbone.routers" in n:
                            p.requires_grad = True
                            unfrozen += 1
                    print(f"[router-freeze] UNFREEZE at opt_step={global_opt_step} | tensors={unfrozen}")

                # 入账逐层活跃 token
                for d in range(num_layers):
                    window_active[d] += float(pending_active[d])
                    window_lora_usage[d] += float(pending_lora_usage[d])  # [新增]
                    pending_active[d] = 0.0
                    pending_lora_usage[d] = 0.0  # [新增]

                window["steps"] += 1

            # ===== 日志（按 micro-step 触发打印，但统计口径以窗口为准）=====
            if (global_micro_step % log_interval) == 0:
                elapsed = time.time() - t0
                toks_s_eff = int(window["tok"] / max(elapsed, 1e-6))  # 用真实有效 token 吞吐

                acc       = (window["correct"] / max(window["tok"], 1)) * 100.0
                avg_loss  = window["loss"]   / max(1, window["steps"])
                avg_main  = window["main"]   / max(1, window["steps"])
                avg_ffn_w = window["ffn"]    / max(1, window["steps"])
                avg_rtr_w = window["router"] / max(1, window["steps"])

                # per-depth keep：用窗口累计的“输入 token”（tok_in）作为分母，更准确
                denom = max(1.0, float(window["tok_in"]))
                active_rates = [f"{(float(cnt) / denom) * 100:.1f}%" for cnt in window_active]
                lora_usage_rates = [f"{(float(cnt) / denom) * 100:.1f}%" for cnt in window_lora_usage]
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"step {global_micro_step} | loss={avg_loss:.4f} | main={avg_main:.4f} | "
                    f"router_w={avg_rtr_w:.6f} | ffn_w={avg_ffn_w:.6f} | acc={acc:.2f}% | toks/s={toks_s_eff}\n"
                    f"[active] per-depth keep: " + " -> ".join(active_rates) + f"\n"  # [修改]
                                                                               f"[active] lora adapter usage: " + " -> ".join(
                        lora_usage_rates) + f"\n"  # [新增]
                                            f"[lr] {lr_now:.3e}\n"  # [修改]
                                            f"[progress] {it}/{micro_steps_per_epoch} ({100.0 * it / micro_steps_per_epoch:.2f}%)"
                )

                # 重置窗口（逐层活跃 & tok 计数一并清零；pending_active 在完成优化步后已清空）
                window = {"loss": 0.0, "main": 0.0, "ffn": 0.0, "router": 0.0, "tok": 0, "tok_in": 0, "correct": 0,
                          "steps": 0}
                window_active = [0.0 for _ in range(num_layers)]
                window_lora_usage = [0.0 for _ in range(num_layers)]  # [新增]
                t0 = time.time()

            # ===== 早期评估（仅第 1 个 epoch）=====
            if (ep == 0 and early_eval
                    and global_micro_step <= early_eval_steps
                    and (global_micro_step % early_eval_interval) == 0):
                val_loss, val_ppl, val_excess = evaluate(
                    model, val_loader, device,
                    precision=precision,
                    max_batches=early_eval_max_batches
                )
                print(f"[early-eval] step {global_micro_step} | val_loss={val_loss:.4f} | val_excess={val_excess:.4f} | val_ppl={val_ppl:.2f}")

        # ===== epoch 结束：评估与存档 =====
        val_loss, val_ppl, val_excess = evaluate(model, val_loader, device, precision=precision, max_batches=50)
        print(f"[epoch {ep + 1}] val_loss={val_loss:.4f} | val_excess={val_excess:.4f} | val_ppl={val_ppl:.2f}")

        ckpt_path = os.path.join(save_dir, f"ep{ep+1}_micro{global_micro_step}_opt{global_opt_step}.pt")
        torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                    "epoch": ep+1, "global_micro_step": global_micro_step,
                    "global_opt_step": global_opt_step}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    print("Training finished.")

if __name__ == "__main__":
    main()
