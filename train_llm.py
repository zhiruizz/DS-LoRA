"""Train native MoR or MoR + DS-LoRA for causal language modeling."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from MoR_LanguageModel import MoR_Config, MoR_LanguageModel


def flatten_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def load_config(path: str) -> MoR_Config:
    with open(path, "r", encoding="utf-8") as handle:
        return MoR_Config(**flatten_config(json.load(handle)))


def cfg_get(cfg: MoR_Config, key: str, default: Any = None) -> Any:
    return getattr(cfg, key, default)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: MoR_Config, tokenizer):
    from datasets import DatasetDict, load_dataset, load_from_disk

    batch_size = int(cfg_get(cfg, "batch_size", 8))
    num_workers = int(cfg_get(cfg, "num_workers", 4))
    max_len = int(cfg_get(cfg, "max_seq_len", 512))
    proc_path = cfg_get(cfg, "processed_dataset_path", None)
    dataset_name = cfg_get(cfg, "dataset_name", None)
    split_train = cfg_get(cfg, "dataset_split_train", "train")
    split_val = cfg_get(cfg, "dataset_split_val", "validation")

    def pick_split(dataset, split):
        if isinstance(dataset, DatasetDict):
            if split in dataset:
                return dataset[split]
            for fallback in ("validation", "valid", "train"):
                if fallback in dataset:
                    return dataset[fallback]
            return next(iter(dataset.values()))
        return dataset

    if proc_path and os.path.exists(proc_path):
        loaded = load_from_disk(proc_path)
        train_ds = pick_split(loaded, split_train)
        val_ds = pick_split(loaded, split_val)
    elif dataset_name:
        loaded = load_dataset(dataset_name)
        train_ds = pick_split(loaded, split_train)
        val_ds = pick_split(loaded, split_val)
    else:
        raise ValueError("Set processed_dataset_path or dataset_name in config.json.")

    def tokenize_if_needed(dataset):
        if "input_ids" in dataset.column_names:
            return dataset
        if "text" not in dataset.column_names:
            raise ValueError("Dataset must contain either input_ids or text.")

        def tokenize(batch):
            encoded = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_len,
                padding=False,
                return_attention_mask=True,
            )
            encoded["labels"] = encoded["input_ids"].copy()
            return encoded

        remove_columns = [col for col in dataset.column_names if col != "text"]
        return dataset.map(tokenize, batched=True, remove_columns=remove_columns)

    train_ds = tokenize_if_needed(train_ds)
    val_ds = tokenize_if_needed(val_ds)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pad_batch(examples):
        max_batch_len = max(len(ex["input_ids"]) for ex in examples)
        input_ids = torch.full((len(examples), max_batch_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_batch_len), dtype=torch.long)
        labels = torch.full((len(examples), max_batch_len), -100, dtype=torch.long)
        for row, ex in enumerate(examples):
            ids = torch.tensor(ex["input_ids"], dtype=torch.long)
            mask = torch.tensor(ex.get("attention_mask", [1] * len(ids)), dtype=torch.long)
            input_ids[row, : ids.numel()] = ids
            attention_mask[row, : mask.numel()] = mask
            labels[row, : ids.numel()] = ids
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=pad_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=pad_batch,
    )
    return train_loader, val_loader


def autocast_context(device: str, precision: str):
    enabled = str(device).startswith("cuda") and precision in ("bf16", "fp16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype, enabled=enabled)


def move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_main_loss(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if attention_mask is not None:
        shift_mask = attention_mask[:, 1:].contiguous().to(torch.bool)
        shift_labels = shift_labels.masked_fill(~shift_mask, -100)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


@torch.no_grad()
def evaluate(model, dataloader, device: str, precision: str, max_batches: int = 50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    vocab_size = None

    for step, batch in enumerate(dataloader, start=1):
        if step > max_batches:
            break
        batch = move_to_device(batch, device)
        with autocast_context(device, precision):
            out = model(**batch)
            logits = out["logits"].float()
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask")
            vocab_size = int(logits.size(-1))
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            valid = shift_labels != -100
            if attention_mask is not None:
                valid = valid & attention_mask[:, 1:].contiguous().to(torch.bool)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.masked_fill(~valid, -100).view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += float(loss)
            total_tokens += int(valid.sum().item())

    model.train()
    if total_tokens == 0:
        return float("nan"), float("nan"), float("nan")
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 80 else float("inf")
    excess = avg_loss - math.log(vocab_size) if vocab_size else float("nan")
    return avg_loss, ppl, excess


def build_optimizer(cfg: MoR_Config, model):
    base_lr = float(cfg_get(cfg, "lr", 3e-5))
    weight_decay = float(cfg_get(cfg, "weight_decay", 0.0))
    beta1 = float(cfg_get(cfg, "beta1", 0.9))
    beta2 = float(cfg_get(cfg, "beta2", 0.95))
    opt_name = str(cfg_get(cfg, "optimizer", "adamw")).lower()

    decay, nodecay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            nodecay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]

    if opt_name in ("adamw8bit", "adamw_8bit"):
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(groups, lr=base_lr, betas=(beta1, beta2))
        except Exception:
            print("[optimizer] bitsandbytes unavailable; falling back to torch AdamW.")

    return torch.optim.AdamW(groups, lr=base_lr, betas=(beta1, beta2))


def build_scheduler(optimizer, max_steps: int, cfg: MoR_Config):
    warmup_steps = int(cfg_get(cfg, "warmup_steps", 0))

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return step / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--max_micro_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg_get(cfg, "seed", 123)))

    device = str(cfg_get(cfg, "device", "cuda:0"))
    precision = str(cfg_get(cfg, "precision", "bf16")).lower()
    tokenizer = AutoTokenizer.from_pretrained(str(cfg_get(cfg, "tokenizer_path", "./tokenizer")), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg.vocab_size = len(tokenizer)

    model = MoR_LanguageModel(cfg).to(device)
    if precision == "bf16":
        model.to(dtype=torch.bfloat16)
    elif precision == "fp16":
        model.to(dtype=torch.float16)

    train_loader, val_loader = build_dataloaders(cfg, tokenizer)
    optimizer = build_optimizer(cfg, model)
    grad_accum = int(cfg_get(cfg, "gradient_accumulation_steps", 1))
    epochs = int(cfg_get(cfg, "train_epochs", 1))
    log_interval = int(cfg_get(cfg, "log_interval", 100))
    save_dir = str(cfg_get(cfg, "save_dir", "./checkpoints"))
    os.makedirs(save_dir, exist_ok=True)

    max_opt_steps = epochs * math.ceil(len(train_loader) / max(1, grad_accum))
    scheduler = build_scheduler(optimizer, max_opt_steps, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and str(device).startswith("cuda"))

    global_micro = 0
    global_opt = 0
    window_loss = 0.0
    window_tokens = 0
    start = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    print(f"[train] device={device} precision={precision} grad_accum={grad_accum} save_dir={save_dir}")
    for epoch in range(epochs):
        for batch in train_loader:
            if args.max_micro_steps is not None and global_micro >= args.max_micro_steps:
                break
            batch = move_to_device(batch, device)
            with autocast_context(device, precision):
                out = model(**batch)
                main_loss = compute_main_loss(out["logits"], batch["labels"], batch.get("attention_mask"))
                aux = out.get("_aux_for_train", {})
                total_loss = main_loss + aux.get("router_aux", out["logits"].new_tensor(0.0))
                loss = total_loss / max(1, grad_accum)

            scaler.scale(loss).backward()
            global_micro += 1
            valid_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
            window_loss += float(total_loss.detach().cpu())
            window_tokens += valid_tokens

            if global_micro % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_opt += 1

            if global_micro % log_interval == 0:
                elapsed = max(time.time() - start, 1e-6)
                avg_loss = window_loss / max(1, log_interval)
                toks_s = int(window_tokens / elapsed)
                stats = out.get("stats", {})
                keep = " -> ".join(f"{x * 100:.1f}%" for x in stats.get("active_keep_rates", []))
                lora = " -> ".join(f"{x * 100:.1f}%" for x in stats.get("lora_adapter_usage_rates", []))
                print(
                    f"step {global_micro} opt {global_opt} | loss={avg_loss:.4f} | toks/s={toks_s} | "
                    f"keep=[{keep}] | ds_lora=[{lora}] | lr={scheduler.get_last_lr()[0]:.3e}"
                )
                window_loss = 0.0
                window_tokens = 0
                start = time.time()

            if bool(cfg_get(cfg, "early_eval", True)) and epoch == 0:
                early_steps = int(cfg_get(cfg, "early_eval_steps", 1000))
                early_interval = int(cfg_get(cfg, "early_eval_interval", 100))
                if global_micro <= early_steps and global_micro % early_interval == 0:
                    val_loss, val_ppl, val_excess = evaluate(
                        model,
                        val_loader,
                        device,
                        precision,
                        max_batches=int(cfg_get(cfg, "early_eval_max_batches", 16)),
                    )
                    print(f"[early-eval] step={global_micro} val_loss={val_loss:.4f} val_excess={val_excess:.4f} val_ppl={val_ppl:.2f}")

        val_loss, val_ppl, val_excess = evaluate(model, val_loader, device, precision)
        print(f"[epoch {epoch + 1}] val_loss={val_loss:.4f} val_excess={val_excess:.4f} val_ppl={val_ppl:.2f}")
        ckpt = os.path.join(save_dir, f"epoch{epoch + 1}_micro{global_micro}_opt{global_opt}.pt")
        torch.save({"model": model.state_dict(), "cfg": vars(cfg), "epoch": epoch + 1}, ckpt)
        print(f"[checkpoint] {ckpt}")

        if args.max_micro_steps is not None and global_micro >= args.max_micro_steps:
            break


if __name__ == "__main__":
    main()
