# fp8_utils.py
from contextlib import contextmanager
import os

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    _HAS_TE = True
except Exception:
    te = None
    DelayedScaling = None
    Format = None
    _HAS_TE = False

@contextmanager
def fp8_autocast_if_available(enabled: bool, margin: int = 0):
    """
    开启 TransformerEngine 的 FP8 autocast（E4M3/E5M2, DelayedScaling）。
    若 TE 未安装或 enabled=False，则空上下文。
    """
    if _HAS_TE and enabled:
        recipe = DelayedScaling(fp8_format=Format.E4M3,  # fwd 激活/权重
                                amax_history_len=16,
                                amax_compute_algo="max",
                                margin=margin)
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            yield
    else:
        yield

def make_linear(hidden_in: int, hidden_out: int, bias: bool, use_te: bool):
    """
    返回 Linear 模块：优先 TE.Linear（支持 FP8），否则 nn.Linear
    """
    import torch.nn as nn
    if _HAS_TE and use_te:
        return te.Linear(hidden_in, hidden_out, bias=bias)
    else:
        return nn.Linear(hidden_in, hidden_out, bias=bias)

def has_te() -> bool:
    return _HAS_TE
