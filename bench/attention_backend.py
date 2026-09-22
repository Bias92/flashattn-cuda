"""Conservative Transformers adapter for dense prefill and single-token decode."""
import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface


def register_backend(name, decode, prefill=None):
    calls = {"decode": 0, "prefill": 0, "fallback": 0}

    def attention(module, query, key, value, attention_mask, scaling=None,
                  dropout=0.0, is_causal=None, **kwargs):
        compatible = (
            query.is_cuda and query.dtype == key.dtype == value.dtype == torch.float16
            and query.shape[-1] in (64, 128) and dropout == 0.0
            and attention_mask is None
            and not kwargs.get("output_attentions", False)
            and kwargs.get("head_mask") is None
            and not (torch.is_grad_enabled() and any(x.requires_grad for x in (query, key, value)))
        )
        # With SDPA mask construction registered, None means no padding or
        # extra cache restrictions. Explicit causal=True with Q=1 has different
        # (upper-left) semantics, so that case must remain with SDPA.
        if compatible and query.shape[2] == 1 and is_causal is not True:
            calls["decode"] += 1
            out = decode.forward_only(query, key, value, scaling)
            return out.transpose(1, 2).contiguous(), None
        default_scale = scaling is None or scaling == query.shape[-1] ** -0.5
        if compatible and prefill is not None and query.shape[2] == key.shape[2] and default_scale:
            causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
            calls["prefill"] += 1
            out = prefill.forward_only(query.contiguous(), key.contiguous(), value.contiguous(), causal)
            return out.transpose(1, 2).contiguous(), None
        calls["fallback"] += 1
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      scaling=scaling, dropout=dropout, is_causal=is_causal, **kwargs)

    AttentionInterface.register(name, attention)
    AttentionMaskInterface.register(name, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    return attention, calls
