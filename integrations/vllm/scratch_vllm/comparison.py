"""Eager-only dense SDPA adapters, including all paged-to-dense costs.

These are integration baselines, not native paged implementations of SDPA.
Prefill and KV writes are still handled by vLLM FlashAttention.
"""

from contextlib import nullcontext
import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl, FlashAttentionMetadataBuilder,
)

from .backend import ScratchBackend


MODES = {
    "SDPA_AUTO": None,
    "SDPA_FLASH": SDPBackend.FLASH_ATTENTION,
    "SDPA_CUDNN": SDPBackend.CUDNN_ATTENTION,
    "SDPA_EFFICIENT": SDPBackend.EFFICIENT_ATTENTION,
    "SDPA_MATH": SDPBackend.MATH,
}
EXPECTED_OPERATORS = {
    "SDPA_FLASH": "aten::_scaled_dot_product_flash_attention",
    "SDPA_CUDNN": "aten::_scaled_dot_product_cudnn_attention",
    "SDPA_EFFICIENT": "aten::_scaled_dot_product_efficient_attention",
    "SDPA_MATH": "aten::_scaled_dot_product_attention_math",
}
logger = init_logger("vllm.scratch_comparison")
_audited_modes = {}


def configure_process_backend(mode):
    # Each server has its own worker process. Select once, not once per layer
    # per token, so repeated Python flag changes do not handicap SDPA.
    for name, setter in (
        ("SDPA_FLASH", torch.backends.cuda.enable_flash_sdp),
        ("SDPA_CUDNN", torch.backends.cuda.enable_cudnn_sdp),
        ("SDPA_EFFICIENT", torch.backends.cuda.enable_mem_efficient_sdp),
        ("SDPA_MATH", torch.backends.cuda.enable_math_sdp),
    ):
        setter(mode in ("SDPA_AUTO", name))


def dense_attention(query, key, value, mode, scale, *, select_backend=True):
    backend = MODES[mode]
    context = nullcontext() if backend is None or not select_backend else sdpa_kernel(backend)
    # PyTorch 2.10 efficient attention does not implement GQA. Its explicit
    # expansion is part of this adapter's measured time and memory usage.
    if mode == "SDPA_EFFICIENT" and query.shape[1] != key.shape[1]:
        group = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(group, dim=1)
        value = value.repeat_interleave(group, dim=1)
    with context:
        return scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False,
            scale=scale, enable_gqa=query.shape[1] != key.shape[1],
        )


def gather_request(kv_cache, page_ids, length):
    """Return compact [1,Hkv,N,D] K/V; gather and repack are intentionally timed."""
    page_size = kv_cache.shape[2]
    count = (length + page_size - 1) // page_size
    pages = kv_cache.index_select(1, page_ids[:count])
    dense = pages.flatten(1, 2)[:, :length].permute(0, 2, 1, 3).contiguous()
    key, value = dense.unbind(0)
    return key.unsqueeze(0), value.unsqueeze(0)


def audited_attention(q, k, v, mode, scale):
    if os.environ.get("SCRATCH_DECODE_AUDIT") != "1" or mode in _audited_modes:
        return dense_attention(q, k, v, mode, scale, select_backend=False)
    # One initial warmup call per process; never profile the timed requests.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        result = dense_attention(q, k, v, mode, scale, select_backend=False)
    operators = sorted(e.key for e in trace.key_averages() if "scaled_dot_product" in e.key)
    if mode in EXPECTED_OPERATORS and EXPECTED_OPERATORS[mode] not in operators:
        raise RuntimeError(f"Forced backend did not execute: {mode}: {operators}")
    _audited_modes[mode] = operators
    logger.info("DENSE_SDPA_OPERATOR_AUDIT mode=%s operators=%s", mode, operators)
    return result


class DenseComparisonMetadataBuilder(FlashAttentionMetadataBuilder):
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return AttentionCGSupport.NEVER

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        result = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if result.max_query_len == 1 and not result.use_cascade:
            common = common_attn_metadata
            # B=1 uses the already available host maximum, avoiding a D2H sync.
            # For B>1 copy once per step in the shared builder, not once per layer.
            lengths = ([common.max_seq_len] if common.num_reqs == 1 else
                       common.seq_lens[:common.num_reqs].cpu().tolist())
            starts = common.query_start_loc_cpu[:common.num_reqs + 1].tolist()
            result.dense_requests = [(i, starts[i], starts[i + 1], length)
                                     for i, length in enumerate(lengths)]
        return result


class DenseComparisonBackend(ScratchBackend):
    @staticmethod
    def get_impl_cls():
        return DenseComparisonImpl

    @staticmethod
    def get_builder_cls():
        return DenseComparisonMetadataBuilder


class DenseComparisonImpl(FlashAttentionImpl):
    can_return_lse_for_decode = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config()
        self.mode = os.environ["SCRATCH_COMPARISON_BACKEND"]
        if self.mode not in MODES:
            raise ValueError(f"Unknown dense comparison mode: {self.mode}")
        if not config.model_config.enforce_eager:
            raise ValueError("Dense comparison requires enforce_eager for ALL baselines")
        if (self.total_cp_world_size != 1 or config.parallel_config.tensor_parallel_size != 1
                or config.parallel_config.pipeline_parallel_size != 1
                or self.alibi_slopes is not None or self.sliding_window != (-1, -1)
                or self.logits_soft_cap != 0 or self.sinks is not None):
            raise ValueError("Dense comparison supports plain single-GPU decoder attention only")
        configure_process_backend(self.mode)
        self.scratch_calls = 0
        self.prefill_calls = 0
        self.last_route = None

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        if (attn_metadata is None or attn_metadata.max_query_len != 1
                or attn_metadata.use_cascade):
            self.prefill_calls += 1
            return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                   output, output_scale, output_block_scale)
        if output is None or output_scale is not None or output_block_scale is not None:
            raise ValueError("Dense comparison requires an FP16 output buffer")
        for request, start, end, length in attn_metadata.dense_requests:
            if end == start:
                continue
            if end - start != 1 or length <= 0:
                raise ValueError("Expected one decode query and nonempty KV")
            k, v = gather_request(kv_cache, attn_metadata.block_table[request], length)
            q = query[start:end].transpose(0, 1).unsqueeze(0)
            result = audited_attention(q, k, v, self.mode, self.scale)
            output[start:end].copy_(result.squeeze(0).transpose(0, 1))
        self.scratch_calls += 1
        self.last_route = dict(backend=self.mode, paged_to_dense=True, eager=True,
                               initial_executed_operators=_audited_modes.get(self.mode))
        if self.scratch_calls == 1:
            logger.info("DENSE_SDPA_DECODE active: mode=%s; paged gather/repack included; "
                        "prefill remains FLASH_ATTN", self.mode)
        return output
