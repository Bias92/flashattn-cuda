"""Decode-only experiment: FlashAttention prefill and vLLM KV writes unchanged."""

import os

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl

from .loader import load_extension

logger = init_logger("vllm.scratch_attention")


class ScratchBackend(FlashAttentionBackend):
    supported_dtypes = [torch.float16]
    supported_kv_cache_dtypes = ["auto", "float16"]

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return ScratchImpl

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16, 32, 64, 128]

    @classmethod
    def supports_head_size(cls, head_size):
        return head_size in (64, 128)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype):
        return kv_cache_dtype in (None, "auto", "float16")

    @classmethod
    def supports_attn_type(cls, attn_type):
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sink(cls):
        return False


class ScratchImpl(FlashAttentionImpl):
    can_return_lse_for_decode = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config()
        self.scratch_capacity = config.model_config.max_model_len
        self.scratch_chunk = int(os.environ.get("SCRATCH_DECODE_CHUNK", "128"))
        if self.scratch_chunk not in (64, 128, 256, 512):
            raise ValueError("SCRATCH_DECODE_CHUNK must be 64,128,256,512")
        if (self.total_cp_world_size != 1 or config.parallel_config.tensor_parallel_size != 1
                or config.parallel_config.pipeline_parallel_size != 1):
            raise ValueError("Initial scratch backend supports one GPU, TP=1/DCP=1 only")
        if (self.alibi_slopes is not None or self.sliding_window != (-1, -1)
                or self.logits_soft_cap != 0 or self.sinks is not None):
            raise ValueError("Scratch decode does not support ALiBi, windows, softcap or sinks")
        self.scratch = load_extension()
        self.scratch_calls = 0
        self.prefill_calls = 0
        self.audit = os.environ.get("SCRATCH_DECODE_AUDIT") == "1"
        self.last_route = None

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        if self.audit:
            self.last_route = None if attn_metadata is None else {
                "max_query_len": attn_metadata.max_query_len,
                "use_cascade": attn_metadata.use_cascade,
            }
        if (attn_metadata is None or attn_metadata.max_query_len != 1
                or attn_metadata.use_cascade):
            self.prefill_calls += 1
            return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                   output, output_scale, output_block_scale)
        if output is None or output_scale is not None or output_block_scale is not None:
            raise ValueError("Scratch decode requires an FP16 output buffer")
        key_cache, value_cache = kv_cache.unbind(0)
        batch = attn_metadata.seq_lens.numel()
        # Fixed split capacity keeps captured launch dimensions valid as KV grows.
        splits = (self.scratch_capacity + self.scratch_chunk - 1) // self.scratch_chunk
        workspace = torch.empty(
            (batch, self.num_heads, splits, self.head_size + 2),
            dtype=torch.float32, device=query.device,
        )
        self.scratch.out(query, key_cache, value_cache, attn_metadata.block_table,
                         attn_metadata.seq_lens, attn_metadata.query_start_loc,
                         output, workspace, self.scratch_capacity, self.scale,
                         self.scratch_chunk)
        self.scratch_calls += 1
        if self.audit and self.scratch_calls == 1:
            logger.info("SCRATCH_PAGED_DECODE active: heads=%d kv_heads=%d D=%d; "
                        "prefill remains FLASH_ATTN", self.num_heads,
                        self.num_kv_heads, self.head_size)
        return output
