"""Prefill and decode attention on the scratch kernels; vLLM keeps everything else.

The batch is reordered so that single-token requests come first, the way vLLM's own
split-kernel backends ask for it. Per attention call:

  - the single-token prefix (decodes, and one-token tails of a chunked prompt)
      -> the paged decode kernel of scratch_vllm, unchanged. A batch that holds nothing else
         goes to the parent class as it is, so it can be captured into a full CUDA graph.
  - every other request, one launch each of cuda/attention_forward.cu:
      nothing in the KV cache yet  -> K and V read from the projections vLLM hands over
      a chunk of a longer prompt, or a prompt with a prefix-cache hit
                                   -> K and V read straight out of the paged cache
    Query and output are vLLM's own buffers, used in place as transposed views.
  - cascade attention, fused output quantization, or a batch whose lengths are not known on
    the host -> vLLM's FlashAttention, counted as a fallback.

The counters say, per layer, how many calls and how many tokens took each route. They count
executions of forward(): steps replayed from a captured CUDA graph do not pass through it,
which only ever applies to batches of nothing but single-token requests.

Not supported (the parent class refuses them when the engine starts): sliding windows, ALiBi,
soft caps, sinks, fp8 caches, tensor/pipeline/context parallelism. Speculative decoding is not
supported either: the host-side sequence lengths used here are exact only without it.
"""

import copy
from dataclasses import dataclass, fields
from typing import NamedTuple

from scratch_vllm.backend import ScratchBackend, ScratchImpl
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from .loader import load_prefill_extension


class PrefillRequest(NamedTuple):
    row: int      # position in the batch, which is also its row of the block table
    start: int    # first token of the request in the flattened batch
    end: int      # one past its last token
    n_kv: int     # tokens it attends to: those already in the cache plus its own


@dataclass
class ScratchPrefillMetadata(FlashAttentionMetadata):
    num_decodes: int = 0
    num_decode_tokens: int = 0
    # Requests after the single-token prefix. None: their lengths are not known on the
    # host, so FlashAttention serves the whole batch.
    prefills: list[PrefillRequest] | None = None
    # The single-token prefix as a batch of its own, for the decode kernel.
    decode_view: FlashAttentionMetadata | None = None


class ScratchPrefillMetadataBuilder(FlashAttentionMetadataBuilder):
    # The prefill route loops over requests on the host, so only batches of nothing but
    # single-token requests may be captured into a full CUDA graph.
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        common = common_attn_metadata
        num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(common, decode_threshold=1)
        md = ScratchPrefillMetadata(**{f.name: getattr(base, f.name) for f in fields(base)},
                                    num_decodes=num_decodes, num_decode_tokens=num_decode_tokens,
                                    prefills=prefill_requests(common, num_decodes))
        if md.prefills and num_decodes:
            view = copy.copy(base)
            view.num_actual_tokens = num_decode_tokens
            view.max_query_len = 1
            view.query_start_loc = base.query_start_loc[:num_decodes + 1]
            view.seq_lens = base.seq_lens[:num_decodes]
            view.block_table = base.block_table[:num_decodes]
            md.decode_view = view
        return md


def prefill_requests(common, num_decodes):
    """The requests after the single-token prefix, with their lengths taken from the host
    copies the model runner keeps (reading them costs no device sync)."""
    if num_decodes == common.num_reqs:
        return []
    if not common.causal or common._seq_lens_cpu is None:
        return None
    starts = common.query_start_loc_cpu.tolist()
    seq_lens = common._seq_lens_cpu.tolist()
    return [PrefillRequest(row, starts[row], starts[row + 1], seq_lens[row])
            for row in range(num_decodes, common.num_reqs)
            if starts[row + 1] > starts[row]]


class ScratchPrefillBackend(ScratchBackend):
    @staticmethod
    def get_impl_cls():
        return ScratchPrefillImpl

    @staticmethod
    def get_builder_cls():
        return ScratchPrefillMetadataBuilder


def as_bhnd(tokens):
    """[n, H, D] token-major rows of one request as the [1, H, n, D] view the kernel takes."""
    return tokens.unsqueeze(0).transpose(1, 2)


class ScratchPrefillImpl(ScratchImpl):
    ROUTES = ("prefill_dense", "prefill_paged", "decode_in_mixed", "decode_only", "flash_fallback")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefill_kernel = load_prefill_extension()
        self.route_calls = dict.fromkeys(self.ROUTES, 0)
        self.route_tokens = dict.fromkeys(self.ROUTES, 0)

    def count(self, route, tokens):
        self.route_calls[route] += 1
        self.route_tokens[route] += tokens

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        md = attn_metadata
        if md is None:   # profiling run: no attention is computed
            return super().forward(layer, query, key, value, kv_cache, md,
                                   output, output_scale, output_block_scale)
        prefills = getattr(md, "prefills", None)
        if (prefills is None or md.use_cascade or output is None
                or output_scale is not None or output_block_scale is not None):
            if md.max_query_len > 1:
                self.count("flash_fallback", md.num_actual_tokens)
            return super().forward(layer, query, key, value, kv_cache, md,
                                   output, output_scale, output_block_scale)
        if not prefills:
            self.count("decode_only", md.num_actual_tokens)
            return super().forward(layer, query, key, value, kv_cache, md, output)

        # The KV cache already holds this step's keys and values: vLLM runs
        # do_kv_cache_update before forward().
        n = md.num_decode_tokens
        if n:
            self.count("decode_in_mixed", n)
            super().forward(layer, query[:n], key[:n], value[:n], kv_cache, md.decode_view,
                            output[:n])
        key_cache, value_cache = kv_cache.unbind(0)
        for p in prefills:
            q, out = as_bhnd(query[p.start:p.end]), as_bhnd(output[p.start:p.end])
            if p.n_kv == p.end - p.start:
                self.count("prefill_dense", p.end - p.start)
                self.prefill_kernel.forward_only(q, as_bhnd(key[p.start:p.end]),
                                                 as_bhnd(value[p.start:p.end]), True, self.scale, out)
            else:
                self.count("prefill_paged", p.end - p.start)
                self.prefill_kernel.forward_paged(q, key_cache, value_cache, md.block_table[p.row],
                                                  p.n_kv, self.scale, out)
        return output
