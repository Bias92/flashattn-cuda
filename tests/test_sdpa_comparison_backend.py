"""CPU routing/metadata checks for the eager SDPA comparison adapter."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from scratch_vllm import register
from scratch_vllm.backend import ScratchBackend
from scratch_vllm.comparison import (
    DenseComparisonBackend, DenseComparisonImpl, DenseComparisonMetadataBuilder,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadataBuilder
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class ComparisonTest(unittest.TestCase):
    def test_registration(self):
        with patch.dict(os.environ, {"SCRATCH_COMPARISON_BACKEND": "SDPA_FLASH"}):
            register()
            self.assertIs(AttentionBackendEnum.CUSTOM.get_class(), DenseComparisonBackend)
        with patch.dict(os.environ):
            os.environ.pop("SCRATCH_COMPARISON_BACKEND", None)
            register()
            self.assertIs(AttentionBackendEnum.CUSTOM.get_class(), ScratchBackend)
        self.assertEqual(DenseComparisonMetadataBuilder.get_cudagraph_support(None, None),
                         AttentionCGSupport.NEVER)

    def test_metadata_ragged_and_single_request(self):
        builder = object.__new__(DenseComparisonMetadataBuilder)
        for lengths, starts in (([7, 33, 51], [0, 1, 1, 2]), ([17], [0, 1])):
            common = SimpleNamespace(num_reqs=len(lengths), max_seq_len=max(lengths),
                seq_lens=torch.tensor(lengths), query_start_loc_cpu=torch.tensor(starts))
            result = SimpleNamespace(max_query_len=1, use_cascade=False)
            with patch.object(FlashAttentionMetadataBuilder, "build", return_value=result):
                meta = builder.build(0, common)
            self.assertEqual(meta.dense_requests,
                             [(i, starts[i], starts[i + 1], n) for i, n in enumerate(lengths)])

    def test_decode_padding_and_prefill(self):
        impl = object.__new__(DenseComparisonImpl)
        impl.mode, impl.scale = "SDPA_MATH", 0.125
        impl.scratch_calls, impl.prefill_calls = 0, 0
        q = torch.randn(2, 8, 64, dtype=torch.float32)
        cache = torch.randn(2, 8, 16, 2, 64)
        meta = SimpleNamespace(max_query_len=1, use_cascade=False,
            dense_requests=[(0, 0, 1, 17), (1, 1, 1, 7), (2, 1, 2, 19)],
            block_table=torch.tensor([[1, 0], [-999, -999], [3, 5]], dtype=torch.int32))
        output = torch.empty_like(q)
        with patch.dict(os.environ, {"SCRATCH_DECODE_AUDIT": "0"}):
            self.assertIs(impl.forward(None, q, None, None, cache, meta, output), output)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(impl.scratch_calls, 1)
        with patch.object(FlashAttentionImpl, "forward", return_value=output) as parent:
            for metadata in (None, SimpleNamespace(max_query_len=3, use_cascade=False),
                             SimpleNamespace(max_query_len=1, use_cascade=True)):
                self.assertIs(impl.forward(None, q, None, None, cache, metadata, output), output)
            self.assertEqual(parent.call_count, 3)
            self.assertEqual(impl.scratch_calls, 1)


if __name__ == "__main__":
    unittest.main()
