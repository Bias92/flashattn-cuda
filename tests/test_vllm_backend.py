"""Backend routing tests. No model weights or CUDA kernels required."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "vllm"))
from scratch_vllm import register
from scratch_vllm.backend import ScratchBackend, ScratchImpl
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class BackendTest(unittest.TestCase):
    def test_registration(self):
        register()
        self.assertIs(AttentionBackendEnum.CUSTOM.get_class(), ScratchBackend)
        self.assertEqual(ScratchBackend.supported_dtypes, [torch.float16])
        self.assertFalse(ScratchBackend.supports_head_size(256))
        self.assertFalse(ScratchBackend.supports_kv_cache_dtype("fp8"))

    def make_impl(self):
        impl = object.__new__(ScratchImpl)
        impl.scratch_capacity, impl.scratch_chunk = 128, 64
        impl.num_heads, impl.num_kv_heads, impl.head_size = 8, 2, 64
        impl.scale = 0.125
        impl.scratch = SimpleNamespace(out=Mock())
        impl.scratch_calls, impl.prefill_calls, impl.audit = 0, 0, False
        return impl

    def test_decode_native_no_gather(self):
        impl = self.make_impl()
        q = torch.zeros(3, 8, 64, dtype=torch.float16)
        cache = torch.zeros(2, 20, 16, 2, 64, dtype=torch.float16)
        output = torch.empty_like(q)
        meta = SimpleNamespace(max_query_len=1, use_cascade=False,
                               seq_lens=torch.tensor([7, 33, 64], dtype=torch.int32),
                               block_table=torch.zeros(3, 8, dtype=torch.int32),
                               query_start_loc=torch.arange(4, dtype=torch.int32))
        returned = impl.forward(None, q, None, None, cache, meta, output)
        self.assertIs(returned, output)
        call = impl.scratch.out.call_args.args
        self.assertEqual(call[1].data_ptr(), cache[0].data_ptr())
        self.assertEqual(call[2].data_ptr(), cache[1].data_ptr())
        self.assertIs(call[3], meta.block_table)
        self.assertIs(call[4], meta.seq_lens)
        self.assertEqual(call[7].numel(), 3 * 8 * 2 * 66)
        self.assertEqual(impl.scratch_calls, 1)

    def test_prefill_and_cascade_fallback(self):
        impl = self.make_impl()
        sentinel = object()
        with patch.object(FlashAttentionImpl, "forward", return_value=sentinel) as base:
            for meta in (None, SimpleNamespace(max_query_len=3, use_cascade=False),
                         SimpleNamespace(max_query_len=1, use_cascade=True)):
                self.assertIs(impl.forward(None, None, None, None, None, meta), sentinel)
            self.assertEqual(base.call_count, 3)
            self.assertEqual(impl.scratch_calls, 0)


if __name__ == "__main__":
    unittest.main()
