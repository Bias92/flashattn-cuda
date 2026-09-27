"""Test dense layout selection on meta tensors, without GPU allocation or launches."""
from pathlib import Path
import unittest

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]


def tensors(b=2, h=8, h_kv=2, n_q=64, n_kv=None, d=64, layout="contiguous"):
    n_kv = n_q if n_kv is None else n_kv

    def tensor(heads, length, role):
        if layout == "token_major" or layout == f"{role}_token_major":
            return torch.empty(b, length, heads, d, device="meta").transpose(1, 2)
        if layout == f"{role}_padded_rows":
            return torch.empty(b, heads, length, d + 8, device="meta")[..., :d]
        if layout == f"{role}_padded_heads":
            return torch.empty(b, heads, length + 1, d, device="meta")[:, :, :length]
        return torch.empty(b, heads, length, d, device="meta")

    return (tensor(h, n_q, "q"), tensor(h_kv, n_kv, "k"),
            tensor(h_kv, n_kv, "v"), tensor(h, n_q, "o"))


class ForwardDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = load(name="forward_dispatch_probe",
                         sources=[str(ROOT / "tests/forward_dispatch_probe.cu")],
                         extra_cuda_cflags=["-O3", "--use_fast_math",
                                           "-gencode=arch=compute_89,code=sm_89"],
                         verbose=False)

    def test_layout_policy(self):
        cases = [
            ({}, False, False, "contiguous_rows"),
            ({"h_kv": 8}, False, False, "contiguous_rows"),
            ({}, False, True, "strided"),
            ({"d": 128}, False, False, "strided"),
            ({}, True, False, "strided"),
            ({"n_q": 65}, True, False, "contiguous_heads"),
            ({"n_q": 65}, True, True, "contiguous_heads"),
            ({"n_q": 65, "d": 128}, True, False, "contiguous_heads"),
            ({"n_q": 65, "h_kv": 8}, True, False, "contiguous_heads"),
            ({"n_q": 65}, False, False, "strided"),
            ({"n_q": 65, "n_kv": 97}, True, False, "strided"),
            ({"n_kv": 96}, False, False, "contiguous_rows"),
            ({"n_kv": 65}, False, False, "strided"),
            ({"layout": "token_major"}, False, False, "strided"),
            ({"n_q": 65, "layout": "token_major"}, True, False, "strided"),
            ({"n_q": 65, "b": 256, "h": 256, "h_kv": 256}, True, False, "strided"),
            ({"n_q": 65, "b": 255, "h": 257, "h_kv": 257}, True, False, "contiguous_heads"),
            ({"n_q": 65, "b": 1, "h_kv": 1, "layout": "k_padded_heads"},
             True, False, "contiguous_heads"),
        ]
        for role in ("q", "k", "v", "o"):
            cases.append(({"layout": f"{role}_padded_rows"}, False, False, "strided"))
            cases.append(({"n_q": 65, "layout": f"{role}_padded_heads"},
                          True, False, "strided"))
        for shape, causal, write_l, expected in cases:
            with self.subTest(shape=shape, causal=causal, write_l=write_l):
                self.assertEqual(self.probe.selected_layout(*tensors(**shape), causal, write_l),
                                 expected)


if __name__ == "__main__":
    unittest.main()
