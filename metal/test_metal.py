"""
FlashAttention Metal — Correctness Test
Compares Metal forward kernel output against naive O(N²) attention.
Same test configs as CUDA baseline (test_forward.py).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metal.flash_attn_metal import MetalFlashAttention


def naive_attention(Q, K, V):
    """O(N²) reference attention. Q,K,V: [B, H, N, D]"""
    D = Q.shape[-1]
    scale = 1.0 / np.sqrt(D)
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale   # [B,H,N,N]
    # Numerically stable softmax
    S_max = S.max(axis=-1, keepdims=True)
    P = np.exp(S - S_max)
    P = P / P.sum(axis=-1, keepdims=True)
    O = np.matmul(P, V)
    return O


def run_test(B, H, N, D=64, atol=1e-5):
    np.random.seed(42)
    Q = np.random.randn(B, H, N, D).astype(np.float32)
    K = np.random.randn(B, H, N, D).astype(np.float32)
    V = np.random.randn(B, H, N, D).astype(np.float32)

    O_ref = naive_attention(Q, K, V)
    O_metal, L_metal = fa.forward(Q, K, V)

    max_diff = np.max(np.abs(O_ref - O_metal))
    passed = max_diff < atol

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] B={B}, H={H}, N={N:>5}, D={D}  |  max_diff={max_diff:.6e}")
    return passed


if __name__ == "__main__":
    print("=" * 70)
    print("FlashAttention Metal — Correctness Test")
    print("=" * 70)

    fa = MetalFlashAttention()

    configs = [
        (1, 1,   32, 64),
        (1, 1,   64, 64),
        (1, 1,  128, 64),
        (1, 1,   63, 64),   # non-aligned
        (1, 1,  127, 64),   # non-aligned
        (2, 4,  256, 64),
        (2, 8,  512, 64),
        (1, 1, 1024, 64),
        (1, 1, 2048, 64),
    ]

    passed = 0
    total = len(configs)

    for B, H, N, D in configs:
        if run_test(B, H, N, D):
            passed += 1

    print("=" * 70)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("All tests passed!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)
