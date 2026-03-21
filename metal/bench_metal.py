"""
FlashAttention Metal — Benchmark
Cross-platform comparison with CUDA baseline numbers.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metal.flash_attn_metal import MetalFlashAttention


# CUDA FP32 baseline numbers (RTX 4060 Ti, B=1, H=8, D=64)
CUDA_BASELINE_MS = {
    128:  0.12,
    256:  0.12,
    512:  0.25,
    1024: 0.94,
    2048: 3.06,
    4096: 11.18,
}


if __name__ == "__main__":
    fa = MetalFlashAttention()

    B, H, D = 1, 8, 64
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    print("=" * 75)
    print(f"FlashAttention Benchmark — Metal ({fa.device_name})")
    print(f"Config: B={B}, H={H}, D={D}, FP32")
    print("=" * 75)
    print(f"{'N':>6} | {'Metal':>10} | {'CUDA BL':>10} | {'Ratio':>8}")
    print("-" * 75)

    for N in seq_lengths:
        np.random.seed(42)
        Q = np.random.randn(B, H, N, D).astype(np.float32)
        K = np.random.randn(B, H, N, D).astype(np.float32)
        V = np.random.randn(B, H, N, D).astype(np.float32)

        ms = fa.bench_forward(Q, K, V, warmup=10, repeats=100)
        cuda_ms = CUDA_BASELINE_MS[N]
        ratio = cuda_ms / ms

        print(f"{N:>6} | {ms:>8.2f}ms | {cuda_ms:>8.2f}ms | {ratio:>6.2f}x")

    print("=" * 75)
    print("Ratio > 1.0 = Metal faster, < 1.0 = CUDA faster")
