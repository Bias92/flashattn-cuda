"""
Paired benchmark: Custom CUDA forward()+L vs PyTorch SDPA.

SDPA is an API with several backends, and they do not perform alike, so this
measures two of them: FLASH_ATTENTION (the FlashAttention-2 kernels PyTorch
vendors, and the one plain `scaled_dot_product_attention` dispatches to here)
and CUDNN_ATTENTION.

The three implementations run in rotating order within each rep, so no one of
them always runs on a cold or a hot clock. The reported gap is the median of
10 per-rep gaps. Dense and causal masking are measured separately, with the
matching is_causal flag on the PyTorch side.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
mod = load(name="attention_forward_cuda", sources=[str(ROOT / "cuda/attention_forward.cu")],
           extra_cuda_cflags=FLAGS, verbose=False)
print(f"so: {mod.__file__}")

REPS = 10


def time_once(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    B, H, D = 1, 8, 64
    torch.manual_seed(42)
    print("=" * 100)
    print(f"Custom CUDA (+L) vs PyTorch SDPA backends — {REPS} paired reps, "
          f"B={B} H={H} D={D} FP16")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print("=" * 100)

    Qb = torch.randn(B, H, 4096, D, device="cuda", dtype=torch.float16)
    for _ in range(100):
        mod.forward_only(Qb, Qb, Qb)
    torch.cuda.synchronize()
    del Qb

    for causal in (False, True):
        print(f"--- causal={causal} "
              f"({'lower-triangular mask' if causal else 'dense'}) ---")
        for N in [1024, 2048, 4096]:
            Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
            K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
            V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
            warmup = 30
            iters = 200 if N <= 1024 else (100 if N <= 2048 else 50)

            def run_ours():
                return time_once(lambda: mod.forward(Q, K, V, causal), warmup, iters)

            def run_backend(backend):
                def f():
                    with sdpa_kernel(backend):
                        return time_once(
                            lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=causal),
                            warmup, iters)
                return f

            runners = {
                "ours": run_ours,
                "flash": run_backend(SDPBackend.FLASH_ATTENTION),
                "cudnn": run_backend(SDPBackend.CUDNN_ATTENTION),
            }
            names = list(runners)
            t = {k: [] for k in names}
            gaps = {"flash": [], "cudnn": []}
            for r in range(REPS):
                order = names[r % len(names):] + names[:r % len(names)]
                rep = {k: runners[k]() for k in order}
                for k in names:
                    t[k].append(rep[k])
                for k in gaps:
                    gaps[k].append((rep["ours"] / rep[k] - 1.0) * 100.0)

            # a causal block computes about half the score matrix
            flops = 4 * N * N * D * H * (0.5 if causal else 1.0)
            tf = {k: flops / (med(v) * 1e-3) / 1e12 for k, v in t.items()}
            print(f"N={N:>5}: ours {med(t['ours']):.4f}ms ({tf['ours']:4.1f} TF)  "
                  f"flash {med(t['flash']):.4f}ms ({tf['flash']:4.1f} TF)  "
                  f"cudnn {med(t['cudnn']):.4f}ms ({tf['cudnn']:4.1f} TF)")
            print(f"        paired gap vs flash {med(gaps['flash']):+.2f}%  "
                  f"vs cudnn {med(gaps['cudnn']):+.2f}%  (positive = ours slower)")
            del Q, K, V
            torch.cuda.empty_cache()

    print("=" * 100)


if __name__ == "__main__":
    main()
