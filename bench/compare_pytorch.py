"""
Paired benchmark: Custom CUDA forward()+L vs PyTorch SDPA.

SDPA is an API with several backends, and they do not perform alike, so this
measures two of them: FLASH_ATTENTION (the FlashAttention-2 kernels PyTorch
vendors, and the one plain `scaled_dot_product_attention` dispatches to here)
and CUDNN_ATTENTION.

The three implementations run in rotating order within each rep, so no one of
them always runs on a cold or a hot clock. The reported gap is the median of
the per-rep gaps, with a bootstrap 95% interval. Every shape is measured with
a dense and with a causal mask, using the matching is_causal flag on the
PyTorch side.

    python3 bench/compare_pytorch.py                 # the README shapes
    python3 bench/compare_pytorch.py --set holdout   # historical development set, D64 only
"""
import argparse
import contextlib
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]

# (B, H, H_kv, N, D)
SHAPE_SETS = {
    # The shapes behind the README tables. Tile sizes were tuned on these.
    "readme": [
        (1, 8, 8, 1024, 64), (1, 8, 8, 2048, 64), (1, 8, 8, 4096, 64),
        (1, 32, 8, 2048, 64), (1, 32, 8, 4096, 64),
        (1, 32, 8, 2048, 128), (1, 32, 8, 4096, 128),
    ],
    # Fixed on 2026-09-19 before any of them was timed. Development data since 2026-09-20:
    # six of its sixteen cases were in the A/B runs that chose between K/V staging variants.
    # They vary batch, head count, GQA ratio and length, and include lengths
    # that are not a multiple of the tile size.
    "holdout": [
        (4, 12, 12, 1536, 64), (2, 16, 4, 3072, 64), (1, 8, 8, 8192, 64), (8, 8, 8, 512, 64),
        (1, 32, 8, 1000, 64), (3, 14, 2, 2500, 64),
        (2, 16, 8, 1536, 128), (1, 28, 4, 3000, 128),
    ],
    # Fixed on 2026-09-20 after the kernel source was final (sha256 in the results record)
    # and before any of them was timed; in no A/B run, test or tuning sweep.
    "holdout2": [
        (2, 24, 8, 2304, 64), (1, 16, 16, 5000, 64), (6, 10, 2, 768, 64), (1, 40, 8, 1700, 64),
        (2, 20, 4, 6144, 64), (5, 9, 3, 1111, 64),
        (1, 24, 8, 2560, 128), (3, 12, 6, 1900, 128),
    ],
}
WINDOW_MS = 60.0   # GPU time one timing sample aims for
MAX_ITERS = 2000   # bound on the calls in one sample, only reached below 0.03 ms per call


def time_once(fn, iters, warmup=3):
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


def median_ci(xs, draws=2000):
    rng = random.Random(0)
    ms = sorted(med([rng.choice(xs) for _ in xs]) for _ in range(draws))
    return ms[int(0.025 * draws)], ms[int(0.975 * draws)]


def measure(mod, shape, causal, reps):
    B, H, H_kv, N, D = shape
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H_kv, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H_kv, N, D, device="cuda", dtype=torch.float16)

    def sdpa():
        return F.scaled_dot_product_attention(Q, K, V, is_causal=causal, enable_gqa=(H != H_kv))

    fns = {"ours": lambda: mod.forward(Q, K, V, causal), "flash": sdpa, "cudnn": sdpa}
    # The backend is selected once around a whole timing sample, never inside the timed
    # calls: entering sdpa_kernel costs CPU time that only the PyTorch side would pay.
    select = {"ours": contextlib.nullcontext,
              "flash": lambda: sdpa_kernel(SDPBackend.FLASH_ATTENTION),
              "cudnn": lambda: sdpa_kernel(SDPBackend.CUDNN_ATTENTION)}

    def sample(k, iters, warmup=3):
        with select[k]():
            return time_once(fns[k], iters, warmup)

    with select["flash"]():
        maxdiff = (fns["ours"]()[0].float() - fns["flash"]().float()).abs().max().item()

    iters = max(10, min(MAX_ITERS, int(WINDOW_MS / sample("flash", 5, warmup=10))))
    names = list(fns)
    t = {k: [] for k in names}
    gaps = {"flash": [], "cudnn": []}
    for r in range(reps):
        order = names[r % len(names):] + names[:r % len(names)]
        rep = {k: sample(k, iters) for k in order}
        for k in names:
            t[k].append(rep[k])
        for k in gaps:
            gaps[k].append((rep["ours"] / rep[k] - 1.0) * 100.0)

    # a causal block computes about half the score matrix
    flops = 4 * B * H * N * N * D * (0.5 if causal else 1.0)
    cells = "  ".join(f"{k} {med(t[k]):.4f}ms ({flops / (med(t[k]) * 1e-3) / 1e12:4.1f} TF)"
                      for k in names)
    gap_cells = "  ".join("vs {} {:+.2f}% [{:+.2f}, {:+.2f}]".format(k, med(g), *median_ci(g))
                          for k, g in gaps.items())
    print(f"B={B} H={H}/{H_kv} N={N:>5} D={D:>3} {'causal' if causal else 'dense ':6}: {cells}")
    print(f"        paired gap {gap_cells}  (positive = ours slower)  "
          f"maxdiff vs flash {maxdiff:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SHAPE_SETS), default="readme")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    mod = load(name="attention_forward_cuda", sources=[str(ROOT / "cuda/attention_forward.cu")],
               extra_cuda_cflags=FLAGS, verbose=False)
    print(f"so: {mod.__file__}")
    torch.manual_seed(42)
    print("=" * 100)
    print(f"Custom CUDA (+L) vs PyTorch SDPA backends — shape set '{args.set}', "
          f"{args.reps} paired reps, FP16")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print("=" * 100)

    Qb = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)
    for _ in range(100):
        mod.forward_only(Qb, Qb, Qb)
    torch.cuda.synchronize()
    del Qb

    for shape in SHAPE_SETS[args.set]:
        if shape[4] != 64:
            continue
        for causal in (False, True):
            measure(mod, shape, causal, args.reps)
            torch.cuda.empty_cache()
    print("=" * 100)


if __name__ == "__main__":
    main()
