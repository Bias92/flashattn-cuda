"""
SDPA-race benchmark: 30 paired reps, randomized order per rep.

Comparisons are O-only on our side, so this is API-latency framing:
  - db_full O-only  (baseline)
  - race-cg O-only  (split-loop + N_STATIC + cp.async.cg)
  - race-ca O-only  (split-loop + N_STATIC + cp.async.ca)
  - SDPA-Flash      (PyTorch API returns O)

Adoption cut: race/sdpa median <= 0.99 at both N=2048/4096,
or <= 0.985 at N=4096 alone. Report median of paired ratios, IQR,
and wins/30.
"""
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load


def gate_clean_gpu(max_util=10, max_mem_mib=2500):
    last = None
    for _ in range(30):
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        util, mem = [int(x) for x in out.split(",")]
        last = (util, mem)
        if util <= max_util and mem <= max_mem_mib:
            print(f"gpu gate ok: util {util}%, mem {mem} MiB")
            return
        time.sleep(2)

    util, mem = last
    print(
        f"ABORT: GPU not idle (util {util}%, mem {mem} MiB). "
        "Close Windows-side GPU users and re-run."
    )
    sys.exit(2)


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


def percentile(xs, p):
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize(name, ratio_sdpa, ratio_full):
    q1 = percentile(ratio_sdpa, 0.25)
    q3 = percentile(ratio_sdpa, 0.75)
    wins = sum(r < 1.0 for r in ratio_sdpa)
    m_sdpa = med(ratio_sdpa)
    m_full = med(ratio_full)
    print(
        f"        {name:>2}/sdpa median={m_sdpa:.4f}  "
        f"IQR={q3 - q1:.4f}  wins={wins:02d}/{len(ratio_sdpa)}  "
        f"{name}/full={m_full:.4f}  full-impr={100 * (1.0 - m_full):+.2f}%"
    )


def main():
    gate_clean_gpu()

    flags = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
    mod_full = load(
        name="flash_attn_fa3_db_full_race30",
        sources=["cuda/flash_attn_fa3_db_full.cu"],
        extra_cuda_cflags=flags,
        verbose=False,
    )
    mod_cg = load(
        name="flash_attn_fa3_race_cg30",
        sources=["cuda/flash_attn_fa3_race.cu"],
        extra_cuda_cflags=flags,
        verbose=False,
    )
    mod_ca = load(
        name="flash_attn_fa3_race_ca30",
        sources=["cuda/flash_attn_fa3_race.cu"],
        extra_cuda_cflags=flags + ["-DUSE_CP_ASYNC_CA=1"],
        verbose=False,
    )
    for m in (mod_full, mod_cg, mod_ca):
        print(f"so: {m.__file__}")

    import os
    reps = int(os.environ.get("RACE_REPS", "30"))
    random.seed(7)
    torch.manual_seed(42)
    B, H, D = 1, 8, 64
    print("=" * 100)
    print(f"SDPA race: {reps} paired reps, randomized order, O-only API latency")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print("=" * 100)

    Qb = torch.randn(B, H, 4096, D, device="cuda", dtype=torch.float16)
    for _ in range(100):
        mod_ca.forward_only(Qb, Qb, Qb)
    torch.cuda.synchronize()

    for N in [2048, 4096]:
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        warmup = 30
        iters = 100 if N <= 2048 else 50

        def run_full():
            return time_once(lambda: mod_full.forward_only(Q, K, V), warmup, iters)

        def run_cg():
            return time_once(lambda: mod_cg.forward_only(Q, K, V), warmup, iters)

        def run_ca():
            return time_once(lambda: mod_ca.forward_only(Q, K, V), warmup, iters)

        def run_sdpa():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return time_once(lambda: F.scaled_dot_product_attention(Q, K, V), warmup, iters)

        runners = {"full": run_full, "cg": run_cg, "ca": run_ca, "sdpa": run_sdpa}
        times = {k: [] for k in runners}
        ratio_sdpa = {"cg": [], "ca": []}
        ratio_full = {"cg": [], "ca": []}

        for _ in range(reps):
            order = list(runners)
            random.shuffle(order)
            rep = {}
            for k in order:
                rep[k] = runners[k]()
            for k, v in rep.items():
                times[k].append(v)
            for k in ("cg", "ca"):
                ratio_sdpa[k].append(rep[k] / rep["sdpa"])
                ratio_full[k].append(rep[k] / rep["full"])

        m = {k: med(v) for k, v in times.items()}
        print(
            f"N={N:>5}: full {m['full']:.4f}  cg {m['cg']:.4f}  "
            f"ca {m['ca']:.4f}  sdpa {m['sdpa']:.4f} ms"
        )
        summarize("cg", ratio_sdpa["cg"], ratio_full["cg"])
        summarize("ca", ratio_sdpa["ca"], ratio_full["ca"])

    print("=" * 100)


if __name__ == "__main__":
    main()
