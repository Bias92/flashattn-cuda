"""
E-route bench: db_full / race-ca / swz-ca / SDPA, paired reps, randomized order.
swz = PAD=0 XOR swizzle + __launch_bounds__(128,6) -> 6 blocks/SM (REG 80, LOCAL 0).
RACE_REPS env controls rep count (default 30). Same adoption cut as race:
swz/sdpa median <= 0.99 at both N, or <= 0.985 at N=4096.
"""
import os
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
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True).strip()
        util, mem = [int(x) for x in out.split(",")]
        last = (util, mem)
        if util <= max_util and mem <= max_mem_mib:
            print(f"gpu gate ok: util {util}%, mem {mem} MiB")
            return
        time.sleep(2)
    print(f"ABORT: GPU not idle {last}.")
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
    pos = (len(xs) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def main():
    gate_clean_gpu()
    flags = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
    ca = ["-DUSE_CP_ASYNC_CA=1"]
    mod_full = load(name="flash_attn_fa3_db_full_swzb", sources=["cuda/flash_attn_fa3_db_full.cu"],
                    extra_cuda_cflags=flags, verbose=False)
    mod_race = load(name="flash_attn_fa3_race_ca_swzb", sources=["cuda/flash_attn_fa3_race.cu"],
                    extra_cuda_cflags=flags + ca, verbose=False)
    mod_swz = load(name="flash_attn_fa3_swz_ca", sources=["cuda/flash_attn_fa3_swz.cu"],
                   extra_cuda_cflags=flags + ca, verbose=False)
    for m in (mod_full, mod_race, mod_swz):
        print(f"so: {m.__file__}")

    reps = int(os.environ.get("RACE_REPS", "30"))
    random.seed(7)
    torch.manual_seed(42)
    B, H, D = 1, 8, 64
    print("=" * 100)
    print(f"E-route: {reps} paired reps, randomized order, O-only API latency")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print("=" * 100)

    Qb = torch.randn(B, H, 4096, D, device="cuda", dtype=torch.float16)
    for _ in range(100):
        mod_swz.forward_only(Qb, Qb, Qb)
    torch.cuda.synchronize()

    for N in [2048, 4096]:
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        warmup = 30
        iters = 100 if N <= 2048 else 50

        def run_full():
            return time_once(lambda: mod_full.forward_only(Q, K, V), warmup, iters)

        def run_race():
            return time_once(lambda: mod_race.forward_only(Q, K, V), warmup, iters)

        def run_swz():
            return time_once(lambda: mod_swz.forward_only(Q, K, V), warmup, iters)

        def run_sdpa():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return time_once(
                    lambda: F.scaled_dot_product_attention(Q, K, V), warmup, iters)

        runners = {"full": run_full, "race": run_race, "swz": run_swz, "sdpa": run_sdpa}
        times = {k: [] for k in runners}
        r_ss, r_sr = [], []
        for r in range(reps):
            order = list(runners)
            random.shuffle(order)
            rep = {}
            for k in order:
                rep[k] = runners[k]()
            for k in runners:
                times[k].append(rep[k])
            r_ss.append(rep["swz"] / rep["sdpa"])
            r_sr.append(rep["swz"] / rep["race"])

        m = {k: med(v) for k, v in times.items()}
        q1, q3 = percentile(r_ss, 0.25), percentile(r_ss, 0.75)
        wins = sum(x < 1.0 for x in r_ss)
        print(f"N={N:>5}: full {m['full']:.4f}  race-ca {m['race']:.4f}  "
              f"swz-ca {m['swz']:.4f}  sdpa {m['sdpa']:.4f} ms")
        print(f"        swz/sdpa median={med(r_ss):.4f}  IQR={q3 - q1:.4f}  "
              f"wins={wins}/{reps}  swz/race={med(r_sr):.4f}")

    print("=" * 100)


if __name__ == "__main__":
    main()
