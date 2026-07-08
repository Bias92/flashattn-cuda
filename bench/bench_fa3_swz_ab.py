"""Root-cause A/B: is the swz regression from the REG squeeze or the XOR swizzle?
Compares race-ca vs swz-ca(minblocks=6, REG 80 + STACK) vs swz-ca-noLB(minblocks=1).
Adoption already rejected; this is attribution only.
"""
import os
import random
import subprocess
import sys
import time

import torch
from torch.utils.cpp_extension import load


def gate():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"], text=True).strip()
    util, mem = [int(x) for x in out.split(",")]
    if util > 10 or mem > 2500:
        print(f"ABORT: GPU not idle (util {util}%, mem {mem} MiB)")
        sys.exit(2)
    print(f"gpu gate ok: util {util}%, mem {mem} MiB")


gate()
flags = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89", "-DUSE_CP_ASYNC_CA=1"]
mod_race = load(name="flash_attn_fa3_race_ca_swzb", sources=["cuda/flash_attn_fa3_race.cu"],
                extra_cuda_cflags=flags, verbose=False)
mod_swz6 = load(name="flash_attn_fa3_swz_ca", sources=["cuda/flash_attn_fa3_swz.cu"],
                extra_cuda_cflags=flags, verbose=False)
mod_swz1 = load(name="flash_attn_fa3_swz_ca_nolb", sources=["cuda/flash_attn_fa3_swz.cu"],
                extra_cuda_cflags=flags + ["-DSWZ_MINBLOCKS=1"], verbose=False)
for m in (mod_race, mod_swz6, mod_swz1):
    print(f"so: {m.__file__}")


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
    random.seed(7)
    torch.manual_seed(42)
    B, H, D = 1, 8, 64
    reps = int(os.environ.get("RACE_REPS", "20"))

    Qb = torch.randn(B, H, 4096, D, device="cuda", dtype=torch.float16)
    for _ in range(100):
        mod_race.forward_only(Qb, Qb, Qb)
    torch.cuda.synchronize()

    N = 4096
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    runners = {
        "race": lambda: time_once(lambda: mod_race.forward_only(Q, K, V), 30, 50),
        "swz6": lambda: time_once(lambda: mod_swz6.forward_only(Q, K, V), 30, 50),
        "swz1": lambda: time_once(lambda: mod_swz1.forward_only(Q, K, V), 30, 50),
    }
    times = {k: [] for k in runners}
    for r in range(reps):
        order = list(runners)
        random.shuffle(order)
        for k in order:
            times[k].append(runners[k]())

    m = {k: med(v) for k, v in times.items()}
    print(f"N=4096 ({reps} reps): race-ca {m['race']:.4f}  "
          f"swz-ca(6blk) {m['swz6']:.4f}  swz-ca(noLB) {m['swz1']:.4f} ms")
    print(f"  swz6/race = {m['swz6']/m['race']:.4f}  (squeeze+swizzle)")
    print(f"  swz1/race = {m['swz1']/m['race']:.4f}  (swizzle only)")
    print(f"  -> squeeze cost = {100*(m['swz6']-m['swz1'])/m['race']:+.2f}%p, "
          f"swizzle cost = {100*(m['swz1']-m['race'])/m['race']:+.2f}%p")


if __name__ == "__main__":
    main()
