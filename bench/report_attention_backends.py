"""Current prefill O-only/+L versus verified SDPA backends; preserve raw pairs."""
import argparse
import contextlib
import json
from pathlib import Path
import random
import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from compare_pytorch import SHAPE_SETS, time_once, median_ci
from prefill_support import CANDIDATE, extension, gpu_state, guard, sha


def dispatch(fn, select):
    with select():
        fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            fn()
        torch.cuda.synchronize()
    return sorted(e.key for e in profile.key_averages() if "scaled_dot_product" in e.key)


def measure(mod, shape, causal, args, seed):
    b, h, hk, n, d = shape
    torch.manual_seed(seed)
    q = torch.randn(b, h, n, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, hk, n, d, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    def native():
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=h != hk)

    fns = {"ours_o": lambda: mod.forward_only(q, k, v, causal),
           "ours_l": lambda: mod.forward(q, k, v, causal),
           "flash": native, "cudnn": native}
    select = {"ours_o": contextlib.nullcontext, "ours_l": contextlib.nullcontext,
              "flash": lambda: sdpa_kernel(SDPBackend.FLASH_ATTENTION),
              "cudnn": lambda: sdpa_kernel(SDPBackend.CUDNN_ATTENTION)}
    ops = {name: dispatch(native, select[name]) for name in ("flash", "cudnn")}
    ops["auto"] = dispatch(native, contextlib.nullcontext)
    assert "aten::_scaled_dot_product_flash_attention" in ops["flash"], ops
    assert "aten::_scaled_dot_product_cudnn_attention" in ops["cudnn"], ops
    actual = fns["ours_o"]()
    maxdiff = {}
    for name, fn in fns.items():
        with select[name]():
            other = fn()
        if name == "ours_l":
            other = other[0]
        torch.testing.assert_close(actual, other, atol=1e-3, rtol=1e-3)
        maxdiff[name] = (actual.float() - other.float()).abs().max().item()
    del actual, other

    def sample(name, count, warmup=3):
        with select[name]():
            return time_once(fns[name], count, warmup)

    for name in fns:
        sample(name, 10, 10)
    count = max(10, min(2000, int(args.window_ms / sample("flash", 5))))
    order = list(fns)
    random.Random(seed).shuffle(order)
    pairs = []
    for rep in range(args.reps):
        rotated = order[rep % 4:] + order[:rep % 4]
        pairs.append({name: sample(name, count) for name in rotated})
    ratios = {}
    for ours in ("ours_o", "ours_l"):
        ratios[ours] = {}
        for target in ("flash", "cudnn"):
            values = [p[ours] / p[target] for p in pairs]
            ratios[ours][target] = {"median": statistics.median(values),
                                    "ci95": median_ci(values)}
    return {"shape": list(shape), "causal": causal, "iterations": count,
            "dispatch": ops, "max_abs_difference_from_ours_o": maxdiff,
            "samples_ms": pairs, "ratios": ratios,
            "median_ms": {name: statistics.median(p[name] for p in pairs) for name in fns},
            "gpu": gpu_state()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--window-ms", type=float, default=30)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    if args.reps < 4 or args.reps % 4 or args.window_ms <= 0:
        parser.error("Use a positive window and reps divisible by four")
    mod, source = extension(CANDIDATE)
    meta = {"run": args.run, "source": source, "runner_sha256": sha(__file__),
            "helpers": {name: sha(Path(__file__).with_name(name))
                        for name in ("compare_pytorch.py", "prefill_support.py")},
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu_before": gpu_state(),
            "guard_before": guard(), "reps": args.reps, "window_ms": args.window_ms,
            "protocol": "warm-cache CUDA events over public APIs; rotating paired order; no graphs",
            "complete": False}
    cases = []
    for set_name, shapes in SHAPE_SETS.items():
        for shape in shapes:
            if shape[4] != 64:
                continue
            for causal in (False, True):
                case = measure(mod, shape, causal, args, 92000 + args.run * 1000 + len(cases))
                case["set"] = set_name
                cases.append(case)
                ratios = case["ratios"]["ours_o"]
                print(f"{set_name} {shape} causal={causal}: O/flash={ratios['flash']['median']:.4f} "
                      f"O/cudnn={ratios['cudnn']['median']:.4f}", flush=True)
                args.output.write_text(json.dumps({"meta": meta, "cases": cases}, indent=2))
                torch.cuda.empty_cache()
    assert sha(CANDIDATE) == source["sha256"], "Kernel changed during measurement"
    meta.update(gpu_after=gpu_state(), guard_after=guard(), complete=True)
    args.output.write_text(json.dumps({"meta": meta, "cases": cases}, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
