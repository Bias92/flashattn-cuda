"""Paired decode microbenchmark. Eager API time and CUDA-graph device time are separate."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_attention_decode import load_decode, reference, decode_source_sha


def exclusive_gpu_check(allow_readonly_pip=False):
    jobs, ignored = [], []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = [a.decode(errors="replace") for a in (p / "cmdline").read_bytes().split(b"\0") if a]
            cmd = " ".join(argv)
            exe = (p / "exe").resolve().name
        except (OSError, RuntimeError):
            continue
        if "python" in exe and "unattended-upgrade" not in cmd:
            readonly_pip = len(argv) >= 4 and argv[1:3] == ["-m", "pip"] and argv[3] in ("index", "show", "list", "freeze", "check", "cache")
            if allow_readonly_pip and readonly_pip:
                ignored.append(cmd)
            else:
                jobs.append(cmd)
    if jobs:
        raise RuntimeError(f"Other Python jobs: timing is not exclusive: {jobs}")
    return ignored


def telemetry():
    fields = "name,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.used"
    return subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=" + fields,
                                    "--format=csv"], text=True).strip()


def timed(fn, iters):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def make_graph(fn, calls=32):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    return graph


def paired_ratio_summary(a, b):
    ratios = [x / y for x, y in zip(a, b)]
    rng = random.Random(812)
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return dict(median=statistics.median(ratios), bootstrap_95ci=[boot[50], boot[1949]], pairs=ratios)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 4096, 8192])
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--output", default="decode_benchmark.json")
    args = ap.parse_args()
    exclusive_gpu_check()
    mod = load_decode()
    torch.manual_seed(321)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = random.Random(333)
    rows = []
    data = dict(torch=torch.__version__, cuda=torch.version.cuda,
                source_sha256=decode_source_sha(),
                extension=mod.__file__, pairs=args.pairs, telemetry_before=telemetry(),
                condition="B=1 Hq=32 Hkv=4 Nq=1; FP16; no mask; scale=1/sqrt(D); O only",
                note="graph_us removes Python issue gaps; eager_us includes them. Warm-cache repeated inputs.",
                valid_for_performance_claims=False, rows=rows)
    output = ROOT / "docs/decoding" / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for d in args.dims:
                for n in args.lengths:
                    exclusive_gpu_check()
                    q = torch.randn(1, 32, 1, d, dtype=torch.float16, device="cuda")
                    k = torch.randn(1, 4, n, d, dtype=torch.float16, device="cuda")
                    v = torch.randn_like(k)
                    variants = {f"custom_{split}": (lambda split=split: mod.forward_only(q, k, v, None, split), None)
                                for split in args.splits}
                    variants.update({"sdpa_default": (lambda: sdpa(q, k, v, enable_gqa=True), None),
                                     "sdpa_flash": (lambda: sdpa(q, k, v, enable_gqa=True), SDPBackend.FLASH_ATTENTION),
                                     "sdpa_cudnn": (lambda: sdpa(q, k, v, enable_gqa=True), SDPBackend.CUDNN_ATTENTION)})
                    ref, _ = reference(q, k, v)
                    graphs, rejected, selected = {}, {}, {}
                    def backend_ctx(backend):
                        return sdpa_kernel(backend) if backend is not None else contextlib.nullcontext()
                    for name, (fn, backend) in variants.items():
                        try:
                            with backend_ctx(backend):
                                torch.testing.assert_close(fn().float(), ref, atol=8e-4, rtol=1e-3)
                                if name.startswith("sdpa"):
                                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                          torch.profiler.ProfilerActivity.CUDA]) as prof:
                                        fn()
                                        torch.cuda.synchronize()
                                    selected[name] = sorted({e.name for e in prof.events()
                                        if "scaled_dot_product" in e.name or str(e.device_type).endswith("CUDA")})
                                graphs[name] = make_graph(fn)
                        except RuntimeError as exc:
                            rejected[name] = str(exc)
                    samples = {mode: {name: [] for name in graphs} for mode in ("graph_us", "eager_us")}
                    # Warm the device after builds and graph capture, outside timing.
                    until = time.monotonic() + 0.3
                    while time.monotonic() < until:
                        for graph in graphs.values():
                            graph.replay()
                    torch.cuda.synchronize()
                    for pair in range(args.pairs):
                        exclusive_gpu_check()
                        for mode in samples:
                            names = list(graphs)
                            rng.shuffle(names)
                            for name in names:
                                fn, backend = variants[name]
                                with backend_ctx(backend):
                                    call = graphs[name].replay if mode == "graph_us" else fn
                                    for _ in range(5):
                                        call()
                                    us = timed(call, 40 if mode == "graph_us" else 100)
                                    samples[mode][name].append(us / 32 if mode == "graph_us" else us)
                    row = dict(d=d, n=n, samples=samples, unavailable=rejected, selected=selected,
                               paired_ratios={mode: {name + "/" + ref_name: paired_ratio_summary(vals[name], vals[ref_name])
                                   for name in graphs if name.startswith("custom")
                                   for ref_name in graphs if ref_name.startswith("sdpa")}
                                   for mode, vals in samples.items()},
                               median={mode: {name: statistics.median(values) for name, values in vals.items()}
                                       for mode, vals in samples.items()})
                    rows.append(row)
                    print(json.dumps(dict(d=d, n=n, median=row["median"], unavailable=rejected)), flush=True)
                    data["telemetry_latest"] = telemetry()
                    output.write_text(json.dumps(data, indent=2) + "\n")
                    del graphs, variants, q, k, v
            exclusive_gpu_check()
        data["valid_for_performance_claims"] = True
    finally:
        data["telemetry_after"] = telemetry()
        output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
