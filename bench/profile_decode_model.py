"""Unprofiled controls plus scoped CPU/CUDA traces of actual TinyLlama decode."""
import argparse
from contextlib import ExitStack, contextmanager
import functools
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from unittest.mock import patch

import torch
import transformers
from transformers import AttentionInterface, AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, ALL_MASK_ATTENTION_FUNCTIONS
from transformers.models.llama import modeling_llama

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_attention_decode import load_decode, decode_source_sha
from attention_backend import register_backend
from bench_attention_decode import exclusive_gpu_check, telemetry, paired_ratio_summary
from decode_trace import summarize_trace


def ranged(fn, phase):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function("phase::" + phase):
            return fn(*args, **kwargs)
    return wrapped


@contextmanager
def phase_ranges(model):
    with ExitStack() as stack:
        stack.enter_context(patch.object(Cache, "update", ranged(Cache.update, "kv_cache")))
        for name, phase in (("apply_rotary_pos_emb", "rope"), ("create_causal_mask", "mask_build")):
            stack.enter_context(patch.object(modeling_llama, name, ranged(getattr(modeling_llama, name), phase)))
        for name, module in model.named_modules():
            phase = None
            if name.endswith(".mlp"):
                phase = "mlp"
            elif name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj")):
                phase = "qkvo_projection"
            elif "layernorm" in name or name == "model.norm":
                phase = "rmsnorm"
            elif name == "lm_head":
                phase = "lm_head"
            elif name == "model.embed_tokens":
                phase = "embedding"
            elif name == "model.rotary_emb":
                phase = "rope"
            if phase:
                stack.enter_context(patch.object(module, "forward", ranged(module.forward, phase)))
        yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--profile-steps", type=int, default=4)
    ap.add_argument("--profile-repeats", type=int, default=2)
    ap.add_argument("--mask", choices=["ones", "none"], default="ones")
    ap.add_argument("--output", default="profile_decode.json")
    args = ap.parse_args()
    ignored_cpu_jobs = []

    def check_jobs():
        ignored_cpu_jobs.extend(exclusive_gpu_check(allow_readonly_pip=True))

    check_jobs()
    decoder = load_decode()
    custom, counts = register_backend("scratch_decode", decoder)
    for mode, fn in (("sdpa", sdpa_attention_forward), ("scratch_decode", custom)):
        AttentionInterface.register("trace_" + mode, ranged(fn, "attention"))
        AttentionMaskInterface.register("trace_" + mode, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, dtype=torch.float16,
                                               attn_implementation="sdpa").cuda().eval()
    torch.manual_seed(137)
    rng = random.Random(829)
    data = dict(model=name, torch=torch.__version__, transformers=transformers.__version__,
                source_sha256=decode_source_sha(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                cpu_affinity=sorted(os.sched_getaffinity(0)), config=vars(args),
                telemetry_before=telemetry(), controls=[], profiles=[], complete=False,
                ignored_readonly_pip_jobs=ignored_cpu_jobs,
                notes=["No CUDA kernel changes. Both modes use SDPA prefill.",
                       "Unprofiled controls include synchronization at the end, not per operation.",
                       "Profiled wall time is not a latency/speedup benchmark."])
    dest = ROOT / "docs/decoding" / args.output
    traces_dir = dest.parent / (dest.stem + "_traces")
    traces_dir.mkdir(parents=True, exist_ok=True)

    def save():
        dest.write_text(json.dumps(data, indent=2) + "\n")

    with torch.inference_mode():
        for n in args.lengths:
            prompt = torch.randint(3, model.config.vocab_size, (1, n), device="cuda")
            tokens = torch.randint(3, model.config.vocab_size, (max(args.steps, args.profile_steps), 1, 1), device="cuda")

            def state(mode):
                model.set_attn_implementation(mode)
                mask = torch.ones_like(prompt) if args.mask == "ones" else None
                output = model(input_ids=prompt, attention_mask=mask, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize()
                for key in counts:
                    counts[key] = 0
                return output.past_key_values, mask

            def steps(cache, mask, count, scoped=False):
                def prepare(mask):
                    return torch.cat([mask, torch.ones(1, 1, device="cuda", dtype=mask.dtype)], -1) if mask is not None else None
                prepare_fn = ranged(prepare, "token_input") if scoped else prepare
                def sample(logits):
                    return logits[:, -1].argmax(-1)
                sample_fn = ranged(sample, "sampling") if scoped else sample
                for step in range(count):
                    mask = prepare_fn(mask)
                    output = model(input_ids=tokens[step], attention_mask=mask, past_key_values=cache,
                                   use_cache=True, logits_to_keep=1)
                    cache = output.past_key_values
                    sample_fn(output.logits)
                return output.logits

            def control(mode):
                cache, mask = state(mode)
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                wall = time.perf_counter()
                logits = steps(cache, mask, args.steps)
                end.record()
                end.synchronize()
                result = dict(wall_ms_per_token=(time.perf_counter()-wall)*1000/args.steps,
                              event_span_ms_per_token=start.elapsed_time(end)/args.steps,
                              calls=dict(counts))
                if mode == "scratch_decode":
                    assert counts["decode"] == args.steps * model.config.num_hidden_layers, counts
                return result, logits.clone()

            for mode in ("sdpa", "scratch_decode"):
                control(mode)
                control(mode)
            samples = {mode: [] for mode in ("sdpa", "scratch_decode")}
            for pair in range(args.pairs):
                check_jobs()
                order = list(samples)
                rng.shuffle(order)
                outputs = {}
                for mode in order:
                    result, logits = control(mode)
                    samples[mode].append(result)
                    outputs[mode] = logits
                torch.testing.assert_close(outputs["scratch_decode"], outputs["sdpa"], atol=0.05, rtol=0.01)
                print(f"CONTROL N={n} #{pair+1} " + json.dumps({m: samples[m][-1]["wall_ms_per_token"] for m in samples}), flush=True)
            data["controls"].append(dict(n=n, samples=samples,
                paired_ratio=paired_ratio_summary([s["wall_ms_per_token"] for s in samples["scratch_decode"]],
                                                 [s["wall_ms_per_token"] for s in samples["sdpa"]])))
            save()
            for repeat in range(args.profile_repeats):
                order = ["sdpa", "scratch_decode"]
                if repeat % 2:
                    order.reverse()
                for mode in order:
                    check_jobs()
                    trace_mode = "trace_" + mode
                    cache, mask = state(trace_mode)
                    with phase_ranges(model):
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA],
                                                    record_shapes=False, with_stack=False, profile_memory=False) as prof:
                            with torch.profiler.record_function("decode.window"):
                                steps(cache, mask, args.profile_steps, scoped=True)
                                with torch.profiler.record_function("phase::final_sync"):
                                    torch.cuda.synchronize()
                    check_jobs()
                    if mode == "scratch_decode":
                        assert counts["decode"] == args.profile_steps * model.config.num_hidden_layers, counts
                    path = traces_dir / f"n{n}_{mode}_{repeat}.json.gz"
                    # PyTorch exports compressed Chrome JSON directly for .gz.
                    prof.export_chrome_trace(str(path))
                    with gzip.open(path, "rt") as handle:
                        trace = json.load(handle)
                    report = summarize_trace(trace, args.profile_steps)
                    report.update(n=n, mode=mode, repeat=repeat, trace=str(path), calls=dict(counts), telemetry=telemetry())
                    report["top_cpu_self_ms_per_token"] = [dict(op=e.key, ms=e.self_cpu_time_total/(1000*args.profile_steps),
                                                               calls=e.count/args.profile_steps)
                        for e in sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)[:15]]
                    data["profiles"].append(report)
                    save()
                    print("PROFILE " + json.dumps({k: report[k] for k in ("n", "mode", "repeat", "trace_wall_ms_per_token",
                        "gpu_active_union_ms_per_token", "gpu_ms_per_token", "gaps_between_gpu_activities_ms_per_token")}), flush=True)
        check_jobs()
    data["complete"] = True
    data["telemetry_after"] = telemetry()
    save()
    print(f"Saved {dest}", flush=True)


if __name__ == "__main__":
    main()
