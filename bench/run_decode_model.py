"""TinyLlama integration with direct, paired single-token latency measurements.

Uses local model files only. TPOT is timed around actual cached model steps,
not estimated by subtracting unrelated prefill and generate measurements.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_attention_decode import load_decode, decode_source_sha
from attention_backend import register_backend
from bench_attention_decode import exclusive_gpu_check, telemetry, paired_ratio_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=7)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--output", default="tinyllama_decode.json")
    args = ap.parse_args()
    exclusive_gpu_check()
    mod = load_decode()
    _, counts = register_backend("scratch_decode", mod)
    name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True, padding_side="left")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True,
        dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    modes = ["sdpa", "scratch_decode"]
    data = dict(model=name, torch=torch.__version__, transformers=transformers.__version__,
        source_sha256=decode_source_sha(),
        extension=mod.__file__, telemetry_before=telemetry(), scope="Both modes use HF SDPA for prefill. Only decode attention changes.",
        valid_for_performance_claims=False, correctness=[], results=[])
    dest = ROOT / "docs/decoding" / args.output
    dest.parent.mkdir(parents=True, exist_ok=True)

    def reset_counts():
        for key in counts:
            counts[key] = 0

    with torch.inference_mode():
        # Independent caches with identical forced tokens isolate numerical
        # differences from divergence caused by different sampled sequences.
        for padded in (False, True):
            texts = ["The key idea behind FlashAttention is"]
            if padded:
                texts.append("Hello")
            batch = tok(texts, return_tensors="pt", padding=True).to("cuda")
            traces = {}
            tokens = []
            for mode in modes:
                model.set_attn_implementation(mode)
                reset_counts()
                mask = batch.attention_mask
                outputs = model(**batch, use_cache=True, logits_to_keep=1)
                logits = [outputs.logits.float()]
                cache = outputs.past_key_values
                for step in range(8):
                    if mode == "sdpa":
                        tokens.append(outputs.logits[:, -1].argmax(-1, keepdim=True))
                    mask = torch.cat([mask, torch.ones(mask.shape[0], 1, dtype=mask.dtype, device="cuda")], -1)
                    outputs = model(input_ids=tokens[step], attention_mask=mask,
                                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = outputs.past_key_values
                    logits.append(outputs.logits.float())
                traces[mode] = torch.cat(logits, dim=1)
            torch.testing.assert_close(traces["scratch_decode"], traces["sdpa"], atol=0.05, rtol=0.01)
            same = torch.equal(traces["scratch_decode"].argmax(-1), traces["sdpa"].argmax(-1))
            assert same, "Greedy tokens differ"
            result = dict(padded=padded, max_logit_error=(traces["scratch_decode"]-traces["sdpa"]).abs().max().item(),
                          same_argmax=same, custom_calls=dict(counts))
            assert counts["decode"] == (0 if padded else 8 * model.config.num_hidden_layers)
            data["correctness"].append(result)
            print("MODEL PASS " + json.dumps(result), flush=True)
        rng = random.Random(512)
        torch.manual_seed(456)
        for n in args.lengths:
            x = torch.randint(3, model.config.vocab_size, (1, n), device="cuda")
            forced = torch.randint(3, model.config.vocab_size, (args.steps, 1, 1), device="cuda")

            def trial(mode):
                model.set_attn_implementation(mode)
                mask = torch.ones_like(x)
                reset_counts()
                # Prefill is setup only: identical HF implementation in both modes.
                output = model(input_ids=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
                torch.cuda.synchronize()
                wall_start = time.perf_counter()
                for step in range(args.steps):
                    starts[step].record()
                    mask = torch.cat([mask, torch.ones(1, 1, device="cuda", dtype=mask.dtype)], -1)
                    output = model(input_ids=forced[step], attention_mask=mask, past_key_values=cache,
                                   use_cache=True, logits_to_keep=1)
                    cache = output.past_key_values
                    # Include greedy token selection, but use the same forced
                    # continuation in both arms of the paired measurement.
                    output.logits[:, -1].argmax(-1)
                    ends[step].record()
                ends[-1].synchronize()
                wall_ms = (time.perf_counter() - wall_start) * 1000 / args.steps
                per_token = [a.elapsed_time(b) for a, b in zip(starts, ends)]
                if mode == "scratch_decode":
                    assert counts["decode"] == args.steps * model.config.num_hidden_layers, counts
                return dict(mean_device_ms=statistics.mean(per_token), wall_ms=wall_ms,
                            per_token_ms=per_token, calls=dict(counts))

            for mode in modes:
                trial(mode)
                trial(mode)
            samples = {mode: [] for mode in modes}
            for pair in range(args.pairs):
                exclusive_gpu_check()
                order = modes.copy()
                rng.shuffle(order)
                for mode in order:
                    samples[mode].append(trial(mode))
                print(f"PAIR N={n} #{pair+1}: " + json.dumps({m: samples[m][-1]["wall_ms"] for m in modes}), flush=True)
            ratios = [samples["scratch_decode"][i]["wall_ms"] / samples["sdpa"][i]["wall_ms"] for i in range(args.pairs)]
            result = dict(n=n, steps=args.steps, samples=samples, paired_wall_ratio_median=statistics.median(ratios),
                paired_wall_ratio=paired_ratio_summary([s["wall_ms"] for s in samples["scratch_decode"]],
                                                      [s["wall_ms"] for s in samples["sdpa"]]),
                median_wall_ms={m: statistics.median(s["wall_ms"] for s in samples[m]) for m in modes},
                telemetry=telemetry())
            data["results"].append(result)
            dest.write_text(json.dumps(data, indent=2) + "\n")
            print("RESULT " + json.dumps({k: v for k, v in result.items() if k not in ("samples", "telemetry")}), flush=True)
        exclusive_gpu_check()
    data["valid_for_performance_claims"] = True
    data["telemetry_after"] = telemetry()
    dest.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved {dest}", flush=True)


if __name__ == "__main__":
    main()
