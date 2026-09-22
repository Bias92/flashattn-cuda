"""Independently recompute client TTFT/TPOT and compare all eager backends."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


def checked_metrics(stats):
    assert stats["completed"] == stats["num_prompts"] and stats["failed"] == 0
    assert not any(stats["errors"])
    ttft = [1000 * value for value in stats["ttfts"]]
    tpot = [1000 * sum(intervals) / (length - 1)
            for intervals, length in zip(stats["itls"], stats["output_lens"], strict=True)]
    assert len(ttft) == len(tpot) == stats["completed"]
    metrics = {"mean_ttft_ms": mean(ttft), "median_ttft_ms": median(ttft),
               "mean_tpot_ms": mean(tpot), "median_tpot_ms": median(tpot)}
    for key, value in metrics.items():
        assert math.isclose(value, stats[key], rel_tol=1e-6, abs_tol=1e-6), (key, value, stats[key])
    # Publish the official summaries after cross-checking them, preserving
    # their rounding at decimal boundaries. Keep the reconstruction residual.
    return {key: stats[key] for key in metrics} | {
        "stream_recompute_max_error_ms": max(abs(value - stats[key]) for key, value in metrics.items())}


def response_differences(actual, reference):
    return [dict(request_index=i, reference=expected, actual=observed)
            for i, (observed, expected) in enumerate(zip(actual, reference, strict=True))
            if observed != expected]


def check_smokes(root):
    baseline = json.loads((root / "sdpa_smoke_FLASH_ATTN.json").read_text())
    records = []
    for mode in ("CUSTOM", "SDPA_AUTO", "SDPA_FLASH", "SDPA_CUDNN", "SDPA_EFFICIENT", "SDPA_MATH"):
        data = json.loads((root / f"sdpa_smoke_{mode}.json").read_text())
        assert not data["graphs"] and len(data["outputs"]) == 3
        max_error = 0
        for a, b in zip(baseline["outputs"], data["outputs"], strict=True):
            assert a["prompt"] == b["prompt"]
            assert a["token_ids"] == b["token_ids"] and len(b["token_ids"]) == 32, mode
            max_error = max(max_error, max(abs(x - y) for x, y in
                             zip(a["selected_logprobs"], b["selected_logprobs"], strict=True)))
        layers = [layer for worker in data["routing"] for layer in worker]
        assert len(layers) == 22 and all(layer["scratch_calls"] == 31 for layer in layers)
        records.append(dict(backend=mode, matching_tokens=96, routed_layers=22,
                            selected_logprob_max_diff=max_error))
    (root / "sdpa_smoke_validation.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = args.directory
    if args.smoke_only:
        check_smokes(root)
        return
    manifest = json.loads((root / "manifest.json").read_text())
    settings = manifest["settings"]
    assert manifest["complete"] and settings["eager"]
    assert len(manifest["results"]) == settings["runs"] * len(settings["backends"])
    repo = Path(__file__).resolve().parents[1]
    for name, digest in manifest["implementation_sha256"].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
    for digest, location in manifest.get("runner_revisions", {}).items():
        path = repo / "bench/bench_vllm_serving.py" if location == "current_workspace_runner" else root / location
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, location
    for event in manifest.get("resume_events", []):
        for name, digest in event["preserved_raw_sha256"].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    rows, samples, clock_resets = [], [], []
    for record in manifest["results"]:
        run, backend = record["run"], record["backend"]
        stats = json.loads((root / f"run{run}_{backend.lower()}.json").read_text())
        baseline = json.loads((root / f"run{run}_flash_attn.json").read_text())
        metrics = checked_metrics(stats)
        if manifest.get("runner_revisions"):
            assert record["runner_sha256"] in manifest["runner_revisions"]
        count = settings["num_prompts"]
        assert stats["completed"] == count
        assert stats["input_lens"] == baseline["input_lens"] == [settings["input_len"] - 1] * count
        assert stats["output_lens"] == baseline["output_lens"] == [settings["output_len"]] * count
        assert "--enforce-eager" in record["server_command"]
        if backend.startswith("SDPA_"):
            assert record["paged_to_dense_included"] and record["operator_audit"]
        differences = response_differences(actual=stats["generated_texts"],
                                            reference=baseline["generated_texts"])
        rows.append(dict(run=run, backend=backend, **metrics,
            matching_generated_texts=sum(x == y for x, y in
                                        zip(stats["generated_texts"], baseline["generated_texts"], strict=True)),
            completed=count, p99_ttft_ms=stats["p99_ttft_ms"], p99_tpot_ms=stats["p99_tpot_ms"],
            mismatched_responses=differences,
            initial_operator_audit=record["operator_audit"]))
        if samples and record["telemetry"][0]["monotonic_s"] < samples[-1]["monotonic_s"]:
            clock_resets.append(dict(before_run=run, before_backend=backend,
                previous_monotonic_s=samples[-1]["monotonic_s"],
                resumed_monotonic_s=record["telemetry"][0]["monotonic_s"]))
        samples.extend(record["telemetry"])
    assert samples and not any(s["other_jobs"] for s in samples)
    summary = dict(settings=settings, results=rows,
        monotonic_clock_resets_between_phases=clock_resets,
        max_sampled_temperature_c=max(float(s["temperature.gpu"]) for s in samples),
        caveats=["All backends eager. Do not combine with earlier CUDA Graph results.",
                 "Prefill is vLLM FlashAttention in all modes; decode only is replaced.",
                 "SDPA includes per-layer paged gather/repack and any GQA expansion.",
                 "These are adapter-inclusive serving results, not isolated kernel speedups.",
                 "Only two independent server runs per backend; exploratory.",
                 "GPU clocks unlocked; Windows graphics activity not eliminated."])
    if clock_resets:
        summary["caveats"].append("The telemetry monotonic clock restarted between interrupted phases; OS-session continuity was not controlled.")
    if any(row["mismatched_responses"] for row in rows):
        summary["caveats"].append("Some generated texts differ despite fixed output token counts; see mismatched_responses. No failing-request logit diagnosis was performed.")
    auto_audits = [line for row in rows if row["backend"] == "SDPA_AUTO"
                   for line in row["initial_operator_audit"]]
    if auto_audits and all("aten::_scaled_dot_product_flash_attention" in line for line in auto_audits):
        summary["caveats"].append("AUTO selected Flash in the initial warmup operator audits; this does not prove every subsequent call chose Flash. Timing differences are not a clean algorithmic ranking.")
    paired = []
    for run in range(settings["runs"]):
        by_backend = {row["backend"]: row for row in rows if row["run"] == run}
        if "CUSTOM" in by_backend:
            paired.append(dict(run=run, custom_over_native_tpot_median=
                by_backend["CUSTOM"]["median_tpot_ms"] / by_backend["FLASH_ATTN"]["median_tpot_ms"]))
    summary["custom_native_paired_ratios"] = paired
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Eager Serving Comparison", "", "All times are client-observed milliseconds.", "",
             "| Backend | Run | TTFT median | TTFT mean | TPOT median | TPOT mean | Matching texts |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for backend in settings["backends"]:
        for row in sorted((r for r in rows if r["backend"] == backend), key=lambda r: r["run"]):
            lines.append(f"| {backend} | {row['run']} | {row['median_ttft_ms']:.4f} | "
                         f"{row['mean_ttft_ms']:.4f} | {row['median_tpot_ms']:.4f} | "
                         f"{row['mean_tpot_ms']:.4f} | {row['matching_generated_texts']}/{row['completed']} |")
    lines += ["", *[f"- {item}" for item in summary["caveats"]], ""]
    lines += ["## Output Differences", ""]
    for row in rows:
        for difference in row["mismatched_responses"]:
            lines += [f"- Run {row['run']}, {row['backend']}, request {difference['request_index']}: "
                      f"reference `{difference['reference']}`, actual `{difference['actual']}`."]
    lines += ["", "## Measurement Limits", ""]
    if (any(p["custom_over_native_tpot_median"] < 1 for p in paired) and
            any(p["custom_over_native_tpot_median"] > 1 for p in paired)):
        lines += ["The lower native TPOT median switches between CUSTOM and FLASH_ATTN across runs. "
                  "This eager experiment does not establish a reproducible native-vLLM win.", ""]
    lines += ["Do not interpret the dense-adapter gap as a pure SDPA/cuDNN kernel speedup.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
