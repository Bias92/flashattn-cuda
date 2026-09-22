"""Recompute all reported prefill ratios from the three independent raw runs."""
import hashlib
import json
from pathlib import Path
import statistics as stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def key(case):
    return tuple(case["shape"]) + (case["causal"],)


def rank(values):
    return {"faster": sum(v < 0.99 for v in values),
            "similar": sum(0.99 <= v <= 1.01 for v in values),
            "slower": sum(v > 1.01 for v in values),
            "min_gap_pct": 100 * (min(values) - 1),
            "max_gap_pct": 100 * (max(values) - 1)}


def main():
    paths = [HERE / f"run{i}.json" for i in (1, 2, 3)]
    runs = [json.loads(p.read_text()) for p in paths]
    assert all(r["meta"]["complete"] and len(r["cases"]) == 46 for r in runs)
    assert len({r["meta"]["source"]["sha256"] for r in runs}) == 1
    assert len({r["meta"]["source"]["so_sha256"] for r in runs}) == 1
    assert len({r["meta"]["runner_sha256"] for r in runs}) == 1
    maps = [{key(c): c for c in r["cases"]} for r in runs]
    assert all(len(m) == 46 and m.keys() == maps[0].keys() for m in maps)
    rows = []
    for shape in maps[0]:
        cases = [m[shape] for m in maps]
        row = {"shape": list(shape[:-1]), "causal": shape[-1], "set": cases[0]["set"],
               "median_ms": {}, "ratios": {}, "ratios_by_run": {}}
        for name in ("ours_o", "ours_l", "flash", "cudnn"):
            row["median_ms"][name] = stats.median(
                stats.median(p[name] for p in c["samples_ms"]) for c in cases)
        for ours in ("ours_o", "ours_l"):
            row["ratios"][ours], row["ratios_by_run"][ours] = {}, {}
            for backend in ("flash", "cudnn"):
                values = [stats.median(p[ours] / p[backend] for p in c["samples_ms"]) for c in cases]
                assert all(abs(v - c["ratios"][ours][backend]["median"]) < 1e-12
                           for v, c in zip(values, cases))
                row["ratios"][ours][backend] = stats.median(values)
                row["ratios_by_run"][ours][backend] = values
        rows.append(row)
    groups = []
    for d in (64, 128):
        for causal in (False, True):
            selected = [r for r in rows if r["shape"][4] == d and r["causal"] == causal]
            groups.append({"D": d, "causal": causal, "count": len(selected),
                           "entries": {ours: {backend: rank([r["ratios"][ours][backend] for r in selected])
                                              for backend in ("flash", "cudnn")}
                                       for ours in ("ours_o", "ours_l")}})
    auto_ops = sorted({tuple(c["dispatch"]["auto"]) for r in runs for c in r["cases"]})
    summary = {"groups": groups, "cases": rows,
               "totals": {ours: {backend: rank([r["ratios"][ours][backend] for r in rows])
                                 for backend in ("flash", "cudnn")}
                          for ours in ("ours_o", "ours_l")},
               "auto_operators": auto_ops,
               "max_abs_difference": max(v for r in runs for c in r["cases"]
                                         for v in c["max_abs_difference_from_ours_o"].values()),
               "raw_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Attention backend overview, 2026-09-22", "",
             "Final project scope is D64. D128 rows below are preserved historical measurements,",
             "not an active optimization target. The original 46-case raw files are unchanged.", "",
             "## Measured Prefill", "",
             "RTX 4060 Ti 8 GB, FP16 inputs, FP32 scratch accumulation, D=64/128.",
             "46 fixed shape/mask cases from the existing readme/holdout/holdout2 sets.",
             "These are comparison sets, not a new untouched holdout. Three independent",
             "process/seed runs; 12 rotating-order paired repetitions per case per run,",
             "30 ms windows, CUDA events over warmed public APIs. No CUDA graphs.",
             "Backend selection, dispatch profiling, correctness, compilation, cuDNN plan",
             "warmup and allocation of input tensors are outside timing. Output allocation",
             "remains part of each public API call. Clocks were not locked or changed.", "",
             "Current source SHA256: `" + runs[0]["meta"]["source"]["sha256"] + "`.",
             "The loaded .so/hash, helper hashes, versions and GPU telemetry are in each raw JSON.",
             "The kernel was unchanged throughout all three runs.", "",
             "Faster/similar/slower use a practical +/-1% band on the median of the three",
             "within-run paired ratios, not a significance test. Counts describe these cases",
             "only; they do not weight workloads by deployment frequency.", "",
             "| D | Mask | Cases | O-only vs Flash F/S/L | O-only vs cuDNN F/S/L |",
             "|---:|---|---:|---|---|"]
    for g in groups:
        cells = [str(g["D"]), "causal" if g["causal"] else "dense", str(g["count"])]
        for backend in ("flash", "cudnn"):
            counts = g["entries"]["ours_o"][backend]
            cells.append("/".join(str(counts[k]) for k in ("faster", "similar", "slower")))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "SDPA default dispatch was audited for every case in every run:",
              "`" + str(auto_ops) + "`.",
              "O-only and +L outputs were checked against both forced backends with",
              "atol=1e-3, rtol=1e-3; no tolerance or kernel precision was changed.",
              f"Largest observed absolute output difference: {summary['max_abs_difference']:.8g}.",
              "The preceding 152-case FP32-reference/sanitizer verification is recorded in",
              "[the regression report](../prefill_dense_2026-09-21/README.md).", ""]
    for ours, title in (("ours_o", "O-only"), ("ours_l", "O + L")):
        lines += [f"## {title} Full Matrix", "", "Positive gap means scratch is slower.",
                  "Times are independently aggregated medians. Gaps come from paired ratios,",
                  "not division of the displayed time medians; their signs can differ near parity.", "",
                  "| Set | B | H/Hkv | N | D | Mask | Scratch ms | Flash ms | cuDNN ms | Flash gap | cuDNN gap |",
                  "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            b, h, hk, n, d = row["shape"]
            t, ratios = row["median_ms"], row["ratios"][ours]
            lines.append(f"| {row['set']} | {b} | {h}/{hk} | {n} | {d} | "
                         f"{'causal' if row['causal'] else 'dense'} | {t[ours]:.5f} | "
                         f"{t['flash']:.5f} | {t['cudnn']:.5f} | "
                         f"{100*(ratios['flash']-1):+.2f}% | {100*(ratios['cudnn']-1):+.2f}% |")
    lines += ["", "## Other Evidence", "",
              "[The 2026-09-20 serving campaign](../prefill_campaign_2026-09-20/REPORT.md)",
              "contains native Flash, scratch decode-only and scratch prefill+decode, eager",
              "and CUDA graphs separately. It predates the latest regression fixes and was",
              "not rerun here. Full/decode-only TTFT differences are configuration-level",
              "comparisons, not a perfectly isolated prefill-kernel experiment.", "",
              "For SCRATCH_FULL versus native FLASH_ATTN with CUDA graphs, long-context",
              "TTFT was 2.2-3.9% lower across five lengths. Throughput across concurrency",
              "2/4/8/16/32 was approximately unchanged (-0.1% to +1.3% output tokens/s).",
              "These are historical engine results, not measurements of today's kernel.", "",
              "No new FlashInfer, standalone FA3, full-model TTFT/TPOT, throughput, or",
              "cross-GPU run was performed here. No production source, old result, commit,",
              "push, GPU clock setting or cloud resource was changed by this report.", "",
              "## Reproduce", "", "```sh",
              "CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.9 \\",
              "python3 bench/report_attention_backends.py --run 1 --output /tmp/backend-new-run1.json",
              "```", "", "Use new output paths and run IDs 1, 2, 3 for independent processes.",
              "The current runner selects the 34 D64 cases; the historical raw runs include D128.",
              "`make_report.py` recomputes this dated report and summary.json from the original run1/2/3.json."]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"groups": groups, "totals": summary["totals"]}, indent=2))


if __name__ == "__main__":
    main()
