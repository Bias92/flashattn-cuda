# README Figures

All plots use existing measurements. Generating them does not compile a kernel,
run a benchmark or modify the original records.

| Figure | Source | Aggregation |
|---|---|---|
| Kernel throughput (README) | [September 22 raw runs](../serving/backend_overview_2026-09-22/) | B=1, H_q=H_kv=8, D=64; all four recorded sequence lengths, dense and causal; QK/PV FLOPs divided by median of three per-run median O-only latencies |
| [Per-case kernel comparison](kernel-comparison.md) | [September 22 raw runs](../serving/backend_overview_2026-09-22/) | All 34 D64 cases; median of three within-run paired O-only ratios; whiskers show the three-run min/max |
| Low latency | [September 20-21 serving records](../serving/prefill_campaign_2026-09-20/serving/graphs/cases/) | TTFT/TPOT in ms; median of three per-run request p50s, CUDA Graphs |
| High throughput | [Serving records](../serving/prefill_campaign_2026-09-20/serving/) | Total output tokens/s; median of three per-run values, eager and CUDA Graphs separately |
| Long context | [Serving records](../serving/prefill_campaign_2026-09-20/serving/graphs/cases/) | TTFT in s and TPOT in ms; median of three per-run request p50s, CUDA Graphs |

The kernel points are recomputed from raw paired samples and checked against
the published summary and case counts. Serving plots use grouped vertical bars:
Native Flash, custom decode-only, and custom prefill + decode at every setting.
All vertical axes start at zero. Eager and CUDA Graph throughput share the same
tokens/s scale. TTFT and TPOT use separate scales. Latency bars show the median
of three per-run p50s, not a percentile of pooled requests.
The script reads only accepted `cases/` records, not superseded or rejected runs.

Figures are sized for a 550-pixel README column. Generation checks label bounds,
text overlap and a minimum displayed font size of 12 pixels at that width.

The [measurement tables](serving-measurements.md) list the values plotted and
the corresponding eager results. Serving percentages in the root README are
calculated from the same unrounded medians as the bars and tables.
The historical campaign's per-run ratios remain in its report and `data.json`.
The 128-token TTFT case has substantial run-to-run variation; a bar is the
median estimate, not a significance claim.
Three-run min/max whiskers are not confidence intervals. The kernel plot's
+/-1% band is a practical threshold, not a significance test.

Source file hashes and the numerical inputs to every performance plot are in
[data.json](data.json). Dates and source-revision boundaries are described in
[the evaluation notes](../evaluation.md).

The [34-case effective TFLOP/s table](kernel-throughput.md) is derived from
the same kernel timings. Its FLOP convention and exclusions are documented
there; it does not estimate whole-model serving FLOP/s.

[Presentation references](presentation-references.md) identify the exact FA2,
FlashInfer, Hydragen and NanoFlow figures used for the README layouts.

## Regenerate

With Python, NumPy and Matplotlib installed, run from the repository root:

```sh
python docs/figures/generate.py
```

To write elsewhere without touching checked-in figures:

```sh
python docs/figures/generate.py --output /tmp/flashattn-figures
```
