# README Figures

All plots use existing measurements. Generating them does not compile a kernel,
run a benchmark or modify the original records.

| Figure | Source | Aggregation |
|---|---|---|
| Kernel throughput (README) | [September 22 raw runs](../serving/backend_overview_2026-09-22/) | B=1, H_q=H_kv=8, D=64; all four recorded sequence lengths, dense and causal; QK/PV FLOPs divided by median of three per-run median O-only latencies |
| [Per-case kernel comparison](kernel-comparison.md) | [September 22 raw runs](../serving/backend_overview_2026-09-22/) | All 34 D64 cases; median of three within-run paired O-only ratios; whiskers show the three-run min/max |
| Low latency | [September 20-21 serving records](../serving/prefill_campaign_2026-09-20/serving/graphs/cases/) | TTFT/TPOT p50 percentage change vs Native Flash; median of three per-run ratios, CUDA Graphs |
| High throughput | [Serving records](../serving/prefill_campaign_2026-09-20/serving/) | Output tokens/s percentage change vs Native Flash; median of three per-run ratios, eager and CUDA Graphs separately |
| Long context | [Serving records](../serving/prefill_campaign_2026-09-20/serving/graphs/cases/) | TTFT/TPOT p50 percentage change vs Native Flash; median of three per-run ratios, CUDA Graphs |

The kernel points are recomputed from raw paired samples and checked against
the published summary and case counts. Serving bars show custom decode-only
and custom prefill + decode relative to the native Flash zero baseline, with
all measured settings. Bars are separated by configuration and labeled with
the percentage change. Eager and CUDA Graph throughput share the same percentage
scale. TTFT and TPOT use separate scales. Latency panels show changes in p50.
The script reads only accepted `cases/` records, not superseded or rejected runs.

Figures are sized for a 550-pixel README column. Generation checks label bounds,
text overlap and a minimum displayed font size of 12 pixels at that width.

Serving bar values match the README's per-run ratio aggregation. Absolute
timings and throughput remain in the original serving report and `data.json`.
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

[Presentation references](presentation-references.md) list the original kernel
articles used to select the README chart format.

## Regenerate

With Python, NumPy and Matplotlib installed, run from the repository root:

```sh
python docs/figures/generate.py
```

To write elsewhere without touching checked-in figures:

```sh
python docs/figures/generate.py --output /tmp/flashattn-figures
```
