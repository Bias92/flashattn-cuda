# README Figures

All plots use existing measurements. Generating them does not compile a kernel,
run a benchmark or modify the original records.

| Figure | Source | Aggregation |
|---|---|---|
| Kernel performance | [September 22 raw runs](../serving/backend_overview_2026-09-22/) | All 34 D64 cases; median of three within-run paired O-only ratios; whiskers show the three-run min/max |
| Low latency | [September 20-21 serving records](../serving/prefill_campaign_2026-09-20/serving/graphs/cases/) | Median of three per-run TTFT/TPOT p50 values, CUDA Graphs |
| High throughput | [Serving records](../serving/prefill_campaign_2026-09-20/serving/) | Median output tokens/s over three runs, eager and CUDA Graphs separately |
| Long context | [Serving records](../serving/prefill_campaign_2026-09-20/serving/graphs/cases/) | Median of three per-run TTFT/TPOT p50 values, CUDA Graphs |

The kernel points are recomputed from raw paired samples and checked against
the published summary and case counts. Serving plots include native Flash,
custom decode-only and custom prefill + decode, with all measured settings.
The script reads only accepted `cases/` records, not superseded or rejected runs.

Absolute metric medians and median per-run ratios are different aggregations:
dividing the plotted values need not reproduce the README's percentage changes.
Three-run min/max whiskers are not confidence intervals. The kernel plot's
+/-1% band is a practical threshold, not a significance test.

Source file hashes and the numerical inputs to every performance plot are in
[data.json](data.json). Dates and source-revision boundaries are described in
[the evaluation notes](../evaluation.md).

## Regenerate

With Python, NumPy and Matplotlib installed, run from the repository root:

```sh
python docs/figures/generate.py
```

To write elsewhere without touching checked-in figures:

```sh
python docs/figures/generate.py --output /tmp/flashattn-figures
```
