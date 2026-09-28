# Per-Case Kernel Comparisons

RTX 4060 Ti 8 GB, D=64, warmed O-only API measurements from September 22.
All 34 cases are included: batch sizes 1-8, MHA/GQA, dense/causal masks,
and sequence lengths 512-8192.

![All 34 kernel cases: latency change versus SDPA-Flash and SDPA-cuDNN](kernel-performance.png?v=aa05629)

Each row is one shape: B is batch size, H is query/KV heads, and N is sequence
length. Negative values mean the custom kernel takes less time. Points are the
median of three within-run paired ratios; whiskers are the three-run min/max,
not confidence intervals. The shaded +/-1% band is a comparison threshold,
not a significance test.

Counts are faster / within 1% / slower:

| Mask | Cases | vs SDPA-Flash | vs SDPA-cuDNN |
|---|---:|---:|---:|
| Dense | 17 | 10 / 4 / 3 | 2 / 4 / 11 |
| Causal | 17 | 15 / 0 / 2 | 16 / 0 / 1 |

[Absolute timings and TFLOP/s](kernel-throughput.md)
| [Raw runs and measurement protocol](../serving/backend_overview_2026-09-22/README.md)
