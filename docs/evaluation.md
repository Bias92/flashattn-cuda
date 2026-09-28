# Evaluation Details

[Project overview](../README.md)

## Serving Evaluation

### Workloads And Metrics

| Workload | Test conditions | Recorded metrics |
|---|---|---|
| **Low latency** | TinyLlama-1.1B; one request at a time; 128 / 512 / 1024 / 1920 input tokens | TTFT, TPOT, inter-token and end-to-end latency; p50 / p95 / p99 |
| **High throughput** | TinyLlama-1.1B; 512 input tokens; 2 / 4 / 8 / 16 / 32 concurrent requests | Total output tokens/s, TTFT, TPOT and request latency distributions |
| **Long context** | Qwen2.5-0.5B; one request at a time; 2048 / 4096 / 8192 / 16384 / 32640 input tokens | TTFT, TPOT, end-to-end latency, completed/failed requests; KV-cache capacity in server logs |

**TTFT** is time to the first token. **TPOT** is the average time per output
token after the first; **inter-token latency** measures individual gaps between
tokens. **End-to-end latency** covers the whole request. p50 is the median;
p95 and p99 describe the slow tail. **Output tokens/s** counts generated tokens
across all requests, excluding prompt tokens.

### Comparison Setup

Only the attention backend changes; model weights and engine settings match.

| Configuration | Prefill | Decode |
|---|---|---|
| Native Flash | vLLM FlashAttention | vLLM FlashAttention |
| Scratch decode-only | vLLM FlashAttention | Custom paged decode |
| Scratch full | Custom prefill | Custom paged decode |

The campaign used vLLM 0.19.0, FP16, 128 output tokens per request,
2048-token chunked prefill and disabled prefix caching. Three runs of each
configuration across 14 workload settings, in both eager and CUDA Graph modes,
produced **252 accepted measurements**. Server startup and warmup are excluded.

### Results

**Scratch full versus native Flash**, from the 2026-09-20/21 campaign.
Each value is the median of three per-run ratios; ranges cover the listed
workload settings. Negative latency changes and positive throughput changes
are better.

| Workload | Metric | Eager change | CUDA Graph change |
|---|---|---:|---:|
| Low latency, 512-1920 input tokens | TTFT p50 | -4.4% to -2.7% | -1.5% to -0.1% |
| Low latency, 512-1920 input tokens | TPOT p50 | -6.6% to -4.4% | +1.2% to +1.8% |
| High throughput, 2-32 concurrent requests | Output tokens/s | +8.6% to +15.7% | -0.1% to +1.3% |
| High throughput, 2-32 concurrent requests | TPOT p50 | -14.7% to -8.9% | -0.8% to +1.6% |
| Long context, 2048-32640 input tokens | TTFT p50 | -5.3% to -3.9% | -3.9% to -2.2% |
| Long context, 2048-32640 input tokens | TPOT p50 | -12.4% to -8.2% | -1.9% to +0.4% |

With CUDA Graphs, the long-context TTFT improvement remains, while the eager
throughput advantage largely disappears. The 128-token TTFT case and
high-concurrency TTFT showed substantial run-to-run variation; individual
runs and tail latencies are in the full report. KV-cache records describe
engine capacity, not per-request peak memory.

These are whole-engine comparisons, including backend dispatch and cache
access. The serving implementation is an earlier revision than the kernel
comparison below.
[Full report](serving/prefill_campaign_2026-09-20/REPORT.md)
| Raw records: [eager](serving/prefill_campaign_2026-09-20/serving/eager/cases/),
[CUDA Graphs](serving/prefill_campaign_2026-09-20/serving/graphs/cases/).

## Kernel Evaluation

**Causal D64 is the strongest path.** Across the measured cases, it is faster
than SDPA-Flash in 15 of 17 cases and SDPA-cuDNN in 16 of 17. Dense D64 is close
to Flash, while cuDNN is faster in most dense cases.

The following measurements are from **2026-09-22**, using PyTorch 2.10.0+cu128
on the RTX 4060 Ti. They time the warmed **O-only public APIs**. Flash and cuDNN
are separately forced PyTorch SDPA backends; default SDPA selected Flash in
every measured case.

B=1, H_q=H_kv=8, D=64, N_q=N_kv=N. Lower time is better.

| Mask | N | Scratch (ms) | SDPA-Flash (ms) | SDPA-cuDNN (ms) |
|---|---:|---:|---:|---:|
| Dense | 1024 | 0.06227 | 0.05998 | 0.05782 |
| Dense | 2048 | 0.22313 | 0.22501 | 0.21361 |
| Dense | 4096 | 0.86249 | 0.87860 | 0.85070 |
| Causal | 1024 | 0.05296 | 0.05270 | 0.05217 |
| Causal | 2048 | 0.14025 | 0.15490 | 0.14524 |
| Causal | 4096 | 0.48298 | 0.51368 | 0.51085 |

The full D64 set covers B=1 through 8, N=512 through 8192, multiple head counts
and GQA ratios, including non-aligned sequence lengths:

| Mask | Cases | vs Flash: faster / similar / slower | vs cuDNN: faster / similar / slower |
|---|---:|---:|---:|
| Dense | 17 | 10 / 4 / 3 | 2 / 4 / 11 |
| Causal | 17 | 15 / 0 / 2 | 16 / 0 / 1 |

Similar means within +/-1% of the paired latency ratio, not a statistical
significance test. These are development/comparison cases, not a fresh holdout.

<details>
<summary>Measurement protocol and original records</summary>

- Three independent process/seed runs, each with 12 rotating-order paired
  repetitions per case and a 30 ms target measurement window.
- CUDA event timing over warmed calls, without CUDA graphs. Compilation,
  cuDNN plan selection, input allocation and correctness checks are outside
  timing; output allocation is part of each API call. GPU clocks were not locked.
- Table times are medians across runs. Faster/similar/slower counts use medians
  of within-run paired ratios, rather than division of the displayed times.
- The timing data predates the standalone-source cleanup. The report retains
  the measured source hash, loaded library paths, backend dispatch and telemetry.

[Full matrix and methodology](serving/backend_overview_2026-09-22/README.md)
| [Summary JSON](serving/backend_overview_2026-09-22/summary.json)
| Raw runs: [1](serving/backend_overview_2026-09-22/run1.json),
[2](serving/backend_overview_2026-09-22/run2.json),
[3](serving/backend_overview_2026-09-22/run3.json).

</details>
