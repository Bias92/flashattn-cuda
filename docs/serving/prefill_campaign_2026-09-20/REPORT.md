# Prefill in a serving engine: what the kernel is worth, 2026-09-20/21

Historical campaign, not a measurement of the current prefill source.
The measured [v7 source snapshot](attention_forward_v7.cu.txt) is evidence only,
not a build input. See the [current index](../README.md) for version boundaries.
Published server logs redact temporary localhost API keys; original private
logs remain in the owner's external archive. Measurement JSON is unchanged.

252 accepted cases: 3 attention configurations x 3 workloads x 3 runs x (eager, CUDA graphs).
RTX 4060 Ti 8 GB, NVIDIA GeForce RTX 4060 Ti, 595.97, python 3.12.3, torch 2.10.0+cu128,
vLLM 0.19.0, transformers 4.57.6.

| configuration | prefill | decode |
|---|---|---|
| `FLASH_ATTN` | vLLM's FlashAttention | vLLM's FlashAttention |
| `SCRATCH_DECODE` | vLLM's FlashAttention | scratch paged decode (`integrations/vllm`) |
| `SCRATCH_FULL` | **scratch prefill** (`cuda/attention_forward.cu`) | scratch paged decode |

The two scratch configurations share the decode kernel, but differ in prefill,
batch dispatch, metadata handling and cache access. This is a configuration-level
comparison, not an isolated prefill-kernel measurement. Sources at measurement time:

- `cuda/attention_forward.cu` `5f1d9815423d9fe1...`
- `cuda/attention_decode_paged.cu` `89db4293f401deed...`
- `integrations/vllm/scratch_vllm/backend.py` `ead06750333bd3c4...`
- `integrations/vllm/scratch_vllm/loader.py` `728f995e06c86761...`
- `integrations/vllm_prefill/scratch_vllm_prefill/backend.py` `49d4345e0ccfa9b4...`
- `integrations/vllm_prefill/scratch_vllm_prefill/loader.py` `d50ba1fbd24a3ae2...`
- `integrations/vllm_prefill/scratch_vllm_prefill/audit.py` `318b8d1b7159f1a4...`
- `bench/bench_serving_prefill.py` `2042275c0c5fa949...`
- `bench/gpu_busy.ps1` `387430f551cf33bb...`

Conditions are the decode campaign's (`bench/bench_serving_workloads.py`, 2026-09-15):
`vllm serve` and `vllm bench serve`, random dataset at request rate inf, FP16, 128 output
tokens, max-num-seqs 32, 2048 batched tokens per step, prefix caching off, GPU memory 0.55,
server seed 42, max(64, 8C) prompts after a warm-up of max(3, 2C) with a different seed.
Server start and warm-up are outside every measured window. One server per
(workload, configuration, run); the configuration order rotates between runs.

Runner: `bench/bench_serving_prefill.py`. Raw client output, server logs and the loaded-kernel
hashes are under `serving/{eager,graphs}/sessions/`, accepted records under `cases/`.

## 1. What actually ran

Per-layer route counters and the server's own token counters are read immediately before and
after each measured client run, and a case is rejected unless they say the configuration's own
kernels did the work. Summed over all 252 accepted cases, tokens per layer:

| route | SCRATCH_FULL |
|---|---:|
| `prefill_dense` (K, V from the projections) | 7,063,041 |
| `prefill_paged` (K, V from the paged cache) | 20,412,903 |
| `decode_in_mixed` | 14,058 |
| `decode_only` | 432,068 |
| **`flash_fallback`** | **0** |

Not one token fell back to FlashAttention, so nothing measured as SCRATCH_FULL was secretly
served by vLLM's kernel. In eager mode the routes are required to add up to exactly the tokens
the server reports processing; that held for all 126 eager cases.

With CUDA graphs a step of nothing but single-token requests is replayed from a captured graph
and never reaches Python, so decode cannot be counted inside the window. What is checked there
instead: prefill (never captured) still accounts for the prompt tokens, the decode counter is
non-zero before the window (startup, capture, warm-up), and every layer reports the expected
implementation class. That is strong evidence, not proof - graph replay is not observable from
Python, and this is stated rather than papered over.

### A verification bug, and what was redone

The first CUDA-graph pass required the decode counter to grow *inside* the window, which that
mode makes impossible. It rejected all 42 `SCRATCH_DECODE` cases and let `SCRATCH_FULL` pass on
its prefill accounting alone. After the fix the whole graphs pass was redone except the 42
`FLASH_ATTN` cases, which the check never touched. The 42 `SCRATCH_FULL` cases accepted under
the old rule were moved to `serving/graphs/superseded_by_routing_fix/` rather than deleted, and
their raw client output is still under `sessions/`. Rejections after the fix: 0. This is why the
graphs report prints two source sets - the second differs only in the runner's accept/reject
predicate, and no kernel, backend or client command changed.

## 2. Full versus decode-only configuration

`SCRATCH_FULL / SCRATCH_DECODE` on TTFT p50. Both carry the same paged decode
kernel, but their backend dispatch and metadata paths differ. Below 1.0 means
lower TTFT for the full configuration. Median of three runs, with each run shown.

### long context (Qwen2.5-0.5B, 2048-token chunked prefill)

| input tokens | eager | graphs | eager per run | graphs per run |
|---:|---:|---:|---|---|
| 2048 | 0.971 | 0.979 | 0.950 / 0.971 / 1.017 | 0.969 / 0.979 / 0.986 |
| 4096 | 0.953 | 0.962 | 0.945 / 0.953 / 0.967 | 0.959 / 0.962 / 0.964 |
| 8192 | 0.960 | 0.959 | 0.943 / 0.960 / 0.963 | 0.959 / 0.959 / 0.961 |
| 16384 | 0.952 | 0.953 | 0.935 / 0.952 / 0.953 | 0.950 / 0.953 / 0.956 |
| 32640 | 0.947 | 0.949 | 0.946 / 0.947 / 0.951 | 0.948 / 0.949 / 0.950 |

### low latency (TinyLlama-1.1B, one request at a time)

| input tokens | eager | graphs | eager per run | graphs per run |
|---:|---:|---:|---|---|
| 128 | 0.961 | 0.988 | 0.748 / 0.961 / 0.969 | 0.968 / 0.988 / 0.997 |
| 512 | 0.973 | 0.994 | 0.957 / 0.973 / 0.981 | 0.992 / 0.994 / 1.007 |
| 1024 | 0.977 | 0.976 | 0.970 / 0.977 / 0.978 | 0.975 / 0.976 / 0.993 |
| 1920 | 0.970 | 0.972 | 0.968 / 0.970 / 0.973 | 0.968 / 0.972 / 0.986 |

The observed difference is **2 to 5% of TTFT, growing with prompt length**,
with similar direction in both execution modes. All three runs agree in sign
at every length from 1024 tokens up.
At 128 and 512 tokens the effect is within the run-to-run spread and should not be read.

This is an end-to-end serving number, including RMSNorm, RoPE, QKV and MLP
projections, cache access and backend work. It cannot be converted into an
attention-only speedup from these measurements.

## 3. The decode gain does not survive CUDA graphs

TPOT p50 against `FLASH_ATTN`. Below 1.0 = faster.

### throughput (TinyLlama-1.1B, 512-token prompts, concurrent requests)

| concurrency | SCRATCH_DECODE eager | graphs | SCRATCH_FULL eager | graphs |
|---:|---:|---:|---:|---:|
| C=2 | 0.933 | 1.001 | 0.911 | 0.999 |
| C=4 | 0.841 | 1.002 | 0.857 | 1.003 |
| C=8 | 0.899 | 0.991 | 0.903 | 0.992 |
| C=16 | 0.888 | 1.032 | 0.853 | 1.016 |
| C=32 | 0.891 | 1.056 | 0.893 | 1.011 |

### long context

| input tokens | SCRATCH_DECODE eager | graphs | SCRATCH_FULL eager | graphs |
|---:|---:|---:|---:|---:|
| 2048 | 0.904 | 0.998 | 0.918 | 0.997 |
| 4096 | 0.875 | 0.990 | 0.876 | 0.990 |
| 8192 | 0.900 | 0.991 | 0.914 | 0.992 |
| 16384 | 0.890 | 0.981 | 0.899 | 0.981 |
| 32640 | 0.885 | 1.002 | 0.899 | 1.004 |

In eager mode both scratch configurations decode 7 to 16% faster (0.841 to 0.933). Under CUDA graphs that gain
is gone: the ratios sit between 0.98 and 1.06, and at 32 concurrent requests `SCRATCH_DECODE`
is 5.6% *slower*. Both configurations move together, which points at the decode kernel they
share rather than at prefill.

The reading: in eager mode every decode step pays Python dispatch and launch overhead, and the
scratch path pays less of it than vLLM's. CUDA graphs capture that host work once and replay
it, so what is left is kernel time, and there the scratch decode kernel is about even and
falls behind as the batch grows.

This matters for how the decode kernel is described. The decode campaign of 2026-09-15 ran
`--enforce-eager`; its numbers are not contradicted here and were measured under stated
conditions. But a decode result measured eager should not be carried over to a CUDA-graph
deployment, which is the default and what serving actually uses. This campaign measured both
and they disagree.

The prefill result (section 2) does not have this problem, and the reason is structural rather
than lucky: prefill cannot be captured into a graph in either configuration.

## 4. End to end against vLLM's FlashAttention

Values are medians over the three runs; ratios are the median of the per-run ratios, with each
run in brackets.

### eager

#### latency
| input | C | config | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | out tok/s | TTFT ratio | TPOT ratio | tok/s ratio |
|---:|---:|---|---:|---:|---:|---:|---|---|---|
| 128 | 1 | FLASH_ATTN | 29.0 | 83.1 | 10.85 | 89.3 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 128 | 1 | SCRATCH_DECODE | 30.0 | 149.9 | 10.52 | 92.2 | 1.034 [1.034 / 1.495 / 0.702] | 0.993 [0.913 / 0.993 / 0.998] | 1.033 [1.103 / 1.033 / 1.010] |
| 128 | 1 | SCRATCH_FULL | 22.5 | 137.1 | 10.67 | 90.2 | 0.774 [0.774 / 1.448 / 0.675] | 0.947 [0.933 / 0.947 / 1.035] | 1.046 [1.053 / 1.046 / 0.985] |
| 512 | 1 | FLASH_ATTN | 34.6 | 135.9 | 11.27 | 85.2 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 1 | SCRATCH_DECODE | 34.4 | 135.8 | 10.71 | 90.5 | 0.993 [0.993 / 0.978 / 0.999] | 0.966 [0.901 / 0.966 / 0.972] | 1.064 [1.097 / 1.064 / 1.044] |
| 512 | 1 | SCRATCH_FULL | 33.0 | 117.9 | 10.94 | 88.8 | 0.956 [0.974 / 0.952 / 0.956] | 0.944 [0.931 / 0.991 / 0.944] | 1.064 [1.064 / 1.039 / 1.069] |
| 1024 | 1 | FLASH_ATTN | 62.1 | 176.6 | 11.28 | 83.7 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 1024 | 1 | SCRATCH_DECODE | 61.8 | 136.6 | 10.37 | 90.9 | 0.996 [1.008 / 0.995 / 0.996] | 0.935 [0.954 / 0.935 / 0.902] | 1.078 [1.035 / 1.078 / 1.101] |
| 1024 | 1 | SCRATCH_FULL | 60.4 | 140.5 | 10.69 | 90.1 | 0.973 [0.977 / 0.973 / 0.973] | 0.956 [0.944 / 0.969 / 0.956] | 1.051 [1.084 / 1.051 / 1.049] |
| 1920 | 1 | FLASH_ATTN | 115.7 | 192.7 | 11.88 | 78.8 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 1920 | 1 | SCRATCH_DECODE | 115.8 | 201.5 | 10.46 | 87.2 | 1.001 [0.997 / 1.001 / 1.004] | 0.915 [0.915 / 0.991 / 0.831] | 1.066 [1.066 / 1.022 / 1.159] |
| 1920 | 1 | SCRATCH_FULL | 112.4 | 196.9 | 10.65 | 84.9 | 0.972 [0.965 / 0.972 / 0.978] | 0.934 [0.896 / 1.007 / 0.934] | 1.054 [1.089 / 1.018 / 1.054] |

#### throughput
| input | C | config | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | out tok/s | TTFT ratio | TPOT ratio | tok/s ratio |
|---:|---:|---|---:|---:|---:|---:|---|---|---|
| 512 | 2 | FLASH_ATTN | 63.3 | 334.6 | 12.09 | 156.7 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 2 | SCRATCH_DECODE | 60.4 | 313.2 | 11.28 | 169.1 | 0.958 [0.958 / 0.810 / 0.959] | 0.933 [0.995 / 0.933 / 0.898] | 1.071 [1.053 / 1.071 / 1.096] |
| 512 | 2 | SCRATCH_FULL | 58.1 | 334.8 | 11.18 | 170.7 | 0.923 [0.923 / 0.903 / 0.925] | 0.911 [0.894 / 0.925 / 0.911] | 1.094 [1.134 / 1.094 / 1.089] |
| 512 | 4 | FLASH_ATTN | 116.8 | 413.5 | 13.09 | 287.8 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 4 | SCRATCH_DECODE | 112.8 | 361.0 | 10.81 | 332.9 | 0.958 [0.954 / 0.997 / 0.958] | 0.841 [0.841 / 0.863 / 0.786] | 1.157 [1.157 / 1.112 / 1.260] |
| 512 | 4 | SCRATCH_FULL | 114.8 | 362.5 | 11.01 | 327.4 | 0.983 [0.986 / 0.967 / 0.983] | 0.857 [0.857 / 0.781 / 0.870] | 1.138 [1.138 / 1.200 / 1.134] |
| 512 | 8 | FLASH_ATTN | 125.9 | 471.8 | 13.46 | 528.5 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 8 | SCRATCH_DECODE | 174.9 | 485.6 | 11.87 | 598.4 | 1.390 [1.253 / 1.390 / 1.524] | 0.899 [0.924 / 0.899 / 0.882] | 1.132 [1.033 / 1.140 / 1.132] |
| 512 | 8 | SCRATCH_FULL | 184.7 | 484.0 | 11.75 | 612.1 | 1.443 [1.342 / 1.467 / 1.443] | 0.903 [0.903 / 0.844 / 0.925] | 1.086 [1.082 / 1.208 / 1.086] |
| 512 | 16 | FLASH_ATTN | 202.0 | 780.8 | 14.66 | 952.8 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 16 | SCRATCH_DECODE | 279.3 | 783.4 | 13.20 | 1026.2 | 0.984 [0.899 / 0.984 / 1.557] | 0.888 [0.977 / 0.888 / 0.820] | 1.098 [1.041 / 1.098 / 1.171] |
| 512 | 16 | SCRATCH_FULL | 242.7 | 656.0 | 12.51 | 1102.0 | 1.201 [1.201 / 0.863 / 1.298] | 0.853 [0.905 / 0.796 / 0.853] | 1.157 [1.106 / 1.220 / 1.157] |
| 512 | 32 | FLASH_ATTN | 327.7 | 1133.2 | 17.43 | 1599.3 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 32 | SCRATCH_DECODE | 330.3 | 1154.1 | 15.89 | 1717.3 | 1.008 [1.008 / 0.882 / 1.074] | 0.891 [0.916 / 0.891 / 0.877] | 1.082 [1.067 / 1.122 / 1.082] |
| 512 | 32 | SCRATCH_FULL | 307.5 | 1171.5 | 15.54 | 1779.2 | 0.939 [0.915 / 0.939 / 1.000] | 0.893 [0.893 / 0.851 / 0.917] | 1.116 [1.116 / 1.162 / 1.080] |

#### long_context
| input | C | config | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | out tok/s | TTFT ratio | TPOT ratio | tok/s ratio |
|---:|---:|---|---:|---:|---:|---:|---|---|---|
| 2048 | 1 | FLASH_ATTN | 63.1 | 178.8 | 14.41 | 67.7 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 2048 | 1 | SCRATCH_DECODE | 62.6 | 169.4 | 13.12 | 73.4 | 0.998 [0.998 / 1.004 / 0.984] | 0.904 [0.911 / 0.883 / 0.904] | 1.098 [1.085 / 1.098 / 1.118] |
| 2048 | 1 | SCRATCH_FULL | 60.8 | 192.8 | 13.16 | 75.3 | 0.955 [1.015 / 0.953 / 0.955] | 0.918 [1.031 / 0.918 / 0.860] | 1.116 [0.998 / 1.116 / 1.150] |
| 4096 | 1 | FLASH_ATTN | 129.0 | 222.1 | 14.83 | 63.4 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 4096 | 1 | SCRATCH_DECODE | 128.1 | 249.3 | 12.97 | 70.2 | 0.994 [1.005 / 0.994 / 0.973] | 0.875 [0.870 / 0.875 / 0.883] | 1.107 [1.114 / 1.107 / 1.104] |
| 4096 | 1 | SCRATCH_FULL | 122.3 | 230.3 | 13.11 | 71.1 | 0.947 [0.949 / 0.947 / 0.941] | 0.876 [0.908 / 0.859 / 0.876] | 1.121 [1.067 / 1.163 / 1.121] |
| 8192 | 1 | FLASH_ATTN | 286.0 | 404.4 | 14.29 | 62.0 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 8192 | 1 | SCRATCH_DECODE | 285.8 | 394.3 | 12.85 | 66.0 | 0.998 [1.026 / 0.998 / 0.998] | 0.900 [0.900 / 0.899 / 0.910] | 1.071 [1.071 / 1.051 / 1.099] |
| 8192 | 1 | SCRATCH_FULL | 274.3 | 419.1 | 13.05 | 65.2 | 0.961 [0.967 / 0.958 / 0.961] | 0.914 [0.949 / 0.899 / 0.914] | 1.078 [1.032 / 1.087 / 1.078] |
| 16384 | 1 | FLASH_ATTN | 722.8 | 831.6 | 14.70 | 49.6 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 16384 | 1 | SCRATCH_DECODE | 724.0 | 820.6 | 12.94 | 53.4 | 1.001 [1.021 / 0.999 / 1.001] | 0.890 [0.859 / 0.896 / 0.890] | 1.083 [1.088 / 1.083 / 1.066] |
| 16384 | 1 | SCRATCH_FULL | 689.7 | 818.3 | 13.22 | 53.2 | 0.953 [0.955 / 0.951 / 0.953] | 0.899 [0.880 / 0.938 / 0.899] | 1.069 [1.111 / 1.069 / 1.066] |
| 32640 | 1 | FLASH_ATTN | 2063.1 | 2193.0 | 14.37 | 32.8 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 32640 | 1 | SCRATCH_DECODE | 2062.7 | 2179.4 | 12.65 | 34.8 | 1.000 [1.003 / 0.998 / 1.000] | 0.885 [0.844 / 0.885 / 0.938] | 1.061 [1.075 / 1.061 / 1.030] |
| 32640 | 1 | SCRATCH_FULL | 1958.0 | 2081.2 | 13.10 | 35.3 | 0.949 [0.949 / 0.950 / 0.947] | 0.899 [0.899 / 0.889 / 0.987] | 1.084 [1.084 / 1.087 / 1.047] |

### CUDA graphs

#### latency
| input | C | config | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | out tok/s | TTFT ratio | TPOT ratio | tok/s ratio |
|---:|---:|---|---:|---:|---:|---:|---|---|---|
| 128 | 1 | FLASH_ATTN | 16.9 | 98.6 | 8.25 | 119.6 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 128 | 1 | SCRATCH_DECODE | 20.6 | 75.8 | 8.31 | 118.7 | 1.216 [1.216 / 1.228 / 1.020] | 1.006 [1.007 / 1.006 / 0.982] | 0.992 [0.991 / 0.992 / 1.016] |
| 128 | 1 | SCRATCH_FULL | 20.0 | 77.1 | 8.30 | 119.0 | 1.188 [1.213 / 1.188 / 1.008] | 1.007 [1.007 / 1.005 / 1.007] | 0.994 [0.995 / 0.991 / 0.994] |
| 512 | 1 | FLASH_ATTN | 33.8 | 140.6 | 8.26 | 117.5 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 1 | SCRATCH_DECODE | 34.3 | 148.9 | 8.36 | 116.4 | 1.007 [0.994 / 1.015 / 1.007] | 1.011 [1.013 / 1.011 / 0.982] | 0.990 [0.990 / 0.990 / 1.016] |
| 512 | 1 | SCRATCH_FULL | 34.2 | 134.1 | 8.35 | 116.5 | 0.999 [0.988 / 1.022 / 0.999] | 1.012 [1.012 / 1.010 / 1.012] | 0.991 [0.991 / 0.992 / 0.987] |
| 1024 | 1 | FLASH_ATTN | 61.6 | 182.3 | 8.28 | 114.3 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 1024 | 1 | SCRATCH_DECODE | 62.6 | 158.0 | 8.43 | 112.8 | 1.009 [1.009 / 1.017 / 1.007] | 1.018 [1.018 / 1.018 / 0.982] | 0.986 [0.986 / 0.986 / 1.015] |
| 1024 | 1 | SCRATCH_FULL | 61.2 | 130.3 | 8.43 | 112.9 | 0.993 [0.984 / 0.993 / 1.000] | 1.018 [1.018 / 1.016 / 1.018] | 0.988 [0.988 / 0.987 / 0.988] |
| 1920 | 1 | FLASH_ATTN | 115.8 | 217.3 | 8.38 | 108.1 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 1920 | 1 | SCRATCH_DECODE | 117.6 | 195.4 | 8.49 | 106.8 | 1.016 [1.017 / 1.016 / 0.999] | 1.013 [1.013 / 1.013 / 0.981] | 0.990 [0.988 / 0.990 / 1.018] |
| 1920 | 1 | SCRATCH_FULL | 114.1 | 181.8 | 8.52 | 106.9 | 0.985 [0.989 / 0.984 / 0.985] | 1.015 [1.017 / 1.015 / 1.013] | 0.989 [0.988 / 0.994 / 0.989] |

#### throughput
| input | C | config | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | out tok/s | TTFT ratio | TPOT ratio | tok/s ratio |
|---:|---:|---|---:|---:|---:|---:|---|---|---|
| 512 | 2 | FLASH_ATTN | 59.2 | 246.5 | 8.54 | 224.5 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 2 | SCRATCH_DECODE | 59.1 | 240.2 | 8.54 | 225.1 | 0.998 [1.010 / 0.857 / 0.998] | 1.001 [1.001 / 1.003 / 0.977] | 1.003 [0.997 / 1.003 / 1.020] |
| 512 | 2 | SCRATCH_FULL | 58.2 | 235.4 | 8.53 | 224.3 | 0.984 [0.984 / 0.998 / 0.981] | 0.999 [0.999 / 1.003 / 0.977] | 0.999 [0.999 / 0.995 / 1.018] |
| 512 | 4 | FLASH_ATTN | 113.2 | 343.6 | 8.62 | 419.0 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 4 | SCRATCH_DECODE | 114.2 | 461.0 | 8.63 | 419.2 | 1.015 [0.980 / 1.015 / 1.029] | 1.002 [1.005 / 1.002 / 0.973] | 1.000 [1.000 / 0.991 / 1.020] |
| 512 | 4 | SCRATCH_FULL | 113.4 | 349.0 | 8.64 | 423.4 | 1.004 [0.999 / 1.012 / 1.004] | 1.003 [1.003 / 1.004 / 0.972] | 1.010 [1.010 / 0.999 / 1.028] |
| 512 | 8 | FLASH_ATTN | 146.0 | 467.4 | 9.46 | 744.5 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 8 | SCRATCH_DECODE | 172.8 | 492.2 | 9.52 | 735.8 | 1.117 [1.117 / 1.395 / 1.005] | 0.991 [1.011 / 0.991 / 0.986] | 0.989 [0.989 / 0.978 / 1.010] |
| 512 | 8 | SCRATCH_FULL | 144.4 | 478.3 | 9.53 | 756.0 | 1.130 [0.801 / 1.160 / 1.130] | 0.992 [1.017 / 0.992 / 0.983] | 1.013 [0.988 / 1.013 / 1.018] |
| 512 | 16 | FLASH_ATTN | 226.5 | 688.7 | 10.89 | 1250.7 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 16 | SCRATCH_DECODE | 183.1 | 763.6 | 11.24 | 1215.8 | 1.014 [1.014 / 1.041 / 0.794] | 1.032 [1.032 / 1.034 / 1.032] | 0.972 [0.965 / 0.972 / 1.008] |
| 512 | 16 | SCRATCH_FULL | 206.1 | 656.9 | 11.02 | 1261.5 | 0.908 [1.229 / 0.908 / 0.755] | 1.016 [0.985 / 1.016 / 1.019] | 1.009 [0.989 / 1.009 / 1.023] |
| 512 | 32 | FLASH_ATTN | 321.4 | 1218.9 | 14.45 | 1855.6 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 512 | 32 | SCRATCH_DECODE | 310.4 | 1133.0 | 15.13 | 1803.6 | 0.966 [0.966 / 0.946 / 0.980] | 1.056 [1.056 / 1.059 / 0.997] | 0.970 [0.966 / 0.970 / 1.000] |
| 512 | 32 | SCRATCH_FULL | 292.0 | 1152.2 | 14.49 | 1875.3 | 0.928 [0.906 / 0.954 / 0.928] | 1.011 [1.011 / 1.014 / 0.969] | 1.004 [1.004 / 1.000 / 1.027] |

#### long_context
| input | C | config | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | out tok/s | TTFT ratio | TPOT ratio | tok/s ratio |
|---:|---:|---|---:|---:|---:|---:|---|---|---|
| 2048 | 1 | FLASH_ATTN | 61.8 | 179.3 | 4.62 | 196.0 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 2048 | 1 | SCRATCH_DECODE | 61.7 | 139.8 | 4.61 | 197.5 | 0.999 [0.999 / 1.015 / 0.973] | 0.998 [0.998 / 0.998 / 0.962] | 1.007 [1.006 / 1.007 / 1.049] |
| 2048 | 1 | SCRATCH_FULL | 60.4 | 145.9 | 4.60 | 198.0 | 0.978 [0.978 / 0.983 / 0.959] | 0.997 [0.997 / 0.997 / 0.962] | 1.010 [1.010 / 1.009 / 1.050] |
| 4096 | 1 | FLASH_ATTN | 125.2 | 242.5 | 4.70 | 176.1 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 4096 | 1 | SCRATCH_DECODE | 126.6 | 135.3 | 4.66 | 178.8 | 1.011 [1.011 / 1.014 / 0.995] | 0.990 [0.990 / 0.991 / 0.959] | 1.016 [1.015 / 1.016 / 1.033] |
| 4096 | 1 | SCRATCH_FULL | 121.8 | 201.5 | 4.66 | 180.0 | 0.972 [0.973 / 0.972 / 0.959] | 0.990 [0.990 / 0.991 / 0.959] | 1.022 [1.022 / 1.019 / 1.040] |
| 8192 | 1 | FLASH_ATTN | 283.0 | 361.4 | 4.87 | 141.4 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 8192 | 1 | SCRATCH_DECODE | 286.9 | 351.8 | 4.83 | 142.5 | 1.014 [1.014 / 1.014 / 0.992] | 0.991 [0.993 / 0.991 / 0.961] | 1.006 [1.006 / 1.004 / 1.032] |
| 8192 | 1 | SCRATCH_FULL | 275.7 | 359.0 | 4.83 | 144.3 | 0.972 [0.974 / 0.972 / 0.952] | 0.992 [0.993 / 0.992 / 0.962] | 1.020 [1.015 / 1.020 / 1.045] |
| 16384 | 1 | FLASH_ATTN | 718.8 | 843.5 | 5.30 | 91.7 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 16384 | 1 | SCRATCH_DECODE | 726.0 | 833.7 | 5.20 | 92.1 | 1.010 [1.010 / 1.011 / 0.993] | 0.981 [0.982 / 0.981 / 0.951] | 1.004 [1.003 / 1.004 / 1.031] |
| 16384 | 1 | SCRATCH_FULL | 692.9 | 805.3 | 5.19 | 94.6 | 0.961 [0.965 / 0.961 / 0.946] | 0.981 [0.981 / 0.983 / 0.951] | 1.032 [1.028 / 1.032 / 1.056] |
| 32640 | 1 | FLASH_ATTN | 2054.7 | 2166.7 | 5.90 | 45.5 | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] | 1.000 [1.000 / 1.000 / 1.000] |
| 32640 | 1 | SCRATCH_DECODE | 2088.4 | 2166.8 | 5.91 | 45.0 | 1.015 [1.016 / 1.015 / 0.997] | 1.002 [1.009 / 1.002 / 0.973] | 0.989 [0.989 / 0.989 / 1.009] |
| 32640 | 1 | SCRATCH_FULL | 1982.9 | 2055.3 | 5.92 | 46.8 | 0.964 [0.965 / 0.964 / 0.945] | 1.004 [1.005 / 1.004 / 0.975] | 1.028 [1.027 / 1.028 / 1.050] |

## 5. What not to read from these tables

- **128-token TTFT.** 17 to 30 ms end to end, and the per-run ratios scatter from 0.68 to 1.50.
  Noise dominates; no conclusion belongs here.
- **TTFT at 8 and 16 concurrent requests.** The per-run ratios disagree in sign (for example
  0.801 / 1.160 / 1.130 in one row). How the scheduler groups requests into steps varies
  between runs and moves TTFT more than the kernel does. Not investigated further; the tables
  print every run so the spread is visible.
- **Output tok/s** tracks TPOT and is not independent evidence.
- One GPU, one architecture (sm_89), two small models on an 8 GB consumer card. Nothing here
  generalizes to other hardware, and no claim is made that it does.

## 6. Summary

- The prefill kernel is wired into vLLM 0.19 well enough that every prefill token in 252
  measured cases went through it, with no silent fallback, in both execution modes, including
  chunked prefill, prefix-cache hits and batches that mix prefill with decode.
- Against the same engine with the same decode kernel it takes **2 to 5% off TTFT, more at
  longer prompts**, consistently across runs and unchanged by CUDA graphs.
- The decode kernel's 7 to 16% TPOT advantage appears only in eager mode and disappears under
  CUDA graphs, where it is even to 5.6% behind at high concurrency.
