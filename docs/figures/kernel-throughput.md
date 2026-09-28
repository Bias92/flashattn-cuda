# Effective Attention Throughput

RTX 4060 Ti 8 GB, FP16 inputs and FP32 accumulation, D=64. These values
are derived from the September 22 warmed O-only API measurements, not a new GPU run.

## FLOP Accounting

For equal query/KV lengths, the QK and PV products together use the
[FlashAttention benchmark convention](https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py):

- Dense: `F = 4 * B * H_q * N^2 * D`.
- Causal: `F = 2 * B * H_q * N^2 * D` (half-dense approximation).
- `effective TFLOP/s = F / (time_ms * 10^9)`; one FMA counts as two FLOPs.

This counts useful matrix-product work, excluding softmax and padding/masked
tile work. It is not a count of all hardware instructions, GPU peak utilization,
or whole-model serving FLOP/s. GQA uses the query-head count H_q.

Each time is the median of three per-run median API latencies. The paired
ratio plot uses a different aggregation and can differ in sign near parity.

## All 34 Cases

| B | H_q/H_kv | N | Mask | GFLOP/call | Custom ms | Custom TFLOP/s | Flash TFLOP/s | cuDNN TFLOP/s |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 8/8 | 1024 | Dense | 2.147 | 0.06227 | 34.48 | 35.80 | 37.14 |
| 1 | 8/8 | 1024 | Causal | 1.074 | 0.05296 | 20.28 | 20.37 | 20.58 |
| 1 | 8/8 | 2048 | Dense | 8.590 | 0.22313 | 38.50 | 38.18 | 40.21 |
| 1 | 8/8 | 2048 | Causal | 4.295 | 0.14025 | 30.62 | 27.73 | 29.57 |
| 1 | 8/8 | 4096 | Dense | 34.360 | 0.86249 | 39.84 | 39.11 | 40.39 |
| 1 | 8/8 | 4096 | Causal | 17.180 | 0.48298 | 35.57 | 33.44 | 33.63 |
| 1 | 8/8 | 8192 | Dense | 137.439 | 3.37455 | 40.73 | 39.76 | 40.20 |
| 1 | 8/8 | 8192 | Causal | 68.719 | 1.81162 | 37.93 | 37.59 | 36.93 |
| 1 | 16/16 | 5000 | Dense | 102.400 | 2.56521 | 39.92 | 39.01 | 39.82 |
| 1 | 16/16 | 5000 | Causal | 51.200 | 1.37299 | 37.29 | 34.60 | 34.79 |
| 1 | 32/8 | 1000 | Dense | 8.192 | 0.22669 | 36.14 | 34.93 | 37.30 |
| 1 | 32/8 | 1000 | Causal | 4.096 | 0.13461 | 30.43 | 28.05 | 28.63 |
| 1 | 32/8 | 2048 | Dense | 34.360 | 0.83484 | 41.16 | 38.52 | 40.31 |
| 1 | 32/8 | 2048 | Causal | 17.180 | 0.46735 | 36.76 | 34.75 | 35.88 |
| 1 | 32/8 | 4096 | Dense | 137.439 | 3.29287 | 41.74 | 40.14 | 41.11 |
| 1 | 32/8 | 4096 | Causal | 68.719 | 1.72846 | 39.76 | 38.50 | 37.23 |
| 1 | 40/8 | 1700 | Dense | 29.594 | 0.75028 | 39.44 | 35.35 | 37.49 |
| 1 | 40/8 | 1700 | Causal | 14.797 | 0.41418 | 35.73 | 30.91 | 31.53 |
| 2 | 16/4 | 3072 | Dense | 77.309 | 1.92892 | 40.08 | 40.31 | 41.72 |
| 2 | 16/4 | 3072 | Causal | 38.655 | 0.99568 | 38.82 | 36.71 | 36.59 |
| 2 | 20/4 | 6144 | Dense | 386.547 | 9.59093 | 40.30 | 42.02 | 43.11 |
| 2 | 20/4 | 6144 | Causal | 193.274 | 4.87668 | 39.63 | 40.26 | 39.18 |
| 2 | 24/8 | 2304 | Dense | 65.230 | 1.64104 | 39.75 | 40.52 | 40.32 |
| 2 | 24/8 | 2304 | Causal | 32.615 | 0.85374 | 38.20 | 35.14 | 36.73 |
| 3 | 14/2 | 2500 | Dense | 67.200 | 1.70956 | 39.31 | 37.49 | 39.14 |
| 3 | 14/2 | 2500 | Causal | 33.600 | 0.90839 | 36.99 | 35.23 | 34.74 |
| 4 | 12/12 | 1536 | Dense | 28.991 | 0.72290 | 40.10 | 40.70 | 40.47 |
| 4 | 12/12 | 1536 | Causal | 14.496 | 0.41489 | 34.94 | 32.76 | 34.04 |
| 5 | 9/3 | 1111 | Dense | 14.219 | 0.37370 | 38.05 | 35.64 | 38.78 |
| 5 | 9/3 | 1111 | Causal | 7.110 | 0.22357 | 31.80 | 29.88 | 29.78 |
| 6 | 10/2 | 768 | Dense | 9.060 | 0.23403 | 38.71 | 38.22 | 39.51 |
| 6 | 10/2 | 768 | Causal | 4.530 | 0.13884 | 32.63 | 29.72 | 29.92 |
| 8 | 8/8 | 512 | Dense | 4.295 | 0.11727 | 36.62 | 37.23 | 37.91 |
| 8 | 8/8 | 512 | Causal | 2.147 | 0.07455 | 28.81 | 26.02 | 26.14 |

[Source measurements](../serving/backend_overview_2026-09-22/README.md)
| [Exact derived values and input hashes](data.json)
