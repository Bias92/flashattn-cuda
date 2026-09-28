# flashattn-cuda

Flash-Attention and Flash-Attention-2 inference kernels written with CUDA/PTX,
especially for **RTX 4060 Ti**.

Evaluation covers both attention-kernel latency against **SDPA-Flash/cuDNN**
and model serving under **low latency, high throughput and long context**.

## Implementation

The FA2 kernel uses FP16 operands and FP32 accumulation.

- Tensor Core `mma.sync` for both QK and PV.
- Register-resident online softmax; no full attention matrix in global memory.
- Four warps per block and double-buffered K/V tiles using `cp.async`.

The standalone forward kernel supports dense/causal attention, GQA and
unequal query/KV lengths. [Read the kernel](cuda/flash_fwd_kernel.h)
| [Source map](cuda/README.md).

## Three Workloads

The serving study used **TinyLlama-1.1B** and **Qwen2.5-0.5B** in vLLM 0.19.0.
It compared native FlashAttention, scratch decode-only, and scratch
prefill+decode with matching model weights and engine settings.
Three runs across 14 settings and two execution modes produced **252 measurements**.

**Results below compare scratch prefill+decode against native FlashAttention,
with CUDA Graphs enabled.** All requests generate 128 output tokens.

### 1. Low Latency

**Test:** TinyLlama, one request at a time, 128-1920 input tokens.

**Metrics:** TTFT, TPOT, inter-token and end-to-end latency, including p50/p95/p99.

**Result:** At 512-1920 tokens, median TTFT was **0.1-1.5% lower**, while TPOT
was **1.2-1.8% higher**. The 128-token TTFT varied substantially between runs.

### 2. High Throughput

**Test:** TinyLlama, 512 input tokens, 2/4/8/16/32 concurrent requests.

**Metrics:** Total output tokens/s, TTFT, TPOT and request-latency distributions.

**Result:** Throughput was **roughly unchanged (-0.1% to +1.3%)**. Median TPOT
changed by -0.8% to +1.6%; higher-concurrency TTFT varied between runs.

### 3. Long Context

**Test:** Qwen2.5-0.5B, one request, 2048-32640 input tokens, 2048-token chunks.

**Metrics:** TTFT, TPOT, end-to-end latency, request failures and KV-cache capacity.

**Result:** Median TTFT was **2.2-3.9% lower** across the five tested lengths.
TPOT changed by -1.9% to +0.4%. KV-cache capacity comes from engine logs,
not a per-request peak-memory measurement.

TTFT is time to the first token; TPOT is average time per subsequent output
token. p50 is the median, while p95/p99 describe the slow tail.

**Execution mode matters:** eager throughput improved by 8.6-15.7%, but that
advantage largely disappeared with CUDA Graphs.
[Detailed comparison](docs/evaluation.md#serving-evaluation)
| [Full serving report and raw runs](docs/serving/prefill_campaign_2026-09-20/REPORT.md).

## Kernel Results

**34 D64 cases:** batch sizes 1-8, sequence lengths 512-8192, dense/causal
attention and multiple MHA/GQA head configurations. Each case uses three runs
of paired, warmed O-only API timings. Flash and cuDNN are forced SDPA backends.

Counts are **faster / within 1% / slower**, based on paired latency ratios:

| Mask | vs SDPA-Flash | vs SDPA-cuDNN |
|---|---:|---:|
| Dense (17 cases) | 10 / 4 / 3 | 2 / 4 / 11 |
| Causal (17 cases) | 15 / 0 / 2 | 16 / 0 / 1 |

Causal attention is the strongest path; cuDNN wins most dense cases.
[Per-shape timings and methodology](docs/serving/backend_overview_2026-09-22/README.md).

## Code And Records

- [Kernel source map](cuda/README.md): forward loop, memory operations, softmax and PTX.
- [Build, API and tests](docs/usage.md): instructions for the measured `sm_89` environment.
- [Evaluation details](docs/evaluation.md): conditions, timing tables and original data.
- [Optimization history](experiments/): earlier kernels, including the preserved FP32 baseline.
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691): algorithm reference.

The active build contains standalone forward attention. Serving results are
from September 20-21; kernel results are from September 22, before the source
cleanup. The earlier vLLM integration and decode kernels are retained in
[Git history](https://github.com/Bias92/flashattn-cuda/tree/d4d4f37f6d4129f4a42f521fe93282e8ae7b7beb),
with their measurements unchanged. Other GPUs have not been evaluated.
