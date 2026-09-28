# flashattn-cuda

Flash-Attention and Flash-Attention-2 inference kernels written in **CUDA C++
with inline PTX**, especially for **RTX 4060 Ti**.

The project evaluates attention-kernel latency against **PyTorch SDPA-Flash
and cuDNN**, then model serving under **low latency, high throughput and
long context**.

## Implementation

- Tensor Core `mma.sync` for QK and PV, with FP16 operands and FP32 accumulation.
- Online softmax and output accumulation in registers.
- Four warps per block; double-buffered K/V tiles loaded with `cp.async`.

The standalone FA2 forward supports **dense/causal attention, GQA and unequal
query/KV lengths**. Earlier FA1/FP32 kernels remain in [experiments](experiments/).

[Read the forward loop](cuda/flash_fwd_kernel.h) | [Source map](cuda/README.md)

## Kernel Performance

**RTX 4060 Ti 8 GB, PyTorch 2.10.0+cu128.** The comparison covers 34 D64 cases:
batch sizes 1-8, sequence lengths 512-8192, MHA/GQA and dense/causal masks.
Flash and cuDNN are separately forced SDPA backends. Timings use warmed
O-only APIs, without CUDA Graphs.

![All 34 kernel cases: custom kernel latency change versus SDPA-Flash and SDPA-cuDNN](docs/figures/kernel-performance.png)

Negative values mean the custom kernel is faster. Points show median paired ratios;
whiskers show the min/max across three runs. H denotes query/KV heads.

**Causal attention is the strongest path; cuDNN wins most dense cases.**
Counts below are faster / within 1% / slower, from three runs of paired timings.
The 1% band is a practical comparison threshold, not a significance test.

| Mask | Cases | vs SDPA-Flash | vs SDPA-cuDNN |
|---|---:|---:|---:|
| Dense | 17 | 10 / 4 / 3 | 2 / 4 / 11 |
| Causal | 17 | 15 / 0 / 2 | 16 / 0 / 1 |

[Per-shape timings and measurement protocol](docs/serving/backend_overview_2026-09-22/README.md)

## Serving Performance

The vLLM 0.19.0 study compared three configurations with the same models and
engine settings:

| Configuration | Prefill | Decode |
|---|---|---|
| Native Flash | vLLM FlashAttention | vLLM FlashAttention |
| Custom decode-only | vLLM FlashAttention | Custom paged decode |
| Custom prefill + decode | Custom prefill | Custom paged decode |

**252 measurements:** 14 settings, three configurations, three runs and two
execution modes. FP16, 128 output tokens per request, 2048-token chunked
prefill, prefix caching off; startup and warmup excluded.

Plots show medians across three runs. These are whole-engine results from the
archived September 20-21 serving
implementation, not the current standalone build. Kernel results above are
from September 22. [Measured revisions and raw records](docs/evaluation.md)

### Low Latency

**TinyLlama-1.1B, one request at a time, 128-1920 input tokens.** Measured TTFT,
TPOT, inter-token and end-to-end latency, including p50/p95/p99.
The plot uses CUDA Graphs; lower latency is better.

![Low latency: TTFT and TPOT for all three configurations with CUDA Graphs](docs/figures/low-latency.png)

With CUDA Graphs, custom prefill + decode had **0.1-1.5% lower TTFT** at
512-1920 tokens, and **1.2-1.8% higher TPOT**, versus native Flash.
The 128-token TTFT varied substantially between runs.

### High Throughput

**TinyLlama-1.1B, 512 input tokens, 2-32 concurrent requests.** Measured total
output tokens/s, TTFT, TPOT and request-latency distributions.
Higher output throughput is better.

![High throughput: output tokens per second under eager execution and CUDA Graphs](docs/figures/high-throughput.png)

Custom prefill + decode improved throughput by **8.6-15.7% in eager mode**.
With CUDA Graphs the difference was **-0.1% to +1.3%**, with TPOT changing by
-0.8% to +1.6%. Higher-concurrency TTFT varied between runs.

### Long Context

**Qwen2.5-0.5B, one request, 2048-32640 input tokens.** Measured TTFT, TPOT,
end-to-end latency, request failures and KV-cache capacity from engine logs.
The plot uses CUDA Graphs; lower latency is better.

![Long context: TTFT and TPOT as input length increases, with CUDA Graphs](docs/figures/long-context.png)

With CUDA Graphs, custom prefill + decode had **2.2-3.9% lower TTFT** across
all five lengths; TPOT changed by -1.9% to +0.4%. KV-cache records describe
engine capacity, not per-request peak memory.

TTFT = time to first token; TPOT = average time per subsequent output token.
Percentage changes use median per-run ratios.
[Full serving report](docs/serving/prefill_campaign_2026-09-20/REPORT.md)

## Code And Records

- [Kernel source map](cuda/README.md): forward loop, memory operations, softmax and PTX.
- [Build, API and tests](docs/usage.md): the measured `sm_89` environment.
- [Evaluation details](docs/evaluation.md): conditions, timing tables and raw results.
- [Figure data and generation](docs/figures/README.md): plots from preserved measurements.
- [Optimization history](experiments/): earlier kernels, including the unchanged FP32 baseline.
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691): algorithm reference.

The active build contains standalone forward attention. The vLLM integration
and decode kernels are preserved in [Git history](https://github.com/Bias92/flashattn-cuda/tree/d4d4f37f6d4129f4a42f521fe93282e8ae7b7beb).
Other GPUs have not been evaluated.
