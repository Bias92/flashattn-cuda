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

**RTX 4060 Ti 8 GB, PyTorch 2.10.0+cu128.** Batch size 1, eight Q/KV heads,
head dimension 64. Flash and cuDNN are separately forced SDPA backends.
FP16 inputs, FP32 accumulation; warmed O-only API time, without CUDA Graphs.

Effective throughput over sequence length; **higher is better**. Each bar uses
the median of three per-run median latencies and the QK/PV FLOP count.

![Attention forward throughput: Custom, SDPA-Flash and SDPA-cuDNN over sequence length, with separate Dense and Causal panels](docs/figures/kernel-throughput.png)

**Execution time (ms), lower is better:**

| Mask | Sequence length | Custom | SDPA-Flash | SDPA-cuDNN |
|---|---:|---:|---:|---:|
| Dense | 1024 | 0.06227 | 0.05998 | 0.05782 |
| Dense | 2048 | 0.22313 | 0.22501 | 0.21361 |
| Dense | 4096 | 0.86249 | 0.87860 | 0.85070 |
| Dense | 8192 | 3.37455 | 3.45639 | 3.41856 |
| Causal | 1024 | 0.05296 | 0.05270 | 0.05217 |
| Causal | 2048 | 0.14025 | 0.15490 | 0.14524 |
| Causal | 4096 | 0.48298 | 0.51368 | 0.51085 |
| Causal | 8192 | 1.81162 | 1.82800 | 1.86083 |

The full benchmark covers **34 cases**, including batch sizes 1-8, MHA/GQA
and sequence lengths 512-8192. Custom throughput spans **34.48-41.74 TFLOP/s
for dense** and **20.28-39.76 TFLOP/s for causal** attention.

[All 34 cases and FLOP accounting](docs/figures/kernel-throughput.md)
| [Per-case comparisons](docs/figures/kernel-comparison.md)
| [Measurement protocol](docs/serving/backend_overview_2026-09-22/README.md)

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

Changes are relative to **Native Flash = 0%**, using the median of three
per-run ratios. These are whole-engine results from the archived September
20-21 serving implementation, not the current standalone build. Kernel results above are
from September 22. [Measured revisions and raw records](docs/evaluation.md)

### Low Latency

**TinyLlama-1.1B, one request at a time, 128-1920 input tokens.** Measured TTFT,
TPOT, inter-token and end-to-end latency, including p50/p95/p99.
The plot uses CUDA Graphs; lower latency is better.

![TTFT and TPOT percentage changes for both custom configurations versus Native Flash, with CUDA Graphs](docs/figures/low-latency.png?v=aa05629)

With CUDA Graphs, custom prefill + decode had **0.1-1.5% lower TTFT** at
512-1920 tokens, and **1.2-1.8% higher TPOT**, versus native Flash.
The 128-token TTFT varied substantially between runs.

### High Throughput

**TinyLlama-1.1B, 512 input tokens, 2-32 concurrent requests.** Measured total
output tokens/s, TTFT, TPOT and request-latency distributions.
Higher output throughput is better.

![Output throughput percentage changes versus Native Flash under eager execution and CUDA Graphs](docs/figures/high-throughput.png?v=aa05629)

Custom prefill + decode improved throughput by **8.6-15.7% in eager mode**.
With CUDA Graphs the difference was **-0.1% to +1.3%**, with TPOT changing by
-0.8% to +1.6%. Higher-concurrency TTFT varied between runs.

### Long Context

**Qwen2.5-0.5B, one request, 2048-32640 input tokens.** Measured TTFT, TPOT,
end-to-end latency, request failures and KV-cache capacity from engine logs.
The plot uses CUDA Graphs; lower latency is better.

![Long-context TTFT and TPOT percentage changes versus Native Flash, with CUDA Graphs](docs/figures/long-context.png?v=aa05629)

With CUDA Graphs, custom prefill + decode had **2.2-3.9% lower TTFT** across
all five lengths; TPOT changed by -1.9% to +0.4%. KV-cache records describe
engine capacity, not per-request peak memory.

TTFT = time to first token; TPOT = average time per subsequent output token.
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
