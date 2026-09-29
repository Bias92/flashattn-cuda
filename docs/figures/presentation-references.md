# Performance Presentation References

Original kernel articles and official projects reviewed on September 28, 2026.
These are format references, not additional benchmark baselines. Public page-view
counts were not available; this is not a ranking by readership.

| Source | Performance presentation |
|---|---|
| [FlashAttention-2, Tri Dao / Stanford CRFM](https://crfm.stanford.edu/2023/07/17/flash2.html#attention-benchmark) | Sequence length vs TFLOP/s; grouped bars by implementation; separate mask/head-dimension panels |
| [FlashAttention-3, Tri Dao](https://tridao.me/blog/2024/flash3/#attention-benchmark) | Sequence length vs TFLOP/s; grouped bars including FlashAttention-2, Triton and cuDNN |
| [Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | Fixed batch/head counts; sequence-length sweep; TFLOP/s curves and numerical tables |
| [CUDA Matmul Worklog, Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM) | Optimization-stage GFLOP/s table and relative cuBLAS performance; matrix-size vs GFLOP/s curves |
| [ThunderKittens, Hazy Research](https://hazyresearch.stanford.edu/blog/2024-05-12-tk) | Sequence length vs attention TFLOP/s; grouped bars; separate forward/backward, causal and head-dimension panels |
| [Block-Sparse-Attention, MIT Han Lab](https://github.com/mit-han-lab/Block-Sparse-Attention#performance) | Speedup relative to dense FlashAttention-2 with workload and hardware specified |

## README Format

The serving figures were revised on September 29 after inspecting 100 primary
reports. No chart geometry had a majority across that sample (46 bars, 43 lines,
9 mixed, 2 scatter); 81 selected figures used absolute performance axes.
The layouts below are specific examples, not a claim of a universal standard.

| This repository | Source figure | Adopted structure |
|---|---|---|
| Kernel throughput | [FlashAttention-2, Fig. 5, PDF p11](https://arxiv.org/pdf/2307.08691#page=11) | Sequence length on x; TFLOP/s on y; grouped bars for implementations; separate mask panels |
| Low-latency and long-context TTFT/TPOT | [FlashInfer, Fig. 7, PDF p8](https://arxiv.org/pdf/2501.01005#page=8) | Absolute latency, grouped implementation bars, separate TTFT and per-token-latency panels |
| Concurrent-request throughput | [Hydragen, Fig. 3(a), PDF p9](https://arxiv.org/pdf/2402.05099#page=9) | Workload size on x; generated tokens/s on y; grouped implementation bars |
| Throughput panel layout | [NanoFlow, Fig. 7, PDF p11](https://arxiv.org/pdf/2408.12757#page=11) | Shared legend, grouped throughput bars, panels for different experimental conditions |

FlashInfer reports ITL; this repository's plotted metric remains TPOT. Hydragen
sweeps batch size; this study sweeps concurrent requests. NanoFlow normalizes
by GPU count; this study plots total output tokens/s on one GPU. Only the
presentation structure is adopted, not those studies' metric definitions or
experimental conditions. Panels are stacked for readability in a narrow README.

The kernel overview uses grouped TFLOP/s bars for the four recorded sequence
lengths at B=1, H_q=H_kv=8, D=64. Dense and causal panels share a zero-based
vertical scale. The accompanying table gives absolute execution times in ms.
The complete 34-case comparison remains linked, including GQA and other batches.

Only the presentation pattern is reused. All displayed measurements come from
this repository's RTX 4060 Ti results; no source figures or numbers are copied.

## Focused Serving Follow-up

On September 29, seven additional official articles were read and twelve chart
images were visually inspected. This targeted follow-up is separate from the
100-report sample above; it is not another prevalence estimate. A multi-panel
image counts once, and the numerical tables are not counted as chart images.

| Official article | Inspected figures | Observed presentation |
|---|---|---|
| [vLLM v0.6.0 performance update](https://vllm.ai/blog/2024-09-05-perf-update) | [A100 8B results](https://vllm.ai/blog-assets/figures/perf-v060/A100_8B.png) | TTFT and TPOT in ms vs QPS as lines with error bars; throughput in requests/s as engine bars; separate workload panels |
| [SGLang Llama 3 serving](https://www.lmsys.org/blog/2024-07-25-sglang-llama3/) | [8B throughput](https://www.lmsys.org/images/blog/sglang_llama3/8b_throughput.svg), [8B latency](https://www.lmsys.org/images/blog/sglang_llama3/8b_latency.svg) | Grouped output-tokens/s bars by workload; median end-to-end latency in seconds vs requests/s as lines |
| [vLLM ROCm attention backends](https://vllm-project.github.io/2026/02/27/rocm-attention-backend.html) | [MHA TTFT](https://vllm-project.github.io/assets/figures/2026-02-27-rocm-attention-backend/mha_ttft_comparison.png), [MHA TPOT](https://vllm-project.github.io/assets/figures/2026-02-27-rocm-attention-backend/mha_tpot_comparison.png), [MHA throughput](https://vllm-project.github.io/assets/figures/2026-02-27-rocm-attention-backend/mha_tps_comparison.png) | Absolute horizontal bars by attention backend; hardware and concurrency panels; ratio annotations alongside absolute axes |
| [NVIDIA TensorRT-LLM benchmarking](https://developer.nvidia.com/blog/?p=102816) | [Figure 1](https://developer-blogs.nvidia.com/wp-content/uploads/2025/06/LLM-Inference-perf-fig-1-png.webp) | Per-GPU tokens/s vs per-user tokens/s trade-off curves; concurrency labels on measured points |
| [SGLang gpt-oss optimization](https://www.lmsys.org/blog/2025-08-27-gpt-oss/) | [Prefill](https://www.lmsys.org/images/blog/gpt_oss/combined_prefill_performance.svg), [decode](https://www.lmsys.org/images/blog/gpt_oss/combined_decode_performance.svg) | Grouped tokens/s bars for before/after implementations by hardware and input length; separate prefill and decode figures |
| [NVIDIA confidential-computing inference](https://developer.nvidia.com/blog/?p=122756) | [Throughput](https://developer-blogs.nvidia.com/wp-content/uploads/2026/09/cc-on-off-output-token-throughput.webp), [TPOT](https://developer-blogs.nvidia.com/wp-content/uploads/2026/09/mean-tpot-cc-on-off.webp) | Paired horizontal bars normalized to a 100% baseline, with absolute tokens/s or ms printed on each bar; concurrency 1-16 |
| [vLLM GLM-5.2 serving](https://vllm.ai/blog/2026-07-23-glm-5.2-nvfp4-b300-pd) | [Results overview](https://vllm.ai/blog-assets/figures/2026-07-23-glm-5.2-nvfp4-b300-pd/02-results-overview.png) | Input-length panels for total throughput, output throughput, TTFT and TPOT; bars, lines and points with explicitly identified units and scales |

### Application to This README

- Keep absolute grouped bars for the three backends at each measured input
  length or concurrency. They separate implementations whose lines would overlap.
- Keep TTFT and TPOT in separate panels. Keep eager and CUDA Graph throughput
  separate, with the same zero-based vertical scale.
- Keep the native baseline visible. Small differences can be read precisely in
  the linked measurement tables; the overview does not need an enlarged change
  axis to distinguish every percentage point.
- Retain kernel TFLOP/s separately from model-serving ms and output tokens/s.

Both lines and bars are established choices. Normalized bars are also used in
official reports; they are not inherently invalid. The choice here follows the
README's fixed-condition comparisons and the need to distinguish three close
series, rather than a claim that one geometry is an industry requirement.

Metric definitions stay local to each experiment. Requests/s, concurrency and
batch size are different axes. Mean latency, median latency and per-token
latency percentiles are not interchangeable. Nor are total tokens/s, output
tokens/s and tokens/s/GPU. Our figures retain the recorded p50 metrics and the
median across three runs; the external articles supply presentation examples,
not replacement definitions or performance evidence for this repository.
