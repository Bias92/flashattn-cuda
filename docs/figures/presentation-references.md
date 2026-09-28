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

The kernel overview uses grouped TFLOP/s bars for the four recorded sequence
lengths at B=1, H_q=H_kv=8, D=64. Dense and causal panels share a zero-based
vertical scale. The accompanying table gives absolute execution times in ms.
The complete 34-case comparison remains linked, including GQA and other batches.

Only the presentation pattern is reused. All displayed measurements come from
this repository's RTX 4060 Ti results; no source figures or numbers are copied.
