# flashattn-cuda

A FlashAttention-2 forward kernel written from scratch in CUDA without CUTLASS, for the RTX 4060 Ti.
It takes fp16 inputs, accumulates in fp32, supports head dim 64 or 128, dense or causal masking,
and grouped-query attention.

Query and key lengths are equal (N_q = N_kv). There is no KV cache and no single-query decode path,
so this is the training and prefill shape, not decode.

## Latency vs PyTorch SDPA

SDPA is an API with several backends and they do not perform alike, so both are measured:
`FLASH_ATTENTION` (the FlashAttention-2 kernels PyTorch vendors, pinned at upstream v2.7.4, and
what plain `scaled_dot_product_attention` dispatches to on this GPU) and `CUDNN_ATTENTION`.

fp16, D=64. Each run is 10 paired reps with the implementations in rotating order; tables are the
median of three runs. Positive gap = custom slower.

B=1, H=8:

| mask | N | Custom ms | flash ms | cuDNN ms | Custom TF | flash TF | cuDNN TF | vs flash | vs cuDNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 0.2170 | 0.2173 | 0.2083 | 39.6 | 39.5 | 41.2 | +0.37% | +4.49% |
| dense | 4096 | 0.8520 | 0.8455 | 0.8141 | 40.3 | 40.6 | 42.2 | +0.80% | +4.39% |
| causal | 2048 | 0.1464 | 0.1528 | 0.1418 | 29.3 | 28.1 | 30.3 | -4.13% | +2.98% |
| causal | 4096 | 0.4905 | 0.5056 | 0.4999 | 35.0 | 34.0 | 34.4 | -2.80% | -2.23% |

B=1, H=32, H_kv=8 (grouped-query, Llama-like):

| mask | N | Custom ms | flash ms | cuDNN ms | Custom TF | flash TF | cuDNN TF | vs flash | vs cuDNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 0.8211 | 0.8489 | 0.8112 | 41.8 | 40.5 | 42.4 | -3.51% | +0.76% |
| dense | 4096 | 3.2757 | 3.2315 | 3.1269 | 42.0 | 42.5 | 43.9 | +1.50% | +4.85% |
| causal | 2048 | 0.4505 | 0.4751 | 0.4684 | 38.1 | 36.2 | 36.7 | -5.12% | -3.72% |
| causal | 4096 | 1.7894 | 1.7424 | 1.7339 | 38.4 | 39.4 | 39.6 | +2.60% | +3.34% |

Where this kernel wins is a tile-size trade-off, not a general one. PyTorch runs
`Flash_fwd_kernel_traits<64, 128, 128, 4>` for every one of these cases, so its 128-row query
block carries a large fully masked triangle under a causal mask, while a 64-row block with
32-column K/V steps throws away less of the diagonal. That is worth a few percent on the causal
rows. The same small tile gets less data reuse, which costs on the dense rows and grows into a
loss at N=4096 with 32 heads, where there is already enough work to keep the machine busy.
cuDNN is ahead of the FlashAttention-2 backend nearly everywhere here.

N=1024 is left out because its per-rep gap swings by more than 10 points;
`bench/compare_pytorch.py` still prints it.

FLOPs counted as 4·N²·D·H, halved for causal. For fp16 multiply with fp32 accumulate this GPU peaks
at 34 SM × 512 FLOP/clk × 3.12 GHz = 54.3 TFLOPS, and the sustained clock under load is lower.

## Peak memory

MiB above the Q/K/V inputs. HF eager = Transformers `eager_attention_forward`, unmodified.

| N | Custom CUDA (O+L) | PyTorch SDPA (FA2) | HF eager |
|---:|---:|---:|---:|
| 1024 | 1.0 | 1.0 | 64.0 |
| 2048 | 2.1 | 2.1 | 256.0 |
| 4096 | 4.1 | 4.1 | 1024.0 |
| 8192 | 8.2 | 8.3 | 4096.0 |

![](docs/profiling/memory.png)

## Kernel

`cuda/attention_forward.cu`. Four warps per block, each owns 16 query rows.
QK and PV on `mma.sync`, softmax and O kept in registers, K/V double-buffered with `cp.async`.
Causal blocks stop at the diagonal and mask only the block that crosses it. Under grouped-query
attention several query heads read one key/value head; with equal head counts that path compiles out.
Shared memory is dynamic, so tiles above the 48 KB static limit are possible.
`ATTN_BR`, `ATTN_BC` and `ATTN_DOUBLE_BUFFER` are build-time knobs; the defaults are what the
numbers above were measured with.
Not implemented: dropout, varlen, backward, head dim other than 64 or 128.

At head dim 128 the kernel is correct but slower than both SDPA backends, by 1.5% on dense at
N=2048 and about 14% on causal at N=2048. A sweep of BR, BC and single versus double buffering
did not close that; see [the record](bench/results/headdim128_2026-09-14_rtx4060ti.md).

## Files

`bench/compare_pytorch.py`, `bench/compare_memory.py`, `tests/test_attention_forward.py`.
Run record: `bench/results/`. Earlier kernels: `experiments/`. Profiles: `docs/profiling/`.
