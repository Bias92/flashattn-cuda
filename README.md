# flashattn-cuda

A FlashAttention-2 forward kernel written from scratch in CUDA without CUTLASS, for the RTX 4060 Ti.
It takes fp16 inputs, accumulates in fp32, supports head dim 64, and runs dense or causal.

Query and key lengths are equal (N_q = N_kv). There is no KV cache and no single-query decode path,
so this is the training and prefill shape, not decode.

## Latency vs PyTorch SDPA (FlashAttention-2 backend)

B=1, H=8, D=64, fp16. Each benchmark run is 10 paired reps with alternating order;
the tables are the median of three such runs. Positive gap = custom slower.

Dense:

| N | Custom ms | Custom TFLOPS | SDPA ms | SDPA TFLOPS | gap |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.0636 | 33.8 | 0.0607 | 35.4 | +4.12% |
| 2048 | 0.2169 | 39.6 | 0.2166 | 39.7 | +0.74% |
| 4096 | 0.8499 | 40.4 | 0.8415 | 40.8 | +0.78% |

Causal:

| N | Custom ms | Custom TFLOPS | SDPA ms | SDPA TFLOPS | gap |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.0552 | 19.5 | 0.0537 | 20.0 | +2.56% |
| 2048 | 0.1460 | 29.4 | 0.1516 | 28.3 | -3.80% |
| 4096 | 0.4858 | 35.4 | 0.5008 | 34.3 | -2.99% |

At N=1024 the per-rep gap swings by more than 10 points in both tables, so those rows carry
launch and dispatch noise rather than kernel time. N=2048 and N=4096 are stable across runs.

The causal rows are the one case where this kernel is ahead, and tile size is the likely reason.
PyTorch runs `Flash_fwd_kernel_traits<64, 128, 128, 4>` for both dense and causal, so its 128-row
query block carries a large fully masked triangle; a 64-row block with 32-column K/V steps throws
away less of the diagonal block. The same small tile is what costs it on the dense rows.

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
Causal blocks stop at the diagonal and mask only the block that crosses it.
Not implemented: dropout, varlen, GQA, backward.

## Files

`bench/compare_pytorch.py`, `bench/compare_memory.py`, `tests/test_attention_forward.py`.
Run record: `bench/results/`. Earlier kernels: `experiments/`. Profiles: `docs/profiling/`.
