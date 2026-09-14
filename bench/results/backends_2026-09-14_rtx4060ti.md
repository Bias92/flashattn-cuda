# SDPA backends, RTX 4060 Ti, 2026-09-14

The earlier records compared only against `SDPBackend.FLASH_ATTENTION`. SDPA is an API with
several backends and they do not perform alike, so `CUDNN_ATTENTION` was measured too. On this
GPU and shape cuDNN is ahead of the FlashAttention-2 backend everywhere, and ahead of this
kernel in three of the four cells below.

## Which backend does plain SDPA use here

With no backend forced, `F.scaled_dot_product_attention` on fp16 B=1 H=8 N=4096 D=64 launches

    void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, ...>>

for both `is_causal=False` and `is_causal=True`, so the default path is the FlashAttention-2
backend, not cuDNN.

## Which FlashAttention-2 this is

torch 2.10.0 carries flash-attention as a git submodule. At tag v2.10.0 the pin is

    third_party/flash-attention @ 979702c87a8713a8e0a5e9fee122b90d2ef13be5

which in Dao-AILab/flash-attention is the commit "Bump to v2.7.4" (2025-01-29). Upstream's
latest release is 2.8.3.post1, but the v2.7.4...v2.8.3 diff does not change this path: in
`flash_fwd_kernel.h` the only edits add `typename` to four dependent-type expressions, the
`flash_fwd_launch_template.h` edit narrows template selection for ALiBi and return_softmax,
and the rest removes head dim 160. For head dim 64, fp16, sm80, no ALiBi/dropout/softcap, the
two versions compile the same forward kernel. PyTorch adds its own dispatch layer on top with
head dims 160 and 224 and without softcap.

## Latency

`python3 bench/compare_pytorch.py`, three runs. Each run is 10 reps and the three
implementations rotate order within a rep, so none of them always meets a cold or hot clock.
GPU utilization before each run was under 10% with no compute process on the device.

Paired median gap per run (positive = custom slower):

    dense    N=2048   vs flash  +0.37  +0.55  -0.37     vs cudnn  +3.85  +4.49  +4.53
             N=4096   vs flash  +0.89  +0.80  +0.32     vs cudnn  +4.58  +4.39  +4.31
    causal   N=2048   vs flash  -4.13  -3.68  -4.22     vs cudnn  +3.21  +2.98  +2.84
             N=4096   vs flash  -2.79  -2.80  -3.04     vs cudnn  -2.31  -1.81  -2.23

Median across the three runs:

| mask | N | custom ms | flash ms | cuDNN ms | custom TF | flash TF | cuDNN TF | vs flash | vs cuDNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 2048 | 0.2170 | 0.2173 | 0.2083 | 39.6 | 39.5 | 41.2 | +0.37% | +4.49% |
| dense | 4096 | 0.8520 | 0.8455 | 0.8141 | 40.3 | 40.6 | 42.2 | +0.80% | +4.39% |
| causal | 2048 | 0.1464 | 0.1528 | 0.1418 | 29.3 | 28.1 | 30.3 | -4.13% | +2.98% |
| causal | 4096 | 0.4905 | 0.5056 | 0.4999 | 35.0 | 34.0 | 34.4 | -2.80% | -2.23% |

Causal N=4096 is the only cell where this kernel leads both backends. Outputs of all three
agree to 4.9e-04 on causal and 1.2e-04 on dense, which is fp16 rounding.

## Scope

One GPU, one shape family (B=1, H=8, D=64, N_q = N_kv), fp16, forward only. Head dim 128,
larger batch and head counts, GQA and variable-length inputs are not implemented here and are
where the larger tiles of the other two kernels would be expected to pay off.

FLOPs counted as 4·N²·D·H, halved for causal.
