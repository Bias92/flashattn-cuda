# flashattn-cuda-metal

This is project FlashAttention forward/backward kernels implemented from scratch in CUDA(profiling it and optimizing with WMMA)
finally, ported to Apple Metal, profiled across discrete(Discrete/Separate Memory Architecture) and unified memory GPUs(Unified Memory Architecture, AKA "UMA").

## Goal

Scratch-implement FlashAttention → profile with ncu → optimize (5 attempts) 
-> port to Metal -> compare kernel-level performance across 
RTX 4060 Ti (discrete GDDR6) vs M4 Pro (unified LPDDR5) vs Jetson AGX Orin (unified LPDDR5).

## Algorithm

### Forward (Algorithm 1)

1. **Tile Q into row blocks ($B_r=32$), K/V into column blocks ($B_c=32$)**
2. **Load Q row into registers** (stays fixed across all K/V blocks)
3. **For each K/V block:**
   - Collaboratively load K, V tiles into shared memory (16KB total)
   - Compute $S = Q \cdot K^T \times \text{scale}$
   - Online softmax update: $m_{new} = \max(m_{old}, m_{block})$, rescale previous accumulator by $e^{m_{old} - m_{new}}$, accumulate new block
4. **Normalize:** $O = \text{acc} / l_i$
5. **Store logsumexp** $L = m + \log(l)$ for backward pass

Thread model: one thread per Q row, grid = $(⌈N/B_r⌉,\; B \times H)$, block = $(B_r,)$.

### Backward (Algorithm 2)

The backward pass computes $dQ$, $dK$, $dV$ without ever materializing the full $N \times N$ attention matrix by **recomputing** $S$ and $P$ from the stored logsumexp $L$.

Three kernels:

1. **Precompute D:** $D_i = \text{rowsum}(dO \odot O)$ — simple per-row dot product
2. **dQ kernel:** One thread per Q row, iterates over all K/V blocks. Recomputes $P = e^{QK^T \times \text{scale} - L}$, then accumulates $dQ \mathrel{+}= P(dO \cdot V^T - D) \times K \times \text{scale}$
3. **dK/dV kernel:** One thread per K/V row, iterates over all Q blocks. Recomputes $P$, then accumulates $dV \mathrel{+}= P^T \times dO$ and $dK \mathrel{+}= (P(dO \cdot V^T - D))^T \times Q \times \text{scale}$

Same tiling strategy ($B_r=32$, $B_c=32$) and shared memory collaborative loading as forward.



## Correctness

### Forward — 9/9 passed

```
[PASS] B=1, H=1, N=   32, D=64  |  max_diff=4.768372e-07
[PASS] B=1, H=1, N=   64, D=64  |  max_diff=3.874302e-07
[PASS] B=1, H=1, N=  128, D=64  |  max_diff=4.768372e-07
[PASS] B=1, H=1, N=   63, D=64  |  max_diff=4.768372e-07
[PASS] B=1, H=1, N=  127, D=64  |  max_diff=4.470348e-07
[PASS] B=2, H=4, N=  256, D=64  |  max_diff=4.768372e-07
[PASS] B=2, H=8, N=  512, D=64  |  max_diff=6.854534e-07
[PASS] B=1, H=1, N= 1024, D=64  |  max_diff=3.576279e-07
[PASS] B=1, H=1, N= 2048, D=64  |  max_diff=4.023214e-07
```

### Backward — 9/9 passed

```
[PASS] B=1, H=1, N=   32, D=64  |  dQ_diff=4.768372e-07  dK_diff=4.172325e-07  dV_diff=3.576279e-07
[PASS] B=1, H=1, N=   64, D=64  |  dQ_diff=5.960464e-07  dK_diff=5.364418e-07  dV_diff=4.768372e-07
[PASS] B=1, H=1, N=  128, D=64  |  dQ_diff=4.768372e-07  dK_diff=5.960464e-07  dV_diff=3.576279e-07
[PASS] B=1, H=1, N=   63, D=64  |  dQ_diff=6.556511e-07  dK_diff=5.364418e-07  dV_diff=4.204921e-07
[PASS] B=1, H=1, N=  127, D=64  |  dQ_diff=6.556511e-07  dK_diff=5.960464e-07  dV_diff=3.576279e-07
[PASS] B=2, H=4, N=  256, D=64  |  dQ_diff=7.152557e-07  dK_diff=1.072884e-06  dV_diff=4.768372e-07
[PASS] B=2, H=8, N=  512, D=64  |  dQ_diff=4.768372e-07  dK_diff=4.768372e-07  dV_diff=3.278255e-07
[PASS] B=1, H=1, N= 1024, D=64  |  dQ_diff=4.768372e-07  dK_diff=3.725290e-07  dV_diff=3.725290e-07
[PASS] B=1, H=1, N= 2048, D=64  |  dQ_diff=4.470348e-07  dK_diff=4.470348e-07  dV_diff=3.278255e-07
```
the reason why Batch is 1. because the kernel processes each (batch, head) pair independently increasing B only launches more identical blocks without changing `per-kernel` behavior. i just fixed B=1 to isolate sequence length scaling. also 
tests cover single-block, multi-block, non-aligned sequence lengths, and multi-batch/multi-head configurations.

## Benchmark Results

### Forward

**GPU:** NVIDIA GeForce RTX 4060 Ti | **Precision:** FP32 | **Config:** B=1, H=8, D=64

| Seq Len | Naive (ms) | Flash (ms) | Speedup | Naive Mem | Flash Mem | Mem Save |
|---------|-----------|-----------|---------|-----------|-----------|----------|
| 128     | 0.62      | 0.12      | 5.26×   | 10.6 MB   | 9.1 MB    | 1.16×    |
| 256     | 0.18      | 0.12      | 1.56×   | 16.1 MB   | 10.1 MB   | 1.59×    |
| 512     | 0.19      | 0.25      | 0.75×   | 36.2 MB   | 12.1 MB   | 2.98×    |
| 1024    | 1.59      | 0.94      | 1.69×   | 112.2 MB  | 16.2 MB   | 6.94×    |
| 2048    | 6.94      | 3.06      | 2.27×   | 408.2 MB  | 24.2 MB   | 16.88×   |
| 4096    | 27.87     | 11.18     | 2.49×   | 1576.4 MB | 40.2 MB   | **39.16×** |

### Backward

**GPU:** NVIDIA GeForce RTX 4060 Ti | **Precision:** FP32 | **Config:** B=1, H=8, D=64

| Seq Len | Naive (ms) | Flash (ms) | Speedup | Naive Mem | Flash Mem | Mem Save |
|---------|-----------|-----------|---------|-----------|-----------|----------|
| 128     | 0.24      | 0.26      | 0.94×   | 12.6 MB   | 10.1 MB   | 1.25×    |
| 256     | 0.43      | 0.52      | 0.82×   | 21.6 MB   | 12.1 MB   | 1.78×    |
| 512     | 0.84      | 1.75      | 0.48×   | 55.2 MB   | 16.2 MB   | 3.41×    |
| 1024    | 2.38      | 5.59      | 0.43×   | 182.2 MB  | 24.2 MB   | 7.53×    |
| 2048    | 8.79      | 17.96     | 0.49×   | 676.2 MB  | 40.2 MB   | 16.80×   |
| 4096    | 34.62     | 69.50     | 0.50×   | 2624.4 MB | 72.4 MB   | **36.26×** |

both of them (fw/bwd) are saving memory 36-39x at seq_len=4096.
backward is slower than naive because of occupancy 7.9% and resiget spill ( may you can see profiling section below.)

## Profiling (Nsight Compute)

in this section. i focused on 4 metrics per kernel : 

- **GPU Speed of Light (SOL)**: how much of the GPU's peak compute and memory bandwidth is actually used. If both are under 60%, the kernel is stalling because there aren't enough warps to hide memory latency.
- **Roofline**: plots the kernel's achieved throughput against hardware limits.
$$\text{Attainable Performance} = \min(\text{Peak FLOP/s},\; \text{Peak Bandwidth} \times \text{Arithmetic Intensity})$$
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$
- **Memory Workload**: shows memory access patterns. High local memory % means registers are spilling to L1/global memory, adding extra latency.
- **Occupancy**: ratio of active warps to the hardware max. Low occupancy means the SM can't switch between enough warps to keep the pipeline busy.

entire kernels are profiled with `ncu --set full --launch-count 1` on N=1024, B=1, H=8, D=64.

### Kernel Comparison Summary

| Metric | Forward | Backward (dQ) | Backward (dK/dV) |
|--------|---------|---------------|-------------------|
| Duration | 1.14 ms | 1.53 ms | 5.64 ms |
| Compute (SM) Throughput | 25.30% | 22.82% | 17.95% |
| Memory Throughput | 25.30% | 73.89% | 32.31% |
| Achieved Occupancy | 7.90% | 8.99% | 9.44% |
| FP32 Peak Achieved | 10% | 11% | 4% |
| Block Limit (Shared Mem) | 5 | 5 | 5 |
| Register Spill (Local Mem) | 28.57% | 26.39% | 71.88% |
| Diagnosis | Latency | High Memory | Latency |

### Forward Kernel

| Before (FP32 Baseline) | After (WMMA+half2 v1) |
|---|---|
| <img src="docs/profiling/gpu_speed_of_light.png" width="400"> | <img src="docs/profiling/wmma_gpu_sol.png" width="400"> |
| Compute 25.30%, Memory 25.30% | Compute 27.40%, Memory 42.55% |
| <img src="docs/profiling/roofline_fp32.png" width="400"> | <img src="docs/profiling/wmma_roofline_fp32.png" width="400"> |
| 10% of FP32 peak | 0% of FP32 peak, Tensor Core 4.54% |
| <img src="docs/profiling/memory_workload.png" width="400"> | <img src="docs/profiling/wmma_memory_workload.png" width="400"> |
| Register spill 28.57% | Register spill 0%, bank conflict 4.6-way |
| <img src="docs/profiling/occupancy.png" width="400"> | <img src="docs/profiling/wmma_occupancy.png" width="400"> |
| Occupancy 7.90% | Occupancy 12.49% |

### Forward Optimization Summary

tried 5 things to make the forward kernel faster:

| # | what i tried | result | why it failed/worked |
|---|---|---|---|
| 1 | tile 32→16 + launch_bounds | 11.18→15.44ms, slower | K/V iteration 2x, spill unchanged |
| 2 | FP16 shared memory | occupancy 2x but 11.18→16.28ms | half2float() conversion kills it |
| 3 | WMMA 16×16 | 11.18→11.83ms | tile too small, softmax breaks pipeline |
| 4 | WMMA + half2 load | 11.18→11.63ms, N=128: 0.09ms | kept this one (v1) |
| 5 | 4 warps/block | 11.18→12.39ms, rolled back | 35KB shmem, only 2 blocks fit per SM |

scratch WMMA can't beat FP32 baseline at large N. that's why Dao Lab uses CUTLASS with 128×64+ tiles and warp-specialized softmax.

------------


### Backward dQ Kernel

#### GPU Speed of Light

![dQ GPU SOL](docs/profiling/bwd_dq_gpu_sol.png)

- Compute (SM) Throughput: 22.82%
- Memory Throughput: 73.89%
- L1/TEX Cache Throughput: 77.84%
- DRAM Throughput: 2.19%
- Diagnosis: High Memory Throughput — L1 bottleneck from register spill traffic

#### Roofline (Single Precision)

![dQ Roofline](docs/profiling/bwd_dq_roofline.png)

Achieves 11% of FP32 peak. Similar arithmetic intensity range as forward but with higher memory pressure from recomputation.

#### Memory Workload

![dQ Memory Workload](docs/profiling/bwd_dq_memory_workload.png)

- Local memory usage: 26.39% of L1TEX — register spill from q_reg[64] + do_reg[64] + dq_acc[64]
- L1/TEX Hit Rate: 90.59% — good cache reuse from tiling
- Local Memory Spilling Requests: 3.1M

#### Occupancy

![dQ Occupancy](docs/profiling/bwd_dq_occupancy.png)

- Theoretical occupancy: 10.42%
- Achieved occupancy: 8.99%
- Active warps per SM: 4.32
- **Bottleneck: shared memory** (Block Limit Shared Mem = 5)
- Estimated speedup from fixing: 26.11%

### Backward dK/dV Kernel

#### GPU Speed of Light

![dK/dV GPU SOL](docs/profiling/bwd_dkdv_gpu_sol.png)

- Compute (SM) Throughput: 17.95%
- Memory Throughput: 32.31%
- L1/TEX Cache Throughput: 37.15%
- DRAM Throughput: 3.95%
- Diagnosis: Latency Issue — FP32 peak only 4%, worst of all three kernels

#### Roofline (Single Precision)

![dK/dV Roofline](docs/profiling/bwd_dkdv_roofline.png)

Achieves only 4% of FP32 peak — the most severe underutilization. Kernel points sit far below the roofline ceiling.

#### Memory Workload

![dK/dV Memory Workload](docs/profiling/bwd_dkdv_memory_workload.png)

- Local memory usage: **71.88%** of L1TEX — severe register spill
- k_reg[64] + v_reg[64] + dk_acc[64] + dv_acc[64] = 256+ registers per thread
- Local Memory Spilling Requests: **22.8M** (7× worse than dQ)
- L1/TEX Hit Rate: 29.51% — poor cache utilization due to massive spill traffic

#### Occupancy

![dK/dV Occupancy](docs/profiling/bwd_dkdv_occupancy.png)

- Theoretical occupancy: 10.42%
- Achieved occupancy: 9.44%
- Active warps per SM: 4.53
- **Bottleneck: shared memory** (Block Limit Shared Mem = 5)
- Estimated speedup from fixing: 67.69%

### Profiling Summary

1. All three kernels share the same bottleneck: **occupancy ~10%** due to shared memory (16KB per block) and register pressure
2. The dK/dV kernel is the worst performer — 71.88% local memory usage from 256+ registers per thread causes 22.8M spill requests
3. Optimization targets: **FP16 Tensor Core** (halves register/shared memory usage, raises throughput ceiling) → **tile size tuning** → **`__launch_bounds__`** for register control

## Project Structure

```
flashattn-cuda-metal/
├── cuda/
│   └── flash_attn_kernel.cu    # Forward + backward CUDA kernels
├── ref/
│   └── naive_attn.py           # O(N²) reference implementation
├── tests/
│   ├── test_forward.py         # Forward correctness tests (9 configs)
│   └── test_backward.py        # Backward correctness tests (9 configs)
├── bench/
│   ├── bench_forward.py        # Forward benchmark with CSV output
│   └── bench_backward.py       # Backward benchmark with CSV output
├── docs/
│   └── profiling/              # NCU screenshots (forward + backward)
├── setup.py                    # PyTorch CUDA extension build
├── LICENSE                     # MIT
└── README.md
```

## Build & Run

Requires: CUDA toolkit matching your PyTorch CUDA version, PyTorch with CUDA support.

```bash
# Build
pip install -e .

# Test correctness
python tests/test_forward.py
python tests/test_backward.py

# Benchmark (outputs CSV to bench/results/)
python bench/bench_forward.py
python bench/bench_backward.py

# Profile with Nsight Compute
ncu --set full --launch-count 1 --kernel-name flash_attn_fwd_kernel \
    --export bench/results/flash_fwd \
    python -c "
import torch, flash_attn_cuda
Q=torch.randn(1,8,1024,64,device='cuda')
K=torch.randn(1,8,1024,64,device='cuda')
V=torch.randn(1,8,1024,64,device='cuda')
flash_attn_cuda.forward(Q,K,V)
"
```

## Current Specs

- Precision: FP32
- Head dimension: D=64 (compile-time constant)
- Tile sizes: B_r=32, B_c=32
- Shared memory: 16KB (sK[32][64] + sV[32][64])
- Target GPU: RTX 4060 Ti (sm_89, Ada Lovelace)

## Roadmap

- [x] Forward kernel (online softmax tiling)
- [x] Backward kernel (dQ, dK, dV with recomputation)
- [x] Nsight Compute profiling (forward + backward)
- [ ] FP16 support with Tensor Core (WMMA/MMA)
- [ ] Occupancy optimization (tile size tuning, register pressure reduction)
- [ ] Warp-level primitives (`__shfl_sync` for reductions)
- [ ] Apple Metal port (M4 Pro)
- [ ] Causal masking support

## Environment

- GPU: NVIDIA GeForce RTX 4060 Ti
- OS: Windows 11 + WSL2 (Ubuntu 24.04)
- CUDA: 12.8
- PyTorch: 2.x (CUDA 12.8 build)
- Profiler: Nsight Compute, Nsight Systems

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022)
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023)
- NVIDIA CUDA C++ Programming Guide
- MIT 6.5940 TinyML (Song Han)

## License

MIT
