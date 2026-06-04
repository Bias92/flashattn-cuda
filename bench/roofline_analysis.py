"""
FlashAttention Forward — Effective Memory Bandwidth & Roofline Analysis
Computes bandwidth utilization across 3 platforms using known GPU times.

Memory access pattern (FlashAttention tiled forward, BR=BC=32):
  Per batch-head:
    Read  Q: N × D              (once, loaded into registers per Q-block)
    Read  K: ceil(N/BR) × N × D (each Q-block iterates all K-blocks)
    Read  V: ceil(N/BR) × N × D (each Q-block iterates all V-blocks)
    Write O: N × D              (once)
    Write L: N                  (once)
"""

import numpy as np

B, H, D = 1, 8, 64
BR = BC = 32

# === Platform specs ===
platforms = {
    "RTX 4060 Ti": {
        "mem_type": "GDDR6",
        "peak_bw_gbps": 288.0,
        "gpu_times_ms": {128: 0.12, 256: 0.12, 512: 0.25, 1024: 0.94, 2048: 3.06, 4096: 11.18},
    },
    # Jetson placeholder — fill in after ncu profiling
    # "Jetson Orin": {
    #     "mem_type": "Unified LPDDR5",
    #     "peak_bw_gbps": 204.8,
    #     "gpu_times_ms": {128: ?, 256: ?, 512: ?, 1024: ?, 2048: ?, 4096: ?},
    # },
}

def compute_bytes(N):
    """Total bytes transferred for FlashAttention forward (FP32)."""
    BH = B * H
    num_q_blocks = (N + BR - 1) // BR
    
    # Per batch-head element counts
    q_read = N * D                         # Q loaded once per row
    k_read = num_q_blocks * N * D          # K fully scanned per Q-block
    v_read = num_q_blocks * N * D          # V fully scanned per Q-block
    o_write = N * D                        # O written once
    l_write = N                            # L written once
    
    total_elements = q_read + k_read + v_read + o_write + l_write
    total_bytes = BH * total_elements * 4  # FP32 = 4 bytes
    return total_bytes

def compute_flops(N):
    """Total FLOPs for FlashAttention forward."""
    BH = B * H
    num_q_blocks = (N + BR - 1) // BR
    num_kv_blocks = (N + BC - 1) // BC
    
    # Per Q-block, per KV-block:
    #   S = Q[BR,D] @ K[BC,D]^T  → 2 * BR * BC * D FLOPs
    #   P @ V: [BR,BC] @ [BC,D]  → 2 * BR * D * BC FLOPs
    #   Softmax: ~5 * BR * BC (exp, max, sum, div, sub)
    
    matmul_flops = num_q_blocks * num_kv_blocks * (2 * BR * BC * D + 2 * BR * D * BC)
    softmax_flops = num_q_blocks * num_kv_blocks * (5 * BR * BC)
    
    total_flops = BH * (matmul_flops + softmax_flops)
    return total_flops

print("=" * 90)
print("FlashAttention Forward — Effective Bandwidth & Roofline Analysis")
print(f"Config: B={B}, H={H}, D={D}, BR=BC={BR}, FP32")
print("=" * 90)

# Print memory transfer amounts
print(f"\n{'N':>6} | {'Total Bytes':>14} | {'Total MB':>10} | {'Total GFLOP':>12} | {'Arith Intensity':>16}")
print("-" * 90)
for N in [128, 256, 512, 1024, 2048, 4096]:
    total_bytes = compute_bytes(N)
    total_flops = compute_flops(N)
    ai = total_flops / total_bytes  # arithmetic intensity (FLOP/byte)
    print(f"{N:>6} | {total_bytes:>14,} | {total_bytes/1e6:>8.1f}MB | {total_flops/1e9:>10.3f}G | {ai:>13.2f} FLOP/B")

# Per-platform analysis
for name, spec in platforms.items():
    peak_bw = spec["peak_bw_gbps"]
    times = spec["gpu_times_ms"]
    
    print(f"\n{'=' * 90}")
    print(f"{name} ({spec['mem_type']}, peak {peak_bw} GB/s)")
    print(f"{'=' * 90}")
    print(f"{'N':>6} | {'GPU Time':>10} | {'Eff BW':>12} | {'BW Util':>10} | {'GFLOPS':>10} | {'Arith Int':>10}")
    print("-" * 90)
    
    for N in [128, 256, 512, 1024, 2048, 4096]:
        if N not in times:
            continue
        gpu_ms = times[N]
        gpu_sec = gpu_ms / 1000.0
        
        total_bytes = compute_bytes(N)
        total_flops = compute_flops(N)
        
        eff_bw_gbps = (total_bytes / gpu_sec) / 1e9
        bw_util = eff_bw_gbps / peak_bw * 100
        gflops = total_flops / gpu_sec / 1e9
        ai = total_flops / total_bytes
        
        print(f"{N:>6} | {gpu_ms:>8.2f}ms | {eff_bw_gbps:>9.2f}GB/s | {bw_util:>7.1f}% | {gflops:>8.2f}G | {ai:>7.2f} F/B")

print(f"\n{'=' * 90}")
print("Notes:")
print("- Effective BW = total_bytes_transferred / GPU_kernel_time")
print("- BW Util = effective_BW / peak_memory_bandwidth × 100")
print("- Arith Intensity = total_FLOPs / total_bytes (FLOP/byte)")
print("- Low BW utilization indicates compute-bound or latency-bound kernel")
print("=" * 90)
