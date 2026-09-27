#pragma once

#include "kernel_traits.h"
#include "attention_forward_ops.cuh"

// ============================================================
// Forward kernel: one block per Q tile and batch/head pair
// ============================================================
template <int kHeadDim_, bool WRITE_L, bool Is_even_MN>
__global__ void __launch_bounds__(kNWarps * 32)
flash_fwd_kernel(
    const half* __restrict__ q_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    half* __restrict__ o_ptr,
    float* __restrict__ softmax_lse_ptr,    // may be nullptr when WRITE_L == false
    int N)
{
    // 1. SETUP: tile dimensions, thread ownership, and copy addresses.
    // -----------------------------------------------------------
    constexpr int LDS = kHeadDim_ + PAD;        // Shared row stride, in half elements.
    constexpr int KSLICES = kHeadDim_ / 16;     // 16-feature slices in QK^T.
    constexpr int NTILES_S = kBlockN / 8;    // 8-key column groups in scores.
    constexpr int KSLICES_PV = kBlockN / 16; // 16-key slices in the weighted V sum.
    constexpr int NTILES_O = kHeadDim_ / 8;     // 8-feature column groups in output.

    // One stage holds K[kBlockN][LDS] followed by V[kBlockN][LDS].
    constexpr int STAGE = 2 * kBlockN * LDS;
    constexpr uint32_t STAGE_BYTES = STAGE * sizeof(half);
    constexpr uint32_t ROW_BYTES = LDS * sizeof(half);

    // Block coordinates and thread/warp coordinates.
    const int tidx  = threadIdx.x;
    const int warp = tidx / 32;
    const int lane = tidx % 32;
    const int bh = blockIdx.y;
    const int q_block = blockIdx.x * kBlockM;

    // Base pointers for the selected batch/head pair. Each tensor slice is N x D.
    const half* Q_bh = q_ptr + (size_t)bh * N * kHeadDim_;
    const half* K_bh = k_ptr + (size_t)bh * N * kHeadDim_;
    const half* V_bh = v_ptr + (size_t)bh * N * kHeadDim_;
    half* O_bh = o_ptr + (size_t)bh * N * kHeadDim_;
    // L_bh is derived inside the WRITE_L epilogue only (no pointer
    // arithmetic on a null softmax_lse_ptr when WRITE_L == false).

    // Two equally sized shared-memory regions. The second region first
    // stages Q; after Q enters registers, both regions stage K/V tiles.
    __shared__ __align__(16) half smem_[2 * STAGE];

    const uint32_t smem_base = smem_u32(smem_);
    uint32_t cur_base  = smem_base;
    uint32_t next_base = smem_base + STAGE_BYTES;

    // Each thread copies 2 rows x 8 half elements from K and from V.
    // r0 = tidx / 8, r1 = r0 + 16, cc = (tidx % 8) * 8.
    const int r0 = tidx >> 3;
    const int r1 = r0 + 16;
    const int cc = (tidx & 7) << 3;
    const uint32_t k0_off = (uint32_t)(r0 * LDS + cc) * sizeof(half);
    const uint32_t k1_off = (uint32_t)(r1 * LDS + cc) * sizeof(half);
    const uint32_t v0_off = (uint32_t)((kBlockN + r0) * LDS + cc) * sizeof(half);
    const uint32_t v1_off = (uint32_t)((kBlockN + r1) * LDS + cc) * sizeof(half);

    // Define the copy operation. sbase: shared destination; kv: first K/V row.
    // Each call requests K and V copies, then commits the copy group.
    auto issue_kv_fast = [&](uint32_t sbase, int kv) {
        if constexpr (Is_even_MN) {
            // N % kBlockN == 0 guarantees every issued row is in range.
            cp_async_16(
                sbase + k0_off, K_bh + (size_t)(kv + r0) * kHeadDim_ + cc, 16);
            cp_async_16(
                sbase + k1_off, K_bh + (size_t)(kv + r1) * kHeadDim_ + cc, 16);
            cp_async_16(
                sbase + v0_off, V_bh + (size_t)(kv + r0) * kHeadDim_ + cc, 16);
            cp_async_16(
                sbase + v1_off, V_bh + (size_t)(kv + r1) * kHeadDim_ + cc, 16);
        } else {
            // A partial final K/V tile: out-of-range rows receive zero bytes
            // from the source and are zero-filled by the copy instruction.
            int g0 = kv + r0, g1 = kv + r1;
            const half* k0 = K_bh + (size_t)(g0 < N ? g0 : 0) * kHeadDim_ + cc;
            const half* k1 = K_bh + (size_t)(g1 < N ? g1 : 0) * kHeadDim_ + cc;
            const half* v0 = V_bh + (size_t)(g0 < N ? g0 : 0) * kHeadDim_ + cc;
            const half* v1 = V_bh + (size_t)(g1 < N ? g1 : 0) * kHeadDim_ + cc;
            cp_async_16(sbase + k0_off, k0, (g0 < N) ? 16 : 0);
            cp_async_16(sbase + k1_off, k1, (g1 < N) ? 16 : 0);
            cp_async_16(sbase + v0_off, v0, (g0 < N) ? 16 : 0);
            cp_async_16(sbase + v1_off, v1, (g1 < N) ? 16 : 0);
        }
        cp_async_fence();
    };

    // 2. LOAD Q ONCE AND INITIALIZE THE RUNNING STATE.
    // -----------------------------------------------------------
    // Start K/V tile 0 while the second shared region stages the Q tile.
    issue_kv_fast(cur_base, 0);
    {
        half (*sQ)[LDS] = reinterpret_cast<half(*)[LDS]>(smem_ + STAGE);
        const int total_h2 = kBlockM * kHeadDim_ / 2;
        for (int i = tidx; i < total_h2; i += blockDim.x) {
            int flat = i * 2;
            int r = flat / kHeadDim_, c = flat % kHeadDim_;
            if constexpr (Is_even_MN) {
                // N % kBlockM == 0 guarantees every Q row in this block is valid.
                *reinterpret_cast<half2*>(&sQ[r][c]) =
                    *reinterpret_cast<const half2*>(&Q_bh[(size_t)(q_block + r) * kHeadDim_ + c]);
            } else {
                int gr = q_block + r;
                half2 val = (gr < N)
                    ? *reinterpret_cast<const half2*>(&Q_bh[(size_t)gr * kHeadDim_ + c])
                    : __float2half2_rn(0.0f);
                *reinterpret_cast<half2*>(&sQ[r][c]) = val;
            }
        }
        __syncthreads();
    }

    // Load Q fragments once for reuse across the entire K/V tile loop.
    uint32_t tSrQ[KSLICES][4];
    {
        half (*sQ)[LDS] = reinterpret_cast<half(*)[LDS]>(smem_ + STAGE);
        int r = warp * 16 + (lane % 16);
        int kbase = (lane < 16) ? 0 : 8;
        #pragma unroll
        for (int ks = 0; ks < KSLICES; ks++) {
            uint32_t addr = smem_u32(&sQ[r][ks * 16 + kbase]);
            ldmatrix_x4(tSrQ[ks][0], tSrQ[ks][1], tSrQ[ks][2], tSrQ[ks][3], addr);
        }
    }
    __syncthreads();

    // Per-lane offsets for reading K and V from the active shared stage.
    const uint32_t qk_lane_base =
        (uint32_t)((lane & 7) * LDS + ((lane < 8) ? 0 : 8)) * sizeof(half);
    const uint32_t pv_lane_base =
        (uint32_t)((kBlockN + (lane & 15)) * LDS) * sizeof(half);

    // Running state, retained across K/V tiles:
    //   acc_o: unnormalized weighted V sum; m: running maximum; l: running sum.
    //   lo/hi: the two Q rows represented by each lane, separated by 8 rows.
    //   Scores and maxima use base-2 scaling for exp2f.
    float acc_o[NTILES_O][4];
    #pragma unroll
    for (int t = 0; t < NTILES_O; t++)
        acc_o[t][0] = acc_o[t][1] = acc_o[t][2] = acc_o[t][3] = 0.0f;

    float m_lo = -INFINITY, m_hi = -INFINITY;
    float l_lo = 0.0f,  l_hi = 0.0f;
    const float scale_softmax_log2 = rsqrtf((float)kHeadDim_) * LOG2Ef;

    const int n_block_max = (N + kBlockN - 1) / kBlockN;

    // 3. FUSED OUTER LOOP: visit one kBlockN-row K/V tile per iteration.
    // -----------------------------------------------------------
    // Q stays fixed. Every iteration performs:
    //   A. Prepare K/V -> B. QK^T -> C. Online softmax -> D. Accumulate V.
    for (int n_block = 0; n_block < n_block_max; n_block++) {
        const int kv = n_block * kBlockN;
        const bool has_next = (n_block + 1) < n_block_max;

        // A. Prefetch the next K/V tile; wait until the current tile is ready.
        if (has_next) {
            issue_kv_fast(next_base, kv + kBlockN);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        const uint32_t qk_base = cur_base + qk_lane_base;
        const uint32_t pv_base = cur_base + pv_lane_base;

        // B. SCORES: QK^T, followed by 1/sqrt(D) and base-2 scaling.
        // Inner loops: 8 key columns per group, then 16 head features per slice.
        float acc_s[NTILES_S][4];

        #pragma unroll
        for (int t = 0; t < NTILES_S; t++) {
            acc_s[t][0] = acc_s[t][1] = acc_s[t][2] = acc_s[t][3] = 0.0f;

            #pragma unroll
            for (int ks = 0; ks < KSLICES; ks++) {
                uint32_t b0, b1;
                ldmatrix_x2(b0, b1,
                    qk_base + (uint32_t)(t * 8) * ROW_BYTES
                            + (uint32_t)(ks * 16) * sizeof(half));
                mma_m16n8k16(
                    acc_s[t][0], acc_s[t][1], acc_s[t][2], acc_s[t][3],
                    tSrQ[ks][0], tSrQ[ks][1], tSrQ[ks][2], tSrQ[ks][3],
                    b0, b1);
            }

            #pragma unroll
            for (int j = 0; j < 4; j++)
                acc_s[t][j] *= scale_softmax_log2;
        }

        // Exclude keys beyond N in a partial final K tile.
        if constexpr (!Is_even_MN) {
            if (kv + kBlockN > N) {
                #pragma unroll
                for (int t = 0; t < NTILES_S; t++) {
                    int col0 = kv + t * 8 + 2 * (lane % 4);
                    if (col0 >= N) {
                        acc_s[t][0] = -INFINITY;
                        acc_s[t][2] = -INFINITY;
                    }
                    if (col0 + 1 >= N) {
                        acc_s[t][1] = -INFINITY;
                        acc_s[t][3] = -INFINITY;
                    }
                }
            }
        }

        // C. ONLINE SOFTMAX: update each Q row's running maximum and sum.
        // C1. Find each Q row's maximum over the current K tile.
        float bm_lo = -INFINITY, bm_hi = -INFINITY;
        #pragma unroll
        for (int t = 0; t < NTILES_S; t++) {
            bm_lo = fmaxf(bm_lo, fmaxf(acc_s[t][0], acc_s[t][1]));
            bm_hi = fmaxf(bm_hi, fmaxf(acc_s[t][2], acc_s[t][3]));
        }
        bm_lo = fmaxf(bm_lo, __shfl_xor_sync(0xffffffff, bm_lo, 1));
        bm_lo = fmaxf(bm_lo, __shfl_xor_sync(0xffffffff, bm_lo, 2));
        bm_hi = fmaxf(bm_hi, __shfl_xor_sync(0xffffffff, bm_hi, 1));
        bm_hi = fmaxf(bm_hi, __shfl_xor_sync(0xffffffff, bm_hi, 2));

        // C2. Merge maxima; alpha rescales contributions from earlier K/V tiles.
        float mn_lo = fmaxf(m_lo, bm_lo);
        float mn_hi = fmaxf(m_hi, bm_hi);
        float alpha_lo = exp2f(m_lo - mn_lo);
        float alpha_hi = exp2f(m_hi - mn_hi);

        // C3. Replace scores with exp2(score - new_max), then sum each row.
        // The loop overwrites acc_s with unnormalized softmax numerators.
        float rs_lo = 0.0f, rs_hi = 0.0f;
        #pragma unroll
        for (int t = 0; t < NTILES_S; t++) {
            acc_s[t][0] = exp2f(acc_s[t][0] - mn_lo);
            acc_s[t][1] = exp2f(acc_s[t][1] - mn_lo);
            acc_s[t][2] = exp2f(acc_s[t][2] - mn_hi);
            acc_s[t][3] = exp2f(acc_s[t][3] - mn_hi);
            rs_lo += acc_s[t][0] + acc_s[t][1];
            rs_hi += acc_s[t][2] + acc_s[t][3];
        }
        rs_lo += __shfl_xor_sync(0xffffffff, rs_lo, 1);
        rs_lo += __shfl_xor_sync(0xffffffff, rs_lo, 2);
        rs_hi += __shfl_xor_sync(0xffffffff, rs_hi, 1);
        rs_hi += __shfl_xor_sync(0xffffffff, rs_hi, 2);

        // C4. Update the running denominator and maximum for the next K/V tile.
        l_lo = l_lo * alpha_lo + rs_lo;
        l_hi = l_hi * alpha_hi + rs_hi;
        m_lo = mn_lo;
        m_hi = mn_hi;

        // D. OUTPUT ACCUMULATION: rescale previous output and add numerator * V.
        // D1. Pack the softmax numerators as FP16 matrix operands.
        uint32_t tOrP[KSLICES_PV][4];
        #pragma unroll
        for (int ks = 0; ks < KSLICES_PV; ks++) {
            tOrP[ks][0] = pack_half2(acc_s[2 * ks][0],     acc_s[2 * ks][1]);
            tOrP[ks][1] = pack_half2(acc_s[2 * ks][2],     acc_s[2 * ks][3]);
            tOrP[ks][2] = pack_half2(acc_s[2 * ks + 1][0], acc_s[2 * ks + 1][1]);
            tOrP[ks][3] = pack_half2(acc_s[2 * ks + 1][2], acc_s[2 * ks + 1][3]);
        }

        // D2. Rescale the previous output sum by alpha.
        #pragma unroll
        for (int t = 0; t < NTILES_O; t++) {
            acc_o[t][0] *= alpha_lo;
            acc_o[t][1] *= alpha_lo;
            acc_o[t][2] *= alpha_hi;
            acc_o[t][3] *= alpha_hi;
        }

        // D3. Add numerator * V for the current K/V tile.
        // Inner loops: 8 output features per group, then 16 K/V rows per slice.
        #pragma unroll
        for (int t = 0; t < NTILES_O; t++) {
            #pragma unroll
            for (int ks = 0; ks < KSLICES_PV; ks++) {
                uint32_t b0, b1;
                ldmatrix_x2_trans(b0, b1,
                    pv_base + (uint32_t)(ks * 16) * ROW_BYTES
                            + (uint32_t)(t * 8) * sizeof(half));
                mma_m16n8k16(
                    acc_o[t][0], acc_o[t][1], acc_o[t][2], acc_o[t][3],
                    tOrP[ks][0], tOrP[ks][1], tOrP[ks][2], tOrP[ks][3],
                    b0, b1);
            }
        }
        __syncthreads();

        // Advance the K/V pipeline by exchanging current/next shared stages.
        uint32_t tmp = cur_base;
        cur_base = next_base;
        next_base = tmp;
    }

    // 4. NORMALIZE AND STORE: all K/V tiles have contributed.
    // -----------------------------------------------------------
    // Divide the accumulated output by the final softmax denominator.
    const float inv_lo = 1.0f / l_lo;
    const float inv_hi = 1.0f / l_hi;
    const int r_lo = q_block + warp * 16 + lane / 4;
    const int r_hi = r_lo + 8;
    const int cbase = 2 * (lane % 4);

    #pragma unroll
    for (int t = 0; t < NTILES_O; t++) {
        int col = t * 8 + cbase;
        if (Is_even_MN || r_lo < N) {
            half2 v = __floats2half2_rn(acc_o[t][0] * inv_lo, acc_o[t][1] * inv_lo);
            *reinterpret_cast<half2*>(&O_bh[(size_t)r_lo * kHeadDim_ + col]) = v;
        }
        if (Is_even_MN || r_hi < N) {
            half2 v = __floats2half2_rn(acc_o[t][2] * inv_hi, acc_o[t][3] * inv_hi);
            *reinterpret_cast<half2*>(&O_bh[(size_t)r_hi * kHeadDim_ + col]) = v;
        }
    }

    // Optional per-Q-row logsumexp, converted back to natural-log units.
    if constexpr (WRITE_L) {
        if (lane % 4 == 0) {
            float* L_bh = softmax_lse_ptr + (size_t)bh * N;
            if (Is_even_MN || r_lo < N)
                L_bh[r_lo] = m_lo * LN2f + logf(l_lo);
            if (Is_even_MN || r_hi < N)
                L_bh[r_hi] = m_hi * LN2f + logf(l_hi);
        }
    }
}
