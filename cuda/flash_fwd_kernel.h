#pragma once

#include "flash_fwd_memory.h"
#include "softmax.h"

// acc_s = Q K^T * softmax_scale * log2(e), for the exp2-based softmax.
template <int D>
__device__ __forceinline__ void compute_qk(
    float (&acc_s)[Flash_fwd_kernel_traits<D>::NTILES_S][4],
    const uint32_t (&tSrQ)[Flash_fwd_kernel_traits<D>::KSLICES][4],
    uint32_t qk_base, float softmax_scale_log2)
{
    using Kernel_traits = Flash_fwd_kernel_traits<D>;
    #pragma unroll
    for (int t = 0; t < Kernel_traits::NTILES_S; t++) {
        acc_s[t][0] = acc_s[t][1] = acc_s[t][2] = acc_s[t][3] = 0.0f;
        #pragma unroll
        for (int ks = 0; ks < Kernel_traits::KSLICES; ks++) {
            uint32_t tSrK0, tSrK1;
            ldmatrix_x2(tSrK0, tSrK1,
                qk_base + (uint32_t)(t * 8) * Kernel_traits::ROW_BYTES
                        + (uint32_t)(ks * 16) * sizeof(half));
            mma_m16n8k16(acc_s[t][0], acc_s[t][1], acc_s[t][2], acc_s[t][3],
                         tSrQ[ks][0], tSrQ[ks][1], tSrQ[ks][2], tSrQ[ks][3], tSrK0, tSrK1);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) acc_s[t][j] *= softmax_scale_log2;
    }
}

template <int NT>
__device__ __forceinline__ void apply_mask(float (&acc_s)[NT][4], int kv, int N_kv, int lane)
{
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        int col0 = kv + t * 8 + 2 * (lane % 4);
        if (col0 >= N_kv)     { acc_s[t][0] = -INFINITY; acc_s[t][2] = -INFINITY; }
        if (col0 + 1 >= N_kv) { acc_s[t][1] = -INFINITY; acc_s[t][3] = -INFINITY; }
    }
}

// Each lane owns rows row_lo and row_lo + 8. Mask tiles crossing the diagonal.
template <int NT>
__device__ __forceinline__ void apply_causal_mask(float (&acc_s)[NT][4], int kv, int row_lo, int lane)
{
    const int row_hi = row_lo + 8;
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        const int col0 = kv + t * 8 + 2 * (lane % 4);
        if (col0     > row_lo) acc_s[t][0] = -INFINITY;
        if (col0 + 1 > row_lo) acc_s[t][1] = -INFINITY;
        if (col0     > row_hi) acc_s[t][2] = -INFINITY;
        if (col0 + 1 > row_hi) acc_s[t][3] = -INFINITY;
    }
}

// Repack P from FP32 C fragments into FP16 A fragments for PV.
// V uses transposed ldmatrix loads.
template <int D>
__device__ __forceinline__ void accumulate_pv(
    float (&acc_o)[Flash_fwd_kernel_traits<D>::NTILES_O][4],
    const float (&acc_s)[Flash_fwd_kernel_traits<D>::NTILES_S][4],
    uint32_t pv_base)
{
    using Kernel_traits = Flash_fwd_kernel_traits<D>;
    uint32_t tOrP[Kernel_traits::KSLICES_PV][4];
    #pragma unroll
    for (int ks = 0; ks < Kernel_traits::KSLICES_PV; ks++) {
        tOrP[ks][0] = pack_half2(acc_s[2 * ks][0],     acc_s[2 * ks][1]);
        tOrP[ks][1] = pack_half2(acc_s[2 * ks][2],     acc_s[2 * ks][3]);
        tOrP[ks][2] = pack_half2(acc_s[2 * ks + 1][0], acc_s[2 * ks + 1][1]);
        tOrP[ks][3] = pack_half2(acc_s[2 * ks + 1][2], acc_s[2 * ks + 1][3]);
    }

    #pragma unroll
    for (int t = 0; t < Kernel_traits::NTILES_O; t++) {
        #pragma unroll
        for (int ks = 0; ks < Kernel_traits::KSLICES_PV; ks++) {
            uint32_t tOrVt0, tOrVt1;
            ldmatrix_x2_trans(tOrVt0, tOrVt1,
                pv_base + (uint32_t)(ks * 16) * Kernel_traits::ROW_BYTES
                        + (uint32_t)(t * 8) * sizeof(half));
            mma_m16n8k16(acc_o[t][0], acc_o[t][1], acc_o[t][2], acc_o[t][3],
                         tOrP[ks][0], tOrP[ks][1], tOrP[ks][2], tOrP[ks][3], tOrVt0, tOrVt1);
        }
    }
}

// Normalize O once; convert the base-2 row maximum to natural-log LSE.
template <int D, bool WRITE_L, bool FULL_TILES>
__device__ __forceinline__ void epilogue(
    const float (&acc_o)[Flash_fwd_kernel_traits<D>::NTILES_O][4],
    const Softmax& softmax,
    half* __restrict__ O_bh, int64_t stride_o_row, float* __restrict__ L_bh,
    int N_q, int q_block, int warp, int lane)
{
    using Kernel_traits = Flash_fwd_kernel_traits<D>;
    const float inv_sum_lo = 1.0f / softmax.row_sum[0];
    const float inv_sum_hi = 1.0f / softmax.row_sum[1];
    const int r_lo = q_block + warp * 16 + lane / 4;
    const int r_hi = r_lo + 8;
    const int cbase = 2 * (lane % 4);

    #pragma unroll
    for (int t = 0; t < Kernel_traits::NTILES_O; t++) {
        int col = t * 8 + cbase;
        if (FULL_TILES || r_lo < N_q) {
            half2 v = __floats2half2_rn(acc_o[t][0] * inv_sum_lo, acc_o[t][1] * inv_sum_lo);
            *reinterpret_cast<half2*>(&O_bh[r_lo * stride_o_row + col]) = v;
        }
        if (FULL_TILES || r_hi < N_q) {
            half2 v = __floats2half2_rn(acc_o[t][2] * inv_sum_hi, acc_o[t][3] * inv_sum_hi);
            *reinterpret_cast<half2*>(&O_bh[r_hi * stride_o_row + col]) = v;
        }
    }
    if constexpr (WRITE_L) {
        if (lane % 4 == 0) {
            if (FULL_TILES || r_lo < N_q) L_bh[r_lo] = softmax.row_max[0] * LN2f + logf(softmax.row_sum[0]);
            if (FULL_TILES || r_hi < N_q) L_bh[r_hi] = softmax.row_max[1] * LN2f + logf(softmax.row_sum[1]);
        }
    }
}

enum class AddressLayout { Strided, ContiguousRows, ContiguousHeads };

template <int D, AddressLayout LAYOUT, bool WRITE_L, bool CAUSAL, bool SQUARE, bool FULL_TILES, bool GQA,
          typename KV, typename... KVFields>
__global__ void __launch_bounds__(kNWarps * 32)
attention_fwd_kernel(
    const half* __restrict__ Q,
    half* __restrict__ O,
    float* __restrict__ L,    // contiguous [B, H_q, N_q]; may be nullptr when WRITE_L == false
    Strides stride_q, Strides stride_o,
    int N_q,
    int N_kv_launch,          // K/V length; ignored when SQUARE
    int H_q,                  // query heads; also used by contiguous GQA addressing
    int kv_group,             // query heads per key/value head; only read when GQA
    float softmax_scale,
    KVFields... kv_fields)    // DenseKV pointers and strides, in declaration order
{
    using Kernel_traits = Flash_fwd_kernel_traits<D>;
    const int N_kv = SQUARE ? N_q : N_kv_launch;
    KV kv_source{kv_fields...};
    constexpr bool FLAT_HEADS = LAYOUT == AddressLayout::ContiguousHeads;
    if constexpr (LAYOUT != AddressLayout::Strided) {
        stride_q.n = stride_o.n = D;
        kv_source.stride_k.n = kv_source.stride_v.n = D;
    }

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int bh = blockIdx.y;
    const int b = FLAT_HEADS ? bh / H_q : blockIdx.z;
    const int h = FLAT_HEADS ? bh % H_q : blockIdx.y;
    const int q_block = blockIdx.x * kBlockM;
    // Bottom-right causal alignment offsets each query by N_kv - N_q.
    const int causal_row0 = q_block + warp * 16 + (N_kv - N_q);

    int h_kv = h;
    if constexpr (GQA) {
        h_kv = h / kv_group;
    }

    const half* Q_bh;
    half* O_bh;
    auto kv_head = kv_source.head(b, h_kv);
    if constexpr (FLAT_HEADS) {
        const int bh_kv = GQA ? b * (H_q / kv_group) + h_kv : bh;
        Q_bh = Q + (int64_t)bh * N_q * D;
        O_bh = O + (int64_t)bh * N_q * D;
        kv_head.K = kv_source.K + (int64_t)bh_kv * N_kv * D;
        kv_head.V = kv_source.V + (int64_t)bh_kv * N_kv * D;
        kv_head.stride_k_row = kv_head.stride_v_row = D;
    } else {
        Q_bh = Q + b * stride_q.b + h * stride_q.h;
        O_bh = O + b * stride_o.b + h * stride_o.h;
    }
    float* L_bh = nullptr;
    if constexpr (WRITE_L) {
        const int64_t l_head = FLAT_HEADS ? bh : (int64_t)b * H_q + h;
        L_bh = L + l_head * N_q;
    }

    // The launcher opts into dynamic shared-memory allocations above 48 KB.
    extern __shared__ __align__(16) char smem_raw[];
    half* smem = reinterpret_cast<half*>(smem_raw);
    const uint32_t smem_base = smem_u32(smem);
    uint32_t cur_base  = smem_base;
    uint32_t next_base = smem_base + Kernel_traits::STAGE_BYTES;

    KVStager<D, FULL_TILES, typename KV::Head> kv_stager(kv_head, N_kv, tid);
    uint32_t tSrQ[Kernel_traits::KSLICES][4];

    if constexpr (kStages == 2) {
        // Q borrows the second stage, so block 0 of K/V can already be in flight.
        kv_stager.issue(cur_base, 0);
        stage_q_to_smem<D, FULL_TILES>(smem + Kernel_traits::STAGE, Q_bh, stride_q.n, q_block, N_q, tid);
        __syncthreads();
        load_q_fragments<D>(tSrQ, smem + Kernel_traits::STAGE, warp, lane);
        __syncthreads();
    } else {
        // Q shares the single K/V buffer, so it has to be consumed first.
        stage_q_to_smem<D, FULL_TILES>(smem, Q_bh, stride_q.n, q_block, N_q, tid);
        __syncthreads();
        load_q_fragments<D>(tSrQ, smem, warp, lane);
        __syncthreads();
        kv_stager.issue(cur_base, 0);
    }

    // Per-lane smem offsets for the B operands of QK^T (K rows) and PV (V rows).
    const uint32_t qk_lane_base =
        (uint32_t)((lane & 7) * Kernel_traits::LDS + ((lane < 8) ? 0 : 8)) * sizeof(half);
    const uint32_t pv_lane_base =
        (uint32_t)((kBlockN + (lane & 15)) * Kernel_traits::LDS) * sizeof(half);

    float acc_o[Kernel_traits::NTILES_O][4];
    #pragma unroll
    for (int t = 0; t < Kernel_traits::NTILES_O; t++)
        acc_o[t][0] = acc_o[t][1] = acc_o[t][2] = acc_o[t][3] = 0.0f;
    Softmax softmax;
    const float softmax_scale_log2 = softmax_scale * LOG2Ef;

    // Skip K/V tiles beyond the last query row's causal boundary.
    const int causal_end = q_block + kBlockM + (N_kv - N_q);
    const int kv_end  = (CAUSAL && causal_end < N_kv) ? causal_end : N_kv;
    const int nblocks = (kv_end + kBlockN - 1) / kBlockN;
    for (int i = 0; i < nblocks; i++) {
        const int kv = i * kBlockN;
        const bool has_next = (i + 1) < nblocks;

        // Wait for the current stage; the next stage may remain in flight.
        if constexpr (kStages == 2) {
            if (has_next) {
                kv_stager.issue(next_base, kv + kBlockN);
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        float acc_s[Kernel_traits::NTILES_S][4];
        compute_qk<D>(acc_s, tSrQ, cur_base + qk_lane_base, softmax_scale_log2);
        if constexpr (!FULL_TILES) {
            if (kv + kBlockN > N_kv) apply_mask(acc_s, kv, N_kv, lane);
        }
        if constexpr (CAUSAL) {
            if (kv + kBlockN > causal_row0) apply_causal_mask(acc_s, kv, causal_row0 + lane / 4, lane);
        }
        softmax_rescale_o(acc_s, acc_o, softmax);   // acc_s becomes P, acc_o *= scores_scale
        accumulate_pv<D>(acc_o, acc_s, cur_base + pv_lane_base);
        __syncthreads();                            // finish reads before reusing the stage

        if constexpr (kStages == 2) {
            uint32_t tmp = cur_base; cur_base = next_base; next_base = tmp;
        } else if (has_next) {
            kv_stager.issue(cur_base, kv + kBlockN);
        }
    }

    epilogue<D, WRITE_L, FULL_TILES>(acc_o, softmax, O_bh, stride_o.n, L_bh, N_q, q_block, warp, lane);
}
