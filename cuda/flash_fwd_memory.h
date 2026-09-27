#pragma once

#include "kernel_traits.h"
#include "attention_forward_ops.cuh"

// Element strides of a [B, H, N, D] tensor whose last dimension is contiguous.
struct Strides {
    int64_t b, h, n;
};

// head() selects a batch/KV head; rows() resolves its K/V row pointers.
// Source fields are separate kernel arguments, in declaration order.

// Ordinary [B, H_kv, N_kv, D] tensors: row r sits at base + r * row stride.
struct DenseKV {
    const half* K;
    const half* V;
    Strides sK, sV;

    struct Head {
        static constexpr bool LINEAR = true;   // the address is linear in the row index
        const half* K;
        const half* V;
        int64_t k_row, v_row;
        __device__ __forceinline__ void rows(int row, const half*& k, const half*& v) const {
            k = K + row * k_row;
            v = V + row * v_row;
        }
    };
    __device__ __forceinline__ Head head(int b, int h_kv) const {
        return {K + b * sK.b + h_kv * sK.h, V + b * sV.b + h_kv * sV.h, sK.n, sV.n};
    }
};

// A vLLM-style paged cache, K and V each [pages, page_size, H_kv, D] with the same strides:
// row r is slot r & (page_size - 1) of page table[r >> page_shift]. One sequence per launch.
struct PagedKV {
    const half* K;
    const half* V;
    const int32_t* table;       // page ids of this sequence, in order
    int page_shift;             // log2(page_size)
    int64_t page, slot, hd;     // element strides of the cache: page, slot in page, head

    struct Head {
        static constexpr bool LINEAR = false;
        const half* K;
        const half* V;
        const int32_t* table;
        int page_shift;
        int64_t page, slot;
        __device__ __forceinline__ void rows(int row, const half*& k, const half*& v) const {
            const int64_t at = table[row >> page_shift] * page
                             + (row & ((1 << page_shift) - 1)) * slot;
            k = K + at;
            v = V + at;
        }
    };
    __device__ __forceinline__ Head head(int /*b*/, int h_kv) const {
        return {K + h_kv * hd, V + h_kv * hd, table, page_shift, page, slot};
    }
};

// Each stage stores kBlockN K rows followed by kBlockN V rows.
// Each thread copies a 16-byte column chunk every ROWSTEP rows of K and V.
template <int D, bool FULL_TILES, typename KVHead>
struct KVStager {
    using Kernel_traits = Flash_fwd_kernel_traits<D>;
    static constexpr int THREADS   = kNWarps * 32;
    static constexpr int COLCHUNKS = D / 8;                 // 16-byte chunks per row
    static constexpr int ROWSTEP   = THREADS / COLCHUNKS;   // rows between a thread's copies
    static_assert(THREADS % COLCHUNKS == 0 && kBlockN % ROWSTEP == 0,
                  "the threads of a block must tile the chunks of a K/V block exactly");

    KVHead src;
    int N_kv;
    int r0, cc;
    uint32_t k0_off;

    __device__ __forceinline__ KVStager(KVHead src_, int N_kv_, int tid) : src(src_), N_kv(N_kv_)
    {
        r0 = tid / COLCHUNKS;
        cc = (tid % COLCHUNKS) * 8;
        k0_off = (uint32_t)(r0 * Kernel_traits::LDS + cc) * sizeof(half);
    }

    // Copy rows [kv, kv + kBlockN) into the stage at byte address sbase.
    __device__ __forceinline__ void issue(uint32_t sbase, int kv) const {
        constexpr uint32_t ROWSTEP_OFF = (uint32_t)(ROWSTEP * Kernel_traits::LDS) * sizeof(half);
        constexpr uint32_t VBLOCK_OFF  = (uint32_t)(kBlockN * Kernel_traits::LDS) * sizeof(half);
        const int row0 = kv + r0;
        if constexpr (FULL_TILES && KVHead::LINEAR) {
            // Full linear tiles use fixed offsets from the first row pointer.
            const half* k = src.K + row0 * src.k_row + cc;
            const half* v = src.V + row0 * src.v_row + cc;
            #pragma unroll
            for (int i = 0; i < kBlockN / ROWSTEP; i++) {
                const uint32_t saddr = sbase + k0_off + (uint32_t)i * ROWSTEP_OFF;
                cp_async_16(saddr, k + (i * ROWSTEP) * src.k_row, 16);
                cp_async_16(saddr + VBLOCK_OFF, v + (i * ROWSTEP) * src.v_row, 16);
            }
        } else {
            // Past N_kv use a valid row-0 pointer and zero-fill the destination (src_size=0).
            #pragma unroll
            for (int i = 0; i < kBlockN / ROWSTEP; i++) {
                const uint32_t saddr = sbase + k0_off + (uint32_t)i * ROWSTEP_OFF;
                const int row = row0 + i * ROWSTEP;
                const bool valid = FULL_TILES || row < N_kv;
                const half* k;
                const half* v;
                src.rows(valid ? row : 0, k, v);
                cp_async_16(saddr, k + cc, valid ? 16 : 0);
                cp_async_16(saddr + VBLOCK_OFF, v + cc, valid ? 16 : 0);
            }
        }
        cp_async_commit();
    }
};

// Stage Q in shared memory before loading the per-warp register fragments.
template <int D, bool FULL_TILES>
__device__ __forceinline__ void stage_q_to_smem(
    half* sQ_raw, const half* __restrict__ Q_bh, int64_t q_row, int q_block, int N, int tid)
{
    using Kernel_traits = Flash_fwd_kernel_traits<D>;
    half (*sQ)[Kernel_traits::LDS] = reinterpret_cast<half(*)[Kernel_traits::LDS]>(sQ_raw);
    constexpr int total_h2 = kBlockM * D / 2;
    for (int i = tid; i < total_h2; i += kNWarps * 32) {
        int flat = i * 2;
        int r = flat / D, c = flat % D;
        if constexpr (FULL_TILES) {
            // N % kBlockM == 0 guarantees every Q row in this block is valid.
            *reinterpret_cast<half2*>(&sQ[r][c]) =
                *reinterpret_cast<const half2*>(&Q_bh[(q_block + r) * q_row + c]);
        } else {
            int gr = q_block + r;
            half2 val = (gr < N)
                ? *reinterpret_cast<const half2*>(&Q_bh[gr * q_row + c])
                : __float2half2_rn(0.0f);
            *reinterpret_cast<half2*>(&sQ[r][c]) = val;
        }
    }
}

template <int D>
__device__ __forceinline__ void load_q_fragments(
    uint32_t (&tSrQ)[Flash_fwd_kernel_traits<D>::KSLICES][4], const half* sQ_raw, int warp, int lane)
{
    using Kernel_traits = Flash_fwd_kernel_traits<D>;
    const half (*sQ)[Kernel_traits::LDS] = reinterpret_cast<const half(*)[Kernel_traits::LDS]>(sQ_raw);
    int r = warp * 16 + (lane % 16);
    int kbase = (lane < 16) ? 0 : 8;
    #pragma unroll
    for (int ks = 0; ks < Kernel_traits::KSLICES; ks++) {
        uint32_t addr = smem_u32(&sQ[r][ks * 16 + kbase]);
        ldmatrix_x4(tSrQ[ks][0], tSrQ[ks][1], tSrQ[ks][2], tSrQ[ks][3], addr);
    }
}
