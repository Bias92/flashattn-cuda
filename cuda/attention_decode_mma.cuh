// Treat the query heads sharing a KV head as rows of a small MMA tile.
// This reuses every loaded K/V tile across up to 16 query heads.
__device__ __forceinline__ uint32_t decode_saddr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void decode_copy(uint32_t dst, const half* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes));
}

template <bool TRANS>
__device__ __forceinline__ void decode_ldmatrix(uint32_t& a, uint32_t& b, uint32_t addr) {
    if constexpr (TRANS)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                     : "=r"(a), "=r"(b) : "r"(addr));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(a), "=r"(b) : "r"(addr));
}

__device__ __forceinline__ void decode_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t decode_pack(float a, float b) {
    const half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

template <int D, bool DIRECT, bool WRITE_L>
__global__ void __launch_bounds__(32) decode_grouped_mma(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, float* __restrict__ partial,
    half* __restrict__ out, float* __restrict__ lse,
    Strides qs, Strides ks, Strides vs, int hq, int hkv, int n,
    int splits, int chunk, float scale_log2) {
    constexpr int BC = 32;
    constexpr int LDS = D + 8;
    const int lane = threadIdx.x;
    const int groups = hq / hkv;
    const int head_tiles = (groups + 15) / 16;
    const int b = blockIdx.y / (hkv * head_tiles);
    const int kh = (blockIdx.y / head_tiles) % hkv;
    const int h0 = (blockIdx.y % head_tiles) * 16;
    const int row0 = lane / 4;
    const int col2 = (lane % 4) * 2;
    const int start = blockIdx.x * chunk;
    const int end = start + min(chunk, n - start);
    q += b * qs.batch + (kh * groups + h0) * qs.head;
    k += b * ks.batch + kh * ks.head;
    v += b * vs.batch + kh * vs.head;
    __shared__ __align__(16) half smem[2 * BC * LDS];
    const uint32_t base = decode_saddr(smem);
    const uint32_t kaddr = base + ((lane & 7) * LDS + (lane < 8 ? 0 : 8)) * 2;
    const uint32_t vaddr = base + (BC + (lane & 15)) * LDS * 2;
    uint32_t qreg[D / 16][4];
    #pragma unroll
    for (int s = 0; s < D / 16; ++s) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = row0 + (i % 2) * 8;
            qreg[s][i] = h0 + row < groups
                ? *reinterpret_cast<const uint32_t*>(q + row * qs.head + s * 16 + col2 + (i / 2) * 8) : 0;
        }
    }
    float o[D / 8][4] = {};
    float m[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float sum[2] = {};
    for (int begin = start; begin < end; begin += BC) {
        #pragma unroll
        for (int copy = 0; copy < BC * (D / 8) / 32; ++copy) {
            const int index = lane + copy * 32;
            const int row = index / (D / 8);
            const int col = (index % (D / 8)) * 8;
            const int r = begin + row;
            const int safe = r < end ? r : 0;
            decode_copy(base + (row * LDS + col) * 2, k + (int64_t)safe * ks.token + col, r < end ? 16 : 0);
            decode_copy(base + ((BC + row) * LDS + col) * 2, v + (int64_t)safe * vs.token + col, r < end ? 16 : 0);
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
        __syncwarp();
        float score[4][4] = {};
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            #pragma unroll
            for (int s = 0; s < D / 16; ++s) {
                uint32_t b0, b1;
                decode_ldmatrix<false>(b0, b1, kaddr + t * 8 * LDS * 2 + s * 16 * 2);
                decode_mma(score[t], qreg[s], b0, b1);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                score[t][i] = begin + t * 8 + col2 + (i % 2) < end
                    ? score[t][i] * scale_log2 : -CUDART_INF_F;
        }
        float next_m[2] = {m[0], m[1]};
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            #pragma unroll
            for (int t = 0; t < 4; ++t)
                next_m[r] = fmaxf(next_m[r], fmaxf(score[t][r * 2], score[t][r * 2 + 1]));
            next_m[r] = fmaxf(next_m[r], __shfl_xor_sync(0xffffffff, next_m[r], 1));
            next_m[r] = fmaxf(next_m[r], __shfl_xor_sync(0xffffffff, next_m[r], 2));
            const float alpha = exp2f(m[r] - next_m[r]);
            sum[r] *= alpha;
            #pragma unroll
            for (int t = 0; t < D / 8; ++t) {
                o[t][r * 2] *= alpha;
                o[t][r * 2 + 1] *= alpha;
            }
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                score[t][r * 2] = exp2f(score[t][r * 2] - next_m[r]);
                score[t][r * 2 + 1] = exp2f(score[t][r * 2 + 1] - next_m[r]);
                sum[r] += score[t][r * 2] + score[t][r * 2 + 1];
            }
            m[r] = next_m[r];
        }
        uint32_t preg[2][4];
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            preg[s][0] = decode_pack(score[2 * s][0], score[2 * s][1]);
            preg[s][1] = decode_pack(score[2 * s][2], score[2 * s][3]);
            preg[s][2] = decode_pack(score[2 * s + 1][0], score[2 * s + 1][1]);
            preg[s][3] = decode_pack(score[2 * s + 1][2], score[2 * s + 1][3]);
        }
        #pragma unroll
        for (int t = 0; t < D / 8; ++t) {
            #pragma unroll
            for (int s = 0; s < 2; ++s) {
                uint32_t b0, b1;
                decode_ldmatrix<true>(b0, b1, vaddr + s * 16 * LDS * 2 + t * 8 * 2);
                decode_mma(o[t], preg[s], b0, b1);
            }
        }
        __syncwarp();
    }
    // Defer the row-sum reduction to here; each quad shares the same maxima.
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        sum[r] += __shfl_xor_sync(0xffffffff, sum[r], 1);
        sum[r] += __shfl_xor_sync(0xffffffff, sum[r], 2);
        const int h = h0 + row0 + r * 8;
        if (h < groups) {
            const int bh = b * hq + kh * groups + h;
            float* dest = nullptr;
            if constexpr (!DIRECT) dest = partial + ((int64_t)bh * splits + blockIdx.x) * (D + 2);
            #pragma unroll
            for (int t = 0; t < D / 8; ++t) {
                const int c = t * 8 + col2;
                if constexpr (DIRECT) {
                    *reinterpret_cast<half2*>(out + (int64_t)bh * D + c) =
                        __floats2half2_rn(o[t][r * 2] / sum[r], o[t][r * 2 + 1] / sum[r]);
                } else {
                    dest[c] = o[t][r * 2];
                    dest[c + 1] = o[t][r * 2 + 1];
                }
            }
            if (lane % 4 == 0) {
                if constexpr (DIRECT) {
                    if constexpr (WRITE_L) lse[bh] = (m[r] + log2f(sum[r])) / LOG2E;
                } else {
                    dest[D] = m[r];
                    dest[D + 1] = sum[r];
                }
            }
        }
    }
}
