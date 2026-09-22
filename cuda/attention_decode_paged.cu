// Scratch CUDA attention over vLLM-style paged KV. No KV gather/copy.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr float LOG2E = 1.4426950408889634f;
struct Strides { int64_t batch, head, token; };
struct CacheStrides { int64_t block, token, head; };

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

// Native paged-cache variant of the validated dense grouped-MMA kernel.
// One query token per request; request lengths and page tables stay on GPU.
template <int D>
__global__ void __launch_bounds__(32) paged_decode_split(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, const int* __restrict__ table,
    const int* __restrict__ lengths, const int* __restrict__ qstarts,
    float* __restrict__ partial, Strides qs, CacheStrides ks, CacheStrides vs,
    int hq, int hkv, int table_stride, int page_shift,
    int splits, int chunk, float scale_log2) {
    constexpr int BC = 32;
    constexpr int LDS = D + 8;
    const int lane = threadIdx.x;
    const int groups = hq / hkv;
    const int head_tiles = (groups + 15) / 16;
    const int b = blockIdx.y / (hkv * head_tiles);
    const int kh = (blockIdx.y / head_tiles) % hkv;
    const int h0 = (blockIdx.y % head_tiles) * 16;
    const int qi = qstarts[b];
    if (qstarts[b + 1] == qi) return;  // CUDA Graph padding request.
    const int n = lengths[b];
    const int row0 = lane / 4;
    const int col2 = (lane % 4) * 2;
    const int start = blockIdx.x * chunk;
    const int end = min(start + chunk, n);
    q += (int64_t)qi * qs.batch + (kh * groups + h0) * qs.head;
    k += kh * ks.head;
    v += kh * vs.head;
    table += (int64_t)b * table_stride;
    const int page_mask = (1 << page_shift) - 1;
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
            const bool valid = r < end;
            // Do not read an unused page-table entry on the last tile.
            const int page = valid ? table[r >> page_shift] : 0;
            const int offset = valid ? (r & page_mask) : 0;
            decode_copy(base + (row * LDS + col) * 2,
                        k + (int64_t)page * ks.block + offset * ks.token + col, valid ? 16 : 0);
            decode_copy(base + ((BC + row) * LDS + col) * 2,
                        v + (int64_t)page * vs.block + offset * vs.token + col, valid ? 16 : 0);
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
            float* dest = partial + ((int64_t)bh * splits + blockIdx.x) * (D + 2);
            #pragma unroll
            for (int t = 0; t < D / 8; ++t) {
                const int c = t * 8 + col2;
                dest[c] = o[t][r * 2];
                dest[c + 1] = o[t][r * 2 + 1];
            }
            if (lane % 4 == 0) {
                dest[D] = m[r];
                dest[D + 1] = sum[r];
            }
        }
    }
}

template <int D>
__global__ void paged_decode_merge(
    const float* __restrict__ partial, half* __restrict__ out,
    const int* __restrict__ lengths, const int* __restrict__ qstarts,
    int hq, int splits, int chunk, int64_t out_token, int64_t out_head) {
    const int bh = blockIdx.x;
    const int b = bh / hq;
    const int qi = qstarts[b];
    if (qstarts[b + 1] == qi) return;
    const int n = lengths[b];
    const int active = min(splits, (n + chunk - 1) / chunk);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float* base = partial + (int64_t)bh * splits * (D + 2);
    __shared__ float maxima[4];
    __shared__ float accum[4][D + 1];
    float m = -CUDART_INF_F;
    for (int s = threadIdx.x; s < active; s += 128)
        m = fmaxf(m, base[(int64_t)s * (D + 2) + D]);
    #pragma unroll
    for (int d = 16; d; d >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, d));
    if (lane == 0) maxima[warp] = m;
    __syncthreads();
    m = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
    float sum = 0.f;
    float value[D / 32] = {};
    for (int s = warp; s < active; s += 4) {
        const float* p = base + (int64_t)s * (D + 2);
        const float weight = exp2f(p[D] - m);
        sum += p[D + 1] * weight;
        #pragma unroll
        for (int i = 0; i < D / 32; ++i)
            value[i] = fmaf(weight, p[lane + i * 32], value[i]);
    }
    if (lane == 0) accum[warp][D] = sum;
    #pragma unroll
    for (int i = 0; i < D / 32; ++i) accum[warp][lane + i * 32] = value[i];
    __syncthreads();
    if (warp == 0) {
        sum = accum[0][D] + accum[1][D] + accum[2][D] + accum[3][D];
        half* dest = out + (int64_t)qi * out_token + (bh % hq) * out_head;
        #pragma unroll
        for (int i = 0; i < D / 32; ++i) {
            const int c = lane + i * 32;
            const float v = accum[0][c] + accum[1][c] + accum[2][c] + accum[3][c];
            dest[c] = __float2half_rn(sum > 0.f ? v / sum : 0.f);
        }
    }
}

void paged_out(torch::Tensor q, torch::Tensor k, torch::Tensor v,
               torch::Tensor table, torch::Tensor lengths, torch::Tensor qstarts,
               torch::Tensor out, torch::Tensor workspace, int max_seq_len,
               double scale, int chunk) {
    for (const auto& t : {q, k, v, table, lengths, qstarts, out, workspace})
        TORCH_CHECK(t.is_cuda() && t.device() == q.device(), "All tensors must share a CUDA device");
    for (const auto& t : {q, k, v, out})
        TORCH_CHECK(t.scalar_type() == at::kHalf, "Q/K/V/output must be FP16");
    for (const auto& t : {table, lengths, qstarts})
        TORCH_CHECK(t.scalar_type() == at::kInt, "Metadata must be int32");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.sizes() == k.sizes(),
                "Expected Q [T,Hq,D], K/V [pages,page_size,Hkv,D]");
    TORCH_CHECK(out.sizes() == q.sizes(), "Output shape must match Q");
    const int hq = q.size(1), hkv = k.size(2), d = q.size(2);
    TORCH_CHECK(q.size(0) > 0 && hq > 0 && hkv > 0 && hq % hkv == 0,
                "Invalid token/head counts");
    TORCH_CHECK((d == 64 || d == 128) && k.size(3) == d, "D must be 64 or 128");
    const int page_size = k.size(1);
    TORCH_CHECK(k.size(0) > 0 &&
                (page_size == 16 || page_size == 32 || page_size == 64 || page_size == 128),
                "Supported page sizes: 16,32,64,128");
    TORCH_CHECK(lengths.dim() == 1 && qstarts.dim() == 1 && table.dim() == 2,
                "Invalid metadata rank");
    const int batch = lengths.numel();
    TORCH_CHECK(batch > 0 && (int64_t)batch * hq <= 65535 &&
                qstarts.numel() == batch + 1 && table.size(0) == batch,
                "Invalid metadata shapes/grid size");
    TORCH_CHECK(lengths.is_contiguous() && qstarts.is_contiguous() && table.stride(1) == 1,
                "Metadata must have contiguous rows");
    TORCH_CHECK(chunk == 64 || chunk == 128 || chunk == 256 || chunk == 512,
                "chunk must be 64,128,256,512");
    TORCH_CHECK(max_seq_len > 0 && max_seq_len <= table.size(1) * page_size &&
                max_seq_len < (1 << 28), "Invalid max_seq_len");
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(static_cast<float>(scale) * LOG2E),
                "Invalid scale");
    for (const auto& t : {q, k, v, out}) {
        TORCH_CHECK(t.stride(t.dim() - 1) == 1 &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                    "Aligned contiguous D dimension required");
        for (int i = 0; i < t.dim() - 1; ++i)
            TORCH_CHECK(t.stride(i) > 0 && t.stride(i) % 8 == 0,
                        "Outer strides must be positive multiples of 8");
    }
    const int splits = (max_seq_len + chunk - 1) / chunk;
    TORCH_CHECK(workspace.scalar_type() == at::kFloat && workspace.is_contiguous() &&
                workspace.numel() >= (int64_t)batch * hq * splits * (d + 2),
                "Workspace too small or invalid");
    const c10::cuda::CUDAGuard guard(q.device());
    const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    const auto qp = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    const auto kp = reinterpret_cast<const half*>(k.data_ptr<at::Half>());
    const auto vp = reinterpret_cast<const half*>(v.data_ptr<at::Half>());
    const auto op = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    const Strides qs{q.stride(0), q.stride(1), 0};
    const CacheStrides ks{k.stride(0), k.stride(1), k.stride(2)};
    const CacheStrides vs{v.stride(0), v.stride(1), v.stride(2)};
    int shift = 0;
    while ((1 << shift) < page_size) ++shift;
    const int blocks = batch * hkv * ((hq / hkv + 15) / 16);
    #define LAUNCH(D) do { \
        paged_decode_split<D><<<dim3(splits, blocks), 32, 0, stream>>>( \
            qp, kp, vp, table.data_ptr<int>(), lengths.data_ptr<int>(), qstarts.data_ptr<int>(), \
            workspace.data_ptr<float>(), qs, ks, vs, hq, hkv, table.stride(0), shift, \
            splits, chunk, static_cast<float>(scale) * LOG2E); \
        paged_decode_merge<D><<<batch * hq, 128, 0, stream>>>( \
            workspace.data_ptr<float>(), op, lengths.data_ptr<int>(), qstarts.data_ptr<int>(), \
            hq, splits, chunk, out.stride(0), out.stride(1)); \
    } while (0)
    if (d == 64) LAUNCH(64); else LAUNCH(128);
    #undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("out", &paged_out, "Paged single-token attention; trusted device metadata, no allocations");
}

