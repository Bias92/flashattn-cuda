// Single-token attention over a dense, valid KV cache. Inference only.
// A CTA computes one query head / KV split, then a second kernel merges
// unnormalized FP32 outputs using their maxima and softmax denominators.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace {
constexpr int THREADS = 128;
constexpr float LOG2E = 1.4426950408889634f;

struct Strides {
    int64_t batch, head, token;
};

__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int d = 16; d; d >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, d));
    return x;
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int d = 16; d; d >>= 1) x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
}

#include "attention_decode_mma.cuh"

__device__ __forceinline__ void load8(const half* p, float (&out)[8]) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    const unsigned words[4] = {v.x, v.y, v.z, v.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(*reinterpret_cast<const half2*>(&words[i]));
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

template <int D, int CHUNK, bool DIRECT, bool WRITE_L>
__global__ void __launch_bounds__(THREADS) decode_split(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, float* __restrict__ partial,
    half* __restrict__ out, float* __restrict__ lse,
    Strides qs, Strides ks, Strides vs, int hq, int hkv, int n,
    int splits, float scale_log2) {
    constexpr int WIDTH = D / 8;
    constexpr int ROWS = THREADS / WIDTH;
    const int tid = threadIdx.x;
    const int col = (tid % WIDTH) * 8;
    const int row = tid / WIDTH;
    const int bh = blockIdx.y;
    const int b = bh / hq;
    const int h = bh % hq;
    const int kh = h / (hq / hkv);
    const int begin = blockIdx.x * CHUNK;
    const int count = min(CHUNK, n - begin);
    q += b * qs.batch + h * qs.head;
    k += b * ks.batch + kh * ks.head + (int64_t)begin * ks.token;
    v += b * vs.batch + kh * vs.head + (int64_t)begin * vs.token;

    __shared__ float scores[CHUNK];
    __shared__ float scratch[ROWS * D];
    float qf[8];
    load8(q + col, qf);
    float local_max = -CUDART_INF_F;
    #pragma unroll 1
    for (int tile = 0; tile < count; tile += ROWS) {
        const int r = tile + row;
        float kf[8] = {};
        if (r < count) load8(k + (int64_t)r * ks.token + col, kf);
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) s = fmaf(qf[i], kf[i], s);
        #pragma unroll
        for (int d = WIDTH / 2; d; d >>= 1)
            s += __shfl_xor_sync(0xffffffff, s, d, WIDTH);
        s = r < count ? s * scale_log2 : -CUDART_INF_F;
        if (tid % WIDTH == 0 && r < count) scores[r] = s;
        local_max = fmaxf(local_max, s);
    }
    float m = warp_max(local_max);
    if (tid % 32 == 0) scratch[tid / 32] = m;
    __syncthreads();
    m = fmaxf(fmaxf(scratch[0], scratch[1]), fmaxf(scratch[2], scratch[3]));
    float sum = 0.f;
    for (int r = tid; r < count; r += THREADS) {
        const float p = exp2f(scores[r] - m);
        scores[r] = p;
        sum += p;
    }
    // All threads must consume the maximum before scratch is reused.
    __syncthreads();
    sum = warp_sum(sum);
    if (tid % 32 == 0) scratch[tid / 32] = sum;
    __syncthreads();
    sum = scratch[0] + scratch[1] + scratch[2] + scratch[3];
    float o[8] = {};
    #pragma unroll 1
    for (int r = row; r < count; r += ROWS) {
        const float p = scores[r];
        float vf[8];
        load8(v + (int64_t)r * vs.token + col, vf);
        #pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = fmaf(p, vf[i], o[i]);
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; ++i) scratch[row * D + col + i] = o[i];
    __syncthreads();
    float* dest = nullptr;
    if constexpr (!DIRECT) dest = partial + ((int64_t)bh * splits + blockIdx.x) * (D + 2);
    if (tid < D) {
        float value = 0.f;
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) value += scratch[r * D + tid];
        if constexpr (DIRECT) out[(int64_t)bh * D + tid] = __float2half_rn(value / sum);
        else dest[tid] = value;
    }
    if (tid == 0) {
        if constexpr (DIRECT) {
            if constexpr (WRITE_L) lse[bh] = (m + log2f(sum)) / LOG2E;
        } else {
            dest[D] = m;
            dest[D + 1] = sum;
        }
    }
}

template <int D, bool WRITE_L>
__global__ void decode_merge(const float* __restrict__ partial,
                             half* __restrict__ out, float* __restrict__ lse,
                             int splits) {
    const int bh = blockIdx.x;
    const int lane = threadIdx.x;
    const float* base = partial + (int64_t)bh * splits * (D + 2);
    float m = -CUDART_INF_F;
    for (int s = lane; s < splits; s += 32) m = fmaxf(m, base[(int64_t)s * (D + 2) + D]);
    m = warp_max(m);
    float sum = 0.f;
    float o[D / 32] = {};
    for (int s = 0; s < splits; ++s) {
        const float* p = base + (int64_t)s * (D + 2);
        const float weight = exp2f(p[D] - m);
        sum += p[D + 1] * weight;
        #pragma unroll
        for (int i = 0; i < D / 32; ++i) o[i] = fmaf(weight, p[lane + i * 32], o[i]);
    }
    #pragma unroll
    for (int i = 0; i < D / 32; ++i)
        out[(int64_t)bh * D + lane + i * 32] = __float2half_rn(o[i] / sum);
    if constexpr (WRITE_L) {
        if (lane == 0) lse[bh] = (m + log2f(sum)) / LOG2E;
    }
}

template <int D, bool WRITE_L>
__global__ void decode_merge_parallel(const float* __restrict__ partial,
    half* __restrict__ out, float* __restrict__ lse, int splits) {
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const float* base = partial + (int64_t)bh * splits * (D + 2);
    __shared__ float maxima[4];
    __shared__ float accum[4][D + 1];
    float m = -CUDART_INF_F;
    for (int s = tid; s < splits; s += 128) m = fmaxf(m, base[(int64_t)s * (D + 2) + D]);
    m = warp_max(m);
    if (lane == 0) maxima[warp] = m;
    __syncthreads();
    m = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
    float sum = 0.f;
    float o[D / 32] = {};
    for (int s = warp; s < splits; s += 4) {
        const float* p = base + (int64_t)s * (D + 2);
        const float weight = exp2f(p[D] - m);
        sum += p[D + 1] * weight;
        #pragma unroll
        for (int i = 0; i < D / 32; ++i) o[i] = fmaf(weight, p[lane + i * 32], o[i]);
    }
    if (lane == 0) accum[warp][D] = sum;
    #pragma unroll
    for (int i = 0; i < D / 32; ++i) accum[warp][lane + i * 32] = o[i];
    __syncthreads();
    if (warp == 0) {
        sum = accum[0][D] + accum[1][D] + accum[2][D] + accum[3][D];
        #pragma unroll
        for (int i = 0; i < D / 32; ++i) {
            const int c = lane + i * 32;
            const float value = accum[0][c] + accum[1][c] + accum[2][c] + accum[3][c];
            out[(int64_t)bh * D + c] = __float2half_rn(value / sum);
        }
        if constexpr (WRITE_L) {
            if (lane == 0) lse[bh] = (m + log2f(sum)) / LOG2E;
        }
    }
}

template <int D, bool WRITE_L>
void launch_merge(const float* partial, half* out, float* lse, int splits, int bh, cudaStream_t stream) {
    if (splits >= 8) decode_merge_parallel<D, WRITE_L><<<bh, 128, 0, stream>>>(partial, out, lse, splits);
    else decode_merge<D, WRITE_L><<<bh, 32, 0, stream>>>(partial, out, lse, splits);
}

bool aligned_rows(const torch::Tensor& x) {
    if (x.stride(3) != 1 || reinterpret_cast<uintptr_t>(x.data_ptr()) % 16) return false;
    for (int i = 0; i < 3; ++i)
        if (x.size(i) > 1 && x.stride(i) % 8 != 0) return false;
    return true;
}

template <int D, int CHUNK, bool WRITE_L>
void launch(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            torch::Tensor& out, torch::Tensor& lse, float scale, bool grouped, cudaStream_t stream) {
    const int bh = q.size(0) * q.size(1);
    const int n = k.size(2);
    const int splits = (n - 1) / CHUNK + 1;
    auto strides = [](const torch::Tensor& x) {
        return Strides{x.stride(0), x.stride(1), x.stride(2)};
    };
    const half* qp = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    const half* kp = reinterpret_cast<const half*>(k.data_ptr<at::Half>());
    const half* vp = reinterpret_cast<const half*>(v.data_ptr<at::Half>());
    half* op = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    float* lp = nullptr;
    if constexpr (WRITE_L) lp = lse.data_ptr<float>();
    if (grouped) {
        const int groups = q.size(1) / k.size(1);
        const int blocks = q.size(0) * k.size(1) * ((groups + 15) / 16);
        if (splits == 1) {
            decode_grouped_mma<D, true, WRITE_L><<<dim3(1, blocks), 32, 0, stream>>>(
                qp, kp, vp, nullptr, op, lp, strides(q), strides(k), strides(v),
                q.size(1), k.size(1), n, splits, CHUNK, scale * LOG2E);
        } else {
            auto tmp = torch::empty({bh, splits, D + 2}, q.options().dtype(torch::kFloat));
            decode_grouped_mma<D, false, WRITE_L><<<dim3(splits, blocks), 32, 0, stream>>>(
                qp, kp, vp, tmp.data_ptr<float>(), op, lp, strides(q), strides(k), strides(v),
                q.size(1), k.size(1), n, splits, CHUNK, scale * LOG2E);
            launch_merge<D, WRITE_L>(tmp.data_ptr<float>(), op, lp, splits, bh, stream);
        }
    } else if (splits == 1) {
        decode_split<D, CHUNK, true, WRITE_L><<<dim3(1, bh), THREADS, 0, stream>>>(
            qp, kp, vp, nullptr, op, lp, strides(q), strides(k), strides(v),
            q.size(1), k.size(1), n, splits, scale * LOG2E);
    } else {
        auto tmp = torch::empty({bh, splits, D + 2}, q.options().dtype(torch::kFloat));
        decode_split<D, CHUNK, false, WRITE_L><<<dim3(splits, bh), THREADS, 0, stream>>>(
            qp, kp, vp, tmp.data_ptr<float>(), op, lp, strides(q), strides(k), strides(v),
            q.size(1), k.size(1), n, splits, scale * LOG2E);
        launch_merge<D, WRITE_L>(tmp.data_ptr<float>(), op, lp, splits, bh, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::pair<torch::Tensor, torch::Tensor> decode_impl(torch::Tensor q, torch::Tensor k,
    torch::Tensor v, std::optional<double> scale, int chunk, bool write_l, bool grouped) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "Q/K/V must share a device");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "Q/K/V must be rank 4");
    TORCH_CHECK(q.scalar_type() == torch::kHalf && k.scalar_type() == torch::kHalf &&
                v.scalar_type() == torch::kHalf, "Decode requires FP16 Q/K/V");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) == 1,
                "Q must have shape [B>0,Hq>0,1,D]");
    TORCH_CHECK(q.size(3) == 64 || q.size(3) == 128, "D must be 64 or 128");
    TORCH_CHECK(k.sizes() == v.sizes() && k.size(0) == q.size(0) && k.size(3) == q.size(3),
                "K/V must have matching [B,Hkv,N,D] shapes");
    TORCH_CHECK(k.size(1) > 0 && q.size(1) % k.size(1) == 0, "Hq must be divisible by Hkv");
    TORCH_CHECK(k.size(2) > 0 && k.size(2) <= std::numeric_limits<int>::max(), "Invalid KV length");
    TORCH_CHECK(q.size(0) * q.size(1) <= 65535, "B*Hq exceeds grid limit");
    const float s = static_cast<float>(scale.value_or(1. / std::sqrt(double(q.size(3)))));
    TORCH_CHECK(std::isfinite(s), "scale must be finite");
    if (chunk == 0) chunk = grouped ? (k.size(2) <= 2048 ? 64 : 128) : 256;
    TORCH_CHECK(chunk == 64 || chunk == 128 || chunk == 256 || chunk == 512 || chunk == 1024,
                "split_size must be 0, 64, 128, 256, 512, or 1024");
    const c10::cuda::CUDAGuard guard(q.device());
    // Keep cache views strided when 16-byte vector loads are safe. clone also
    // fixes an unaligned storage offset on an otherwise contiguous tensor.
    if (!aligned_rows(q)) q = q.clone(torch::MemoryFormat::Contiguous);
    if (!aligned_rows(k)) k = k.clone(torch::MemoryFormat::Contiguous);
    if (!aligned_rows(v)) v = v.clone(torch::MemoryFormat::Contiguous);
    auto out = torch::empty(q.sizes(), q.options());
    torch::Tensor lse;
    if (write_l) lse = torch::empty({q.size(0), q.size(1), 1}, q.options().dtype(torch::kFloat));
    const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    auto dispatch = [&](auto d, auto wl) {
        constexpr int D = decltype(d)::value;
        constexpr bool WL = decltype(wl)::value;
        switch (chunk) {
            case 64: launch<D, 64, WL>(q, k, v, out, lse, s, grouped, stream); break;
            case 128: launch<D, 128, WL>(q, k, v, out, lse, s, grouped, stream); break;
            case 256: launch<D, 256, WL>(q, k, v, out, lse, s, grouped, stream); break;
            case 512: launch<D, 512, WL>(q, k, v, out, lse, s, grouped, stream); break;
            case 1024: launch<D, 1024, WL>(q, k, v, out, lse, s, grouped, stream); break;
        }
    };
    #define DISPATCH(D) do { \
        if (write_l) dispatch(std::integral_constant<int, D>{}, std::true_type{}); \
        else dispatch(std::integral_constant<int, D>{}, std::false_type{}); \
    } while (0)
    if (q.size(3) == 64) DISPATCH(64); else DISPATCH(128);
    #undef DISPATCH
    return {out, lse};
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_only", [](torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              std::optional<double> scale, int split_size) {
        return decode_impl(q, k, v, scale, split_size, false, true).first;
    }, "Single-token dense-cache attention (O)", pybind11::arg("Q"), pybind11::arg("K"),
       pybind11::arg("V"), pybind11::arg("scale") = pybind11::none(), pybind11::arg("split_size") = 0);
    m.def("forward", [](torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         std::optional<double> scale, int split_size) {
        auto [o, l] = decode_impl(q, k, v, scale, split_size, true, true);
        return std::vector<torch::Tensor>{o, l};
    }, "Single-token dense-cache attention (O, L)", pybind11::arg("Q"), pybind11::arg("K"),
       pybind11::arg("V"), pybind11::arg("scale") = pybind11::none(), pybind11::arg("split_size") = 0);
    m.def("scalar_only", [](torch::Tensor q, torch::Tensor k, torch::Tensor v,
                            std::optional<double> scale, int split_size) {
        return decode_impl(q, k, v, scale, split_size, false, false).first;
    }, "Scalar decode ablation", pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
       pybind11::arg("scale") = pybind11::none(), pybind11::arg("split_size") = 0);
}
