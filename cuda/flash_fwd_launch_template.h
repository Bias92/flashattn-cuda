#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <utility>
#include "flash_fwd_kernel.h"

static Strides strides_of(const torch::Tensor& t) {
    return {t.stride(0), t.stride(1), t.stride(2)};
}

template <bool... B>
using Flags = std::integer_sequence<bool, B...>;

// Expand runtime flags into f.run<flags...>().
template <typename F, bool... Known>
static void with_flags(F& f, Flags<Known...>) {
    f.template run<Known...>();
}

template <typename F, bool... Known, typename... Rest>
static void with_flags(F& f, Flags<Known...>, bool flag, Rest... rest) {
    if (flag) with_flags(f, Flags<Known..., true>{}, rest...);
    else      with_flags(f, Flags<Known..., false>{}, rest...);
}

// Flag order: WRITE_L, CAUSAL, SQUARE, FULL_TILES, GQA.
template <int D, AddressLayout LAYOUT, typename KV, typename Launch, typename... KVFields>
struct KernelChoice {
    Launch& launch;
    template <bool... FLAGS>
    void run() {
        launch(attention_fwd_kernel<D, LAYOUT, FLAGS..., KV, KVFields...>,
               Flash_fwd_kernel_traits<D>::SMEM_BYTES, LAYOUT == AddressLayout::ContiguousHeads);
    }
};

struct ForwardLaunch {
    const torch::Tensor& Q;
    const torch::Tensor& O;
    float* L;
    int N_kv;
    int kv_group;
    bool causal;
    float scale;

    template <typename Dispatch, typename... KVFields>
    void run(KVFields... kv_fields) const {
        const int B = Q.size(0), H = Q.size(1), N_q = Q.size(2), D = Q.size(3);
        TORCH_CHECK(B <= 65535 && H <= 65535, "B and H must each be <= 65535 (grid limits)");
        TORCH_CHECK(D == 64 || D == 128, "Head dimension must be 64 or 128, got ", D);

        const dim3 block(kNWarps * 32);
        const auto stream = at::cuda::getCurrentCUDAStream();
        auto launch = [&](auto kernel, int smem_bytes, bool flatten_heads) {
            const dim3 grid = flatten_heads ? dim3((N_q + kBlockM - 1) / kBlockM, B * H)
                                           : dim3((N_q + kBlockM - 1) / kBlockM, H, B);
            if (smem_bytes > 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
            }
            kernel<<<grid, block, smem_bytes, stream>>>(
                reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
                reinterpret_cast<half*>(O.data_ptr<at::Half>()), L,
                strides_of(Q), strides_of(O), N_q, N_kv, H, kv_group, scale, kv_fields...);
        };
        if (D == 64)
            Dispatch::template run<64>(launch, *this, kv_fields...);
        else
            Dispatch::template run<128>(launch, *this, kv_fields...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
};
