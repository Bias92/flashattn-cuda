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

// Turns run-time flags into template arguments, one instantiation per combination:
// with_flags(f, Flags<>{}, a, b) ends in f.template run<a, b>().
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
// run<flags...>() hands the kernel instantiation with those flags to `launch`.
template <int D, AddressLayout LAYOUT, typename KV, typename Launch, typename... KVFields>
struct KernelChoice {
    Launch& launch;
    template <bool... FLAGS>
    void run() {
        launch(attention_fwd_kernel<D, LAYOUT, FLAGS..., KV, KVFields...>,
               Flash_fwd_kernel_traits<D>::SMEM_BYTES, LAYOUT == AddressLayout::ContiguousHeads);
    }
};

template <int D, typename KV, bool ALL_MODES, typename Launch, typename... KVFields>
static void choose_layout(Launch& launch, bool packed_rows, bool contiguous_rows,
                          bool want_L, bool causal, bool square, bool full_tiles, bool gqa) {
    KernelChoice<D, AddressLayout::Strided, KV, Launch, KVFields...> general{launch};
    if constexpr (ALL_MODES) {
        if (packed_rows) {
            KernelChoice<D, AddressLayout::ContiguousHeads, KV, Launch, KVFields...> flat{launch};
            if (want_L)
                with_flags(flat, Flags</*WRITE_L=*/true, /*CAUSAL=*/true,
                                       /*SQUARE=*/true, /*FULL_TILES=*/false>{}, gqa);
            else
                with_flags(flat, Flags</*WRITE_L=*/false, /*CAUSAL=*/true,
                                       /*SQUARE=*/true, /*FULL_TILES=*/false>{}, gqa);
            return;
        }
        if constexpr (D == 64) {
            if (contiguous_rows) {
                KernelChoice<D, AddressLayout::ContiguousRows, KV, Launch, KVFields...> rows{launch};
                if (square)
                    with_flags(rows, Flags</*WRITE_L=*/false, /*CAUSAL=*/false,
                                           /*SQUARE=*/true, /*FULL_TILES=*/true>{}, gqa);
                else
                    with_flags(rows, Flags</*WRITE_L=*/false, /*CAUSAL=*/false,
                                           /*SQUARE=*/false, /*FULL_TILES=*/true>{}, gqa);
                return;
            }
        }
        with_flags(general, Flags<>{}, want_L, causal, square, full_tiles, gqa);
    } else {
        with_flags(general, Flags</*WRITE_L=*/false, /*CAUSAL=*/true, /*SQUARE=*/false>{},
                   full_tiles, gqa);
    }
}

// Picks the kernel instantiation for the run-time flags and launches it over
// (Q blocks, query heads, batch). ALL_MODES = false keeps only what inference needs (causal,
// no L, two lengths), so a K/V source used that way does not pay for the other instantiations.
// Specializations, all decided here: D; L wanted; causal; square (N_q == N_kv); full tiles
// (N_q a multiple of kBlockM and N_kv of kBlockN, so no row or column needs a guard); GQA.
template <typename KV, bool ALL_MODES, typename... KVFields>
static void launch_forward(const torch::Tensor& Q_h, const torch::Tensor& O_h, float* L_ptr,
                           int N_kv, int kv_group, bool causal, float scale, KVFields... kv_fields)
{
    const int B = Q_h.size(0), H = Q_h.size(1), N_q = Q_h.size(2), D = Q_h.size(3);
    const bool want_L = (L_ptr != nullptr);
    const bool gqa = (kv_group != 1);
    const bool square = (N_q == N_kv);
    const bool full_tiles = (N_q % kBlockM == 0) && (N_kv % kBlockN == 0);
    bool packed_rows = false;
    bool contiguous_rows = false;
    if constexpr (ALL_MODES) {
        const KV source{kv_fields...};
        // D=64 output-only full tiles benefit from constant copy strides. Keep
        // the general path for D=128 and +L, where this layout raises latency.
        contiguous_rows = D == 64 && !want_L && !causal && full_tiles
                       && Q_h.stride(2) == D && O_h.stride(2) == D
                       && source.sK.n == D && source.sV.n == D;
        const int H_kv = H / kv_group;
        const bool k_contiguous = source.sK.n == D
            && (H_kv == 1 || source.sK.h == (int64_t)N_kv * D)
            && (B == 1 || source.sK.b == (int64_t)H_kv * N_kv * D);
        const bool v_contiguous = source.sV.n == D
            && (H_kv == 1 || source.sV.h == (int64_t)N_kv * D)
            && (B == 1 || source.sV.b == (int64_t)H_kv * N_kv * D);
        // Contiguous addressing helps masked causal copies. Full tiles already hoist
        // their row addresses; keep their existing code generation and residency.
        packed_rows = causal && square && !full_tiles
                   && Q_h.is_contiguous() && O_h.is_contiguous()
                   && k_contiguous && v_contiguous && (int64_t)B * H <= 65535;
    }
    TORCH_CHECK(B <= 65535 && H <= 65535, "B and H must each be <= 65535 (grid limits)");
    TORCH_CHECK(ALL_MODES || (causal && !want_L), "this K/V source is built for causal, O-only use");

    dim3 block(kNWarps * 32);
    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto kernel, int smem_bytes, bool contiguous) {
        const dim3 grid = contiguous ? dim3((N_q + kBlockM - 1) / kBlockM, B * H)
                                     : dim3((N_q + kBlockM - 1) / kBlockM, H, B);
        if (smem_bytes > 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        }
        kernel<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(Q_h.data_ptr<at::Half>()),
            reinterpret_cast<half*>(O_h.data_ptr<at::Half>()), L_ptr,
            strides_of(Q_h), strides_of(O_h), N_q, N_kv, H, kv_group, scale, kv_fields...);
    };
    if (D == 64)
        choose_layout<64, KV, ALL_MODES, decltype(launch), KVFields...>(
            launch, packed_rows, contiguous_rows, want_L, causal, square, full_tiles, gqa);
    else
        choose_layout<128, KV, ALL_MODES, decltype(launch), KVFields...>(
            launch, packed_rows, contiguous_rows, want_L, causal, square, full_tiles, gqa);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
