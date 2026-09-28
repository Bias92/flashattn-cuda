// PyTorch entry points and tensor validation. The forward loop is in flash_fwd_kernel.h.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>
#include <vector>

#include "flash_fwd_dense_dispatch.h"

// Byte alignment required by half2 loads/stores (Q/O) and cp.async copies (K/V).
constexpr int64_t QO_ROW_ALIGN = 4;
constexpr int64_t KV_ROW_ALIGN = 16;

// True when every row of t starts on an `align`-byte boundary.
static bool rows_aligned(const torch::Tensor& t, int64_t align) {
    const int64_t elems = align / (int64_t)sizeof(at::Half);
    bool ok = reinterpret_cast<uintptr_t>(t.data_ptr()) % align == 0;
    for (int d = 0; d < 3; d++) {
        ok = ok && (t.size(d) == 1 || t.stride(d) % elems == 0);
    }
    return ok;
}

// Sufficient overlap check: each stride must clear all smaller-stride dimensions.
static bool no_internal_overlap(const torch::Tensor& t) {
    std::vector<int> dims;
    for (int d = 0; d < t.dim(); d++) {
        if (t.size(d) > 1) dims.push_back(d);
    }
    std::sort(dims.begin(), dims.end(), [&](int a, int b) { return t.stride(a) < t.stride(b); });
    int64_t span = 1;   // elements covered by the dimensions seen so far
    for (int d : dims) {
        if (t.stride(d) < span) return false;
        span += (t.size(d) - 1) * t.stride(d);
    }
    return true;
}

// Byte range [lo, hi) that the elements of t fall in.
static std::pair<uintptr_t, uintptr_t> byte_span(const torch::Tensor& t) {
    int64_t last = 0;
    for (int d = 0; d < t.dim(); d++) {
        last += (t.size(d) - 1) * t.stride(d);
    }
    const auto lo = reinterpret_cast<uintptr_t>(t.data_ptr());
    return {lo, lo + (uintptr_t)(last + 1) * t.element_size()};
}

// Conservative: compares the enclosing byte ranges, so interleaved but disjoint views
// (two slots of one fused buffer) count as overlapping.
static bool spans_overlap(const torch::Tensor& a, const torch::Tensor& b) {
    const auto [alo, ahi] = byte_span(a);
    const auto [blo, bhi] = byte_span(b);
    return alo < bhi && blo < ahi;
}

// fp16 rows the kernel can read in place: contiguous last dimension, aligned row starts.
// Anything else is copied into a fresh contiguous tensor.
static torch::Tensor as_half_rows(const torch::Tensor& t, int64_t align) {
    auto h = t.to(torch::kHalf);
    return (h.stride(3) == 1 && rows_aligned(h, align))
        ? h : h.clone(at::MemoryFormat::Contiguous);
}

static torch::Tensor output_for(const torch::Tensor& Q_h, const std::optional<torch::Tensor>& out,
                                std::initializer_list<torch::Tensor> inputs) {
    if (!out.has_value()) {
        return torch::empty(Q_h.sizes(), Q_h.options());
    }
    const torch::Tensor& O_h = *out;
    TORCH_CHECK(O_h.is_cuda() && O_h.device() == Q_h.device() && O_h.scalar_type() == torch::kHalf,
                "out must be an fp16 CUDA tensor on the device of Q");
    TORCH_CHECK(O_h.sizes() == Q_h.sizes() && O_h.stride(3) == 1 && rows_aligned(O_h, QO_ROW_ALIGN),
                "out must have the shape of Q, a contiguous last dimension and rows "
                "that start on a ", QO_ROW_ALIGN, "-byte boundary");
    // Reject aliasing: blocks read inputs and write output concurrently.
    TORCH_CHECK(no_internal_overlap(O_h), "out has elements that share memory");
    for (const auto& t : inputs) {
        TORCH_CHECK(!spans_overlap(O_h, t), "the address range of out must not overlap an input");
    }
    return O_h;
}

// Return the number of query heads per K/V head.
static int check_query(const torch::Tensor& Q, int64_t H_kv, int64_t N_kv, bool causal) {
    TORCH_CHECK(Q.is_cuda() && Q.dim() == 4, "Q must be a 4D CUDA tensor [B, H_q, N_q, D]");
    const int64_t H_q = Q.size(1), N_q = Q.size(2), D = Q.size(3);
    TORCH_CHECK(Q.size(0) > 0 && H_q > 0, "batch and query head counts must be > 0");
    TORCH_CHECK(D == 64 || D == 128, "Head dimension must be 64 or 128, got ", D);
    TORCH_CHECK(N_q > 0 && N_kv > 0, "N_q and N_kv must be > 0");
    TORCH_CHECK(!causal || N_kv >= N_q,
                "a causal mask is aligned to the bottom right, which needs N_kv >= N_q");
    TORCH_CHECK(H_kv > 0 && H_q % H_kv == 0,
                "Query heads (", H_q, ") must be a multiple of key/value heads (", H_kv, ")");
    return (int)(H_q / H_kv);
}

static std::pair<torch::Tensor, torch::Tensor> attention_forward_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool want_L, bool causal,
    std::optional<double> softmax_scale, std::optional<torch::Tensor> out)
{
    TORCH_CHECK(K.is_cuda() && V.is_cuda() && K.dim() == 4 && V.dim() == 4,
                "K and V must be 4D CUDA tensors [B, H_kv, N_kv, D]");
    TORCH_CHECK(K.sizes() == V.sizes(), "K and V must have the same shape");
    const int kv_group = check_query(Q, K.size(1), K.size(2), causal);
    TORCH_CHECK(K.size(0) == Q.size(0) && K.size(3) == Q.size(3),
                "K and V must match Q in batch and head dimension");
    TORCH_CHECK(K.device() == Q.device() && V.device() == Q.device(),
                "Q/K/V must be on the same device");
    const int D = Q.size(3);
    const float scale = softmax_scale.has_value() ? (float)*softmax_scale
                                                  : 1.0f / std::sqrt((float)D);

    const at::cuda::CUDAGuard guard(Q.device());
    auto Q_h = as_half_rows(Q, QO_ROW_ALIGN);
    auto K_h = as_half_rows(K, KV_ROW_ALIGN);
    auto V_h = as_half_rows(V, KV_ROW_ALIGN);
    auto O_h = output_for(Q_h, out, {Q_h, K_h, V_h});
    torch::Tensor L;
    if (want_L) {
        L = torch::empty({Q.size(0), Q.size(1), Q.size(2)}, Q.options().dtype(torch::kFloat));
    }

    const ForwardLaunch launch{
        Q_h, O_h, want_L ? L.data_ptr<float>() : nullptr, (int)K.size(2), kv_group, causal, scale};
    launch.run<DenseForwardDispatch>(
        reinterpret_cast<const half*>(K_h.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V_h.data_ptr<at::Half>()), strides_of(K_h), strides_of(V_h));
    return {O_h, L};
}

std::vector<torch::Tensor> attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                             bool causal, std::optional<double> softmax_scale,
                                             std::optional<torch::Tensor> out) {
    auto [O, L] = attention_forward_impl(Q, K, V, /*want_L=*/true, causal, softmax_scale, out);
    return {O, L};
}

torch::Tensor attention_forward_only(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                     bool causal, std::optional<double> softmax_scale,
                                     std::optional<torch::Tensor> out) {
    auto [O, L] = attention_forward_impl(Q, K, V, /*want_L=*/false, causal, softmax_scale, out);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Custom CUDA forward: returns O half, L float",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("causal") = false, pybind11::arg("softmax_scale") = pybind11::none(),
          pybind11::arg("out") = pybind11::none());
    m.def("forward_only", &attention_forward_only, "Custom CUDA forward, true O-only",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("causal") = false, pybind11::arg("softmax_scale") = pybind11::none(),
          pybind11::arg("out") = pybind11::none());
}
