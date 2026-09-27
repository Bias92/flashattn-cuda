// Fused attention forward: FP16 kernel operands/output, FP32 accumulation.
// Matching identifiers follow Dao-AILab FlashAttention v2.8.3 (060c9188).
// Kernel-specific identifiers retain their original names.
//
// Reading order inside flash_fwd_kernel:
//   1. Set up tile storage and cooperative copies.
//   2. Load the block's Q tile and initialize running state.
//   3. Scan K/V tiles: scores -> online softmax -> output accumulation.
//   4. Normalize and store the output.
//
// QK^T, online softmax, and the weighted V sum stay in one fused kernel.
// CUDA instruction wrappers are in attention_forward_ops.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "flash_fwd_launch_template.h"

std::vector<torch::Tensor> mha_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    auto [out, softmax_lse] = attention_forward_impl(q, k, v, /*want_L=*/true);
    return {out, softmax_lse};
}

torch::Tensor attention_forward_only(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    auto [out, softmax_lse] = attention_forward_impl(q, k, v, /*want_L=*/false);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mha_fwd, "Custom CUDA forward: returns O half, L float");
    m.def("forward_only", &attention_forward_only, "Custom CUDA forward, true O-only");
}
