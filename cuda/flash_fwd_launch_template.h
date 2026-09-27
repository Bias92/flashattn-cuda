#pragma once

#include "flash_fwd_kernel.h"

// ============================================================
// Host interface: validate tensors, choose a specialization, and launch
// ============================================================
static std::pair<torch::Tensor, torch::Tensor> attention_forward_impl(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, bool want_L)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4, "Q must be 4D [B, H, N, D]");

    int batch_size = q.size(0), num_heads = q.size(1), N = q.size(2), head_size = q.size(3);
    TORCH_CHECK(head_size == kHeadDim, "Head dimension must be ", kHeadDim);
    TORCH_CHECK(N > 0, "N must be > 0");
    TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(),
                "K and V must have the same shape as Q (self-attention only)");
    TORCH_CHECK(k.device() == q.device() && v.device() == q.device(),
                "Q/K/V must be on the same device");
    int64_t BH = (int64_t)batch_size * num_heads;
    TORCH_CHECK(BH <= 65535, "B*H must be <= 65535 (gridDim.y limit)");

    const at::cuda::CUDAGuard guard(q.device());
    auto Q_h = q.to(torch::kHalf).reshape({BH, N, head_size}).contiguous();
    auto K_h = k.to(torch::kHalf).reshape({BH, N, head_size}).contiguous();
    auto V_h = v.to(torch::kHalf).reshape({BH, N, head_size}).contiguous();

    auto O_h = torch::empty({BH, N, head_size}, Q_h.options());
    torch::Tensor softmax_lse;
    float* softmax_lse_ptr = nullptr;
    if (want_L) {
        softmax_lse = torch::empty({BH, N}, q.options().dtype(torch::kFloat));
        softmax_lse_ptr = softmax_lse.data_ptr<float>();
    }

    dim3 grid((N + kBlockM - 1) / kBlockM, BH);
    dim3 block(kNWarps * 32);
    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto kernel) {
        kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(Q_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(K_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(V_h.data_ptr<at::Half>()),
            reinterpret_cast<half*>(O_h.data_ptr<at::Half>()),
            softmax_lse_ptr,
            N);
    };
    // Full Q/K/V tiles omit boundary checks; partial tiles use guarded paths.
    const bool is_even_MN = (N % kBlockM == 0) && (N % kBlockN == 0);
    if (want_L) {
        if (is_even_MN) launch(flash_fwd_kernel<kHeadDim, true, true>);
        else            launch(flash_fwd_kernel<kHeadDim, true, false>);
    } else {
        if (is_even_MN) launch(flash_fwd_kernel<kHeadDim, false, true>);
        else            launch(flash_fwd_kernel<kHeadDim, false, false>);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {O_h.reshape({batch_size, num_heads, N, head_size}),
            want_L ? softmax_lse.reshape({batch_size, num_heads, N}) : torch::Tensor()};
}
