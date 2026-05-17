#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>
#include <float.h>

using namespace nvcuda;

// ============================================================
// Constants
// ============================================================
constexpr int HD = 64;

// WMMA forward: 1 warp per block, 16×16 tiles
constexpr int BR_W = 16;   // Q tile rows (= WMMA M)
constexpr int BC_W = 16;   // K/V tile rows (= WMMA N)

// Backward: 32 threads, 32×32 tiles (FP16 mixed precision, non-WMMA)
constexpr int BR_B = 32;
constexpr int BC_B = 32;

// ============================================================
// Forward Kernel — WMMA Tensor Core
//   1 warp (32 threads) per block processes 16 Q rows
//   WMMA for S = Q @ K^T and O_partial = P @ V
//   Scalar online softmax between the two matmuls
// ============================================================
template <int Br, int Bc, int D>
__global__ void flash_attn_fwd_wmma_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ L,
    int N)
{
    int lane = threadIdx.x;   // 0-31 within warp
    int bh = blockIdx.y;
    int q_start = blockIdx.x * Br;

    const half* Q_bh = Q + bh * N * D;
    const half* K_bh = K + bh * N * D;
    const half* V_bh = V + bh * N * D;
    half* O_bh = O + bh * N * D;
    float* L_bh = L + bh * N;

    // Shared memory layout (~12KB total)
    __shared__ half sQ[Br][D + 8];    // 16×72×2 = 2.25KB  — Q tile (loaded once)
    __shared__ half sK[Bc][D + 8];    // 16×72×2 = 2.25KB  — K tile (per KV block)
    __shared__ half sV[Bc][D + 8];    // 16×72×2 = 2.25KB  — V tile (per KV block)
    __shared__ float sS[Br][Bc];      // 16×16×4 = 1KB  — S scores / temp buffer
    __shared__ half sP[Br][Bc];       // 16×16×2 = 0.5KB — P after softmax
    __shared__ float sO[Br][D];       // 16×64×4 = 4KB  — running output accumulator
    __shared__ float sm[Br];          // running max per row
    __shared__ float sl[Br];          // running sum per row
    __shared__ float s_alpha[Br];     // rescale factor per row

    // Initialize accumulator and softmax state
    for (int i = lane; i < Br * D; i += 32)
        sO[i / D][i % D] = 0.0f;
    if (lane < Br) {
        sm[lane] = -FLT_MAX;
        sl[lane] = 0.0f;
    }
    __syncthreads();

    // Load Q tile to shared memory — vectorized half2 (2x bandwidth)
    {
        const int total_h2 = Br * D / 2;  // 16*64/2 = 512
        for (int i = lane; i < total_h2; i += 32) {
            int flat = i * 2;
            int r = flat / D;
            int c = flat % D;
            int global_r = q_start + r;
            if (global_r < N) {
                const half2* src = reinterpret_cast<const half2*>(&Q_bh[global_r * D + c]);
                *reinterpret_cast<half2*>(&sQ[r][c]) = *src;
            } else {
                *reinterpret_cast<half2*>(&sQ[r][c]) = __float2half2_rn(0.0f);
            }
        }
    }
    __syncthreads();

    float scale = rsqrtf((float)D);
    int num_kv_blocks = (N + Bc - 1) / Bc;

    for (int j = 0; j < num_kv_blocks; j++) {
        int kv_start = j * Bc;

        // Load K, V tiles — vectorized half2
        {
            const int total_h2 = Bc * D / 2;
            for (int i = lane; i < total_h2; i += 32) {
                int flat = i * 2;
                int r = flat / D;
                int c = flat % D;
                int global_r = kv_start + r;
                if (global_r < N) {
                    *reinterpret_cast<half2*>(&sK[r][c]) = *reinterpret_cast<const half2*>(&K_bh[global_r * D + c]);
                    *reinterpret_cast<half2*>(&sV[r][c]) = *reinterpret_cast<const half2*>(&V_bh[global_r * D + c]);
                } else {
                    *reinterpret_cast<half2*>(&sK[r][c]) = __float2half2_rn(0.0f);
                    *reinterpret_cast<half2*>(&sV[r][c]) = __float2half2_rn(0.0f);
                }
            }
        }
        __syncthreads();

        // ======== WMMA: S[16×16] = Q[16×64] @ K[16×64]^T ========
        // Split D=64 into 4 chunks of 16
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);

        for (int dk = 0; dk < D; dk += 16) {
            wmma::load_matrix_sync(q_frag, &sQ[0][dk], D + 8);
            wmma::load_matrix_sync(k_frag, &sK[0][dk], D + 8);
            wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
        }

        // Apply scale to S
        for (int i = 0; i < s_frag.num_elements; i++)
            s_frag.x[i] *= scale;

        // Store S to shared memory
        wmma::store_matrix_sync(&sS[0][0], s_frag, Bc, wmma::mem_row_major);
        __syncthreads();

        // ======== Online Softmax (Phase 1: threads 0-15 compute stats + P) ========
        if (lane < Br) {
            int r = lane;
            int global_r = q_start + r;

            if (global_r < N) {
                // Block max
                float block_max = -FLT_MAX;
                for (int c = 0; c < Bc; c++) {
                    int global_c = kv_start + c;
                    if (global_c < N && sS[r][c] > block_max)
                        block_max = sS[r][c];
                }

                float m_new = fmaxf(sm[r], block_max);
                s_alpha[r] = expf(sm[r] - m_new);

                sl[r] *= s_alpha[r];

                // Compute P and update sum
                for (int c = 0; c < Bc; c++) {
                    int global_c = kv_start + c;
                    float p_val = (global_c < N) ? expf(sS[r][c] - m_new) : 0.0f;
                    sl[r] += p_val;
                    sP[r][c] = __float2half(p_val);
                }

                sm[r] = m_new;
            } else {
                s_alpha[r] = 1.0f;
                for (int c = 0; c < Bc; c++)
                    sP[r][c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // ======== Rescale accumulator (Phase 2: all 32 threads) ========
        for (int i = lane; i < Br * D; i += 32) {
            int r = i / D;
            sO[r][i % D] *= s_alpha[r];
        }
        __syncthreads();

        // ======== WMMA: O_partial = P[16×16] @ V_chunk[16×16] ========
        // Load P fragment once, reuse for all D chunks
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
        wmma::load_matrix_sync(p_frag, &sP[0][0], Bc);

        for (int dk = 0; dk < D; dk += 16) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;
            wmma::fill_fragment(o_frag, 0.0f);

            wmma::load_matrix_sync(v_frag, &sV[0][dk], D + 8);
            wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);

            // Store partial result to sS (reused as temp buffer)
            wmma::store_matrix_sync(&sS[0][0], o_frag, Bc, wmma::mem_row_major);
            __syncthreads();

            // Accumulate into sO (all 32 threads)
            for (int i = lane; i < Br * 16; i += 32) {
                int r = i / 16;
                int c = i % 16;
                sO[r][dk + c] += sS[r][c];
            }
            __syncthreads();
        }
    }

    // ======== Write Output ========
    for (int i = lane; i < Br * D; i += 32) {
        int r = i / D;
        int c = i % D;
        int global_r = q_start + r;
        if (global_r < N)
            O_bh[global_r * D + c] = __float2half(sO[r][c] / sl[r]);
    }

    if (lane < Br) {
        int global_r = q_start + lane;
        if (global_r < N)
            L_bh[global_r] = sm[lane] + logf(sl[lane]);
    }
}

// ============================================================
// Forward Host Launcher
// ============================================================
std::vector<torch::Tensor> flash_attn_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V)
{
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, N, D]");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);
    TORCH_CHECK(D == HD, "Head dimension must be " + std::to_string(HD));

    int BH = B * H;

    auto Q_h = Q.to(torch::kHalf).reshape({BH, N, D}).contiguous();
    auto K_h = K.to(torch::kHalf).reshape({BH, N, D}).contiguous();
    auto V_h = V.to(torch::kHalf).reshape({BH, N, D}).contiguous();

    auto O_h = torch::zeros({BH, N, D}, Q_h.options());
    auto L = torch::zeros({BH, N}, Q.options().dtype(torch::kFloat));

    int num_q_blocks = (N + BR_W - 1) / BR_W;
    dim3 grid(num_q_blocks, BH);
    dim3 block(32);   // 1 warp per block

    flash_attn_fwd_wmma_kernel<BR_W, BC_W, HD><<<grid, block>>>(
        reinterpret_cast<const half*>(Q_h.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K_h.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V_h.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O_h.data_ptr<at::Half>()),
        L.data_ptr<float>(),
        N);

    auto O_out = O_h.to(torch::kFloat).reshape({B, H, N, D});
    auto L_out = L.reshape({B, H, N});

    return {O_out, L_out};
}

// ============================================================
// Backward Kernels — FP16 Mixed Precision (non-WMMA)
//   Same design as FP16 backward: half storage, float accumulation
// ============================================================

__global__ void flash_attn_precompute_D_kernel(
    const half* __restrict__ O,
    const half* __restrict__ dO,
    float* __restrict__ Di,
    int N, int D_dim)
{
    int bh = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const half* O_row = O + bh * N * D_dim + row * D_dim;
    const half* dO_row = dO + bh * N * D_dim + row * D_dim;

    float sum = 0.0f;
    for (int d = 0; d < D_dim; d++)
        sum += __half2float(O_row[d]) * __half2float(dO_row[d]);

    Di[bh * N + row] = sum;
}

template <int B_r, int B_c, int D>
__global__ void flash_attn_bwd_dq_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ Di,
    half* __restrict__ dQ,
    int N)
{
    int tid = threadIdx.x;
    int bh = blockIdx.y;
    int row = blockIdx.x * B_r + tid;

    const half* Q_bh  = Q  + bh * N * D;
    const half* K_bh  = K  + bh * N * D;
    const half* V_bh  = V  + bh * N * D;
    const half* dO_bh = dO + bh * N * D;
    const float* L_bh  = L  + bh * N;
    const float* Di_bh = Di + bh * N;
    half* dQ_bh = dQ + bh * N * D;

    bool valid = (row < N);
    float scale = rsqrtf((float)D);

    float q_reg[D], do_reg[D];
    float l_i = 0.0f, d_i = 0.0f;
    if (valid) {
        for (int d = 0; d < D; d++) {
            q_reg[d]  = __half2float(Q_bh[row * D + d]);
            do_reg[d] = __half2float(dO_bh[row * D + d]);
        }
        l_i = L_bh[row];
        d_i = Di_bh[row];
    }

    float dq_acc[D];
    for (int d = 0; d < D; d++)
        dq_acc[d] = 0.0f;

    __shared__ half sK[B_c][D];
    __shared__ half sV[B_c][D];

    int num_kv_blocks = (N + B_c - 1) / B_c;

    for (int j = 0; j < num_kv_blocks; j++) {
        int kv_start = j * B_c;

        for (int c = tid; c < B_c; c += B_r) {
            int global_c = kv_start + c;
            for (int d = 0; d < D; d++) {
                sK[c][d] = (global_c < N) ? K_bh[global_c * D + d] : __float2half(0.0f);
                sV[c][d] = (global_c < N) ? V_bh[global_c * D + d] : __float2half(0.0f);
            }
        }
        __syncthreads();

        if (valid) {
            for (int c = 0; c < B_c; c++) {
                int global_c = kv_start + c;
                if (global_c >= N) break;

                float s_c = 0.0f;
                for (int d = 0; d < D; d++)
                    s_c += q_reg[d] * __half2float(sK[c][d]);
                s_c *= scale;

                float p_c = expf(s_c - l_i);

                float dp_c = 0.0f;
                for (int d = 0; d < D; d++)
                    dp_c += do_reg[d] * __half2float(sV[c][d]);

                float ds_c = p_c * (dp_c - d_i);

                for (int d = 0; d < D; d++)
                    dq_acc[d] += ds_c * __half2float(sK[c][d]);
            }
        }
        __syncthreads();
    }

    if (valid) {
        for (int d = 0; d < D; d++)
            dQ_bh[row * D + d] = __float2half(scale * dq_acc[d]);
    }
}

template <int B_r, int B_c, int D>
__global__ void flash_attn_bwd_dkdv_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ Di,
    half* __restrict__ dK,
    half* __restrict__ dV,
    int N)
{
    int tid = threadIdx.x;
    int bh = blockIdx.y;
    int col = blockIdx.x * B_c + tid;

    const half* Q_bh  = Q  + bh * N * D;
    const half* K_bh  = K  + bh * N * D;
    const half* V_bh  = V  + bh * N * D;
    const half* dO_bh = dO + bh * N * D;
    const float* L_bh  = L  + bh * N;
    const float* Di_bh = Di + bh * N;
    half* dK_bh = dK + bh * N * D;
    half* dV_bh = dV + bh * N * D;

    bool valid = (col < N);
    float scale = rsqrtf((float)D);

    float k_reg[D], v_reg[D];
    if (valid) {
        for (int d = 0; d < D; d++) {
            k_reg[d] = __half2float(K_bh[col * D + d]);
            v_reg[d] = __half2float(V_bh[col * D + d]);
        }
    }

    float dk_acc[D], dv_acc[D];
    for (int d = 0; d < D; d++) {
        dk_acc[d] = 0.0f;
        dv_acc[d] = 0.0f;
    }

    __shared__ half sQ[B_r][D];
    __shared__ half sdO[B_r][D];
    __shared__ float sL[B_r];
    __shared__ float sD[B_r];

    int num_q_blocks = (N + B_r - 1) / B_r;

    for (int i = 0; i < num_q_blocks; i++) {
        int q_start = i * B_r;

        for (int r = tid; r < B_r; r += B_c) {
            int global_r = q_start + r;
            if (global_r < N) {
                for (int d = 0; d < D; d++) {
                    sQ[r][d]  = Q_bh[global_r * D + d];
                    sdO[r][d] = dO_bh[global_r * D + d];
                }
                sL[r] = L_bh[global_r];
                sD[r] = Di_bh[global_r];
            } else {
                for (int d = 0; d < D; d++) {
                    sQ[r][d]  = __float2half(0.0f);
                    sdO[r][d] = __float2half(0.0f);
                }
                sL[r] = 0.0f;
                sD[r] = 0.0f;
            }
        }
        __syncthreads();

        if (valid) {
            for (int r = 0; r < B_r; r++) {
                int global_r = q_start + r;
                if (global_r >= N) break;

                float s_val = 0.0f;
                for (int d = 0; d < D; d++)
                    s_val += __half2float(sQ[r][d]) * k_reg[d];
                s_val *= scale;

                float p_val = expf(s_val - sL[r]);

                for (int d = 0; d < D; d++)
                    dv_acc[d] += p_val * __half2float(sdO[r][d]);

                float dp_val = 0.0f;
                for (int d = 0; d < D; d++)
                    dp_val += __half2float(sdO[r][d]) * v_reg[d];

                float ds_val = p_val * (dp_val - sD[r]);

                for (int d = 0; d < D; d++)
                    dk_acc[d] += ds_val * __half2float(sQ[r][d]);
            }
        }
        __syncthreads();
    }

    if (valid) {
        for (int d = 0; d < D; d++) {
            dK_bh[col * D + d] = __float2half(scale * dk_acc[d]);
            dV_bh[col * D + d] = __float2half(dv_acc[d]);
        }
    }
}

// ============================================================
// Backward Host Launcher
// ============================================================
std::vector<torch::Tensor> flash_attn_backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L)
{
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(dO.is_cuda(), "dO must be a CUDA tensor");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);
    TORCH_CHECK(D == HD, "Head dimension must be " + std::to_string(HD));

    int BH = B * H;

    auto Q_h  = Q.to(torch::kHalf).reshape({BH, N, D}).contiguous();
    auto K_h  = K.to(torch::kHalf).reshape({BH, N, D}).contiguous();
    auto V_h  = V.to(torch::kHalf).reshape({BH, N, D}).contiguous();
    auto O_h  = O.to(torch::kHalf).reshape({BH, N, D}).contiguous();
    auto dO_h = dO.to(torch::kHalf).reshape({BH, N, D}).contiguous();
    auto L_flat = L.to(torch::kFloat).reshape({BH, N}).contiguous();

    auto dQ_h = torch::zeros({BH, N, D}, Q_h.options());
    auto dK_h = torch::zeros({BH, N, D}, K_h.options());
    auto dV_h = torch::zeros({BH, N, D}, V_h.options());

    auto Di = torch::zeros({BH, N}, Q.options().dtype(torch::kFloat));

    // Kernel 1: Precompute D_i
    {
        int threads = 256;
        int blocks_per_bh = (N + threads - 1) / threads;
        dim3 grid(blocks_per_bh, BH);
        flash_attn_precompute_D_kernel<<<grid, threads>>>(
            reinterpret_cast<const half*>(O_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(dO_h.data_ptr<at::Half>()),
            Di.data_ptr<float>(),
            N, D);
    }

    // Kernel 2: dQ
    {
        int num_q_blocks = (N + BR_B - 1) / BR_B;
        dim3 grid(num_q_blocks, BH);
        dim3 block(BR_B);
        flash_attn_bwd_dq_kernel<BR_B, BC_B, HD><<<grid, block>>>(
            reinterpret_cast<const half*>(Q_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(K_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(V_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(dO_h.data_ptr<at::Half>()),
            L_flat.data_ptr<float>(),
            Di.data_ptr<float>(),
            reinterpret_cast<half*>(dQ_h.data_ptr<at::Half>()),
            N);
    }

    // Kernel 3: dK, dV
    {
        int num_kv_blocks = (N + BC_B - 1) / BC_B;
        dim3 grid(num_kv_blocks, BH);
        dim3 block(BC_B);
        flash_attn_bwd_dkdv_kernel<BR_B, BC_B, HD><<<grid, block>>>(
            reinterpret_cast<const half*>(Q_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(K_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(V_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(dO_h.data_ptr<at::Half>()),
            L_flat.data_ptr<float>(),
            Di.data_ptr<float>(),
            reinterpret_cast<half*>(dK_h.data_ptr<at::Half>()),
            reinterpret_cast<half*>(dV_h.data_ptr<at::Half>()),
            N);
    }

    auto dQ_out = dQ_h.to(torch::kFloat).reshape({B, H, N, D});
    auto dK_out = dK_h.to(torch::kFloat).reshape({B, H, N, D});
    auto dV_out = dV_h.to(torch::kFloat).reshape({B, H, N, D});

    return {dQ_out, dK_out, dV_out};
}

// ============================================================
// PyBind11 Bindings
// ============================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_forward, "FlashAttention forward (WMMA Tensor Core)");
    m.def("backward", &flash_attn_backward, "FlashAttention backward (FP16 mixed precision)");
}
