#include <metal_stdlib>
using namespace metal;

// ============================================================
// FlashAttention Forward Kernel — Metal (1:1 port from CUDA FP32 baseline)
//
// Algorithm 1 from Dao et al. (NeurIPS 2022):
//   Tiled Q×K^T with online softmax, O(N) memory.
//
// Thread model: 1 thread per Q row, threadgroup of 32 threads.
//   Identical to CUDA baseline: B_r=32, B_c=32, D=64.
// ============================================================

constant constexpr int BR = 32;   // Q row block size
constant constexpr int BC = 32;   // K/V column block size
constant constexpr int HD = 64;   // head dimension

struct FlashAttnParams {
    int N;      // sequence length
    int BH;     // batch * heads
};

kernel void flash_attn_fwd_kernel(
    device const float* Q        [[buffer(0)]],   // [BH, N, D]
    device const float* K        [[buffer(1)]],   // [BH, N, D]
    device const float* V        [[buffer(2)]],   // [BH, N, D]
    device float*       O        [[buffer(3)]],   // [BH, N, D]
    device float*       L        [[buffer(4)]],   // [BH, N]
    device const FlashAttnParams& params [[buffer(5)]],
    uint3 thread_pos       [[thread_position_in_threadgroup]],
    uint3 grid_pos      [[threadgroup_position_in_grid]])
{
    int tid       = thread_pos.x;
    int block_row = grid_pos.x;          // which Q block
    int bh        = grid_pos.y;          // batch * head index
    int row       = block_row * BR + tid;
    int N         = params.N;

    // Pointers for this batch-head
    device const float* Q_bh = Q + bh * N * HD;
    device const float* K_bh = K + bh * N * HD;
    device const float* V_bh = V + bh * N * HD;
    device float*       O_bh = O + bh * N * HD;
    device float*       L_bh = L + bh * N;

    // All threads must participate in threadgroup loads + barriers
    bool valid = (row < N);

    // Load Q row into thread-local storage
    float q_reg[HD];
    if (valid) {
        for (int d = 0; d < HD; d++)
            q_reg[d] = Q_bh[row * HD + d];
    }

    float scale = rsqrt(float(HD));

    // Running online softmax state
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc[HD];
    for (int d = 0; d < HD; d++)
        acc[d] = 0.0f;

    // Threadgroup (shared) memory for K and V tiles
    threadgroup float sK[BC][HD];    // 32×64×4 = 8 KB
    threadgroup float sV[BC][HD];    // 32×64×4 = 8 KB  (total 16 KB)

    int num_kv_blocks = (N + BC - 1) / BC;

    for (int j = 0; j < num_kv_blocks; j++) {
        int kv_start = j * BC;

        // Collaborative load: all threads participate
        for (int c = int(tid); c < BC; c += BR) {
            int global_c = kv_start + c;
            for (int d = 0; d < HD; d++) {
                sK[c][d] = (global_c < N) ? K_bh[global_c * HD + d] : 0.0f;
                sV[c][d] = (global_c < N) ? V_bh[global_c * HD + d] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid) {
            // Compute S[c] = dot(q_reg, sK[c]) * scale
            float s[BC];
            for (int c = 0; c < BC; c++) {
                float dot = 0.0f;
                for (int d = 0; d < HD; d++)
                    dot += q_reg[d] * sK[c][d];
                s[c] = dot * scale;
            }

            // Find block max
            float block_max = -FLT_MAX;
            for (int c = 0; c < BC; c++) {
                int global_c = kv_start + c;
                if (global_c < N && s[c] > block_max)
                    block_max = s[c];
            }

            // Online softmax update
            float m_new = max(m_i, block_max);
            float alpha = exp(m_i - m_new);

            l_i = l_i * alpha;
            for (int d = 0; d < HD; d++)
                acc[d] = acc[d] * alpha;

            for (int c = 0; c < BC; c++) {
                int global_c = kv_start + c;
                float p = (global_c < N) ? exp(s[c] - m_new) : 0.0f;
                l_i += p;
                for (int d = 0; d < HD; d++)
                    acc[d] += p * sV[c][d];
            }

            m_i = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    if (valid) {
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < HD; d++)
            O_bh[row * HD + d] = acc[d] * inv_l;
        L_bh[row] = m_i + log(l_i);
    }
}
