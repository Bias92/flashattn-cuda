#pragma once

#include <cuda_runtime.h>
#include <cmath>

// Four lanes share each MMA accumulator row.
__device__ __forceinline__ float quad_allreduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return v;
}

__device__ __forceinline__ float quad_allreduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v;
}

// Scores and row_max are scaled for exp2; row_sum is the unnormalized sum.
struct Softmax {
    float row_max[2];  // [0] = row lo, [1] = row hi
    float row_sum[2];

    __device__ __forceinline__ Softmax() {
        row_max[0] = row_max[1] = -INFINITY;
        row_sum[0] = row_sum[1] = 0.0f;
    }
};

// Update row statistics, replace scores with P, and rescale the previous O.
// The caller adds PV; normalization by row_sum happens in the epilogue.
template <int NT, int NO>
__device__ __forceinline__ void softmax_rescale_o(
    float (&acc_s)[NT][4], float (&acc_o)[NO][4], Softmax& softmax)
{
    // Reduce across each thread's fragments, then across the four-lane row.
    float block_max[2] = {-INFINITY, -INFINITY};
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        block_max[0] = fmaxf(block_max[0], fmaxf(acc_s[t][0], acc_s[t][1]));
        block_max[1] = fmaxf(block_max[1], fmaxf(acc_s[t][2], acc_s[t][3]));
    }
    block_max[0] = quad_allreduce_max(block_max[0]);
    block_max[1] = quad_allreduce_max(block_max[1]);

    float scores_max_cur[2], scores_scale[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        scores_max_cur[r] = fmaxf(softmax.row_max[r], block_max[r]);
        scores_scale[r]   = exp2f(softmax.row_max[r] - scores_max_cur[r]);
    }

    float scores_sum[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        acc_s[t][0] = exp2f(acc_s[t][0] - scores_max_cur[0]);
        acc_s[t][1] = exp2f(acc_s[t][1] - scores_max_cur[0]);
        acc_s[t][2] = exp2f(acc_s[t][2] - scores_max_cur[1]);
        acc_s[t][3] = exp2f(acc_s[t][3] - scores_max_cur[1]);
        scores_sum[0] += acc_s[t][0] + acc_s[t][1];
        scores_sum[1] += acc_s[t][2] + acc_s[t][3];
    }
    scores_sum[0] = quad_allreduce_sum(scores_sum[0]);
    scores_sum[1] = quad_allreduce_sum(scores_sum[1]);

    #pragma unroll
    for (int r = 0; r < 2; r++) {
        softmax.row_sum[r] = softmax.row_sum[r] * scores_scale[r] + scores_sum[r];
        softmax.row_max[r] = scores_max_cur[r];
    }

    #pragma unroll
    for (int t = 0; t < NO; t++) {
        acc_o[t][0] *= scores_scale[0];
        acc_o[t][1] *= scores_scale[0];
        acc_o[t][2] *= scores_scale[1];
        acc_o[t][3] *= scores_scale[1];
    }
}
