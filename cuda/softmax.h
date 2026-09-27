#pragma once

#include <cuda_runtime.h>
#include <cmath>

// Row-wide reductions over the 4 lanes that share an accumulator row
// (FA2: quad_allreduce_ with MaxOp / SumOp).
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

// ============================================================
// Step 5: online softmax (FlashAttention-2 form, names as in FA2 softmax.h)
//
// Scores and row_max use base-2 units; row_sum is an ordinary exponential sum.
//   row_max = running max of scaled scores
//   row_sum = running sum of exp2(acc_s - row_max)
// One softmax_rescale_o per K/V block:
//   scores_max_cur = max(row_max, blockmax(acc_s))
//   scores_scale   = exp2(row_max - scores_max_cur)   // rescale factor for old contributions
//   acc_s          = exp2(acc_s - scores_max_cur)      // acc_s now holds P
//   row_sum        = scores_scale * row_sum + rowsum(P)
//   acc_o          = scores_scale * acc_o              // P V is added afterwards by accumulate_pv
// acc_o is NOT divided by row_sum here; that happens once in the epilogue.
// ============================================================
struct Softmax {
    float row_max[2];  // [0] = row lo, [1] = row hi
    float row_sum[2];

    __device__ __forceinline__ Softmax() {
        row_max[0] = row_max[1] = -INFINITY;
        row_sum[0] = row_sum[1] = 0.0f;
    }
};

template <int NT, int NO>
__device__ __forceinline__ void softmax_rescale_o(
    float (&acc_s)[NT][4], float (&acc_o)[NO][4], Softmax& softmax)
{
    // reduce_max: block max per row (thread-local, then across the quad).
    float block_max[2] = {-INFINITY, -INFINITY};
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        block_max[0] = fmaxf(block_max[0], fmaxf(acc_s[t][0], acc_s[t][1]));
        block_max[1] = fmaxf(block_max[1], fmaxf(acc_s[t][2], acc_s[t][3]));
    }
    block_max[0] = quad_allreduce_max(block_max[0]);
    block_max[1] = quad_allreduce_max(block_max[1]);

    // New running max and the rescale factor for what was accumulated so far.
    float scores_max_cur[2], scores_scale[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        scores_max_cur[r] = fmaxf(softmax.row_max[r], block_max[r]);
        scores_scale[r]   = exp2f(softmax.row_max[r] - scores_max_cur[r]);
    }

    // scale_apply_exp2 + reduce_sum: acc_s = exp2(acc_s - scores_max_cur), and its row sums.
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

    // Rescale the unnormalized output accumulator.
    #pragma unroll
    for (int t = 0; t < NO; t++) {
        acc_o[t][0] *= scores_scale[0];
        acc_o[t][1] *= scores_scale[0];
        acc_o[t][2] *= scores_scale[1];
        acc_o[t][3] *= scores_scale[1];
    }
}
