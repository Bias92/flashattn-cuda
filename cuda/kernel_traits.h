#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#ifndef ATTN_BR
#define ATTN_BR 64
#endif
#ifndef ATTN_BC
#define ATTN_BC 32
#endif
// One K/V stage instead of two. Halves shared memory, which buys residency at
// larger tiles, and gives up the overlap of the next tile's copies with this
// tile's math.
#ifndef ATTN_DOUBLE_BUFFER
#define ATTN_DOUBLE_BUFFER 1
#endif
constexpr int kStages = ATTN_DOUBLE_BUFFER ? 2 : 1;

constexpr int kBlockM = ATTN_BR;      // Q rows per block
constexpr int kBlockN = ATTN_BC;      // K/V rows per streamed block
constexpr int kSmemPad = 8;           // smem row padding (halves) to reduce ldmatrix bank conflicts
constexpr int kNWarps = kBlockM / 16; // one warp per 16 Q rows

#define LN2f 0.69314718056f
#define LOG2Ef 1.44269504089f

// ------------------------------------------------------------
// Compile-time tile geometry
// ------------------------------------------------------------
template <int D>
struct Flash_fwd_kernel_traits {
    static constexpr int LDS        = D + kSmemPad;    // smem row stride (halves)
    static constexpr int KSLICES    = D / 16;          // k16 steps for S = Q K^T
    static constexpr int NTILES_S   = kBlockN / 8;     // n8 tiles of S per K/V block
    static constexpr int KSLICES_PV = kBlockN / 16;    // k16 steps for O += P V
    static constexpr int NTILES_O   = D / 8;           // n8 tiles of O
    static constexpr int STAGE      = 2 * kBlockN * LDS; // halves per stage: K block then V block
    static constexpr uint32_t STAGE_BYTES = STAGE * sizeof(half);
    static constexpr uint32_t ROW_BYTES   = LDS * sizeof(half);

    // Q is staged in shared memory before the loop starts. With two stages it
    // borrows the second one, so that stage has to hold a whole Q block; with
    // one stage it reuses the single K/V buffer, which is sized for whichever
    // is larger. Getting this wrong overruns shared memory at run time, so it
    // is checked at compile time instead.
    static_assert(kStages == 1 || kBlockM * LDS <= STAGE,
                  "with two stages BR must be <= 2*BC: Q is staged in one K/V stage");
    static constexpr int SMEM_HALVES = (kStages == 2)
        ? 2 * STAGE
        : (STAGE > kBlockM * LDS ? STAGE : kBlockM * LDS);
    static constexpr int SMEM_BYTES = SMEM_HALVES * (int)sizeof(half);
};
