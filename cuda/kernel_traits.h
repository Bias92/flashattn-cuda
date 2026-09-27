#pragma once

// Tile configuration for this kernel; values are unchanged.
// Tile dimensions, in elements. Each warp owns 16 Q rows.
constexpr int kHeadDim = 64;  // Head dimension.
constexpr int kBlockM = 64;   // Q rows per block.
constexpr int kBlockN = 32;   // K/V rows per outer-loop iteration.
constexpr int PAD = 8;             // Extra half elements per shared-memory row.
constexpr int kNWarps = kBlockM / 16;

#define LN2f 0.69314718056f
#define LOG2Ef 1.44269504089f
