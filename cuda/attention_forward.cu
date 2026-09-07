// ============================================================
// attention_forward.cu -- Custom CUDA attention forward
//
// FlashAttention-2 style forward: each warp owns 16 Q rows and streams
// K/V in BC-row blocks with online softmax. The kernel body is split into
// named __forceinline__ steps (stage K/V, QK^T, mask, online softmax,
// PV, normalize) so the algorithm reads top to bottom; the hardware
// details live in the helpers.
//
// Two-stage K/V copies with precomputed shared-memory addresses.
// FULL_TILES removes load, mask, and store guards when N is divisible by BR and BC.
// ============================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

constexpr int HD = 64;          // head dimension
constexpr int BR = 64;          // Q rows per block
constexpr int BC = 32;          // K/V rows per streamed block
constexpr int PAD = 8;          // smem row padding (halves) to avoid ldmatrix bank conflicts
constexpr int NWARPS = BR / 16; // one warp per 16 Q rows

#define LN2f 0.69314718056f
#define LOG2Ef 1.44269504089f

// ------------------------------------------------------------
// Compile-time tile geometry
// ------------------------------------------------------------
template <int D>
struct Tile {
    static constexpr int LDS        = D + PAD;      // smem row stride (halves)
    static constexpr int KSLICES    = D / 16;       // k16 steps for S = Q K^T
    static constexpr int NTILES_S   = BC / 8;       // n8 tiles of S per K/V block
    static constexpr int KSLICES_PV = BC / 16;      // k16 steps for O += P V
    static constexpr int NTILES_O   = D / 8;        // n8 tiles of O
    static constexpr int STAGE      = 2 * BC * LDS; // halves per stage: K block then V block
    static constexpr uint32_t STAGE_BYTES = STAGE * sizeof(half);
    static constexpr uint32_t ROW_BYTES   = LDS * sizeof(half);
};

// ------------------------------------------------------------
// Fragment layout reminder (mma.m16n8k16, fp32 accumulators)
//
// In every m16n8 accumulator tile a lane holds four floats:
//   [0] = (row lo, col c)   [1] = (row lo, col c+1)
//   [2] = (row hi, col c)   [3] = (row hi, col c+1)
// with lo = lane/4, hi = lo + 8, c = 2*(lane%4).
// So each thread tracks softmax statistics for two rows, index 0 = lo,
// index 1 = hi, and a row-wide reduction is a reduction over the
// 4 lanes that share lane/4 (a "quad").
// ------------------------------------------------------------

// ============================================================
// PTX wrappers
// ============================================================
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_half2(float x, float y) {
    half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void cp_async_16(uint32_t saddr, const void* gptr, int src_size) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(saddr), "l"(gptr), "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int NGROUPS>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NGROUPS));
}

// Row-wide reductions over the 4 lanes that share an accumulator row.
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return v;
}

__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v;
}

// ============================================================
// Step 1: K/V block staging (global -> smem via cp.async)
//
// One stage holds a BC-row K block followed by a BC-row V block.
// Each thread copies two 16-byte chunks of K and two of V per block;
// the smem offsets depend only on tid, so they are computed once.
// ============================================================
template <int D, bool FULL_TILES>
struct KVStager {
    using T = Tile<D>;
    const half* K_bh;
    const half* V_bh;
    int N;
    int r0, r1, cc;
    uint32_t k0_off, k1_off, v0_off, v1_off;

    __device__ __forceinline__ KVStager(const half* K, const half* V, int N_, int tid)
        : K_bh(K), V_bh(V), N(N_)
    {
        r0 = tid >> 3;
        r1 = r0 + 16;
        cc = (tid & 7) << 3;
        k0_off = (uint32_t)(r0 * T::LDS + cc) * sizeof(half);
        k1_off = (uint32_t)(r1 * T::LDS + cc) * sizeof(half);
        v0_off = (uint32_t)((BC + r0) * T::LDS + cc) * sizeof(half);
        v1_off = (uint32_t)((BC + r1) * T::LDS + cc) * sizeof(half);
    }

    // Issue the copies for K/V rows [kv, kv + BC) into the stage at sbase.
    __device__ __forceinline__ void issue(uint32_t sbase, int kv) const {
        if constexpr (FULL_TILES) {
            // N % BC == 0 guarantees every issued row is in range.
            cp_async_16(sbase + k0_off, K_bh + (size_t)(kv + r0) * D + cc, 16);
            cp_async_16(sbase + k1_off, K_bh + (size_t)(kv + r1) * D + cc, 16);
            cp_async_16(sbase + v0_off, V_bh + (size_t)(kv + r0) * D + cc, 16);
            cp_async_16(sbase + v1_off, V_bh + (size_t)(kv + r1) * D + cc, 16);
        } else {
            // Out-of-range rows read row 0 with src_size 0 (zero-fill).
            int g0 = kv + r0, g1 = kv + r1;
            const half* k0 = K_bh + (size_t)(g0 < N ? g0 : 0) * D + cc;
            const half* k1 = K_bh + (size_t)(g1 < N ? g1 : 0) * D + cc;
            const half* v0 = V_bh + (size_t)(g0 < N ? g0 : 0) * D + cc;
            const half* v1 = V_bh + (size_t)(g1 < N ? g1 : 0) * D + cc;
            cp_async_16(sbase + k0_off, k0, (g0 < N) ? 16 : 0);
            cp_async_16(sbase + k1_off, k1, (g1 < N) ? 16 : 0);
            cp_async_16(sbase + v0_off, v0, (g0 < N) ? 16 : 0);
            cp_async_16(sbase + v1_off, v1, (g1 < N) ? 16 : 0);
        }
        cp_async_commit();
    }
};

// ============================================================
// Step 2: Q -> smem -> per-warp A fragments (kept in registers for the whole loop)
// ============================================================
template <int D, bool FULL_TILES>
__device__ __forceinline__ void stage_q_to_smem(
    half* sQ_raw, const half* __restrict__ Q_bh, int q_block, int N, int tid)
{
    using T = Tile<D>;
    half (*sQ)[T::LDS] = reinterpret_cast<half(*)[T::LDS]>(sQ_raw);
    constexpr int total_h2 = BR * D / 2;
    for (int i = tid; i < total_h2; i += NWARPS * 32) {
        int flat = i * 2;
        int r = flat / D, c = flat % D;
        if constexpr (FULL_TILES) {
            // N % BR == 0 guarantees every Q row in this block is valid.
            *reinterpret_cast<half2*>(&sQ[r][c]) =
                *reinterpret_cast<const half2*>(&Q_bh[(size_t)(q_block + r) * D + c]);
        } else {
            int gr = q_block + r;
            half2 val = (gr < N)
                ? *reinterpret_cast<const half2*>(&Q_bh[(size_t)gr * D + c])
                : __float2half2_rn(0.0f);
            *reinterpret_cast<half2*>(&sQ[r][c]) = val;
        }
    }
}

template <int D>
__device__ __forceinline__ void load_q_fragments(
    uint32_t (&qf)[Tile<D>::KSLICES][4], const half* sQ_raw, int warp, int lane)
{
    using T = Tile<D>;
    const half (*sQ)[T::LDS] = reinterpret_cast<const half(*)[T::LDS]>(sQ_raw);
    int r = warp * 16 + (lane % 16);
    int kbase = (lane < 16) ? 0 : 8;
    #pragma unroll
    for (int ks = 0; ks < T::KSLICES; ks++) {
        uint32_t addr = smem_u32(&sQ[r][ks * 16 + kbase]);
        ldmatrix_x4(qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], addr);
    }
}

// ============================================================
// Step 3: S = (Q K^T) * scale * log2(e)
//
// The log2(e) factor is folded into the scale so softmax can use exp2.
// ============================================================
template <int D>
__device__ __forceinline__ void compute_qk(
    float (&s)[Tile<D>::NTILES_S][4],
    const uint32_t (&qf)[Tile<D>::KSLICES][4],
    uint32_t qk_base, float scale_log2)
{
    using T = Tile<D>;
    #pragma unroll
    for (int t = 0; t < T::NTILES_S; t++) {
        s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.0f;
        #pragma unroll
        for (int ks = 0; ks < T::KSLICES; ks++) {
            uint32_t b0, b1;
            ldmatrix_x2(b0, b1,
                qk_base + (uint32_t)(t * 8) * T::ROW_BYTES
                        + (uint32_t)(ks * 16) * sizeof(half));
            mma_m16n8k16(s[t][0], s[t][1], s[t][2], s[t][3],
                         qf[ks][0], qf[ks][1], qf[ks][2], qf[ks][3], b0, b1);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) s[t][j] *= scale_log2;
    }
}

// ============================================================
// Step 4: mask columns past N in the last (partial) K/V block
// ============================================================
template <int NT>
__device__ __forceinline__ void mask_tail_columns(float (&s)[NT][4], int kv, int N, int lane)
{
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        int col0 = kv + t * 8 + 2 * (lane % 4);
        if (col0 >= N)     { s[t][0] = -INFINITY; s[t][2] = -INFINITY; }
        if (col0 + 1 >= N) { s[t][1] = -INFINITY; s[t][3] = -INFINITY; }
    }
}

// ============================================================
// Step 5: online softmax (FlashAttention-2 form)
//
// Running statistics per row, in the log2 domain:
//   m = running max of scaled scores
//   l = running sum of exp2(s - m)
// One step per K/V block:
//   m_new = max(m, blockmax(s))
//   alpha = exp2(m - m_new)          // rescale factor for old contributions
//   P     = exp2(s - m_new)          // written back into s
//   l     = alpha * l + rowsum(P)
//   O     = alpha * O                // P V is added afterwards by accumulate_pv
// O is NOT divided by l here; that happens once in normalize_and_store.
// ============================================================
struct SoftmaxState {
    float m[2];  // [0] = row lo, [1] = row hi
    float l[2];

    __device__ __forceinline__ SoftmaxState() {
        m[0] = m[1] = -INFINITY;
        l[0] = l[1] = 0.0f;
    }
};

template <int NT, int NO>
__device__ __forceinline__ void online_softmax_step(
    float (&s)[NT][4], float (&o_acc)[NO][4], SoftmaxState& st)
{
    // Block max per row (thread-local, then across the quad).
    float bm[2] = {-INFINITY, -INFINITY};
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        bm[0] = fmaxf(bm[0], fmaxf(s[t][0], s[t][1]));
        bm[1] = fmaxf(bm[1], fmaxf(s[t][2], s[t][3]));
    }
    bm[0] = quad_max(bm[0]);
    bm[1] = quad_max(bm[1]);

    // New running max and the rescale factor for what was accumulated so far.
    float mn[2], alpha[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        mn[r]    = fmaxf(st.m[r], bm[r]);
        alpha[r] = exp2f(st.m[r] - mn[r]);
    }

    // P = exp2(s - m_new), and its row sums.
    float rs[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        s[t][0] = exp2f(s[t][0] - mn[0]);
        s[t][1] = exp2f(s[t][1] - mn[0]);
        s[t][2] = exp2f(s[t][2] - mn[1]);
        s[t][3] = exp2f(s[t][3] - mn[1]);
        rs[0] += s[t][0] + s[t][1];
        rs[1] += s[t][2] + s[t][3];
    }
    rs[0] = quad_sum(rs[0]);
    rs[1] = quad_sum(rs[1]);

    #pragma unroll
    for (int r = 0; r < 2; r++) {
        st.l[r] = st.l[r] * alpha[r] + rs[r];
        st.m[r] = mn[r];
    }

    // Rescale the unnormalized output accumulator.
    #pragma unroll
    for (int t = 0; t < NO; t++) {
        o_acc[t][0] *= alpha[0];
        o_acc[t][1] *= alpha[0];
        o_acc[t][2] *= alpha[1];
        o_acc[t][3] *= alpha[1];
    }
}

// ============================================================
// Step 6: O += P V
//
// P (fp32 accumulator layout) is repacked in registers into the fp16
// A-fragment layout of the next mma; V is read transposed from smem.
// ============================================================
template <int D>
__device__ __forceinline__ void accumulate_pv(
    float (&o_acc)[Tile<D>::NTILES_O][4],
    const float (&p)[Tile<D>::NTILES_S][4],
    uint32_t pv_base)
{
    using T = Tile<D>;
    uint32_t pf[T::KSLICES_PV][4];
    #pragma unroll
    for (int ks = 0; ks < T::KSLICES_PV; ks++) {
        pf[ks][0] = pack_half2(p[2 * ks][0],     p[2 * ks][1]);
        pf[ks][1] = pack_half2(p[2 * ks][2],     p[2 * ks][3]);
        pf[ks][2] = pack_half2(p[2 * ks + 1][0], p[2 * ks + 1][1]);
        pf[ks][3] = pack_half2(p[2 * ks + 1][2], p[2 * ks + 1][3]);
    }

    #pragma unroll
    for (int t = 0; t < T::NTILES_O; t++) {
        #pragma unroll
        for (int ks = 0; ks < T::KSLICES_PV; ks++) {
            uint32_t b0, b1;
            ldmatrix_x2_trans(b0, b1,
                pv_base + (uint32_t)(ks * 16) * T::ROW_BYTES
                        + (uint32_t)(t * 8) * sizeof(half));
            mma_m16n8k16(o_acc[t][0], o_acc[t][1], o_acc[t][2], o_acc[t][3],
                         pf[ks][0], pf[ks][1], pf[ks][2], pf[ks][3], b0, b1);
        }
    }
}

// ============================================================
// Step 7: O = O / l (the single normalization), optional L = m*ln2 + log(l)
// ============================================================
template <int D, bool WRITE_L, bool FULL_TILES>
__device__ __forceinline__ void normalize_and_store(
    const float (&o_acc)[Tile<D>::NTILES_O][4],
    const SoftmaxState& st,
    half* __restrict__ O_bh, float* __restrict__ L,
    int bh, int N, int q_block, int warp, int lane)
{
    using T = Tile<D>;
    const float inv_lo = 1.0f / st.l[0];
    const float inv_hi = 1.0f / st.l[1];
    const int r_lo = q_block + warp * 16 + lane / 4;
    const int r_hi = r_lo + 8;
    const int cbase = 2 * (lane % 4);

    #pragma unroll
    for (int t = 0; t < T::NTILES_O; t++) {
        int col = t * 8 + cbase;
        if (FULL_TILES || r_lo < N) {
            half2 v = __floats2half2_rn(o_acc[t][0] * inv_lo, o_acc[t][1] * inv_lo);
            *reinterpret_cast<half2*>(&O_bh[(size_t)r_lo * D + col]) = v;
        }
        if (FULL_TILES || r_hi < N) {
            half2 v = __floats2half2_rn(o_acc[t][2] * inv_hi, o_acc[t][3] * inv_hi);
            *reinterpret_cast<half2*>(&O_bh[(size_t)r_hi * D + col]) = v;
        }
    }
    if constexpr (WRITE_L) {
        if (lane % 4 == 0) {
            float* L_bh = L + (size_t)bh * N;
            if (FULL_TILES || r_lo < N) L_bh[r_lo] = st.m[0] * LN2f + logf(st.l[0]);
            if (FULL_TILES || r_hi < N) L_bh[r_hi] = st.m[1] * LN2f + logf(st.l[1]);
        }
    }
}

// ============================================================
// Forward kernel
// ============================================================
template <int D, bool WRITE_L, bool FULL_TILES>
__global__ void __launch_bounds__(NWARPS * 32)
attention_fwd_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    float* __restrict__ L,    // may be nullptr when WRITE_L == false
    int N)
{
    using T = Tile<D>;

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int bh = blockIdx.y;
    const int q_block = blockIdx.x * BR;

    const half* Q_bh = Q + (size_t)bh * N * D;
    const half* K_bh = K + (size_t)bh * N * D;
    const half* V_bh = V + (size_t)bh * N * D;
    half* O_bh = O + (size_t)bh * N * D;
    // L is only dereferenced inside normalize_and_store when WRITE_L.

    // Two K/V stages, double-buffered. Q is staged through stage 1 before
    // the loop starts using it.
    __shared__ __align__(16) half smem[2 * T::STAGE];
    const uint32_t smem_base = smem_u32(smem);
    uint32_t cur_base  = smem_base;
    uint32_t next_base = smem_base + T::STAGE_BYTES;

    // ---- Prologue: kick off K/V block 0, stage Q, pull Q into registers ----
    KVStager<D, FULL_TILES> kv_stager(K_bh, V_bh, N, tid);
    kv_stager.issue(cur_base, 0);

    stage_q_to_smem<D, FULL_TILES>(smem + T::STAGE, Q_bh, q_block, N, tid);
    __syncthreads();
    uint32_t qf[T::KSLICES][4];
    load_q_fragments<D>(qf, smem + T::STAGE, warp, lane);
    __syncthreads();

    // Per-lane smem offsets for the B operands of QK^T (K rows) and PV (V rows).
    const uint32_t qk_lane_base =
        (uint32_t)((lane & 7) * T::LDS + ((lane < 8) ? 0 : 8)) * sizeof(half);
    const uint32_t pv_lane_base =
        (uint32_t)((BC + (lane & 15)) * T::LDS) * sizeof(half);

    float o_acc[T::NTILES_O][4];
    #pragma unroll
    for (int t = 0; t < T::NTILES_O; t++)
        o_acc[t][0] = o_acc[t][1] = o_acc[t][2] = o_acc[t][3] = 0.0f;
    SoftmaxState st;
    const float scale_log2 = rsqrtf((float)D) * LOG2Ef;

    // ---- Main loop over K/V blocks ----
    const int nblocks = (N + BC - 1) / BC;
    for (int i = 0; i < nblocks; i++) {
        const int kv = i * BC;
        const bool has_next = (i + 1) < nblocks;

        // Prefetch the next block into the other stage, wait for this one.
        if (has_next) {
            kv_stager.issue(next_base, kv + BC);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        float s[T::NTILES_S][4];
        compute_qk<D>(s, qf, cur_base + qk_lane_base, scale_log2);
        if constexpr (!FULL_TILES) {
            if (kv + BC > N) mask_tail_columns(s, kv, N, lane);
        }
        online_softmax_step(s, o_acc, st);          // s becomes P, o_acc *= alpha
        accumulate_pv<D>(o_acc, s, cur_base + pv_lane_base);
        __syncthreads();                            // everyone done reading this stage

        uint32_t tmp = cur_base; cur_base = next_base; next_base = tmp;
    }

    normalize_and_store<D, WRITE_L, FULL_TILES>(o_acc, st, O_bh, L, bh, N, q_block, warp, lane);
}

// ============================================================
// Host launchers
// ============================================================
static std::pair<torch::Tensor, torch::Tensor> attention_forward_impl(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool want_L)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, N, D]");

    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    TORCH_CHECK(D == HD, "Head dimension must be ", HD);
    TORCH_CHECK(N > 0, "N must be > 0");
    TORCH_CHECK(K.sizes() == Q.sizes() && V.sizes() == Q.sizes(),
                "K and V must have the same shape as Q (self-attention only)");
    TORCH_CHECK(K.device() == Q.device() && V.device() == Q.device(),
                "Q/K/V must be on the same device");
    int64_t BH = (int64_t)B * H;
    TORCH_CHECK(BH <= 65535, "B*H must be <= 65535 (gridDim.y limit)");

    const at::cuda::CUDAGuard guard(Q.device());
    auto Q_h = Q.to(torch::kHalf).reshape({BH, N, D}).contiguous();
    auto K_h = K.to(torch::kHalf).reshape({BH, N, D}).contiguous();
    auto V_h = V.to(torch::kHalf).reshape({BH, N, D}).contiguous();

    auto O_h = torch::empty({BH, N, D}, Q_h.options());
    torch::Tensor L;
    float* L_ptr = nullptr;
    if (want_L) {
        L = torch::empty({BH, N}, Q.options().dtype(torch::kFloat));
        L_ptr = L.data_ptr<float>();
    }

    dim3 grid((N + BR - 1) / BR, BH);
    dim3 block(NWARPS * 32);
    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto kernel) {
        kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(Q_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(K_h.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(V_h.data_ptr<at::Half>()),
            reinterpret_cast<half*>(O_h.data_ptr<at::Half>()),
            L_ptr,
            N);
    };
    const bool full_tiles = (N % BR == 0) && (N % BC == 0);
    if (want_L) {
        if (full_tiles) launch(attention_fwd_kernel<HD, true, true>);
        else            launch(attention_fwd_kernel<HD, true, false>);
    } else {
        if (full_tiles) launch(attention_fwd_kernel<HD, false, true>);
        else            launch(attention_fwd_kernel<HD, false, false>);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {O_h.reshape({B, H, N, D}),
            want_L ? L.reshape({B, H, N}) : torch::Tensor()};
}

std::vector<torch::Tensor> attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto [O, L] = attention_forward_impl(Q, K, V, /*want_L=*/true);
    return {O, L};
}

torch::Tensor attention_forward_only(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    auto [O, L] = attention_forward_impl(Q, K, V, /*want_L=*/false);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Custom CUDA forward: returns O half, L float");
    m.def("forward_only", &attention_forward_only, "Custom CUDA forward, true O-only");
}
