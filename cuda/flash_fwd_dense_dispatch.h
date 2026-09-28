#pragma once

#include "flash_fwd_launch_template.h"

struct DenseForwardDispatch {
    static AddressLayout select_layout(const ForwardLaunch& args, const DenseKV& source) {
        const int B = args.Q.size(0), H_q = args.Q.size(1);
        const int N_q = args.Q.size(2), D = args.Q.size(3), N_kv = args.N_kv;
        const bool full_tiles = (N_q % kBlockM == 0) && (N_kv % kBlockN == 0);
        const bool contiguous_rows = D == 64 && args.L == nullptr && !args.causal && full_tiles
            && args.Q.stride(2) == D && args.O.stride(2) == D
            && source.stride_k.n == D && source.stride_v.n == D;

        const int H_kv = H_q / args.kv_group;
        const bool k_contiguous = source.stride_k.n == D
            && (H_kv == 1 || source.stride_k.h == (int64_t)N_kv * D)
            && (B == 1 || source.stride_k.b == (int64_t)H_kv * N_kv * D);
        const bool v_contiguous = source.stride_v.n == D
            && (H_kv == 1 || source.stride_v.h == (int64_t)N_kv * D)
            && (B == 1 || source.stride_v.b == (int64_t)H_kv * N_kv * D);
        const bool packed_rows = args.causal && N_q == N_kv && !full_tiles
            && args.Q.is_contiguous() && args.O.is_contiguous()
            && k_contiguous && v_contiguous && (int64_t)B * H_q <= 65535;

        if (packed_rows) return AddressLayout::ContiguousHeads;
        if (contiguous_rows) return AddressLayout::ContiguousRows;
        return AddressLayout::Strided;
    }

    template <int D, typename Launch, typename... KVFields>
    static void run(Launch& launch, const ForwardLaunch& args, KVFields... kv_fields) {
        const DenseKV source{kv_fields...};
        const AddressLayout layout = select_layout(args, source);
        const bool want_L = args.L != nullptr;
        const bool square = args.Q.size(2) == args.N_kv;
        const bool full_tiles = args.Q.size(2) % kBlockM == 0 && args.N_kv % kBlockN == 0;
        const bool gqa = args.kv_group != 1;

        if (layout == AddressLayout::ContiguousHeads) {
            KernelChoice<D, AddressLayout::ContiguousHeads, DenseKV, Launch, KVFields...> flat{launch};
            if (want_L)
                with_flags(flat, Flags</*WRITE_L=*/true, /*CAUSAL=*/true,
                                       /*SQUARE=*/true, /*FULL_TILES=*/false>{}, gqa);
            else
                with_flags(flat, Flags</*WRITE_L=*/false, /*CAUSAL=*/true,
                                       /*SQUARE=*/true, /*FULL_TILES=*/false>{}, gqa);
            return;
        }
        if constexpr (D == 64) {
            if (layout == AddressLayout::ContiguousRows) {
                KernelChoice<D, AddressLayout::ContiguousRows, DenseKV, Launch, KVFields...> rows{launch};
                if (square)
                    with_flags(rows, Flags</*WRITE_L=*/false, /*CAUSAL=*/false,
                                           /*SQUARE=*/true, /*FULL_TILES=*/true>{}, gqa);
                else
                    with_flags(rows, Flags</*WRITE_L=*/false, /*CAUSAL=*/false,
                                           /*SQUARE=*/false, /*FULL_TILES=*/true>{}, gqa);
                return;
            }
        }
        KernelChoice<D, AddressLayout::Strided, DenseKV, Launch, KVFields...> general{launch};
        with_flags(general, Flags<>{}, want_L, args.causal, square, full_tiles, gqa);
    }
};
