#pragma once

#include "flash_fwd_launch_template.h"

struct PagedForwardDispatch {
    template <int D, typename Launch, typename... KVFields>
    static void run(Launch& launch, const ForwardLaunch& args, KVFields...) {
        TORCH_CHECK(args.causal && args.L == nullptr, "this K/V source is built for causal, O-only use");
        const bool full_tiles = args.Q.size(2) % kBlockM == 0 && args.N_kv % kBlockN == 0;
        const bool gqa = args.kv_group != 1;
        KernelChoice<D, AddressLayout::Strided, PagedKV, Launch, KVFields...> paged{launch};
        with_flags(paged, Flags</*WRITE_L=*/false, /*CAUSAL=*/true, /*SQUARE=*/false>{},
                   full_tiles, gqa);
    }
};
