#include "../cuda/flash_fwd_dense_dispatch.h"

static std::string selected_layout(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                   torch::Tensor O, bool causal, bool write_l) {
    float unused_l;
    const ForwardLaunch args{Q, O, write_l ? &unused_l : nullptr,
        (int)K.size(2), (int)(Q.size(1) / K.size(1)), causal, 1.0f};
    const DenseKV source{nullptr, nullptr, strides_of(K), strides_of(V)};
    switch (DenseForwardDispatch::select_layout(args, source)) {
        case AddressLayout::ContiguousHeads: return "contiguous_heads";
        case AddressLayout::ContiguousRows: return "contiguous_rows";
        default: return "strided";
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selected_layout", &selected_layout);
}
