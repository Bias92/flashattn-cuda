# Forward Source Map

Start with `flash_fwd_kernel.h` and its `attention_fwd_kernel` function.
The loop reads: stage K/V, QK, mask, online softmax, PV, store O (and optionally L).

| File | Responsibility |
|---|---|
| [attention_forward.cu](attention_forward.cu) | Tensor validation and Python entry points: forward, forward_only, forward_paged |
| [flash_fwd_launch_template.h](flash_fwd_launch_template.h) | ForwardLaunch: launch grid, stream and shared-memory allocation |
| [flash_fwd_dense_dispatch.h](flash_fwd_dense_dispatch.h) | DenseForwardDispatch: layout selection for ordinary K/V tensors |
| [flash_fwd_paged_dispatch.h](flash_fwd_paged_dispatch.h) | PagedForwardDispatch: causal, O-only dispatch for paged K/V |
| [flash_fwd_kernel.h](flash_fwd_kernel.h) | QK, masks, PV, epilogue and the forward loop |
| [kernel_traits.h](kernel_traits.h) | Compile-time tile dimensions and shared-memory sizes |
| [flash_fwd_memory.h](flash_fwd_memory.h) | Dense/paged K/V addressing, asynchronous staging and Q fragments |
| [softmax.h](softmax.h) | Quad reductions and running row statistics |
| [attention_forward_ops.cuh](attention_forward_ops.cuh) | Inline PTX: mma, ldmatrix and cp.async |

## Call Path

```text
forward / forward_only -> ForwardLaunch::run<DenseForwardDispatch>
forward_paged          -> ForwardLaunch::run<PagedForwardDispatch>
                                           |
                         selected attention_fwd_kernel specialization
```

The dense dispatcher selects the address layout. The paged dispatcher uses the
strided layout with causal masking and no L output. Both use the same launcher
and calculation kernel; neither depends on vLLM or benchmark code.

Kernel flags:

| Flag | Meaning |
|---|---|
| `WRITE_L` | Also store L, the natural-log softmax normalizer |
| `CAUSAL` | Apply the bottom-right causal mask |
| `SQUARE` | Query and K/V lengths are equal |
| `FULL_TILES` | No partial query or K/V tile; causal masking still applies |
| `GQA` | Multiple query heads share one K/V head |

`ContiguousRows` uses fixed D-element row strides; `ContiguousHeads` also
flattens the batch/head grid. `Strided` retains explicit tensor strides.
The existing `ATTN_BR`, `ATTN_BC`, and `ATTN_DOUBLE_BUFFER` compile-time
options remain unchanged; the default is a 64-by-32 tile with two K/V stages.

## Names

These names correspond to the roles in upstream FA2, not to CuTe tensor types:

| Name | Meaning here |
|---|---|
| `kBlockM`, `kBlockN` | Query rows per block, K/V rows per streamed tile |
| `kNWarps` | Warps per block |
| `Flash_fwd_kernel_traits<D>` | Shared-memory geometry and fragment counts |
| `tSrQ`, `tSrK*` | Register operands for QK |
| `acc_s` | FP32 score accumulator, then unnormalized probabilities |
| `row_max`, `row_sum` | Running softmax statistics |
| `scores_scale` | Rescale factor for earlier tiles |
| `tOrP`, `tOrVt*` | Register operands for PV |
| `acc_o` | FP32 output accumulator, normalized once in the epilogue |

Dense and paged decode remain in `attention_decode.cu` and
`attention_decode_paged.cu`; their implementation has not been reorganized.

## FA1 and FA2 Organization

The tagged upstream [FA1 v1.0.9 source](https://github.com/Dao-AILab/flash-attention/tree/v1.0.9/csrc/flash_attn/src)
already separates kernel bodies, launch templates, per-head-dimension translation units
and `fmha/` helpers. It is not a single-file implementation.

The tagged [FA2 v2.0.0 source](https://github.com/Dao-AILab/flash-attention/tree/v2.0.0/csrc/flash_attn/src)
uses `flash_fwd_kernel.h`, `flash_fwd_launch_template.h`, `kernel_traits.h`,
`softmax.h` and head-dimension/dtype-specific translation units.
This repository borrows those organizational conventions, not the upstream
CuTe/CUTLASS implementation or its full template/build hierarchy.

## Integration Boundary

`cuda/attention_forward.cu` owns tensor validation and the three Python APIs.
`integrations/vllm_prefill/` adapts vLLM batches to those APIs. `bench/` calls the
APIs for measurement. Neither directory is a dependency of the CUDA extension.

Dispatch uses compile-time policies, not virtual calls. The existing kernel
specializations and Python signatures are retained.

Build records hash the entry point and local headers. Dated benchmark records
retain their original source hashes; reorganizing sources is not a fresh measurement.
