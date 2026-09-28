# FA2 Forward

Read `attention_fwd_kernel` in [flash_fwd_kernel.h](flash_fwd_kernel.h).
Each block keeps its Q tile in registers and streams K/V tiles through shared memory:

```text
load Q -> [load K/V -> QK -> mask -> online softmax -> PV] -> normalize/store O
```

`forward` also returns the row log-sum-exp L. `forward_only` returns O.

| File | Responsibility |
|---|---|
| [attention_forward.cu](attention_forward.cu) | Python APIs and tensor validation |
| [flash_fwd_dense_dispatch.h](flash_fwd_dense_dispatch.h) | Select the kernel for input layout, mask, lengths and heads |
| [flash_fwd_launch_template.h](flash_fwd_launch_template.h) | CUDA stream, grid, shared memory and launch |
| [flash_fwd_kernel.h](flash_fwd_kernel.h) | QK, masks, PV, output and the forward loop |
| [flash_fwd_memory.h](flash_fwd_memory.h) | Tensor addressing, asynchronous K/V copies and Q loading |
| [softmax.h](softmax.h) | Running row maximum/sum and output rescaling |
| [kernel_traits.h](kernel_traits.h) | Tile dimensions and fragment counts |
| [attention_forward_ops.cuh](attention_forward_ops.cuh) | PTX wrappers for mma, ldmatrix and cp.async |

## Notation

- `N_q`, `N_kv`: query and key/value lengths; `H_q`, `H_kv`: head counts.
- `stride_q/k/v/o`: tensor strides in elements. `sQ`: Q in shared memory.
- `tSrQ`, `tSrK*`: register operands for QK; `tOrP`, `tOrVt*`: operands for PV.
- `acc_s`: FP32 scores, then unnormalized probabilities; `acc_o`: FP32 output sum.
- `kBlockM`, `kBlockN`: query/KV tile rows; `kNWarps`: warps per block.

## Specializations

`WRITE_L`, `CAUSAL`, `SQUARE`, `FULL_TILES` and `GQA` select output, masking,
equal-length, boundary-check and head-sharing paths at compile time.
`ContiguousRows` fixes row strides to D; `ContiguousHeads` also flattens the
batch/head grid. `Strided` uses explicit strides. All paths compute the same
attention operation; the selection depends on input properties.

The default tile is 64 by 32 with four warps and two K/V stages.
The extension has no serving-engine dependency. Dated measurements retain
their original source hashes and are not rerun by reorganizing this code.
