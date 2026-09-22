# Prefill layout regression experiment, 2026-09-21

Historical A/B record, superseded by the subsequent dense regression fix.
Candidate copies and the temporary experiment runner were moved to the owner's
external archive at finalization. Commands below require that archive; they
are not current entry points. Raw measurements and validation logs are retained.
Use the [root README](../../../README.md) for the active kernel and benchmark.

## Scope

Recover the contiguous, square, guarded causal paths that slowed down when the
forward kernel gained general strides. This is a paired kernel/API comparison,
not a new SDPA/cuDNN comparison or an end-to-end serving campaign. The decode
backend, precision, tile geometry and attention arithmetic are unchanged.

Measured hardware: RTX 4060 Ti, 8188 MiB reported by nvidia-smi (8 GB), WSL2,
torch 2.10.0+cu128, CUDA 12.8. Older reports saying 16 GB do not describe the
memory capacity observed during this experiment.

## Change

The selected candidate is `attention_forward_guarded.cu`, SHA256
`b5a8bb5620105af913bf5567a6f2c42e9582fd460f4a0516787eca46e29e3c41`.
It is now applied to `cuda/attention_forward.cu`. The installed source also adds
an explicit nonempty batch/head check before dispatch and corrects a parameter
comment. Those final host-validation-only changes are recorded separately in
`applied_final_tests.log`; the measured experiment source is preserved unchanged.
Final installed source SHA256:
`17e49362ed66ff1853e2cbd2356bbf8e776c63e06a1ef68c5db238f2158d7b19`.
The measured and final builds have identical function-label/SASS-instruction
streams (86,752 lines, SHA256
`f476d0d369ce4401cdb56aee4e412f7ec5302741d9a50c01624a3bbd7dca9584`).
Both contain 80 kernels; the maximum reported LOCAL allocation is zero.

Its extra specialization requires all of the following:

- Causal, square attention (`N_q == N_kv`) with a partial tile.
- Q/K/V/O have contiguous `[B,H,N,D]` addressing after existing input validation.
- The flattened `B*H` launch fits CUDA's grid-y limit.

This path uses a flattened head index and compile-time row strides. It shares
the same QK, softmax, PV and epilogue functions with the general path. There is
no exact sequence length, batch size, head count or model-name dispatch table.
Full tiles, dense attention, non-square attention, strided views and paged KV
keep the general path. The narrower scope is deliberate: the broader candidates
below regressed other paths. Only eight additional kernel instantiations are
needed (D, L output, GQA), rather than doubling every mode.

## Paired Results

`bench/bench_prefill_layout.py`, FP16 inputs / FP32 accumulation, unchanged
`forward` and `forward_only` APIs, warm cache, CUDA events, 18 alternating-order
paired rounds, approximately 30 ms per sample. Host launch/API costs can affect
the smallest cases; these are not CUDA-graph or engine timings. Compilation and
correctness checks are outside timing. Each JSON records source and shared
library hashes, timings, paired ratios, GPU state and the external-GPU-user gate.
Bootstrap intervals describe the paired samples within a run, not deployment
variability. Two development runs and one previously untimed holdout were used.

Reference v7 source:
`bench/results/raw_2026-09-20_v5/attention_forward_v7.cu`, SHA256
`5f1d9815423d9fe133575609f7cc96e1c83bf5ff8f733f042d64c92a0cd1ed24`.
The original HEAD source is separately included in the development A/B runs.

Times below are candidate / v7 medians (less than 1 is faster), causal:

| B | H/Hkv | N | D | +L run 1 / 2 | O-only run 1 / 2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 32/8 | 1000 | 64 | 0.9921 / 0.9890 | 0.9870 / 0.9893 |
| 3 | 14/2 | 2500 | 64 | 0.9917 / 0.9918 | 0.9893 / 0.9920 |
| 1 | 8/8 | 2047 | 64 | 0.9812 / 0.9795 | 0.9769 / 0.9787 |
| 1 | 28/4 | 3000 | 128 | 0.9724 / 0.9713 | 0.9819 / 0.9811 |
| 2 | 16/8 | 1500 | 128 | 0.9740 / 0.9741 | 0.9839 / 0.9817 |

Previously untimed holdout, fixed in the runner before candidate selection:

| B | H/Hkv | N | D | mask | +L | O-only |
|---:|---:|---:|---:|---|---:|---:|
| 2 | 6/2 | 1537 | 64 | causal | 0.9925 | 0.9928 |
| 1 | 12/3 | 2305 | 128 | causal | 0.9757 | 0.9824 |
| 2 | 12/4 | 960 | 64 | dense | 0.9969 | 0.9985 |
| 1 | 20/5 | 3584 | 128 | dense | 1.0006 | 0.9986 |
| 1 | 6/2 | 3073 | 64 | dense | 1.0010 | 1.0002 |
| 3 | 4/2 | 192 | 128 | causal | 1.0021 | 0.9977 |

Token-major Q/K/V and token-major preallocated `out=` were also measured across
24 entries: median candidate/v7 ratios 0.9961-1.0099. The two entries above 1.005
have paired intervals crossing 1. No serving speedup is inferred from this.

Raw files: `guarded_run1.json`, `guarded_run2.json`, `guarded_holdout.json`,
`guarded_token_major.json`, with corresponding `.log` files. The tiny N=128
development control scattered from 1.0497 to 0.9949 (+L) between runs. Its
host-sensitive timing is not a stable improvement or regression finding.

## Rejected Candidates

- `attention_forward_packed.cu`: constant row strides alone. It did not recover
  the guarded causal loss and regressed some D=128 paths. `packed_run1.json`.
- `attention_forward_contiguous.cu`: flattened contiguous addressing applied
  everywhere. It recovered guarded causal performance but lost existing aligned
  causal and D=128 dense gains. `contiguous_run1.json`.

Both remain as experiments, not production dispatch alternatives. The first
benchmark attempt stopped before timing because the read-only Windows GPU guard
could not run under that shell's execution policy; the retry is kept separately.

## Verification

The existing dense suite (66), existing chunked/paged suite (40), and 20 added
layout cases passed. Added cases cover head/batch padding, independently strided
Q/K/V, strided output, custom scale, both head dimensions and both masks against
FP32 arithmetic. The test harness injects the explicitly hashed extension into
the unchanged suites; it does not substitute the numerical reference.
Two final rejection cases cover empty batch/head tensors before launch.
The installed source passed all 128 cases in `applied_final_tests.log`.

Memcheck completed: 126 cases passed, zero errors (`guarded_memcheck.log`).
Targeted racecheck completed: 14 boundary/GQA/paged cases plus the 20 layout
cases, zero hazards/errors/warnings (`guarded_racecheck_smoke.log`). This run
filters on `kns=attention_fwd_kernel`, not PyTorch's reference kernels. The
unfiltered full-suite racecheck was explicitly stopped for cost before it
completed; `guarded_racecheck.log` is an incomplete attempt, not a passing run.
After the final host validation change, the installed source passed the same
34 targeted cases and two empty-input rejection cases under filtered racecheck:
36 cases, zero hazards/errors/warnings (`applied_final_racecheck.log`).
The 126-case memcheck above used the frozen measured candidate, not the final
host-validation revision; their compiled device instruction streams are identical.

## Remaining Work

Follow-up: the dense O-only loss below was subsequently recovered in the
[dense regression experiment](../prefill_dense_2026-09-21/README.md). The results
in this document describe the earlier source revision and are kept unchanged.

The D=64, H=32/8, N=2048 dense O-only path remains approximately 3% behind the
original pre-stride HEAD. This patch does not fix every historical regression.
There is no TTFT/TPOT rerun, no latest generic-decode integration, and no claim
that this change beats SDPA/cuDNN. Existing sealed serving results are untouched.
No commit, push, cloud job, clock or power change was made.

## Reproduce

From the repository in WSL with CUDA 12.8 and the existing torch environment:

```sh
export CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.9
python3 bench/bench_prefill_layout.py --action test --check-empty
python3 bench/bench_prefill_layout.py --candidate docs/serving/prefill_layout_2026-09-21/attention_forward_guarded.cu --action test
python3 bench/bench_prefill_layout.py --candidate docs/serving/prefill_layout_2026-09-21/attention_forward_guarded.cu --action bench --head --output /tmp/prefill-layout-new.json
```

The runner refuses to overwrite an existing result. Use `--set holdout` or
`--layout token_major` for the other measurements. The HEAD comparison only
supports contiguous inputs because the old API has no strided `out=` support.
