# Dense O-only regression recovery, 2026-09-21

Historical A/B record. Candidate source copies and the temporary experiment
runner were moved to the owner's external archive at finalization. Commands
below describe that archived experiment, not current entry points. Retained
raw results and validation logs identify their original source hashes.
For the active kernel and runnable benchmark, use the [root README](../../../README.md).

## Result

The remaining D=64, B=1, H/Hkv=32/8, N=2048 dense O-only regression is recovered.
Two paired runs improved on the immediately preceding kernel by 3.8% and 3.3%.
Direct paired comparisons with the original pre-stride HEAD were 0.9995 and
0.9989; both within-run 95% bootstrap intervals include 1. This is recovery of
the old latency, not a new comparison against Flash/cuDNN or a serving result.

| N | B | H/Hkv | Current / preceding, run 1 | Run 2 |
|---:|---:|---:|---:|---:|
| 1024 | 1 | 32/8 | 0.9700 | 0.9683 |
| 1536 | 2 | 12/4 | 0.9670 | 0.9622 |
| 2048 | 1 | 32/8 | 0.9620 | 0.9665 |
| 4096 | 1 | 8/8 | 0.9947 | 0.9936 |

All rows are D=64, non-causal, O-only. Ratios are medians of paired samples,
not ratios of independently aggregated medians. At N=2048, run 1 median times
were 0.870001 ms preceding, 0.837588 ms current, 0.839114 ms original HEAD.
Run 2: 0.874603, 0.846716, 0.847239 ms respectively.

## Change And Selection

The general row strides introduced extra address instructions. A row-contiguous
specialization makes Q/K/V/O row strides compile-time constants while retaining
runtime batch/head strides and the existing 3D launch. QK, softmax, PV, output
normalization, precision, tile dimensions and buffering are unchanged.

The new path requires D=64, O-only, non-causal, full tiles, and row stride D for
all four tensors. It supports square and non-square attention, GQA and ordinary
heads, including padded batch/head layouts. No exact N, B, head count, model
name, or benchmark-case lookup is used. Custom scale and output buffers remain
supported. Inputs with other strides use the existing general path.

A broader candidate (`attention_forward_rows.cu`) was tested first. It recovered
the target but caused about 3.4-3.9% losses on some D=128 paths; some D=64 +L
cases also lost time. It was rejected. The selected candidate leaves +L and
D=128 on their existing implementations. The earlier guarded-causal recovery
is retained.

The original 80 compiled device specializations have identical SASS instruction
lines after matching geometry, flags and KV source across the bool-to-enum name
change. Only four specializations are added, making 84. `codegen_audit.json`
records the per-function hashes. The target full-tile GQA O-only kernel uses
93 registers versus 96 in the general path; all 84 report LOCAL=0. The broader
prototype's function-level dumps show 765 to 709 static instructions and 136
to 110 IMADs, with 32 HMMA and eight LDGSTS instructions unchanged. These are
static counts, not a profiler's dynamic stall attribution.

## New Shapes And Controls

The eight holdout shapes were declared before the first dense candidate was
measured, then timed only after freezing the selected candidate. Sixteen entries
cover both output APIs. No further performance tuning followed these results.

| D64 dense O-only holdout | Current / preceding | Current / original HEAD |
|---|---:|---:|
| B=3, H/Hkv=10/2, N=1792 | 0.9613 | 0.9998 |
| B=2, H/Hkv=15/5, N=2816 | 0.9648 | 1.0021 |
| B=1, H/Hkv=24/6, N=4544 | 0.9684 | 1.0036 |

The three eligible unseen shapes improve 3.2-3.9% over the preceding version.
The table retains their small remaining differences from original HEAD rather
than claiming exact equality on every shape. D=128, causal, +L and partial-tile
controls are included in the raw files. Across both 24-entry development runs
and the 16-entry holdout, no candidate/preceding median exceeds 1.0023.

Token-major Q/K/V with a preallocated strided output were measured separately:
24 entries, ratios 0.9962-1.0021. This path is not eligible for the new row-stride
specialization. There is no evidence here of a serving speedup; TTFT/TPOT and
CUDA-graph serving were not rerun.

## Verification

- Dense suite: 66 cases; existing chunked/paged suite: 40.
- Existing layout checks: 20; new dense full-tile layout checks: 24.
- Empty batch/head rejection: two. Total: 152 passed.
- New checks include square/non-square keys, custom scale, independently strided
  Q/K/V, head padding and strided output, compared with FP32 reference arithmetic.
- Full 152-case memcheck: zero errors (`dense_memcheck.log`).
- Target-kernel-filtered racecheck: 60 cases, zero hazards/errors/warnings
  (`dense_racecheck.log`). This does not represent a full reference-kernel audit.

`dense_tests_final.log` records the selected candidate tests. The first two
narrowed-candidate builds failed during host template compilation, before any
kernel ran (`dense_tests.log`, `dense_tests_retry.log`). Their source snapshots
are retained. Moving dispatch from the nested generic lambda to a named helper
resolved that build issue; device arithmetic was not changed.

After applying the patch, the live source was rebuilt and passed all 152 cases
again (`applied_tests.log`, including its loaded .so path/hash). All 84 installed
device instruction streams match the measured candidate exactly
(`applied_codegen_audit.json`). The source hashes differ only because of line
endings, not code; the sanitizer runs above used the preserved candidate.

The initial code-generation audit did not match mangled symbols because the
new enum shifts type-substitution indices. That unsuccessful audit is retained
as `codegen_audit_symbol_mismatch.json`; the corrected audit matches all flags
and the KV source and rejects duplicate keys.

## Provenance

RTX 4060 Ti, 8188 MiB (8 GB), WSL2, torch 2.10.0+cu128, CUDA 12.8, sm_89.
No GPU clock/power settings were changed. The external process guard recorded
1-3% during the selected runs. Raw files retain temperatures, clocks and memory.

- Immediately preceding source: `attention_forward_before.cu`, SHA256
  `17e49362ed66ff1853e2cbd2356bbf8e776c63e06a1ef68c5db238f2158d7b19`.
- Measured candidate: `attention_forward_dense.cu`, SHA256
  `8126e61f7c96f4d4b2162598207fdcea44524d0a890340915fbaff4d112db0bf`.
- Installed `cuda/attention_forward.cu`, SHA256
  `6fc64206c542c7aa86c2ce609b4fe15222f0080d73c3fdceaf30ce0f1eff0ae0`.
  It differs from the candidate only in line endings.
- Original pre-stride HEAD: `attention_forward_HEAD_32056b7.cu` from
  `bench/results/raw_2026-09-20_v5/`, SHA256
  `215471067d4b750961314f220017a96c75cfc313ce7cfb4adbb560cf86afaa3c`.

The paired protocol is unchanged: warmed public API calls, CUDA events, 18
rotating-order rounds, approximately 30 ms per sample, no CUDA graphs. Builds,
warm-up, correctness checks and sanitizer runs are outside timing. Bootstrap
intervals describe the paired samples within a run, not all future deployments.

Raw selected results: `dense_run1.json`, `dense_run2.json`, `dense_holdout.json`,
`dense_token_major.json`, with corresponding logs. `summary.json` recomputes
direct candidate/preceding and candidate/HEAD ratios from individual samples.
The broader rejected candidate is recorded in `rows_run1.json`.

FP32 baseline, decode implementations, old sealed campaign results and root
README benchmark numbers are untouched. No commit, push or cloud resources.

## Reproduce

From the repository in its existing WSL environment:

```sh
export CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.9
python3 bench/bench_prefill_layout.py --action test --check-empty
python3 bench/bench_prefill_layout.py --candidate docs/serving/prefill_dense_2026-09-21/attention_forward_dense.cu --baseline docs/serving/prefill_dense_2026-09-21/attention_forward_before.cu --action bench --set dense-development --head --output /tmp/dense-new.json
```

Use `--set dense-holdout` for the unseen shapes, or drop `--head` and use
`--layout token_major` for the strided control. Result paths must not already
exist. `summarize.py` accepts raw result paths and writes recomputed JSON to
stdout. Each raw JSON contains source hashes, loaded .so paths/hashes and flags.
