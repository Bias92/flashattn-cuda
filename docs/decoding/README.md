# Single-token CUDA decoding

Whole-model follow-up: [decode profiling findings](PROFILE_FINDINGS.md) separates
attention, KV cache, projections, MLP, and CPU submission instead of treating
CUDA-event elapsed time as pure GPU compute.

## Scope and implementation

Implemented locally on 2026-09-14 in `C:\Users\PC\flashattn-cuda-main`.
The existing prefill implementation and Claude's uncommitted `bench/run_model.py`
were not modified. This is a separate, scratch CUDA inference path, not an
implementation of the upstream FlashAttention library.

- Q: `[B,Hq,1,D]`; K/V: `[B,Hkv,N,D]`, FP16, D=64/128, Hq divisible by Hkv.
- Every supplied cache token is valid and visible to the query. No built-in
  padding, arbitrary attention mask, dropout, paged cache, or backward.
- Cache storage/allocation/appending is handled by the caller (Transformers in
  the integration benchmark). Aligned strided cache views avoid a full copy.
- `forward_only(Q,K,V,scale=None,split_size=0)` returns FP16 O.
  `forward(...)` additionally returns FP32 natural-log logsumexp L.
- FP32 QK/PV accumulation and softmax statistics. The MMA path rounds softmax
  weights to FP16 before PV, as does the existing prefill implementation.
- Default split size: 64 cache tokens through N=2048, 128 beyond that.
  Explicit sizes 64/128/256/512/1024 are available for ablation.

`attention_decode.cu` contains validation, allocation, scalar split attention,
and the final reduction. `attention_decode_mma.cuh` groups up to 16 query heads
sharing a KV head into one MMA row tile. One warp streams 32 K/V tokens at a
time via `cp.async`, using `mma.sync.m16n8k16` for QK and PV. The row-sum shuffle
is deferred to the epilogue. No attention score matrix is materialized globally.
The merge rescales FP32 partial outputs using their maxima and denominators;
at 8 or more splits it distributes the work across four warps.

The split/merge algorithm follows the published
[Flash-Decoding principle](https://pytorch.org/blog/flash-decoding/).
This record claims implementation/performance evidence, not novelty of that algorithm.

## Decisions

1. A vectorized scalar split-K kernel was built first and passed 50 numerical
   cases. It was slower than SDPA on longer caches in the first tuning run.
   The implementation remains available as `scalar_only` for ablation.
2. Group query heads sharing K/V into a Tensor Core tile. This avoids loading
   the same tile independently for every query head. See `tuning_mma.json`.
3. Parallelize the split merge and select 64/128-token splits from the measured
   trade-off. Benchmark outputs retain the exact combined source hash and .so
   path; current source hash is
   `ab7d1630373290954cd3bc691056f477200ea32a28d70a6c196806b3d67ada16`.

An initial scalar tail implementation incorrectly used a full-warp shuffle
inside a divergent row loop and hung on N=1. The test process was terminated;
the loop was made uniform before any reported performance measurements.

## Validation

`correctness.json`: 56 shape/numerical cases for both scalar and grouped paths,
plus a non-default stream and 6 invalid-input checks. Includes N=1, non-tile
aligned lengths, GQA groups crossing the 16-head tile, D64/128, transposed and
sliced caches, misaligned storage offsets, custom scales, and amplitude-16
inputs. Reference: FP32 arithmetic on the same FP16 inputs, TF32 disabled.
O tolerance is `atol=8e-4*amplitude, rtol=1e-3`; L uses
`atol=3e-4, rtol=3e-6`. O-only and O+L outputs must match bitwise.
These are tested tolerances, not a global error guarantee.

Compute Sanitizer also completed the 48-case grouped-kernel workload (including
1/33/129/257 cache lengths and 3/8/17 query heads) with zero memcheck errors and
zero racecheck hazards. See `memcheck.log`, `racecheck.log`, and
`tests/check_decode_sanitizer.py`. This is not exhaustive formal verification.

The Transformers adapter registers both an attention callback and the existing
SDPA mask factory. It delegates masked/padded requests, explicit upper-left
causal decode, dropout, and grad-enabled inputs to Transformers' actual SDPA
helper. Six adapter regression cases cover masks (including fully masked rows),
scaling, and explicit causal semantics. The prefill path in the decode model
benchmark is unchanged HF SDPA in both arms.

## Measurements

RTX 4060 Ti, torch 2.10.0+cu128, CUDA 12.8; B=1, Hq=32, Hkv=4, Nq=1.
All microbenchmarks use FP16 inputs and return O only. Automatic SDPA, forced
Flash, and forced cuDNN are measured separately. Actual backend operator and
kernel names are captured outside timing. In these tests automatic SDPA
selected Flash's split-KV path.

The two final microbenchmark files use 10 randomized paired repetitions each:
`decode_benchmark_run1.json` and `decode_benchmark_run2.json`.
CUDA graph times are device-side, repeated warm-cache attention calls. Eager
API measurements are also included and fluctuate substantially more because
they include host issue gaps. Neither is a whole-model generation speedup.

| D64 cache length | Custom graph us, run 1 / 2 | SDPA default us, run 1 / 2 | cuDNN us, run 1 / 2 |
|---:|---:|---:|---:|
| 512 | 4.16 / 4.52 | 7.84 / 8.15 | 5.23 / 5.67 |
| 2048 | 5.85 / 6.12 | 9.09 / 9.20 | 5.72 / 5.82 |
| 4096 | 7.86 / 7.81 | 15.07 / 15.02 | 7.09 / 7.03 |

At D64/N4096 the custom graph path takes about 52% of automatic SDPA/Flash time
in both runs. It remains slower than cuDNN there. D128 results are in the same
JSON files and must not be replaced by the faster D64 figures.

`run_decode_model.py` checks TinyLlama-1.1B logits and argmax with separate KV
caches and identical forced continuations. It then measures actual cached
model steps, including mask extension and greedy selection, recording both
CUDA-event intervals and synchronized wall time per token. It does not estimate
TPOT by subtracting a separate prefill call from `generate` time. This is an
uncompiled Transformers model benchmark, not a serving-engine benchmark.

Whole-model results do **not** establish a reproducible TPOT speedup:

| Prompt | Run 1 paired custom/SDPA wall ratio (7 pairs, 24 steps) | Run 2 (11 pairs, 32 steps) | Run 2 bootstrap 95% CI |
|---:|---:|---:|---:|
| 512 | 0.822 | 0.998 | [0.795, 1.106] |
| 2048 | 0.858 | 1.034 | [0.865, 1.176] |

These are medians of within-pair ratios, not ratios of independent medians.
The apparent run-1 improvement did not reproduce. Do not turn the attention
microbenchmark improvement into a whole-model headline. Both model runs passed
the unpadded eight-step logits/argmax check (max logit difference 0.015625), and
the padded fallback check matched SDPA exactly. The unpadded check counted
176 custom decode calls (22 layers times 8 tokens); the padded case counted none.

## Reproduction

From the repository in WSL, with no other GPU Python workload running:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1
python3 tests/test_attention_decode.py
python3 tests/test_decode_adapter.py
python3 bench/bench_attention_decode.py --pairs 10
HF_HUB_OFFLINE=1 python3 bench/run_decode_model.py --pairs 11 --steps 32
```

TinyLlama weights must already exist locally; the model benchmark does not
download them. Editable installation also registers `attention_decode_cuda`:

```bash
CUDA_HOME=/usr/local/cuda-12.8 pip install -e . --break-system-packages
```

After installation, the adapter can also connect the existing custom prefill
kernel and the new decode kernel together:

```python
import attention_forward_cuda as prefill
import attention_decode_cuda as decode
from bench.attention_backend import register_backend

register_backend("scratch_cuda", decode, prefill=prefill)
model.set_attn_implementation("scratch_cuda")
```

This combined route is not the measured TPOT baseline: the tests above keep
prefill on SDPA to isolate the effect of changing decode.

The benchmark rejects other Linux Python jobs and records GPU telemetry. This
is a contamination check, not an exclusive hardware lock; Windows desktop GPU
activity and CPU scheduling can still affect timings. Do not run it concurrently
with Claude's benchmarks or profiling.

## Compiler resources and limits

Verified against the hash-named .so with CUDA 12.8 `cuobjdump`:

| Kernel | Registers/thread | Static shared bytes | LOCAL | STACK |
|---|---:|---:|---:|---:|
| D64 grouped split, O or O+L | 156 | 9216 | 0 | 0 |
| D128 grouped split, O or O+L | 252 | 17408 | 0 | 0 |
| D64 parallel merge | 40 | 1056 | 0 | 0 |
| D128 parallel merge | 40 | 2080 | 0 | 0 |

Direct one-split specializations differ slightly (D64 155-156, D128 253-254).
No architecture-wide occupancy claim is inferred from register count alone.
Only sm_89 is built by the current scripts. Other GPU results are unknown.
