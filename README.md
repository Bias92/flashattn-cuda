# flashattn-cuda

Scratch CUDA attention for **head dimension 64 on an RTX 4060 Ti 8 GB**.
The forward kernel uses the FlashAttention-2 algorithm, inline `mma.sync`,
register-resident softmax and `cp.async` double buffering, without CUTLASS.
Inputs are FP16; matrix products and softmax accumulate in FP32.

The project includes dense/causal prefill, GQA, paged decode, and a vLLM 0.19
integration. D128 remains in the implementation for compatibility, but is
outside the final performance scope. No further D128 tuning is planned.

## Kernel Results

Measured on 2026-09-22 with PyTorch 2.10.0+cu128. These are **O-only public API
latencies**, not full-model serving results. Flash and cuDNN are separately
forced PyTorch SDPA backends; default SDPA selected Flash in the measured cases.

| D64 workload | Cases | vs SDPA-Flash: faster / similar / slower | vs SDPA-cuDNN: faster / similar / slower |
|---|---:|---:|---:|
| Dense | 17 | 10 / 4 / 3 | 2 / 4 / 11 |
| Causal | 17 | 15 / 0 / 2 | 16 / 0 / 1 |

Similar means within +/-1%, a practical band rather than a significance test.
The set varies batch size from 1 to 8, sequence length from 512 to 8192, head
count and GQA ratio. It includes aligned and non-aligned lengths. These are
development/comparison cases, not an untouched holdout.

Representative B=1, H=8, D=64 cases:

| Mask | N | Scratch ms | Flash ms | cuDNN ms | Paired gap vs Flash | Paired gap vs cuDNN |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 1024 | 0.06227 | 0.05998 | 0.05782 | +2.71% | +7.90% |
| Dense | 4096 | 0.86249 | 0.87860 | 0.85070 | -1.48% | +2.64% |
| Causal | 2048 | 0.14025 | 0.15490 | 0.14524 | -9.51% | -3.45% |
| Causal | 4096 | 0.48298 | 0.51368 | 0.51085 | -5.73% | -5.23% |

Negative gaps mean lower latency. Times and paired ratios are aggregated
independently, so dividing the displayed time medians does not reproduce the
gap columns. There are three independent runs, each with 12 rotating-order
paired repetitions and 30 ms windows. CUDA events time warmed calls without
CUDA graphs; compilation, plan selection and correctness checks are excluded.

Causal D64 is the strongest part of this implementation. Dense D64 is close
to Flash, while cuDNN is faster on most dense cases.
[Full matrix, +L results, methodology and raw JSON](docs/serving/backend_overview_2026-09-22/README.md).

Measured prefill source SHA256:
`6fc64206c542c7aa86c2ce609b4fe15222f0080d73c3fdceaf30ce0f1eff0ae0`.

## Serving

The vLLM integration has three configurations: native FlashAttention, scratch
decode-only, and scratch prefill+decode. The full backend handles fresh prompts,
chunked prefill, prefix-cache reads and mixed prefill/decode batches. vLLM still
owns scheduling, KV-cache writes, RMSNorm, RoPE, projections and sampling.

The **2026-09-20/21 campaign used an earlier prefill source**, not the source in
the kernel table above. It recorded 252 accepted cases across low latency,
throughput and long context, with three repeats and separate eager/graph runs.
Models were TinyLlama-1.1B and Qwen2.5-0.5B, both D64.

With CUDA graphs, the full backend's long-context TTFT was 2.2-3.9% lower than
native FlashAttention over the five tested lengths. Output throughput at
concurrency 2/4/8/16/32 was approximately unchanged (-0.1% to +1.3%). These are
whole-configuration comparisons, including different cache access paths.
They are not isolated kernel speedups or a rerun of the latest source.

[Integration, reproduction and limitations](docs/serving/README.md) |
[Campaign report and original results](docs/serving/prefill_campaign_2026-09-20/REPORT.md).

## Implementation

- Prefill: four warps per block, QK and PV through `mma.sync`, online softmax,
  and double-buffered K/V. Causal work stops at the diagonal.
- Dense input: strided Q/K/V, GQA, runtime scale, caller-provided output,
  and unequal query/KV lengths with bottom-right causal alignment.
- Paged prefill: direct block-table reads for one sequence per launch.
  The serving backend dispatches individual requests from a mixed batch.
- Decode: a paged split-KV attention kernel and a separate merge kernel.
  A dense-cache decode implementation is also retained.

The last tensor dimension must be contiguous. Alignment requirements can
cause input copies; unsupported `out=` alignment/overlap is rejected.
This is forward-only FP16 inference: no dropout, backward, arbitrary masks,
FP8, or multi-sequence varlen prefill in one launch. The build targets sm_89.

## Build And Check

WSL2/Linux, Python 3.12, CUDA toolkit 12.8 and a matching CUDA PyTorch installation:

```sh
CUDA_HOME=/usr/local/cuda-12.8 python3 -m pip install -e . --break-system-packages
python3 tests/test_attention_forward.py
python3 tests/test_attention_chunked_paged.py
```

Reproduce the D64 kernel comparison with a new output filename for each run:

```sh
CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.9 \
python3 bench/report_attention_backends.py --run 1 --output /tmp/d64-run1.json
```

Use run IDs 1, 2 and 3 in separate processes. Run alone on an idle GPU.
The runner records source/build hashes, loaded library paths, actual backend
dispatch, paired samples and GPU telemetry. Existing results are not overwritten.

## Repository Map

| Path | Role |
|---|---|
| `cuda/attention_forward.cu` | Current prefill kernel |
| `cuda/attention_decode_paged.cu` | Paged decode actually loaded by vLLM |
| `cuda/attention_decode.cu` | Dense-cache decode |
| `integrations/vllm_prefill/` | Full prefill+decode backend |
| `integrations/vllm/` | Decode-only backend and SDPA comparison adapters |
| `tests/`, `bench/` | Correctness and measurement tools |
| `docs/serving/` | Dated measurements, raw evidence and integration guide |
| `experiments/` | Historical optimization stages, including the preserved FP32 baseline |

Only the root README identifies the current scope and headline results.
Dated records describe their recorded source hashes, not whichever kernel is
currently checked out. Superseded diagnostic scripts and unselected candidates
were removed from the active tree during finalization; historical result data
and the original FP32 baseline were preserved.
