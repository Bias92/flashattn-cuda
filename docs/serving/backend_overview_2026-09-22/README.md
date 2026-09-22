# Attention backend overview, 2026-09-22

Final project scope is D64. D128 rows below are preserved historical measurements,
not an active optimization target. The original 46-case raw files are unchanged.

## Measured Prefill

RTX 4060 Ti 8 GB, FP16 inputs, FP32 scratch accumulation, D=64/128.
46 fixed shape/mask cases from the existing readme/holdout/holdout2 sets.
These are comparison sets, not a new untouched holdout. Three independent
process/seed runs; 12 rotating-order paired repetitions per case per run,
30 ms windows, CUDA events over warmed public APIs. No CUDA graphs.
Backend selection, dispatch profiling, correctness, compilation, cuDNN plan
warmup and allocation of input tensors are outside timing. Output allocation
remains part of each public API call. Clocks were not locked or changed.

Current source SHA256: `6fc64206c542c7aa86c2ce609b4fe15222f0080d73c3fdceaf30ce0f1eff0ae0`.
The loaded .so/hash, helper hashes, versions and GPU telemetry are in each raw JSON.
The kernel was unchanged throughout all three runs.

Faster/similar/slower use a practical +/-1% band on the median of the three
within-run paired ratios, not a significance test. Counts describe these cases
only; they do not weight workloads by deployment frequency.

| D | Mask | Cases | O-only vs Flash F/S/L | O-only vs cuDNN F/S/L |
|---:|---|---:|---|---|
| 64 | dense | 17 | 10/4/3 | 2/4/11 |
| 64 | causal | 17 | 15/0/2 | 16/0/1 |
| 128 | dense | 6 | 1/0/5 | 0/0/6 |
| 128 | causal | 6 | 0/0/6 | 0/0/6 |

SDPA default dispatch was audited for every case in every run:
`[('aten::_scaled_dot_product_flash_attention', 'aten::scaled_dot_product_attention')]`.
O-only and +L outputs were checked against both forced backends with
atol=1e-3, rtol=1e-3; no tolerance or kernel precision was changed.
Largest observed absolute output difference: 0.0009765625.
The preceding 152-case FP32-reference/sanitizer verification is recorded in
[the regression report](../prefill_dense_2026-09-21/README.md).

## O-only Full Matrix

Positive gap means scratch is slower.
Times are independently aggregated medians. Gaps come from paired ratios,
not division of the displayed time medians; their signs can differ near parity.

| Set | B | H/Hkv | N | D | Mask | Scratch ms | Flash ms | cuDNN ms | Flash gap | cuDNN gap |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| readme | 1 | 8/8 | 1024 | 64 | dense | 0.06227 | 0.05998 | 0.05782 | +2.71% | +7.90% |
| readme | 1 | 8/8 | 1024 | 64 | causal | 0.05296 | 0.05270 | 0.05217 | +1.05% | +2.38% |
| readme | 1 | 8/8 | 2048 | 64 | dense | 0.22313 | 0.22501 | 0.21361 | -0.64% | +4.23% |
| readme | 1 | 8/8 | 2048 | 64 | causal | 0.14025 | 0.15490 | 0.14524 | -9.51% | -3.45% |
| readme | 1 | 8/8 | 4096 | 64 | dense | 0.86249 | 0.87860 | 0.85070 | -1.48% | +2.64% |
| readme | 1 | 8/8 | 4096 | 64 | causal | 0.48298 | 0.51368 | 0.51085 | -5.73% | -5.23% |
| readme | 1 | 32/8 | 2048 | 64 | dense | 0.83484 | 0.89203 | 0.85239 | -5.67% | -1.54% |
| readme | 1 | 32/8 | 2048 | 64 | causal | 0.46735 | 0.49443 | 0.47880 | -6.06% | -4.80% |
| readme | 1 | 32/8 | 4096 | 64 | dense | 3.29287 | 3.42422 | 3.34291 | -2.89% | -0.44% |
| readme | 1 | 32/8 | 4096 | 64 | causal | 1.72846 | 1.78491 | 1.84577 | -4.04% | -6.25% |
| readme | 1 | 32/8 | 2048 | 128 | dense | 1.82320 | 1.74606 | 1.62777 | +3.40% | +10.19% |
| readme | 1 | 32/8 | 2048 | 128 | causal | 1.00747 | 0.89886 | 0.87708 | +12.43% | +14.49% |
| readme | 1 | 32/8 | 4096 | 128 | dense | 7.04444 | 6.62458 | 6.34761 | +6.45% | +10.93% |
| readme | 1 | 32/8 | 4096 | 128 | causal | 3.77907 | 3.38550 | 3.33006 | +11.72% | +14.13% |
| holdout | 4 | 12/12 | 1536 | 64 | dense | 0.72290 | 0.71223 | 0.71644 | -0.06% | +1.52% |
| holdout | 4 | 12/12 | 1536 | 64 | causal | 0.41489 | 0.44243 | 0.42589 | -5.46% | -5.10% |
| holdout | 2 | 16/4 | 3072 | 64 | dense | 1.92892 | 1.91807 | 1.85309 | -0.12% | +2.00% |
| holdout | 2 | 16/4 | 3072 | 64 | causal | 0.99568 | 1.05309 | 1.05650 | -3.87% | -6.27% |
| holdout | 1 | 8/8 | 8192 | 64 | dense | 3.37455 | 3.45639 | 3.41856 | -3.64% | -0.80% |
| holdout | 1 | 8/8 | 8192 | 64 | causal | 1.81162 | 1.82800 | 1.86083 | -4.24% | -3.37% |
| holdout | 8 | 8/8 | 512 | 64 | dense | 0.11727 | 0.11535 | 0.11328 | +2.88% | +4.75% |
| holdout | 8 | 8/8 | 512 | 64 | causal | 0.07455 | 0.08255 | 0.08217 | -9.76% | -9.22% |
| holdout | 1 | 32/8 | 1000 | 64 | dense | 0.22669 | 0.23451 | 0.21964 | -2.29% | +3.99% |
| holdout | 1 | 32/8 | 1000 | 64 | causal | 0.13461 | 0.14604 | 0.14306 | -8.98% | -5.96% |
| holdout | 3 | 14/2 | 2500 | 64 | dense | 1.70956 | 1.79268 | 1.71699 | -3.32% | +2.98% |
| holdout | 3 | 14/2 | 2500 | 64 | causal | 0.90839 | 0.95371 | 0.96732 | -4.58% | -3.78% |
| holdout | 2 | 16/8 | 1536 | 128 | dense | 1.02707 | 1.00030 | 0.93648 | +3.12% | +10.01% |
| holdout | 2 | 16/8 | 1536 | 128 | causal | 0.59700 | 0.52689 | 0.51636 | +13.66% | +15.66% |
| holdout | 1 | 28/4 | 3000 | 128 | dense | 3.18257 | 3.23005 | 3.05327 | -1.35% | +5.51% |
| holdout | 1 | 28/4 | 3000 | 128 | causal | 1.75931 | 1.64560 | 1.63371 | +6.44% | +7.47% |
| holdout2 | 2 | 24/8 | 2304 | 64 | dense | 1.64104 | 1.60990 | 1.61796 | -0.62% | +1.31% |
| holdout2 | 2 | 24/8 | 2304 | 64 | causal | 0.85374 | 0.92823 | 0.88794 | -8.24% | -3.99% |
| holdout2 | 1 | 16/16 | 5000 | 64 | dense | 2.56521 | 2.62514 | 2.57171 | -1.97% | -0.22% |
| holdout2 | 1 | 16/16 | 5000 | 64 | causal | 1.37299 | 1.47991 | 1.47161 | -6.82% | -5.79% |
| holdout2 | 6 | 10/2 | 768 | 64 | dense | 0.23403 | 0.23706 | 0.22928 | -1.31% | +2.71% |
| holdout2 | 6 | 10/2 | 768 | 64 | causal | 0.13884 | 0.15242 | 0.15142 | -7.96% | -7.55% |
| holdout2 | 1 | 40/8 | 1700 | 64 | dense | 0.75028 | 0.83722 | 0.78939 | -10.38% | -5.29% |
| holdout2 | 1 | 40/8 | 1700 | 64 | causal | 0.41418 | 0.47869 | 0.46932 | -13.38% | -11.53% |
| holdout2 | 2 | 20/4 | 6144 | 64 | dense | 9.59093 | 9.19961 | 8.96565 | +3.99% | +6.93% |
| holdout2 | 2 | 20/4 | 6144 | 64 | causal | 4.87668 | 4.80081 | 4.93352 | +1.74% | -1.17% |
| holdout2 | 5 | 9/3 | 1111 | 64 | dense | 0.37370 | 0.39896 | 0.36671 | -6.98% | +0.85% |
| holdout2 | 5 | 9/3 | 1111 | 64 | causal | 0.22357 | 0.23797 | 0.23878 | -7.27% | -7.12% |
| holdout2 | 1 | 24/8 | 2560 | 128 | dense | 2.14220 | 2.04962 | 1.94309 | +3.60% | +10.16% |
| holdout2 | 1 | 24/8 | 2560 | 128 | causal | 1.17572 | 1.05148 | 1.03739 | +12.26% | +14.12% |
| holdout2 | 3 | 12/6 | 1900 | 128 | dense | 1.70624 | 1.67676 | 1.63028 | +1.50% | +5.58% |
| holdout2 | 3 | 12/6 | 1900 | 128 | causal | 0.96252 | 0.91036 | 0.88041 | +5.14% | +8.93% |
## O + L Full Matrix

Positive gap means scratch is slower.
Times are independently aggregated medians. Gaps come from paired ratios,
not division of the displayed time medians; their signs can differ near parity.

| Set | B | H/Hkv | N | D | Mask | Scratch ms | Flash ms | cuDNN ms | Flash gap | cuDNN gap |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| readme | 1 | 8/8 | 1024 | 64 | dense | 0.06230 | 0.05998 | 0.05782 | +4.59% | +7.34% |
| readme | 1 | 8/8 | 1024 | 64 | causal | 0.05312 | 0.05270 | 0.05217 | +1.25% | +3.18% |
| readme | 1 | 8/8 | 2048 | 64 | dense | 0.22761 | 0.22501 | 0.21361 | +1.21% | +4.95% |
| readme | 1 | 8/8 | 2048 | 64 | causal | 0.14068 | 0.15490 | 0.14524 | -9.29% | -3.20% |
| readme | 1 | 8/8 | 4096 | 64 | dense | 0.86260 | 0.87860 | 0.85070 | -2.39% | +2.04% |
| readme | 1 | 8/8 | 4096 | 64 | causal | 0.48392 | 0.51368 | 0.51085 | -6.49% | -5.09% |
| readme | 1 | 32/8 | 2048 | 64 | dense | 0.84381 | 0.89203 | 0.85239 | -4.46% | -1.00% |
| readme | 1 | 32/8 | 2048 | 64 | causal | 0.45854 | 0.49443 | 0.47880 | -6.92% | -4.57% |
| readme | 1 | 32/8 | 4096 | 64 | dense | 3.31833 | 3.42422 | 3.34291 | -3.25% | -0.81% |
| readme | 1 | 32/8 | 4096 | 64 | causal | 1.79461 | 1.78491 | 1.84577 | -2.13% | -3.09% |
| readme | 1 | 32/8 | 2048 | 128 | dense | 1.72412 | 1.74606 | 1.62777 | -2.34% | +6.01% |
| readme | 1 | 32/8 | 2048 | 128 | causal | 0.96794 | 0.89886 | 0.87708 | +8.31% | +10.70% |
| readme | 1 | 32/8 | 4096 | 128 | dense | 6.76953 | 6.62458 | 6.34761 | +1.90% | +5.52% |
| readme | 1 | 32/8 | 4096 | 128 | causal | 3.60555 | 3.38550 | 3.33006 | +6.38% | +7.95% |
| holdout | 4 | 12/12 | 1536 | 64 | dense | 0.71605 | 0.71223 | 0.71644 | -0.49% | +1.06% |
| holdout | 4 | 12/12 | 1536 | 64 | causal | 0.40936 | 0.44243 | 0.42589 | -5.37% | -6.16% |
| holdout | 2 | 16/4 | 3072 | 64 | dense | 1.85964 | 1.91807 | 1.85309 | -3.24% | +1.43% |
| holdout | 2 | 16/4 | 3072 | 64 | causal | 0.99150 | 1.05309 | 1.05650 | -5.40% | -3.79% |
| holdout | 1 | 8/8 | 8192 | 64 | dense | 3.35776 | 3.45639 | 3.41856 | -3.70% | -1.50% |
| holdout | 1 | 8/8 | 8192 | 64 | causal | 1.76161 | 1.82800 | 1.86083 | -4.17% | -3.24% |
| holdout | 8 | 8/8 | 512 | 64 | dense | 0.11679 | 0.11535 | 0.11328 | +1.72% | +3.21% |
| holdout | 8 | 8/8 | 512 | 64 | causal | 0.07463 | 0.08255 | 0.08217 | -8.95% | -8.65% |
| holdout | 1 | 32/8 | 1000 | 64 | dense | 0.22614 | 0.23451 | 0.21964 | -2.05% | +3.49% |
| holdout | 1 | 32/8 | 1000 | 64 | causal | 0.13274 | 0.14604 | 0.14306 | -9.79% | -7.93% |
| holdout | 3 | 14/2 | 2500 | 64 | dense | 1.71063 | 1.79268 | 1.71699 | -4.20% | +0.64% |
| holdout | 3 | 14/2 | 2500 | 64 | causal | 0.93658 | 0.95371 | 0.96732 | -4.42% | -3.50% |
| holdout | 2 | 16/8 | 1536 | 128 | dense | 0.97752 | 1.00030 | 0.93648 | -2.04% | +4.71% |
| holdout | 2 | 16/8 | 1536 | 128 | causal | 0.57445 | 0.52689 | 0.51636 | +8.96% | +11.03% |
| holdout | 1 | 28/4 | 3000 | 128 | dense | 3.26640 | 3.23005 | 3.05327 | +0.76% | +6.96% |
| holdout | 1 | 28/4 | 3000 | 128 | causal | 1.73334 | 1.64560 | 1.63371 | +5.59% | +5.95% |
| holdout2 | 2 | 24/8 | 2304 | 64 | dense | 1.57113 | 1.60990 | 1.61796 | -2.60% | -2.71% |
| holdout2 | 2 | 24/8 | 2304 | 64 | causal | 0.89300 | 0.92823 | 0.88794 | -4.03% | -1.19% |
| holdout2 | 1 | 16/16 | 5000 | 64 | dense | 2.66185 | 2.62514 | 2.57171 | -0.99% | +3.27% |
| holdout2 | 1 | 16/16 | 5000 | 64 | causal | 1.38342 | 1.47991 | 1.47161 | -6.32% | -5.11% |
| holdout2 | 6 | 10/2 | 768 | 64 | dense | 0.23518 | 0.23706 | 0.22928 | -0.11% | +3.37% |
| holdout2 | 6 | 10/2 | 768 | 64 | causal | 0.13918 | 0.15242 | 0.15142 | -7.70% | -7.54% |
| holdout2 | 1 | 40/8 | 1700 | 64 | dense | 0.77866 | 0.83722 | 0.78939 | -7.53% | -2.45% |
| holdout2 | 1 | 40/8 | 1700 | 64 | causal | 0.41490 | 0.47869 | 0.46932 | -12.78% | -9.86% |
| holdout2 | 2 | 20/4 | 6144 | 64 | dense | 9.59655 | 9.19961 | 8.96565 | +3.72% | +7.15% |
| holdout2 | 2 | 20/4 | 6144 | 64 | causal | 4.86268 | 4.80081 | 4.93352 | +1.06% | -0.52% |
| holdout2 | 5 | 9/3 | 1111 | 64 | dense | 0.38543 | 0.39896 | 0.36671 | -3.69% | +2.60% |
| holdout2 | 5 | 9/3 | 1111 | 64 | causal | 0.21759 | 0.23797 | 0.23878 | -8.89% | -8.87% |
| holdout2 | 1 | 24/8 | 2560 | 128 | dense | 2.02715 | 2.04962 | 1.94309 | -2.42% | +4.44% |
| holdout2 | 1 | 24/8 | 2560 | 128 | causal | 1.12073 | 1.05148 | 1.03739 | +6.09% | +7.95% |
| holdout2 | 3 | 12/6 | 1900 | 128 | dense | 1.68713 | 1.67676 | 1.63028 | +1.10% | +5.55% |
| holdout2 | 3 | 12/6 | 1900 | 128 | causal | 0.96754 | 0.91036 | 0.88041 | +6.06% | +10.40% |

## Other Evidence

[The 2026-09-20 serving campaign](../prefill_campaign_2026-09-20/REPORT.md)
contains native Flash, scratch decode-only and scratch prefill+decode, eager
and CUDA graphs separately. It predates the latest regression fixes and was
not rerun here. Full/decode-only TTFT differences are configuration-level
comparisons, not a perfectly isolated prefill-kernel experiment.

For SCRATCH_FULL versus native FLASH_ATTN with CUDA graphs, long-context
TTFT was 2.2-3.9% lower across five lengths. Throughput across concurrency
2/4/8/16/32 was approximately unchanged (-0.1% to +1.3% output tokens/s).
These are historical engine results, not measurements of today's kernel.

No new FlashInfer, standalone FA3, full-model TTFT/TPOT, throughput, or
cross-GPU run was performed here. No production source, old result, commit,
push, GPU clock setting or cloud resource was changed by this report.

## Reproduce

```sh
CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=8.9 \
python3 bench/report_attention_backends.py --run 1 --output /tmp/backend-new-run1.json
```

Use new output paths and run IDs 1, 2, 3 for independent processes.
The current runner selects the 34 D64 cases; the historical raw runs include D128.
`make_report.py` recomputes this dated report and summary.json from the original run1/2/3.json.
