# Why the decode kernel win did not become a TPOT win

## What was measured

Codex measured the local TinyLlama-1.1B FP16 model on the RTX 4060 Ti on
2026-09-14. No CUDA kernel, model weight, cache implementation, or precision
setting was changed for this investigation. The decode source hash remains
`ab7d1630373290954cd3bc691056f477200ea32a28d70a6c196806b3d67ada16`.

The complete run is `profile_decode_final.json`: contexts 512 and 2048, two
scoped traces per implementation/context, three actual cached decode steps per
trace. Both arms use Transformers SDPA prefill and the same forced continuation.
The custom arm uses the scratch decoder in all 22 layers during each decode
step. Unprofiled controls are recorded separately. Their short, noisy samples
are not evidence of a speedup.
For the context-2048 trace the three decoded queries see 2049, 2050, and 2051
valid cache tokens, rather than the standalone microbenchmark's exact N=2048.

`bench/profile_decode_model.py` labels attention, KV append, Q/K/V/O projections,
MLP, RMSNorm, RoPE, embeddings, mask construction, logits, and token selection.
`bench/decode_trace.py` attributes CUDA activities through launch correlation
IDs, falling back to the associated CPU external ID. GPU annotations are not
counted again as CPU ranges. Three CPU-only analyzer tests cover asynchronous
launch attribution, duplicate GPU annotations, interval unions, missing GPU
data, and external-ID fallback. All complete-run traces had zero unattributed
GPU activities.

## Findings at context 2048

Milliseconds per token, GPU activity durations, not CPU range durations:

| Phase | SDPA, two traces | Scratch decode, two traces |
|---|---:|---:|
| Attention across 22 layers | 0.3135 / 0.3139 | 0.1496 / 0.1497 |
| MLP | 5.7833 / 5.7556 | 5.7888 / 5.7904 |
| Q/K/V/O projections | 1.6925 / 1.6894 | 1.6927 / 1.6901 |
| Vocabulary projection | 0.4872 / 0.4770 | 0.4809 / 0.6182 |
| KV append | 0.2623 / 0.2633 | 0.2634 / 0.2661 |
| RMSNorm | 0.4331 / 0.4326 | 0.4311 / 0.4298 |
| RoPE | 0.3147 / 0.3143 | 0.3140 / 0.3132 |
| Total active GPU intervals, union | 9.3435 / 9.3032 | 9.1775 / 9.3141 |

The attention improvement is real inside the model, not only in the standalone
microbenchmark: about 0.164 ms saved per token in the measured attention phase.
But attention accounts for only about 3.4% of SDPA's active GPU work and about
1.6% after replacement. The removed attention work is roughly 1.8% of the
baseline active GPU time. This percentage is **not** an end-to-end speedup
prediction: CPU/GPU execution overlaps and launch/dispatch behavior also matters.

The MLP and projections still account for approximately 7.92-8.10 ms per token.
Most of that time appears in cuBLAS GEMV kernels. The trace identifies where the
time is spent; it does not by itself prove the GEMVs' memory bandwidth limit.
KV append is implemented as two `torch.cat` calls per layer and contributes
44 copy kernels per token. It is measurable, but not the dominant GPU cost at
this context length.

## CPU submission and small kernels

Both implementations launch **927 CUDA kernels per token** in these traces:

| Phase | Kernels or activities per token |
|---|---:|
| RMSNorm | 360 |
| RoPE | 229 |
| MLP (GEMV plus activations) | 110 |
| Q/K/V/O projections | 88 |
| KV append | 44 |
| Attention | 44 |

RMSNorm and RoPE alone account for 589 launches. Making attention's two kernels
per layer faster does not remove those launches. CUDA runtime calls and gaps
between GPU activities are substantial in the instrumented timelines.

**Do not quote profiled wall time as normal TPOT.** At context 2048 the traced
window was 32.76-42.92 ms/token, much higher than many unprofiled samples. Python
scope wrappers, Kineto, and CUDA activity collection perturb submission. GPU
gaps therefore cannot all be called "Python overhead" or claimed recoverable.
Host phase durations, CUDA API durations, and GPU durations overlap and must
not be added.

Mask construction also contains one device-to-host transfer and a stream
synchronization per token when passed the all-ones mask. Removing that is only
valid when the input/cache really has no padding or extra mask restrictions;
silently dropping arbitrary masks is not an optimization.

## Fixed-cache replay control

`decode_replay_control.json` records a separate, unprofiled 15-pair experiment at
context 2048. It repeatedly computes the same one-token query with the same
valid cache prefix. Both eager and graph use no padding mask and the same
precomputed position tensors. Eager cache reset happens outside timing.

| Implementation | Eager synchronized wall median, ms | Graph replay median, ms |
|---|---:|---:|
| SDPA | 13.9318 | 9.7610 |
| Scratch decode | 15.4403 | 9.5750 |

Within-pair graph/eager ratio medians were 0.710 for SDPA and 0.623 for scratch.
Their bootstrap 95% intervals were [0.484, 0.772] and [0.350, 0.746]. The eager
samples were much noisier than replay. This supports a material benefit from
changing the host submission path under this controlled workload; it does not
assign every saved microsecond specifically to Python versus the CUDA driver.

The two graph paths differ by only about 0.19 ms at their medians. The paired
scratch/SDPA graph ratio was 0.982, with bootstrap interval [0.967, 0.985],
consistent with attention being only a small fraction of total model GPU work.
Each graph output matched its own eager output bitwise, and custom versus SDPA
had matching argmax with maximum logit difference 0.01171875.

**This is not autoregressive TPOT:** replays do not advance the cache. It is one
completed 15-pair diagnostic run. An attempted second independent run was
blocked by an external `check_qdq.py` workload, so independent-run replication
has not been established. No growing-cache Graph integration or RMSNorm/RoPE
fusion was implemented in this profiling change.

## Noise and contamination

The second context-512 pair showed roughly 16.7 ms of active GPU work instead
of roughly 9.1 ms. A post-trace sample also recorded a 5001 MHz memory clock
instead of 8751 MHz. This is evidence of unstable operating conditions, not a
precise in-kernel clock measurement: telemetry was sampled after trace export.
Do not pool these absolute times into a fixed-clock result. The context-2048
component figures above were much more consistent.

Other llm-compressor pytest jobs were detected and blocked multiple attempts.
`profile_decode_initial.json` and `profile_decode_probe.json` are incomplete
debug records, not complete comparison runs. The completed scoped run has
`complete: true`. Read-only pip metadata commands may be ignored by the scoped
diagnostic runner and are recorded explicitly; the performance benchmark's
default process guard remains strict. Polling is not a global GPU lock and
cannot rule out very short overlapping workloads or Windows desktop activity.

## Next experiments

Historical proposals below are preserved for provenance. On 2026-09-14/15 the
user chose to reuse vLLM's model operations and serving engine, rather than
implement RMSNorm/RoPE fusion here. The implemented paged integration and current
unexecuted workload plan are documented in the
[serving handoff](../serving/WORKLOAD_HANDOFF_2026-09-15.md).

1. Fuse RMSNorm, then RoPE, while preserving their FP16 rounding behavior and
   retesting logits/argmax. These target hundreds of small launches, not another
   few microseconds of attention. Performance improvement remains to be measured.
2. Use a fixed-cache CUDA Graph replay as a diagnostic control for submission
   overhead. `bench/bench_decode_replay.py` restores the same cache outside eager
   timing and captures the identical single-token math. Graph outputs must equal
   eager outputs bitwise. This is **not** a growing-cache generation loop.
3. A production graph path would need graph-compatible cache updates and valid
   cache-length handling. The current kernel accepts N as a host launch argument;
   replaying it with a growing cache without redesigning that contract is wrong.
4. Profile the remaining GEMVs before changing precision or choosing a new
   linear layer implementation. Precision changes must be a separate comparison.

## Reproduce

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1
export HF_HUB_OFFLINE=1
python3 tests/test_decode_trace.py
python3 bench/profile_decode_model.py --lengths 512 2048 --pairs 3 --steps 12 \
    --profile-steps 3 --profile-repeats 2 --output profile_decode_final.json
python3 bench/bench_decode_replay.py --length 2048 --pairs 15
```

Compressed Chrome/Perfetto traces are under `profile_decode_final_traces/`.
Weights are loaded from the existing local Hugging Face cache only.
The profiler API follows the official
[PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
