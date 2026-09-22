# vLLM Attention Integration

Final scope: D64, RTX 4060 Ti 8 GB, FP16 inputs and FP32 accumulation.
The integration is pinned to vLLM 0.19.0, PyTorch 2.10.0+cu128 and Python 3.12.
This is an out-of-tree attention backend, not a new serving engine.

## Runtime Paths

| Configuration | Prefill | Decode |
|---|---|---|
| Native `FLASH_ATTN` | vLLM FlashAttention | vLLM FlashAttention |
| Scratch decode-only | vLLM FlashAttention | `cuda/attention_decode_paged.cu` for pure decode batches |
| Scratch full | `cuda/attention_forward.cu` | `cuda/attention_decode_paged.cu` |

`integrations/vllm/scratch_vllm/loader.py` loads the paged decode source.
`integrations/vllm_prefill/` adds scratch prefill and mixed-batch handling.
The full backend reads fresh K/V from the projection tensors, and cached K/V
through page tables for chunks and prefix hits. Each prefill request gets its
own launch. The decode-only backend retains native handling for mixed batches.

Both leave vLLM's KV-cache writes, RMSNorm, RoPE, projections, MLP, scheduling,
sampling and CUDA Graph machinery intact. Unsupported attention features are
not covered by the measurements below.

## Which Results Apply

| Record | Prefill SHA256 prefix | Meaning |
|---|---|---|
| [2026-09-22 kernel comparison](backend_overview_2026-09-22/README.md) | `6fc64206c542` | Current kernel, forced Flash/cuDNN, O-only and +L |
| [2026-09-20/21 serving campaign](prefill_campaign_2026-09-20/REPORT.md) | `5f1d9815423d` | Earlier source, 252 accepted serving cases |
| [2026-09-21 regression analysis](prefill_dense_2026-09-21/README.md) | See report | Earlier-to-current source A/B and correctness evidence |

The paged decode SHA256 is
`89db4293f401deede3cb833ff7d7eb0647c1aabf480ee57b37df78e15208d146`.
The latest prefill changes were not rerun through the full serving campaign.
Old measurements remain valid records of their old configurations.

The campaign used TinyLlama-1.1B for low latency and C1-C32 throughput,
and Qwen2.5-0.5B for 2048-32640-token context. Eager and CUDA Graph results
are separate. With graphs, full/native long-context TTFT was 2.2-3.9% lower;
throughput at C2-C32 was roughly equal. Eager decode gains mostly disappeared
under graphs. See the report for per-case latency, memory and raw samples.

Fresh prefill and paged prefill use different cache access paths from native
FlashAttention. Full/decode-only comparisons also include backend dispatch and
metadata work; they do not isolate just the arithmetic of the prefill kernel.

## Validation And Limits

- The campaign checks route counters outside measured windows. Eager counters
  account for all processed tokens. Python counters do not count CUDA Graph
  replays; the graph check also verifies the implementation and capture/warmup.
- Numerical checks include FP32 references, boundary lengths, GQA, strided
  tensors, shuffled pages, poisoned unused slots, output overlap and sanitizers.
- Generated text need not match token for token. The campaign report retains
  numerical exceptions, including a Qwen logprob check exceeding its threshold.
- Low-level paged decode trusts device-side metadata values supplied by vLLM;
  host checks cannot validate every page ID or sequence length without a sync.
- D128 implementation and historical results remain for compatibility only;
  D64 is the maintained measurement scope.

## Reproduce

Install into a dedicated environment with the pinned versions above:

```sh
CUDA_HOME=/usr/local/cuda-12.8 python3 -m pip install -e . --break-system-packages
python3 -m pip install -e integrations/vllm -e integrations/vllm_prefill
python3 tests/test_decode_paged.py
python3 bench/check_vllm_routes.py --help
```

The campaign runner starts the local server and client, warms each session,
checks routing and saves raw records. Model weights must already be available
or separately downloaded. Use fresh output directories:

```sh
CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=1 \
python3 bench/bench_serving_prefill.py --mode eager --runs 3 --output-dir /tmp/serving-eager-new
CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=1 \
python3 bench/bench_serving_prefill.py --mode graphs --runs 3 --output-dir /tmp/serving-graphs-new
```

These commands measure the checked-out source, not the historical source
automatically. Historical commands, hashes and raw outputs are retained in
the dated report. Kernel-only SDPA numbers and serving adapters that gather
paged KV are separate comparisons.
