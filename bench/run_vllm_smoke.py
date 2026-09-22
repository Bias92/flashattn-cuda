"""Correctness smoke run, not a latency benchmark. One backend per process."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from bench_attention_decode import exclusive_gpu_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("FLASH_ATTN", "CUSTOM", "SDPA_AUTO", "SDPA_FLASH",
                        "SDPA_CUDNN", "SDPA_EFFICIENT", "SDPA_MATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphs", action="store_true")
    args = parser.parse_args()
    dense = args.backend.startswith("SDPA_")
    if dense and args.graphs:
        parser.error("Dense SDPA adapters require eager execution")
    exclusive_gpu_check()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("VLLM_PLUGINS", "scratch_attention")
    os.environ.setdefault("SCRATCH_DECODE_AUDIT", "1")
    os.environ.pop("SCRATCH_COMPARISON_BACKEND", None)
    if dense:
        os.environ["SCRATCH_COMPARISON_BACKEND"] = args.backend
    import torch
    from vllm import LLM, SamplingParams
    from scratch_vllm.loader import SOURCE

    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    engine = LLM(
        model=model, dtype="float16", max_model_len=2048,
        gpu_memory_utilization=0.55, max_num_seqs=8, max_num_batched_tokens=2048,
        attention_config={"backend": "CUSTOM" if dense else args.backend}, enforce_eager=not args.graphs,
        worker_extension_cls="scratch_vllm.audit.AuditWorker",
        enable_prefix_caching=False, enable_chunked_prefill=False, seed=42,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8]},
    )
    prompts = ["Explain what a GPU memory bank conflict is.",
               "Write a short Python function that adds two integers.",
               "The capital of France is"]
    samples = engine.generate(prompts, SamplingParams(temperature=0, max_tokens=32,
                              ignore_eos=True, logprobs=1), use_tqdm=False)
    result = dict(backend=args.backend, prefill_backend="FLASH_ATTN", graphs=args.graphs,
                  model=model, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  routing=engine.collective_rpc("scratch_routing_snapshot"),
                  outputs=[dict(prompt=x.prompt, token_ids=list(x.outputs[0].token_ids),
                                text=x.outputs[0].text,
                                selected_logprobs=[p[t].logprob for t, p in
                                    zip(x.outputs[0].token_ids, x.outputs[0].logprobs)])
                           for x in samples])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.backend == "CUSTOM" or dense:
        print(json.dumps(result["routing"], indent=2), flush=True)
        assert any(layer["scratch_calls"] for worker in result["routing"] for layer in worker), \
            "No custom decode calls; do not report this as a custom-kernel run"
    print(f"Saved {args.output}; smoke only, no latency claim", flush=True)


if __name__ == "__main__":
    main()
