"""Which kernel serves what inside vLLM 0.19, and whether the numbers agree with FlashAttention.

Four situations a server meets, run on one engine per backend with chunked prefill and the
prefix cache on and a small per-step token budget:

    fresh     one prompt shorter than the budget, nothing cached
    chunked   one prompt several budgets long, so it is prefilled in chunks
    mixed     several requests at once, so steps hold decodes and prefill chunks together
    prefix    a prompt whose first part is already in the prefix cache

For SCRATCH_FULL the worker reads the per-route counters around each situation and records a
failure when a token went to the FlashAttention fallback or when the route the situation is
there to exercise was not used. Numbers are compared against FLASH_ATTN on the same prompts:
the logprob of every prompt token (this reads every row the prefill wrote, not only the last
one) and of the first generated token. vLLM skips the prefix cache for requests that ask for
prompt logprobs, so `prefix` compares the first generated token only.

This is a wiring check: a wrong row, page or mask moves logprobs by O(1). Kernel arithmetic is
checked against FP32 in tests/. Timing is not measured here.

    python3 bench/check_vllm_routes.py --output-dir docs/serving/<dir>/routes_eager --eager
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BACKENDS = {
    # name: (VLLM_PLUGINS, vLLM attention backend)
    "FLASH_ATTN": ("", "FLASH_ATTN"),
    "TRITON_ATTN": ("", "TRITON_ATTN"),   # a second vLLM kernel: how far two FP16 kernels drift apart
    "SCRATCH_DECODE": ("scratch_attention", "CUSTOM"),
    "SCRATCH_FULL": ("scratch_prefill", "CUSTOM"),
}
REVISIONS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
}
# Per situation: routes SCRATCH_FULL has to use, and routes it must not use.
REQUIRED = {
    "fresh": (("prefill_dense",), ("prefill_paged", "decode_in_mixed")),
    "chunked": (("prefill_dense", "prefill_paged"), ("decode_in_mixed",)),
    "mixed": (("decode_in_mixed",), ()),
    "prefix": (("prefill_paged",), ("prefill_dense",)),
}
BLOCK = 16   # vLLM's KV block size; a prefix is shared in whole blocks


def worker(args):
    plugins, backend = BACKENDS[args.backend]
    os.environ.update(HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", VLLM_PLUGINS=plugins)
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    budget = args.step_budget
    graphs = {} if args.eager else {"compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8]}}
    engine = LLM(
        model=args.model, revision=REVISIONS.get(args.model), dtype="float16", seed=42,
        max_model_len=args.max_model_len, gpu_memory_utilization=0.55, max_num_seqs=8,
        max_num_batched_tokens=budget, enable_chunked_prefill=True, enable_prefix_caching=True,
        enforce_eager=args.eager, attention_config={"backend": backend},
        worker_extension_cls="scratch_vllm_prefill.audit.RoutingWorker", **graphs,
    )
    vocab = engine.llm_engine.model_config.get_vocab_size()
    rng = torch.Generator().manual_seed(2718)   # same prompts for every backend

    def tokens(n):
        return torch.randint(10, vocab - 10, (n,), generator=rng).tolist()

    def routing():
        """Calls and tokens per route, summed over layers; empty for vLLM's own implementation."""
        totals = {"route_calls": {}, "route_tokens": {}}
        for rank in engine.collective_rpc("attention_routing"):
            for layer in rank:
                for key, total in totals.items():
                    for route, n in (layer[key] or {}).items():
                        total[route] = total.get(route, 0) + n
        return totals

    def run(prompts, new_tokens, prompt_logprobs=True):
        params = SamplingParams(temperature=0, max_tokens=new_tokens, ignore_eos=True, logprobs=1,
                                prompt_logprobs=1 if prompt_logprobs else None)
        before = routing()
        outputs = engine.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params,
                                  use_tqdm=False)
        after = routing()
        used = {key: {r: n - before[key][r] for r, n in after[key].items()} for key in after}
        requests = []
        for out in outputs:
            ids, first = out.prompt_token_ids, out.outputs[0]
            requests.append(dict(
                prompt_tokens=len(ids), token_ids=list(first.token_ids),
                first_logprob=first.logprobs[0][first.token_ids[0]].logprob,
                prompt_logprobs=[row[t].logprob for row, t in zip(out.prompt_logprobs[1:], ids[1:])]
                if prompt_logprobs else None))
        return dict(requests=requests, **used)

    shared = (3 * budget // 2) // BLOCK * BLOCK
    seed_prompt = tokens(2 * budget)
    situations = {
        "fresh": run([tokens(3 * budget // 5)], 8),
        "chunked": run([tokens(7 * budget // 2)], 8),
        "mixed": run([tokens(budget * k // 20) for k in (8, 13, 18, 5, 20, 11)], 24),
    }
    run([seed_prompt], 1, prompt_logprobs=False)
    situations["prefix"] = run([seed_prompt[:shared] + tokens(3 * budget // 10)], 8,
                               prompt_logprobs=False)

    failures = []
    if args.backend == "SCRATCH_FULL":
        for name, (needed, unused) in REQUIRED.items():
            calls, toks = situations[name]["route_calls"], situations[name]["route_tokens"]
            if toks["flash_fallback"]:
                failures.append(f"{name}: {toks['flash_fallback']} tokens went to the fallback")
            failures += [f"{name}: route {r} was not used" for r in needed if not calls[r]]
            failures += [f"{name}: route {r} was used" for r in unused if calls[r]]
    args.json.write_text(json.dumps(dict(
        backend=args.backend, model=args.model, eager=args.eager, step_budget=budget,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, situations=situations,
        route_failures=failures)) + "\n")


def compare(result, reference):
    """Per situation: worst logprob gaps against the reference, and greedy agreement."""
    rows = {}
    for name, situation in result["situations"].items():
        pairs = list(zip(situation["requests"], reference["situations"][name]["requests"]))
        gaps = [abs(x - y) for a, b in pairs if a["prompt_logprobs"]
                for x, y in zip(a["prompt_logprobs"], b["prompt_logprobs"])]
        rows[name] = dict(
            requests=len(pairs), prompt_positions=len(gaps),
            prompt_logprob_gap_max=max(gaps) if gaps else None,
            prompt_logprob_gap_mean=sum(gaps) / len(gaps) if gaps else None,
            first_logprob_gap_max=max(abs(a["first_logprob"] - b["first_logprob"]) for a, b in pairs),
            first_token_equal=sum(a["token_ids"][0] == b["token_ids"][0] for a, b in pairs),
            all_tokens_equal=sum(a["token_ids"] == b["token_ids"] for a, b in pairs))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--backends", nargs="+", choices=sorted(BACKENDS),
                    default=["FLASH_ATTN", "SCRATCH_FULL"], help="the first one is the reference")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--step-budget", type=int, default=512,
                    help="max_num_batched_tokens; the longest prompt is 3.5 times this")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--gap-limit", type=float, default=0.05,
                    help="largest logprob gap to the reference that still passes")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", choices=sorted(BACKENDS), help=argparse.SUPPRESS)
    ap.add_argument("--json", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if 7 * args.step_budget // 2 + 8 > args.max_model_len:
        ap.error("max-model-len has to hold 3.5 step budgets plus 8 tokens")
    if args.worker:
        return worker(args)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = {}
    for backend in args.backends:
        path = out / f"{backend.lower()}.json"
        cmd = [sys.executable, __file__, "--worker", "--backend", backend, "--json", str(path),
               "--model", args.model, "--max-model-len", str(args.max_model_len),
               "--step-budget", str(args.step_budget)] + (["--eager"] if args.eager else [])
        print(f"{backend} ({'eager' if args.eager else 'CUDA graphs'})", flush=True)
        with (out / f"{backend.lower()}.log").open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        results[backend] = json.loads(path.read_text())

    failed = False
    reference = results[args.backends[0]]
    for backend in args.backends[1:]:
        result = results[backend]
        print(f"\n{backend} against {args.backends[0]}, {args.model}, "
              f"{'eager' if args.eager else 'CUDA graphs'}, step budget {args.step_budget}")
        for name, row in compare(result, reference).items():
            situation = result["situations"][name]
            worst = max(row["prompt_logprob_gap_max"] or 0.0, row["first_logprob_gap_max"])
            failed |= worst > args.gap_limit
            print(f"  {name:<8} calls {situation['route_calls']}\n"
                  f"  {'':<8} tokens {situation['route_tokens']}\n"
                  f"  {'':<8} {json.dumps(row)}{'  <-- over the gap limit' if worst > args.gap_limit else ''}")
        for failure in result["route_failures"]:
            print(f"  ROUTE FAILURE {failure}")
        failed |= bool(result["route_failures"])
    (out / "comparison.json").write_text(json.dumps(
        {b: compare(results[b], reference) for b in args.backends[1:]}, indent=2) + "\n")
    print("\nFAILED" if failed else "\nall routes as required, all gaps within the limit")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
