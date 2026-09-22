"""Shape/model checks owned and monitored by bench_serving_workloads.py."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "bench"), str(ROOT / "integrations/vllm")]


def kernels():
    import torch
    from test_decode_paged import make_case, invoke, check
    from scratch_vllm.loader import load_extension
    from scratch_vllm.comparison import MODES, gather_request, dense_attention

    torch.manual_seed(4901)
    torch.backends.cuda.matmul.allow_tf32 = False
    extension = load_extension()
    records = []
    shapes = [(32, 4, tuple(511 + i % 33 for i in range(batch)))
              for batch in (1, 2, 4, 8, 16, 32)]
    shapes += [(14, 2, (n,)) for n in (1, 33, 2047, 2048, 2175, 4095, 4096,
                                      4223, 8192, 8319, 16384, 16511, 32640, 32768)]
    for hnd in (False, True):
        for hq, hkv, lengths in shapes:
            case = make_case(hq=hq, hkv=hkv, lengths=lengths, hnd=hnd)
            invoke(extension, case)
            item = dict(hq=hq, hkv=hkv, lengths=lengths, hnd=hnd,
                        custom_max_diff=check(case), sdpa={})
            # Validate the candidate's GQA=7 against an independent FP32 oracle.
            if hq == 14:
                n = lengths[0]
                ids = case["table"][0, :(n + 15) // 16].long()
                k = case["k"][ids].reshape(-1, hkv, 64)[:n].transpose(0, 1)
                v = case["v"][ids].reshape(-1, hkv, 64)[:n].transpose(0, 1)
                q = case["q"][0].reshape(1, hq, 1, 64)
                scores = torch.matmul(q.float(), k.repeat_interleave(7, 0).float()
                                      .transpose(-2, -1)) * case["scale"]
                oracle = scores.softmax(-1) @ v.repeat_interleave(7, 0).float()
                cache = torch.stack((case["k"], case["v"]))
                kg, vg = gather_request(cache, case["table"][0], n)
                torch.testing.assert_close(kg[0], k, rtol=0, atol=0)
                torch.testing.assert_close(vg[0], v, rtol=0, atol=0)
                for mode in MODES:
                    if mode == "SDPA_CUDNN" and n == 1:
                        item["sdpa"][mode] = dict(status="known_unsupported_length_1")
                        continue
                    out = dense_attention(q, kg, vg, mode=mode, scale=case["scale"])
                    torch.testing.assert_close(out.float(), oracle, atol=8e-4, rtol=1e-3)
                    item["sdpa"][mode] = dict(status="passed", max_diff=(out.float() - oracle).abs().max().item())
            records.append(item)
            print(json.dumps(item), flush=True)
            del case
    return dict(passed=len(records), cases=records, so=extension.__file__)


def model(args):
    os.environ["VLLM_PLUGINS"] = "scratch_attention"
    os.environ["SCRATCH_DECODE_AUDIT"] = "1"
    os.environ.pop("SCRATCH_COMPARISON_BACKEND", None)
    if args.backend.startswith("SDPA_"):
        os.environ["SCRATCH_COMPARISON_BACKEND"] = args.backend
    from vllm import LLM, SamplingParams

    engine = LLM(model=args.model, revision=args.revision, dtype="half",
                 max_model_len=args.capacity, gpu_memory_utilization=0.55,
                 max_num_seqs=32, max_num_batched_tokens=2048, enforce_eager=True,
                 attention_config={"backend": "CUSTOM" if args.backend.startswith("SDPA_") else args.backend},
                 worker_extension_cls="scratch_vllm.workload_audit.WorkloadAuditWorker",
                 enable_prefix_caching=False, seed=42)
    prompts = ["Explain what a GPU memory bank conflict is.",
               "Write a short Python function that adds two integers.",
               "The capital of France is"]
    outputs = engine.generate(prompts, SamplingParams(temperature=0, max_tokens=32,
                              ignore_eos=True, logprobs=5), use_tqdm=False)
    result = dict(model=args.model, revision=args.revision, backend=args.backend,
                  capacity=args.capacity, snapshot=engine.collective_rpc("workload_snapshot"),
                  outputs=[dict(prompt=x.prompt, ids=list(x.outputs[0].token_ids),
                      text=x.outputs[0].text, logprobs=[{str(k): v.logprob for k, v in p.items()}
                      for p in x.outputs[0].logprobs]) for x in outputs])
    assert all(len(x["ids"]) == 32 for x in result["outputs"])
    if args.backend != "FLASH_ATTN":
        layers = [layer for worker in result["snapshot"] for layer in worker["routing"]]
        assert layers and all((layer["scratch_calls"] or 0) > 0 for layer in layers)
    return result


def main():
    import psutil
    parent = psutil.Process().parent()
    expected = ROOT / "bench/bench_serving_workloads.py"
    if (os.environ.get("WORKLOAD_OWNER_PID") != str(parent.pid)
            or str(expected) not in parent.cmdline()):
        raise RuntimeError("Run through the monitored workload runner, not unguarded")
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("kernels", "model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--backend")
    parser.add_argument("--capacity", type=int, default=2048)
    args = parser.parse_args()
    result = kernels() if args.phase == "kernels" else model(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validation saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
