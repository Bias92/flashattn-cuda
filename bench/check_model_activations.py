"""Attention error against FP32 on the Q, K, V of a real model: this kernel and FlashAttention.

The unit tests use random tensors of unit scale. A trained model is different: a few heads of
Qwen2.5 carry very large keys, and FP16 kernels lose more there. This script runs a model in
FP16 through Hugging Face on one random prompt, records what every attention layer hands to
scaled_dot_product_attention, and per layer compares three results on exactly those tensors:

    reference   plain attention in FP32
    flash       PyTorch SDPA restricted to FlashAttention, FP16
    scratch     cuda/attention_forward.cu, FP16

so a gap between the two FP16 kernels inside a served model can be held against how far each
of them is from FP32.

    python3 bench/check_model_activations.py --model Qwen/Qwen2.5-0.5B-Instruct --tokens 3000
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]


def reference(q, k, v, scale):
    """Causal attention in FP32; K and V heads are repeated for grouped queries."""
    group = q.size(1) // k.size(1)
    q, k, v = q.float(), k.float().repeat_interleave(group, 1), v.float().repeat_interleave(group, 1)
    scores = (q @ k.transpose(-1, -2)) * scale
    n = q.size(2)
    scores.masked_fill_(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tokens", type=int, default=3000)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 torch_dtype=torch.float16,
                                                 attn_implementation="sdpa").cuda().eval()
    recorded = []
    sdpa = F.scaled_dot_product_attention

    def recording_sdpa(q, k, v, *rest, **kwargs):
        recorded.append((q.detach().clone(), k.detach().clone(), v.detach().clone(), kwargs.get("scale")))
        return sdpa(q, k, v, *rest, **kwargs)

    ids = torch.randint(10, model.config.vocab_size - 10, (1, args.tokens),
                        generator=torch.Generator().manual_seed(99)).cuda()
    F.scaled_dot_product_attention = recording_sdpa
    try:
        with torch.no_grad():
            model(ids, use_cache=False)
    finally:
        F.scaled_dot_product_attention = sdpa
    del model
    torch.cuda.empty_cache()

    scratch = load(name="attention_forward_cuda", sources=[str(ROOT / "cuda/attention_forward.cu")],
                   extra_cuda_cflags=FLAGS, verbose=False)
    rows = []
    print(f"{args.model}, {args.tokens} random tokens, {len(recorded)} attention layers, "
          f"Q {tuple(recorded[0][0].shape)}, K {tuple(recorded[0][1].shape)}")
    print("| layer | max abs Q | max abs K | max abs out | flash max err | scratch max err | "
          "flash mean err | scratch mean err |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for layer, (q, k, v, scale) in enumerate(recorded):
        scale = scale if scale is not None else q.size(-1) ** -0.5
        ref = reference(q, k, v, scale)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            flash = sdpa(q, k, v, is_causal=True, scale=scale, enable_gqa=q.size(1) != k.size(1))
        ours = scratch.forward_only(q, k, v, True, scale)
        row = dict(layer=layer, q_max=q.abs().max().item(), k_max=k.abs().max().item(),
                   out_max=ref.abs().max().item())
        for name, out in (("flash", flash), ("scratch", ours)):
            err = (out.float() - ref).abs()
            row[f"{name}_max_err"], row[f"{name}_mean_err"] = err.max().item(), err.mean().item()
        rows.append(row)
        print(f"| {layer} | {row['q_max']:.1f} | {row['k_max']:.1f} | {row['out_max']:.2f} | "
              f"{row['flash_max_err']:.2e} | {row['scratch_max_err']:.2e} | "
              f"{row['flash_mean_err']:.2e} | {row['scratch_mean_err']:.2e} |")
    for name in ("flash", "scratch"):
        worst = max(rows, key=lambda r: r[f"{name}_max_err"])
        print(f"{name}: worst layer {worst['layer']} with max err {worst[f'{name}_max_err']:.3e}; "
              f"mean over layers of the mean err {sum(r[f'{name}_mean_err'] for r in rows) / len(rows):.3e}")
    if args.json:
        args.json.write_text(json.dumps(dict(model=args.model, tokens=args.tokens, layers=rows),
                                        indent=2) + "\n")


if __name__ == "__main__":
    main()
