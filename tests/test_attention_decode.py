"""FP32 oracle, GQA, cache views, split tails, streams, and host validation."""
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]


def decode_source_sha():
    return hashlib.sha256(b"".join((ROOT / "cuda" / name).read_bytes() for name in
        ("attention_decode.cu", "attention_decode_mma.cuh"))).hexdigest()


def load_decode():
    src = ROOT / "cuda/attention_decode.cu"
    sha = decode_source_sha()
    mod = load(name="attention_decode_" + sha[:12], sources=[str(src)],
               extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                                  "-gencode=arch=compute_89,code=sm_89"], verbose=False)
    print(f"decode source sha256={sha}\nloaded={mod.__file__}", flush=True)
    return mod


def reference(q, k, v, scale=None):
    groups = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(groups, dim=1)
    v = v.float().repeat_interleave(groups, dim=1)
    scores = (q.float() @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5 if scale is None else scale)
    return scores.softmax(-1) @ v, scores.logsumexp(-1)


def main():
    mod = load_decode()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(8192)
    results = []
    cases = []
    for d in (64, 128):
        for n in (1, 7, 127, 128, 129, 255, 256, 257, 511, 1025, 4095, 4096, 8193):
            cases.append((1, 32, 4, n, d, 1., None, "contiguous", 0))
        for h, hk in ((1, 1), (8, 8), (12, 4), (16, 1), (17, 1), (40, 2)):
            cases.append((2, h, hk, 513, d, 1., None, "transposed", 0))
        for split in (64, 128, 256, 512, 1024):
            cases.append((1, 8, 2, 1031, d, 16., None, "contiguous", split))
        for scale in (0., 1., -0.2):
            cases.append((1, 4, 2, 257, d, 1., scale, "sliced", 0))
        cases.append((1, 4, 1, 33, d, 1., None, "unaligned", 0))
    with torch.inference_mode():
        for b, h, hk, n, d, amp, scale, layout, split in cases:
            def make(heads, tokens):
                if layout == "transposed":
                    return (torch.randn(b, tokens, heads, d, device="cuda", dtype=torch.float16) * amp).transpose(1, 2)
                if layout == "sliced":
                    return (torch.randn(b, heads, 2 * tokens, d, device="cuda", dtype=torch.float16) * amp)[:, :, ::2]
                if layout == "unaligned":
                    return (torch.randn(b * heads * tokens * d + 1, device="cuda", dtype=torch.float16) * amp)[1:].view(b, heads, tokens, d)
                return torch.randn(b, heads, tokens, d, device="cuda", dtype=torch.float16) * amp
            q, k, v = make(h, 1), make(hk, n), make(hk, n)
            ref, ref_l = reference(q, k, v, scale)
            out, l = mod.forward(q, k, v, scale, split)
            only = mod.forward_only(q, k, v, scale, split)
            scalar = mod.scalar_only(q, k, v, scale, split)
            torch.testing.assert_close(out.float(), ref, atol=8e-4 * amp, rtol=1e-3)
            torch.testing.assert_close(l, ref_l, atol=3e-4, rtol=3e-6)
            torch.testing.assert_close(scalar.float(), ref, atol=8e-4 * amp, rtol=1e-3)
            assert torch.equal(out, only)
            result = dict(b=b, h=h, hk=hk, n=n, d=d, amp=amp, scale=scale,
                          layout=layout, split=split, o_error=(out.float()-ref).abs().max().item(),
                          l_error=(l-ref_l).abs().max().item())
            results.append(result)
            print(f"PASS {result}", flush=True)
        # A non-default stream must own all allocation, work, and dependencies.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            q = torch.randn(1, 8, 1, 64, device="cuda", dtype=torch.float16)
            k = torch.randn(1, 2, 333, 64, device="cuda", dtype=torch.float16)
            v = torch.randn_like(k)
            out = mod.forward_only(q, k, v)
            ref, _ = reference(q, k, v)
            torch.testing.assert_close(out.float(), ref, atol=8e-4, rtol=1e-3)
        stream.synchronize()
        bad = [lambda: mod.forward_only(q.float(), k, v),
               lambda: mod.forward_only(q.expand(-1, -1, 2, -1), k, v),
               lambda: mod.forward_only(q, k[:, :, :0], v[:, :, :0]),
               lambda: mod.forward_only(q[:, :7], k, v),
               lambda: mod.forward_only(q, k, v, float("nan")),
               lambda: mod.forward_only(q, k, v, None, 3)]
        for fn in bad:
            try:
                fn()
            except RuntimeError:
                pass
            else:
                raise AssertionError("invalid input was accepted")
    print(f"PASS {len(results)} numerical cases + stream + {len(bad)} validation cases", flush=True)
    path = ROOT / "docs/decoding"
    path.mkdir(parents=True, exist_ok=True)
    (path / "correctness.json").write_text(json.dumps({"torch": torch.__version__,
        "source_sha256": decode_source_sha(),
        "cases": results, "stream": "pass", "invalid_inputs": len(bad)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
