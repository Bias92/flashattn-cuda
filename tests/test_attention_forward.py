"""Test full-tile, guarded and causal paths against FP32 arithmetic on FP16-rounded inputs."""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]

mod = load(
    name="attention_forward_cuda",
    sources=[str(ROOT / "cuda/attention_forward.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"],
    verbose=False,
)

BR, BC = 64, 32


def naive_attention(Q, K, V, causal):
    D = Q.shape[-1]
    scale = D ** -0.5
    S = Q @ K.transpose(-2, -1) * scale
    if causal:
        N = Q.shape[-2]
        keep = torch.ones(N, N, device=Q.device, dtype=torch.bool).tril()
        S = S.masked_fill(~keep, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return P @ V, torch.logsumexp(S, dim=-1)


def test_config(B, H, N, D, device="cuda", dtype=torch.float32, amp=1.0, causal=False):
    torch.manual_seed(42)
    Q = (torch.randn(B, H, N, D, device=device, dtype=torch.float32) * amp).to(dtype)
    K = (torch.randn(B, H, N, D, device=device, dtype=torch.float32) * amp).to(dtype)
    V = (torch.randn(B, H, N, D, device=device, dtype=torch.float32) * amp).to(dtype)

    Qh, Kh, Vh = Q.half().float(), K.half().float(), V.half().float()
    O_ref, L_ref = naive_attention(Qh, Kh, Vh, causal)
    O_mma, L_mma = mod.forward(Q, K, V, causal)
    O_mma = O_mma.float()
    O_only = mod.forward_only(Q, K, V, causal).float()
    oo_same = torch.equal(O_only, O_mma)

    O_diff = (O_mma - O_ref).abs().max().item()
    L_diff = (L_mma - L_ref).abs().max().item()
    ok = (torch.allclose(O_mma, O_ref, atol=2e-3 * max(amp, 1.0), rtol=2e-3)
          and torch.allclose(L_mma, L_ref, atol=2e-3, rtol=1e-3)
          and oo_same)
    path = "full" if (N % BR == 0 and N % BC == 0) else "guarded"
    mask = "causal" if causal else "dense"
    tag = f" dtype={str(dtype).split('.')[-1]}" if dtype != torch.float32 else ""
    tag += f" amp={amp:g}" if amp != 1.0 else ""
    print(f"[{'PASS' if ok else 'FAIL'}] B={B}, H={H}, N={N:>5}, D={D} "
          f"[{path:>7}, {mask:>6}]{tag}  |  "
          f"O_diff={O_diff:.3e}  L_diff={L_diff:.3e}  o_only_match={oo_same}")
    return ok


def main():
    print("=" * 100)
    print("Custom CUDA correctness test")
    print("=" * 100)
    configs = [
        # full path
        (1, 1, 64, 64), (1, 1, 128, 64), (2, 8, 512, 64),
        (1, 1, 1024, 64), (1, 1, 2048, 64), (1, 1, 4096, 64),
        # guarded path
        (1, 1, 1, 64), (1, 1, 2, 64), (1, 1, 7, 64), (1, 1, 31, 64),
        (1, 1, 33, 64), (1, 1, 63, 64), (1, 1, 127, 64), (1, 1, 4095, 64),
        (2, 4, 256, 64),  # multi-head full-tile case
    ]
    passed = total = 0
    for causal in (False, True):
        for c in configs:
            passed += test_config(*c, causal=causal)
            total += 1
        print("-" * 100)

    extras = [
        dict(B=1, H=1, N=1024, D=64, dtype=torch.float16),   # full, fp16 direct
        dict(B=1, H=1, N=127, D=64, dtype=torch.float16),    # guarded, fp16 direct
        dict(B=1, H=1, N=2048, D=64, amp=16.0),              # full, stress
        dict(B=1, H=1, N=4095, D=64, amp=16.0),              # guarded, stress
        dict(B=1, H=1, N=1024, D=64, dtype=torch.float16, causal=True),
        dict(B=1, H=1, N=4095, D=64, amp=16.0, causal=True),
        dict(B=1, H=1, N=65, D=64, causal=True),             # first row of the second Q block
    ]
    for e in extras:
        passed += test_config(**e)
        total += 1

    print("=" * 100)
    print(f"Result: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
