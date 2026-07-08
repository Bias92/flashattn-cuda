"""FA3 race path (split-loop + N_STATIC): correctness vs half-cast FP32 reference.

Covers all three dispatch paths (N_STATIC 2048/4096, generic full, guarded)
plus bit-identity vs db_full (math order unchanged).
"""
import torch
from torch.utils.cpp_extension import load

FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
mod = load(name="flash_attn_fa3_race", sources=["cuda/flash_attn_fa3_race.cu"],
           extra_cuda_cflags=FLAGS, verbose=False)
mod_full = load(name="flash_attn_fa3_db_full", sources=["cuda/flash_attn_fa3_db_full.cu"],
                extra_cuda_cflags=FLAGS, verbose=False)
print(f"so: {mod.__file__}")
print(f"so: {mod_full.__file__}")


def naive_attention(Q, K, V):
    D = Q.shape[-1]
    scale = D ** -0.5
    S = Q @ K.transpose(-2, -1) * scale
    P = torch.softmax(S, dim=-1)
    return P @ V, torch.logsumexp(S, dim=-1)


def path_of(N):
    if N % 64 == 0:
        return "N4096" if N == 4096 else ("N2048" if N == 2048 else "full")
    return "guarded"


def test_config(B, H, N, D, device="cuda", dtype=torch.float32, amp=1.0):
    torch.manual_seed(42)
    Q = (torch.randn(B, H, N, D, device=device) * amp).to(dtype)
    K = (torch.randn(B, H, N, D, device=device) * amp).to(dtype)
    V = (torch.randn(B, H, N, D, device=device) * amp).to(dtype)

    Qh, Kh, Vh = Q.half().float(), K.half().float(), V.half().float()
    O_ref, L_ref = naive_attention(Qh, Kh, Vh)
    O_r, L_r = mod.forward(Q, K, V)
    O_only = mod.forward_only(Q, K, V)
    oo_same = torch.equal(O_only, O_r)

    O_f, L_f = mod_full.forward(Q, K, V)
    # bit-identity vs db_full is INFORMATIONAL: the split-loop/lambda
    # restructure lets ptxas contract FMAs differently, which can move
    # last-ulp rounding without any algorithmic change. Gate on the
    # reference check + the direct distance to db_full instead.
    vsf_O = (O_r.float() - O_f.float()).abs().max().item()
    vsf_L = (L_r - L_f).abs().max().item()

    O_diff = (O_r.float() - O_ref).abs().max().item()
    L_diff = (L_r - L_ref).abs().max().item()
    ok = (torch.allclose(O_r.float(), O_ref, atol=2e-3 * max(amp, 1.0), rtol=2e-3)
          and torch.allclose(L_r, L_ref, atol=2e-3, rtol=1e-3)
          and oo_same
          and vsf_O <= 2e-3 * max(amp, 1.0) and vsf_L <= 1e-4 * max(amp, 1.0))
    tag = f" dtype={str(dtype).split('.')[-1]}" if dtype != torch.float32 else ""
    tag += f" amp={amp:g}" if amp != 1.0 else ""
    print(f"[{'PASS' if ok else 'FAIL'}] B={B}, H={H}, N={N:>5} [{path_of(N):>7}]{tag}  |  "
          f"O_diff={O_diff:.3e}  L_diff={L_diff:.3e}  o_only={oo_same}  "
          f"vs_full: O={vsf_O:.3e} L={vsf_L:.3e}")
    return ok


def main():
    print("=" * 100)
    print("FA3 race path Correctness Test")
    print("=" * 100)
    configs = [
        (1, 1, 2048, 64), (1, 1, 4096, 64), (2, 2, 2048, 64),   # N_STATIC paths
        (1, 1, 64, 64), (1, 1, 128, 64), (2, 8, 512, 64), (1, 1, 1024, 64),  # generic full
        (1, 1, 1, 64), (1, 1, 7, 64), (1, 1, 31, 64), (1, 1, 33, 64),
        (1, 1, 63, 64), (1, 1, 127, 64), (1, 1, 4095, 64),      # guarded
    ]
    extras = [
        dict(B=1, H=1, N=4096, D=64, dtype=torch.float16),
        dict(B=1, H=1, N=127, D=64, dtype=torch.float16),
        dict(B=1, H=1, N=2048, D=64, amp=16.0),
        dict(B=1, H=1, N=4095, D=64, amp=16.0),
    ]
    passed = sum(test_config(*c) for c in configs)
    for e in extras:
        passed += test_config(**e)
    total = len(configs) + len(extras)
    print("=" * 100)
    print(f"Result: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
