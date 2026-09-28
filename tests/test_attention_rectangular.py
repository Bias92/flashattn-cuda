"""Unequal query/KV lengths against FP32 attention with bottom-right causal masking."""
import torch
from test_attention_forward import BC, BR, mod, naive_attention


def rand_half(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.float32).half()


def test_rectangular(B, H_q, H_kv, N_q, N_kv, D, causal, token_major=False):
    torch.manual_seed(7)
    Q = rand_half(B, H_q, N_q, D)
    K, V = rand_half(B, H_kv, N_kv, D), rand_half(B, H_kv, N_kv, D)
    O_ref, L_ref = naive_attention(Q.float(), K.float(), V.float(), causal)
    if token_major:
        Q, K, V = (X.transpose(1, 2).contiguous().transpose(1, 2) for X in (Q, K, V))
    O, L = mod.forward(Q, K, V, causal)
    O_only = mod.forward_only(Q, K, V, causal)
    ok = (torch.allclose(O.float(), O_ref, atol=2e-3, rtol=2e-3)
          and torch.allclose(L, L_ref, atol=2e-3, rtol=1e-3)
          and torch.equal(O, O_only))
    path = "full" if N_q % BR == 0 and N_kv % BC == 0 else "guarded"
    print(f"[{'PASS' if ok else 'FAIL'}] B={B} H={H_q}/{H_kv} N_q={N_q} N_kv={N_kv} "
          f"D={D} {path} causal={causal} token_major={token_major} "
          f"O_diff={(O.float() - O_ref).abs().max().item():.3e} "
          f"L_diff={(L - L_ref).abs().max().item():.3e}")
    return ok


def test_rejection():
    Q, K, V = rand_half(1, 2, 100, 64), rand_half(1, 2, 40, 64), rand_half(1, 2, 40, 64)
    try:
        mod.forward_only(Q, K, V, True)
        torch.cuda.synchronize()
    except RuntimeError:
        print("[PASS] rejects causal N_kv < N_q")
        return True
    print("[FAIL] accepted causal N_kv < N_q")
    return False


def main():
    assert not hasattr(mod, "forward_paged"), "Loaded an extension with the removed paged API"
    cases = [
        # B, H_q, H_kv, N_q, N_kv, D, causal
        (1, 4, 4, 64, 128, 64, True), (1, 4, 4, 64, 96, 64, True),
        (2, 8, 2, 512, 2560, 64, True), (1, 8, 2, 2048, 4096, 64, True),
        (1, 4, 4, 1, 1000, 64, True), (1, 4, 2, 5, 517, 64, True),
        (1, 4, 4, 63, 64, 64, True), (1, 4, 4, 65, 129, 64, True),
        (2, 8, 2, 100, 1000, 64, True), (1, 14, 2, 2048, 3000, 64, True),
        (1, 4, 2, 200, 1000, 128, True), (1, 4, 4, 128, 1024, 128, True),
        (1, 4, 4, 64, 128, 64, False), (2, 8, 2, 100, 300, 64, False),
        (1, 4, 2, 50, 20, 64, False), (1, 4, 2, 200, 1000, 128, False),
    ]
    passed = sum(test_rectangular(*case) for case in cases)
    passed += test_rectangular(2, 8, 2, 100, 1000, 64, True, token_major=True)
    passed += test_rejection()
    total = len(cases) + 2
    print(f"Result: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
