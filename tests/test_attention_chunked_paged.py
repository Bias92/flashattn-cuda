"""Query shorter than the keys, and K/V read out of a paged cache, against FP32 arithmetic.

Both are what a serving engine asks for: a chunk of a longer prompt attends to everything
before it, and the earlier keys and values live in a paged cache. The reference is the same
plain FP32 attention as in test_attention_forward.py, with the causal mask aligned to the
bottom right. Unused cache slots are NaN, so reading any of them shows up in the output.
"""
import torch
from test_attention_forward import BC, BR, mod, naive_attention

DEV = "cuda"


def rand_half(*shape):
    return torch.randn(*shape, device=DEV, dtype=torch.float32).half()


def close(O, O_ref):
    return torch.allclose(O.float(), O_ref, atol=2e-3, rtol=2e-3)


def path_of(N_q, N_kv):
    return "full" if (N_q % BR == 0 and N_kv % BC == 0) else "guarded"


def test_chunk(B, H, H_kv, N_q, N_kv, D, causal, token_major=False):
    """N_q query rows against N_kv keys held in ordinary tensors."""
    torch.manual_seed(7)
    Q, K, V = rand_half(B, H, N_q, D), rand_half(B, H_kv, N_kv, D), rand_half(B, H_kv, N_kv, D)
    O_ref, L_ref = naive_attention(Q.float(), K.float(), V.float(), causal)
    if token_major:
        Q, K, V = (X.transpose(1, 2).contiguous().transpose(1, 2) for X in (Q, K, V))
    O, L = mod.forward(Q, K, V, causal)
    O_only = mod.forward_only(Q, K, V, causal)
    ok = (close(O, O_ref) and torch.allclose(L, L_ref, atol=2e-3, rtol=1e-3)
          and torch.equal(O, O_only))
    print(f"[{'PASS' if ok else 'FAIL'}] chunk B={B} H={H}/{H_kv} N_q={N_q:>4} N_kv={N_kv:>5} D={D:>3} "
          f"[{path_of(N_q, N_kv):>7}, {'causal' if causal else 'dense':>6}]"
          f"{' token_major' if token_major else ''}  |  "
          f"O_diff={(O.float() - O_ref).abs().max().item():.3e}  "
          f"L_diff={(L - L_ref).abs().max().item():.3e}")
    return ok


def paged_cache(K, V, page_size, cache_layout, spare_pages=5):
    """Scatter K, V [1, H_kv, N_kv, D] over shuffled pages of a vLLM-shaped cache.

    Returns key_cache, value_cache [pages, page_size, H_kv, D] (views of one tensor, as vLLM
    hands them over) and the page table. Every slot that holds no row is NaN.
    """
    _, H_kv, N_kv, D = K.shape
    used = -(-N_kv // page_size)
    pages = used + spare_pages
    if cache_layout == "NHD":
        kv = torch.full((2, pages, page_size, H_kv, D), float("nan"), device=DEV, dtype=torch.float16)
    else:   # "HND": heads before slots in memory, seen through the same [.., slot, head, ..] view
        kv = torch.full((2, pages, H_kv, page_size, D), float("nan"), device=DEV,
                        dtype=torch.float16).permute(0, 1, 3, 2, 4)
    key_cache, value_cache = kv.unbind(0)
    order = torch.randperm(pages, device=DEV)[:used]
    rows = torch.arange(N_kv, device=DEV)
    key_cache[order[rows // page_size], rows % page_size] = K[0].transpose(0, 1)
    value_cache[order[rows // page_size], rows % page_size] = V[0].transpose(0, 1)
    # vLLM's table row is longer than the sequence needs; the tail is never read.
    table = torch.cat([order.int(), torch.zeros(3, device=DEV, dtype=torch.int32)])
    return key_cache, value_cache, table


def test_paged(H, H_kv, N_q, N_kv, D, page_size, cache_layout="NHD", into_out=False):
    """The last N_q of N_kv tokens attend causally to a paged cache."""
    torch.manual_seed(11)
    Q, K, V = rand_half(1, H, N_q, D), rand_half(1, H_kv, N_kv, D), rand_half(1, H_kv, N_kv, D)
    O_ref, _ = naive_attention(Q.float(), K.float(), V.float(), True)
    key_cache, value_cache, table = paged_cache(K, V, page_size, cache_layout)
    Q_tm = Q.transpose(1, 2).contiguous().transpose(1, 2)   # token-major, as vLLM holds it
    out = torch.empty(1, N_q, H, D, device=DEV, dtype=torch.float16).transpose(1, 2) \
        if into_out else None
    O = mod.forward_paged(Q_tm, key_cache, value_cache, table, N_kv, None, out)
    wrote_out = (not into_out) or O.data_ptr() == out.data_ptr()
    # Same arithmetic in the same order as the dense source, so the bits have to agree.
    same_as_dense = torch.equal(O, mod.forward_only(Q, K, V, True))
    ok = close(O, O_ref) and same_as_dense and wrote_out
    print(f"[{'PASS' if ok else 'FAIL'}] paged H={H}/{H_kv} N_q={N_q:>4} N_kv={N_kv:>5} D={D:>3} "
          f"page={page_size:>3} {cache_layout} [{path_of(N_q, N_kv):>7}]{' out=' if into_out else ''}  |  "
          f"O_diff={(O.float() - O_ref).abs().max().item():.3e}  bits equal dense source={same_as_dense}")
    return ok


def test_rejections():
    """Inputs the paged entry point cannot serve have to raise."""
    Q, K, V = rand_half(1, 4, 40, 64), rand_half(1, 2, 100, 64), rand_half(1, 2, 100, 64)
    kc, vc, table = paged_cache(K, V, 16, "NHD")
    odd = torch.zeros(2, 8, 48, 2, 64, device=DEV, dtype=torch.float16)
    calls = {
        "block table shorter than N_kv needs": lambda: mod.forward_paged(Q, kc, vc, table[:3], 100),
        "block table is int64": lambda: mod.forward_paged(Q, kc, vc, table.long(), 100),
        "two sequences in Q": lambda: mod.forward_paged(Q.repeat(2, 1, 1, 1), kc, vc, table, 100),
        "value cache strided differently": lambda: mod.forward_paged(Q, kc, vc.contiguous().clone()
                                                                     .transpose(1, 2).contiguous()
                                                                     .transpose(1, 2), table, 100),
        "page size not a power of two": lambda: mod.forward_paged(Q, odd[0], odd[1], table, 100),
        "N_kv smaller than N_q": lambda: mod.forward_paged(Q, kc, vc, table, 39),
        "dense causal with N_kv < N_q": lambda: mod.forward_only(K, Q[:, :2], Q[:, :2], True),
    }
    passed = 0
    for name, call in calls.items():
        try:
            call()
            torch.cuda.synchronize()
            ok = False
        except RuntimeError:
            ok = True
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] rejected: {name}")
    return passed, len(calls)


def main():
    print("=" * 100)
    print("N_q < N_kv and paged K/V")
    print("=" * 100)
    passed = total = 0
    chunks = [
        # (B, H, H_kv, N_q, N_kv, D, causal)
        (1, 4, 4, 64, 128, 64, True), (1, 4, 4, 64, 96, 64, True),       # full tiles
        (2, 8, 2, 512, 2560, 64, True), (1, 8, 2, 2048, 4096, 64, True),
        (1, 4, 4, 1, 1000, 64, True), (1, 4, 2, 5, 517, 64, True),        # a few query rows
        (1, 4, 4, 63, 64, 64, True), (1, 4, 4, 65, 129, 64, True),        # around a tile edge
        (2, 8, 2, 100, 1000, 64, True), (1, 14, 2, 2048, 3000, 64, True),
        (1, 4, 2, 200, 1000, 128, True), (1, 4, 4, 128, 1024, 128, True),
        (1, 4, 4, 64, 128, 64, False), (2, 8, 2, 100, 300, 64, False),    # no mask
        (1, 4, 2, 50, 20, 64, False), (1, 4, 2, 200, 1000, 128, False),   # more queries than keys
    ]
    for c in chunks:
        passed += test_chunk(*c)
        total += 1
    passed += test_chunk(2, 8, 2, 100, 1000, 64, True, token_major=True)
    total += 1
    print("-" * 100)

    paged = [
        # (H, H_kv, N_q, N_kv, D, page_size)
        (8, 2, 512, 512, 64, 16), (8, 2, 1000, 1000, 64, 16),             # whole prompt from the cache
        (8, 2, 512, 2560, 64, 16), (8, 2, 2048, 4096, 64, 16),           # a chunk of a longer prompt
        (14, 2, 2048, 8192, 64, 16), (8, 2, 100, 1000, 64, 16),
        (4, 4, 1, 777, 64, 16), (4, 2, 5, 517, 64, 16),
        (8, 2, 300, 1500, 64, 32), (8, 2, 300, 1500, 64, 64), (8, 2, 300, 1500, 64, 128),
        (4, 2, 200, 1000, 128, 16), (4, 4, 128, 1024, 128, 32),
    ]
    for c in paged:
        passed += test_paged(*c)
        total += 1
    passed += test_paged(8, 2, 300, 1500, 64, 16, cache_layout="HND")
    passed += test_paged(4, 2, 200, 1000, 128, 32, cache_layout="HND")
    passed += test_paged(8, 2, 300, 1500, 64, 16, into_out=True)
    total += 3
    print("-" * 100)

    rejected, candidates = test_rejections()
    passed += rejected
    total += candidates
    print("=" * 100)
    print(f"Result: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
