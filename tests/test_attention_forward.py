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


def repeat_kv(X, groups):
    """Expand [B, H_kv, N, D] to [B, H_kv*groups, N, D], the GQA mapping."""
    if groups == 1:
        return X
    B, H_kv, N, D = X.shape
    return X[:, :, None].expand(B, H_kv, groups, N, D).reshape(B, H_kv * groups, N, D)


def lay_out(Q, K, V, layout):
    """Views equal to Q, K, V whose memory is laid out the way a caller might hold them."""
    if layout == "contiguous":
        return Q, K, V
    if layout == "token_major":      # [B, N, H, D] buffers seen as [B, H, N, D]
        return tuple(X.transpose(1, 2).contiguous().transpose(1, 2) for X in (Q, K, V))
    if layout == "packed_qkv":       # slices of one fused projection output per token
        B, H, N, D = Q.shape
        H_kv = K.shape[1]
        buf = torch.cat([X.transpose(1, 2).reshape(B, N, -1) for X in (Q, K, V)], dim=-1)
        q, k, v = buf.split([H * D, H_kv * D, H_kv * D], dim=-1)
        return (q.view(B, N, H, D).transpose(1, 2), k.view(B, N, H_kv, D).transpose(1, 2),
                v.view(B, N, H_kv, D).transpose(1, 2))
    if layout == "odd_offset":       # every tensor starts one element (2 bytes) into its storage
        return shifted(Q, 1), shifted(K, 1), shifted(V, 1)
    if layout.startswith("kv_shift_"):   # only K and V start `n` elements into their storage
        n = int(layout.rsplit("_", 1)[1])
        return Q, shifted(K, n), shifted(V, n)
    if layout == "padded_rows":      # rows 4 elements apart: 136-byte row stride at D=64
        return padded_rows(Q), padded_rows(K), padded_rows(V)
    raise ValueError(layout)


def shifted(X, elements):
    buf = torch.empty(X.numel() + elements, device=X.device, dtype=X.dtype)
    buf[elements:].copy_(X.reshape(-1))
    return buf[elements:].view(X.shape)


def padded_rows(X):
    wide = torch.empty(*X.shape[:-1], X.shape[-1] + 4, device=X.device, dtype=X.dtype)
    wide[..., :X.shape[-1]].copy_(X)
    return wide[..., :X.shape[-1]]


# Layouts whose rows miss the alignment the device code needs (4 bytes for Q, 16 for K and V).
# The host has to copy those; every other layout has to be read in place.
COPIED_LAYOUTS = {"odd_offset", "kv_shift_2", "kv_shift_4", "padded_rows"}


def bytes_allocated_by(call):
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    call()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - before


def naive_attention(Q, K, V, causal, scale=None):
    D = Q.shape[-1]
    scale = D ** -0.5 if scale is None else scale
    K = repeat_kv(K, Q.shape[1] // K.shape[1])
    V = repeat_kv(V, Q.shape[1] // V.shape[1])
    S = Q @ K.transpose(-2, -1) * scale
    if causal:
        # Aligned to the bottom right: query i sees keys up to i + (N_kv - N_q).
        N_q, N_kv = Q.shape[-2], K.shape[-2]
        keep = torch.ones(N_q, N_kv, device=Q.device, dtype=torch.bool).tril(N_kv - N_q)
        S = S.masked_fill(~keep, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return P @ V, torch.logsumexp(S, dim=-1)


def test_config(B, H, N, D, device="cuda", dtype=torch.float32, amp=1.0, causal=False,
                H_kv=None, layout="contiguous", scale=None, into_out=False):
    torch.manual_seed(42)
    H_kv = H if H_kv is None else H_kv
    Q = (torch.randn(B, H, N, D, device=device, dtype=torch.float32) * amp).to(dtype)
    K = (torch.randn(B, H_kv, N, D, device=device, dtype=torch.float32) * amp).to(dtype)
    V = (torch.randn(B, H_kv, N, D, device=device, dtype=torch.float32) * amp).to(dtype)

    Qh, Kh, Vh = Q.half().float(), K.half().float(), V.half().float()
    O_ref, L_ref = naive_attention(Qh, Kh, Vh, causal, scale)
    Q, K, V = lay_out(Q, K, V, layout)
    # into_out: the caller owns a token-major [B, N, H, D] buffer and the kernel fills it.
    out = torch.empty(B, N, H, D, device=device, dtype=torch.float16).transpose(1, 2) \
        if into_out else None
    O_mma, L_mma = mod.forward(Q, K, V, causal, scale, out)
    wrote_out = (not into_out) or O_mma.data_ptr() == out.data_ptr()
    O_mma = O_mma.float()
    out_only = torch.empty_like(out) if into_out else None
    O_only = mod.forward_only(Q, K, V, causal, scale, out_only)
    wrote_out = wrote_out and ((not into_out) or O_only.data_ptr() == out_only.data_ptr())
    oo_same = torch.equal(O_only.float(), O_mma)

    # fp16 inputs reach the kernel as the views built above, so the host must allocate
    # nothing for an in-place layout and a copy for a misaligned one.
    layout_ok = True
    if dtype == torch.float16:
        scratch = torch.empty(B, H, N, D, device=device, dtype=torch.float16)
        copied = bytes_allocated_by(lambda: mod.forward_only(Q, K, V, causal, scale, scratch)) > 0
        layout_ok = copied == (layout in COPIED_LAYOUTS)

    O_diff = (O_mma - O_ref).abs().max().item()
    L_diff = (L_mma - L_ref).abs().max().item()
    ok = (torch.allclose(O_mma, O_ref, atol=2e-3 * max(amp, 1.0), rtol=2e-3)
          and torch.allclose(L_mma, L_ref, atol=2e-3, rtol=1e-3)
          and oo_same and wrote_out and layout_ok)
    path = "full" if (N % BR == 0 and N % BC == 0) else "guarded"
    mask = "causal" if causal else "dense"
    tag = f" dtype={str(dtype).split('.')[-1]}" if dtype != torch.float32 else ""
    tag += f" amp={amp:g}" if amp != 1.0 else ""
    tag += f" H_kv={H_kv}" if H_kv != H else ""
    tag += f" layout={layout}" if layout != "contiguous" else ""
    tag += f" scale={scale:g}" if scale is not None else ""
    tag += " out=" if into_out else ""
    tag += "" if layout_ok else " (wrong copy/in-place decision)"
    print(f"[{'PASS' if ok else 'FAIL'}] B={B}, H={H}, N={N:>5}, D={D} "
          f"[{path:>7}, {mask:>6}]{tag}  |  "
          f"O_diff={O_diff:.3e}  L_diff={L_diff:.3e}  o_only_match={oo_same}")
    return ok


def test_rejections():
    """An `out` the kernel cannot write safely has to raise, not corrupt memory."""
    Q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
    K, V = torch.randn_like(Q), torch.randn_like(Q)
    one_head = torch.empty(1, 1, 128, 64, device="cuda", dtype=torch.float16)
    bad_outs = {
        "out is K": K,
        "out rows share memory (expand)": one_head.expand(1, 4, 128, 64),
        "out starts off the 4-byte boundary": shifted(torch.empty_like(Q), 1),
        "out rows 65 elements apart": torch.empty(1, 4, 128, 65, device="cuda",
                                                  dtype=torch.float16)[..., :64],
        "out is fp32": torch.empty_like(Q, dtype=torch.float32),
        "out has another shape": torch.empty(1, 4, 64, 64, device="cuda", dtype=torch.float16),
    }
    passed = 0
    for name, out in bad_outs.items():
        try:
            mod.forward_only(Q, K, V, False, None, out)
            ok = False
        except RuntimeError:
            ok = True
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] rejected: {name}")
    return passed, len(bad_outs)


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
        # grouped-query attention: several query heads share one K/V head
        dict(B=1, H=8, N=1024, D=64, H_kv=1),                # multi-query, one K/V head
        dict(B=1, H=8, N=1024, D=64, H_kv=2),
        dict(B=2, H=8, N=2048, D=64, H_kv=4, causal=True),
        dict(B=1, H=4, N=4095, D=64, H_kv=2, causal=True),   # guarded path under GQA
        dict(B=1, H=8, N=127, D=64, H_kv=8),                 # equal heads still takes the plain path
        # head dimension 128
        dict(B=1, H=4, N=1024, D=128),
        dict(B=1, H=4, N=1024, D=128, causal=True),
        dict(B=1, H=4, N=127, D=128, causal=True),           # guarded path at D=128
        dict(B=1, H=8, N=512, D=128, H_kv=2, causal=True),   # D=128 with GQA
        dict(B=1, H=2, N=2048, D=128, amp=16.0),             # D=128 stress
        # memory layouts read in place (fp16, so the kernel gets these exact views)
        dict(B=2, H=8, N=512, D=64, H_kv=2, dtype=torch.float16, layout="token_major", causal=True),
        dict(B=2, H=8, N=127, D=64, H_kv=2, dtype=torch.float16, layout="token_major"),
        dict(B=2, H=8, N=1024, D=64, H_kv=2, dtype=torch.float16, layout="packed_qkv", causal=True),
        dict(B=1, H=6, N=1000, D=128, H_kv=3, dtype=torch.float16, layout="packed_qkv", causal=True),
        # rows off the alignment the device code needs: copied by the host, still correct
        dict(B=1, H=4, N=257, D=64, dtype=torch.float16, layout="odd_offset", causal=True),
        dict(B=1, H=4, N=257, D=64, dtype=torch.float16, layout="kv_shift_2", causal=True),
        dict(B=1, H=4, N=257, D=64, dtype=torch.float16, layout="kv_shift_4", causal=True),
        dict(B=1, H=4, N=257, D=64, dtype=torch.float16, layout="padded_rows", causal=True),
        # K and V 16 bytes into their storage: aligned, so read in place
        dict(B=1, H=4, N=257, D=64, dtype=torch.float16, layout="kv_shift_8", causal=True),
        # output written into a caller-owned token-major buffer
        dict(B=2, H=8, N=512, D=64, H_kv=2, dtype=torch.float16, layout="token_major",
             causal=True, into_out=True),
        dict(B=1, H=4, N=127, D=128, dtype=torch.float16, layout="packed_qkv", into_out=True),
        # softmax scale other than 1/sqrt(D)
        dict(B=1, H=4, N=512, D=64, causal=True, scale=0.2),
        dict(B=1, H=4, N=127, D=128, scale=0.05),
    ]
    for e in extras:
        passed += test_config(**e)
        total += 1
    print("-" * 100)

    rejected, candidates = test_rejections()
    passed += rejected
    total += candidates

    print("=" * 100)
    print(f"Result: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
