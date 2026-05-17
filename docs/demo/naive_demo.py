import torch, time

B, H, D, N = 1, 8, 64, 8192
ITERS = 30

print(f"NAIVE ATTENTION — N={N}, materialized N×N score matrix\n")

Q = torch.randn(B, H, N, D, device='cuda')
K = torch.randn(B, H, N, D, device='cuda')
V = torch.randn(B, H, N, D, device='cuda')
scale = D ** -0.5

for _ in range(3):
    S = (Q @ K.transpose(-1, -2)) * scale
    P = torch.softmax(S, dim=-1)
    O = P @ V
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

print(f"Running {ITERS} iterations...")
t0 = time.perf_counter()
for i in range(ITERS):
    S = (Q @ K.transpose(-1, -2)) * scale
    P = torch.softmax(S, dim=-1)
    O = P @ V
    torch.cuda.synchronize()
    if (i+1) % 5 == 0:
        print(f"  [{i+1:>3}/{ITERS}]  {time.perf_counter()-t0:>5.2f}s")
total = time.perf_counter() - t0
peak = torch.cuda.max_memory_allocated() / 1e6

print("=" * 60)
print(f"  Per iter:    {total/ITERS*1000:>7.2f} ms")
print(f"  Peak memory: {peak:>7.1f} MB")
print("=" * 60)
