"""Small racecheck/memcheck target including partial tiles and GQA head tails."""
import torch
from test_attention_decode import load_decode

mod = load_decode()
with torch.inference_mode():
    for d in (64, 128):
        for n in (1, 33, 129, 257):
            for h in (3, 8, 17):
                q = torch.randn(1, h, 1, d, device="cuda", dtype=torch.float16)
                k = torch.randn(1, 1, n, d, device="cuda", dtype=torch.float16)
                v = torch.randn_like(k)
                for split in (64, 256):
                    o, l = mod.forward(q, k, v, None, split)
                    assert torch.isfinite(o).all() and torch.isfinite(l).all()
    torch.cuda.synchronize()
print("PASS sanitizer workload", flush=True)
