"""The custom adapter must not silently drop masks, scaling, or cache restrictions."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from attention_backend import register_backend
from test_attention_decode import load_decode


def main():
    mod = load_decode()
    fn, counts = register_backend("scratch_decode_test", mod)
    assert ALL_MASK_ATTENTION_FUNCTIONS["scratch_decode_test"] is ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
    torch.manual_seed(121)
    with torch.inference_mode():
        q = torch.randn(2, 4, 1, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(2, 2, 31, 64, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        mask = torch.ones(2, 1, 1, 31, dtype=torch.bool, device="cuda")
        mask[0, :, :, :7] = False
        mask[1] = False
        additive = torch.zeros_like(mask, dtype=q.dtype).masked_fill(~mask, -float("inf"))
        configs = [dict(attention_mask=None), dict(attention_mask=None, scaling=1.),
                   dict(attention_mask=None, is_causal=True), dict(attention_mask=mask),
                   dict(attention_mask=additive), dict(attention_mask=mask, scaling=0.3)]
        for i, kwargs in enumerate(configs):
            out, _ = fn(module, q, k, v, **kwargs)
            ref, _ = sdpa_attention_forward(module, q, k, v, **kwargs)
            torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
            print(f"PASS adapter {i} counts={counts}", flush=True)
        assert counts == {"decode": 2, "prefill": 0, "fallback": 4}
    print("PASS 6 adapter cases + SDPA mask-factory registration", flush=True)


if __name__ == "__main__":
    main()
