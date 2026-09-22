"""Validate paged gather, FP32-reference accuracy and actual SDPA dispatch."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from serving_gpu_guard import telemetry, wait_idle
from scratch_vllm.comparison import MODES, dense_attention, gather_request


EXPECTED = {
    "SDPA_FLASH": "aten::_scaled_dot_product_flash_attention",
    "SDPA_CUDNN": "aten::_scaled_dot_product_cudnn_attention",
    "SDPA_EFFICIENT": "aten::_scaled_dot_product_efficient_attention",
    "SDPA_MATH": "aten::_scaled_dot_product_attention_math",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness-only-busy-ok", action="store_true",
                        help="Allow concurrent CPU builds; never records performance numbers")
    args = parser.parse_args()
    if args.correctness_only_busy_ok:
        assert float(telemetry()["temperature.gpu"]) <= 65
    else:
        wait_idle()
    torch.manual_seed(721)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for mode in MODES:
        cases = []
        for d, n, page_size, layout in (
            (64, 1, 16, "NHD"), (64, 17, 16, "HND"),
            (64, 511, 16, "NHD"), (64, 512, 32, "HND"),
            (64, 543, 16, "NHD"), (64, 2048, 16, "NHD"),
            (128, 127, 16, "HND"),
        ):
            page_count = (n + page_size - 1) // page_size
            cache = torch.randn(2, page_count + 4, page_size, 4, d,
                                device="cuda", dtype=torch.float16)
            if layout == "HND":
                cache = cache.transpose(2, 3).contiguous().transpose(2, 3)
            ids = torch.randperm(page_count + 4, device="cuda", dtype=torch.int32)[:page_count]
            ids = torch.cat((ids, torch.full((3,), -999, dtype=torch.int32, device="cuda")))
            k, v = gather_request(cache, ids, n)
            reference_pages = torch.stack([cache[:, int(i)] for i in ids[:page_count].cpu()], 1)
            expected = reference_pages.flatten(1, 2)[:, :n].permute(0, 2, 1, 3)
            assert torch.equal(k[0], expected[0]) and torch.equal(v[0], expected[1])
            assert k.is_contiguous() and v.is_contiguous()
            q = torch.randn(1, 32, 1, d, device="cuda", dtype=torch.float16)
            scale = d ** -0.5
            kf = k.float().repeat_interleave(8, dim=1)
            vf = v.float().repeat_interleave(8, dim=1)
            oracle = torch.softmax((q.float() @ kf.transpose(-1, -2)) * scale, -1) @ vf
            try:
                actual = dense_attention(q, k, v, mode, scale)
            except RuntimeError as error:
                # PyTorch 2.10 explicitly rejects cuDNN KV length 1. Preserve
                # this limitation instead of silently switching algorithms.
                if mode != "SDPA_CUDNN" or n != 1 or "No available kernel" not in str(error):
                    raise
                cases.append(dict(d=d, n=n, page_size=page_size, layout=layout,
                                  status="unsupported", reason="cuDNN SDPA rejects KV length 1"))
                continue
            torch.testing.assert_close(actual.float(), oracle, atol=8e-4, rtol=1e-3)
            cases.append(dict(d=d, n=n, page_size=page_size, layout=layout, status="passed",
                              max_abs_error=(actual.float() - oracle).abs().max().item()))
        # Inspect the executed operator, not just enabled backend flags.
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            dense_attention(q, k, v, mode, scale)
            torch.cuda.synchronize()
        operators = sorted(event.key for event in trace.key_averages()
                           if "scaled_dot_product" in event.key)
        if mode in EXPECTED:
            assert EXPECTED[mode] in operators, (mode, operators)
            assert not any(op in operators for other, op in EXPECTED.items() if other != mode)
        results.append(dict(mode=mode, cases=cases, executed_operators=operators))
        print(f"CHECKED {mode}: {[c['status'] for c in cases]}; {operators}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(torch=torch.__version__, cudnn=torch.backends.cudnn.version(),
                                          gpu=torch.cuda.get_device_name(), results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
