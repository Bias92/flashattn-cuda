"""CPU-only checks for GPU attribution, interval unions, and async scope timing."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from decode_trace import summarize_trace, union_duration


class TraceTests(unittest.TestCase):
    def test_union(self):
        self.assertEqual(union_duration([(0, 3), (1, 2), (2, 5), (8, 9)]), 6)

    def test_async_kernel_uses_launch_scope(self):
        def event(name, cat, ts, dur, tid=1, **args):
            return dict(name=name, cat=cat, ts=ts, dur=dur, tid=tid, ph="X", args=args)
        trace = dict(traceEvents=[
            event("decode.window", "user_annotation", 0, 100),
            event("decode.window", "gpu_user_annotation", 30, 60, tid=9),
            event("phase::attention", "user_annotation", 0, 20),
            event("phase::attention", "gpu_user_annotation", 30, 10, tid=9),
            event("phase::mlp", "user_annotation", 20, 70),
            event("cudaLaunchKernel", "cuda_runtime", 5, 2, correlation=7),
            event("attention_kernel", "kernel", 30, 10, tid=9, correlation=7),
            event("cudaLaunchKernel", "cuda_runtime", 45, 2, correlation=8),
            event("mlp_kernel", "kernel", 70, 20, tid=9, correlation=8),
        ])
        result = summarize_trace(trace, 1)
        self.assertEqual(result["gpu_ms_per_token"], {"mlp": 0.02, "attention": 0.01})
        self.assertAlmostEqual(result["gpu_active_union_ms_per_token"], 0.03)
        self.assertAlmostEqual(result["gaps_between_gpu_activities_ms_per_token"], 0.03)
        self.assertFalse(result["unattributed"])
        self.assertEqual(result["host_phase_ms_per_token"]["attention"], 0.02)

    def test_external_id_fallback_and_missing_gpu(self):
        def event(name, cat, ts, dur, tid=1, **args):
            return dict(name=name, cat=cat, ts=ts, dur=dur, tid=tid, ph="X", args=args)
        trace = dict(traceEvents=[
            event("decode.window", "user_annotation", 0, 100),
            event("phase::kv_cache", "user_annotation", 0, 20),
            event("aten::cat", "cpu_op", 5, 10, **{"External id": 12}),
            event("cache_kernel", "kernel", 30, 10, tid=9, **{"External id": 12}),
        ])
        self.assertEqual(summarize_trace(trace, 1)["gpu_ms_per_token"], {"kv_cache": 0.01})
        trace["traceEvents"].pop()
        with self.assertRaises(ValueError):
            summarize_trace(trace, 1)


if __name__ == "__main__":
    unittest.main()
