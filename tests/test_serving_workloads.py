"""CPU-only regressions for unattended workload accounting."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from bench_serving_workloads import (StopRequested, capture_runtime, check_runtime,
    classify_jobs, model_comparison, percentile, prometheus_values, request_count,
    routes, validated_metrics)


class WorkloadTests(unittest.TestCase):
    def fixture(self):
        return json.loads((ROOT / "tests/fixtures/workload_flash.json").read_text())

    def test_existing_official_metrics(self):
        result = validated_metrics(self.fixture(), 16, 512, 32)
        self.assertGreater(result["output_throughput"], 0)
        self.assertGreaterEqual(result["p95_tpot_ms"], result["p50_tpot_ms"])

    def test_separate_first_token_clocks(self):
        stats = self.fixture()
        gaps_ms = [0.2] * 16
        stats["mean_e2el_ms"] -= sum(gaps_ms) / 16
        for key in ("mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms"):
            stats[key] -= gaps_ms[0] / 31
        stats["p99_e2el_ms"] -= gaps_ms[0]
        result = validated_metrics(stats, 16, 512, 32)
        self.assertGreater(result["first_token_timestamp_offset_sum_ms"], 3.1)

    def test_reject_tpot_outside_observed_offset_bound(self):
        stats = self.fixture()
        stats["p99_tpot_ms"] -= 0.1
        with self.assertRaises(ValueError):
            validated_metrics(stats, 16, 512, 32)

    def test_reject_tpot_e2el_identity_violation(self):
        stats = self.fixture()
        stats["mean_tpot_ms"] *= 2
        with self.assertRaises(ValueError):
            validated_metrics(stats, 16, 512, 32)

    def test_reject_negative_timestamp_offset(self):
        stats = self.fixture()
        stats["mean_e2el_ms"] += 1
        with self.assertRaises(ValueError):
            validated_metrics(stats, 16, 512, 32)

    def test_reject_wrong_tokens(self):
        with self.assertRaises(ValueError):
            validated_metrics(self.fixture(), 16, 512, 128)

    def test_zero_special_token_model(self):
        stats = self.fixture()
        stats['input_lens'] = [512] * 16
        stats['total_input_tokens'] = 512 * 16
        result = validated_metrics(stats, 16, 512, 32, special_tokens=0)
        self.assertEqual(result['input_tokens'], 8192)
        with self.assertRaises(ValueError):
            validated_metrics(stats, 16, 512, 32, special_tokens=1)

    def test_reject_fake_throughput(self):
        stats = self.fixture()
        stats["output_throughput"] *= 2
        with self.assertRaises(ValueError):
            validated_metrics(stats, 16, 512, 32)

    def test_reject_negative_duration(self):
        stats = self.fixture()
        stats["duration"] = -1
        with self.assertRaises(ValueError):
            validated_metrics(stats, 16, 512, 32)

    def test_percentile(self):
        self.assertEqual(percentile([0, 100], 95), 95)
        self.assertEqual(percentile([8], 50), 8)
        with self.assertRaises(ValueError):
            percentile([], 50)

    def test_runtime_fingerprint_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime"
            path.write_bytes(b"original")
            snapshot = capture_runtime([path])
            check_runtime(snapshot)
            self.assertEqual(len(snapshot["files"][str(path)]["sha256"]), 64)

    def test_runtime_replacement_stops_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime"
            path.write_bytes(b"original")
            snapshot = capture_runtime([path])
            replacement = path.with_suffix(".new")
            replacement.write_bytes(b"original")
            replacement.replace(path)
            with self.assertRaises(StopRequested):
                check_runtime(snapshot)

    def test_missing_runtime_stops_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime"
            path.write_bytes(b"original")
            snapshot = capture_runtime([path])
            path.unlink()
            with self.assertRaises(StopRequested):
                check_runtime(snapshot)

    def test_sustained_request_counts(self):
        self.assertEqual(request_count(64, 8), 64)
        self.assertEqual(request_count(64, 16), 128)
        self.assertEqual(request_count(64, 32), 256)

    def test_only_proven_owned_sessions_are_excluded(self):
        jobs = [dict(pid=101, command=['python','resource_tracker']), dict(pid=202, command=['python','resource_tracker'])]
        with patch('bench_serving_workloads.os.getpgid', side_effect=lambda pid: 100 if pid == 101 else 200), \
             patch('bench_serving_workloads.os.getsid', side_effect=lambda pid: 100 if pid == 101 else 200):
            foreign, owned = classify_jobs(jobs, {100: 10}, 9)
            self.assertEqual([j['pid'] for j in foreign], [202])
            self.assertEqual([j['pid'] for j in owned], [101])
            foreign, owned = classify_jobs(jobs, {100: 10}, 11)
            self.assertEqual(len(foreign), 2)
            self.assertFalse(owned)

    def test_smoke_identical(self):
        data = dict(outputs=[dict(prompt="x", ids=[1, 2], logprobs=[{"1": -0.1}, {"2": -0.1}])])
        self.assertTrue(model_comparison(data, data)["all_identical"])

    def test_smoke_small_margin_recorded(self):
        ref = dict(outputs=[dict(prompt="x", ids=[1, 2], logprobs=[{"1": -0.1}, {"2": -0.1, "3": -0.12}])])
        actual = copy.deepcopy(ref)
        actual["outputs"][0]["ids"][1] = 3
        result = model_comparison(actual, ref)
        self.assertFalse(result["all_identical"])
        self.assertAlmostEqual(result["prompts"][0]["reference_logprob_margin"], 0.02)

    def test_smoke_large_divergence_rejected(self):
        ref = dict(outputs=[dict(prompt="x", ids=[1], logprobs=[{"1": -0.1, "2": -1.0}])])
        actual = copy.deepcopy(ref)
        actual["outputs"][0]["ids"] = [2]
        with self.assertRaises(ValueError):
            model_comparison(actual, ref)

    def test_routing_delta_and_native_unknown(self):
        before = [dict(routing=[dict(name="x", implementation="custom", scratch_calls=2, prefill_calls=3),
                               dict(name="y", implementation="native", scratch_calls=None, prefill_calls=None)])]
        after = copy.deepcopy(before)
        after[0]["routing"][0].update(scratch_calls=9, prefill_calls=4)
        result = routes(before, after)
        self.assertEqual(result[0]["scratch_calls"], 7)
        self.assertEqual(result[0]["fallback_calls"], 1)
        self.assertIsNone(result[1]["scratch_calls"])
        with self.assertRaises(ValueError):
            routes(after, before)

    def test_prometheus_structured_parse(self):
        raw = '# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc{engine="0"} 0.25\n'
        raw += '# TYPE unrelated gauge\nunrelated 100\n'
        result = prometheus_values(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["value"], 0.25)


if __name__ == "__main__":
    unittest.main()
