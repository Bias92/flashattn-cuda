"""Coordination checks must reject contention, not terminate unrelated jobs."""

import sys
import itertools
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import serving_gpu_guard as guard


class GuardTest(unittest.TestCase):
    def test_quiet_window_restarts_after_other_job(self):
        jobs = [[], [], [{"pid": 42}], [], [], [], [], [], []]
        with patch.object(guard, "telemetry", side_effect=lambda: {
                 "temperature.gpu": "40", "utilization.gpu": "1"}), \
             patch.object(guard, "other_python_jobs", side_effect=jobs), \
             patch.object(guard.time, "monotonic", side_effect=itertools.count()), \
             patch.object(guard.time, "sleep"):
            result = guard.wait_idle(quiet_seconds=6)
        self.assertGreaterEqual(result[-1]["quiet_seconds_observed"], 6)
        self.assertEqual(len(result), 7)

    def test_idle_requires_consecutive_samples(self):
        values = [40, 1, 1, 50, 1, 1, 1]
        samples = [dict(**{"temperature.gpu": "40", "utilization.gpu": str(v)}) for v in values]
        with patch.object(guard, "telemetry", side_effect=samples), \
             patch.object(guard, "other_python_jobs", return_value=[]), \
             patch.object(guard.time, "sleep"):
            result = guard.wait_idle()
        self.assertEqual(len(result), 7)

    def test_contention_stops_only_passed_client(self):
        proc = Mock()
        proc.poll.return_value = None
        with patch.object(guard, "telemetry", return_value={"temperature.gpu": "40"}), \
             patch.object(guard, "other_python_jobs", return_value=[{"pid": 12345}]), \
             self.assertRaisesRegex(RuntimeError, "another Python job"):
            guard.monitored_wait(proc)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()

    def test_low_power_desktop_is_explicit(self):
        sample = {"temperature.gpu": "44", "utilization.gpu": "25", "pstate": "P8",
                  "clocks.sm": "210", "clocks.mem": "405", "power.draw": "12"}
        with patch.object(guard, "telemetry", side_effect=lambda: dict(sample)), \
             patch.object(guard, "other_python_jobs", return_value=[]), \
             patch.object(guard.time, "sleep"):
            result = guard.wait_idle()
        self.assertEqual(len(result), 3)
        self.assertTrue(all(s["low_power_desktop_gate"] for s in result))

    def test_hot_gpu_stops_client(self):
        proc = Mock()
        proc.poll.return_value = None
        with patch.object(guard, "telemetry", return_value={"temperature.gpu": "81"}), \
             patch.object(guard, "other_python_jobs", return_value=[]), \
             self.assertRaisesRegex(RuntimeError, "temperature"):
            guard.monitored_wait(proc)
        proc.terminate.assert_called_once()

    def test_telemetry_error_stops_client(self):
        proc = Mock()
        proc.poll.return_value = None
        with patch.object(guard, "telemetry", side_effect=RuntimeError("GPU query failed")), \
             self.assertRaisesRegex(RuntimeError, "GPU query failed"):
            guard.monitored_wait(proc)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
