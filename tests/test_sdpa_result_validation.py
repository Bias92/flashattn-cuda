import copy
import json
from pathlib import Path
import unittest

from validate_sdpa_serving_results import checked_metrics, response_differences


class ResultTest(unittest.TestCase):
    def test_difference_labels_preserve_reference_direction(self):
        self.assertEqual(response_differences(actual=["same", "1998"], reference=["same", "1999"]),
                         [dict(request_index=1, reference="1999", actual="1998")])

    def fixture(self):
        return dict(completed=2, num_prompts=2, failed=0, errors=["", ""],
                    ttfts=[0.02, 0.04], itls=[[0.01, 0.03], [0.02, 0.04]], output_lens=[3, 3],
                    mean_ttft_ms=30.0, median_ttft_ms=30.0,
                    mean_tpot_ms=25.0, median_tpot_ms=25.0)

    def test_recompute(self):
        self.assertAlmostEqual(checked_metrics(self.fixture())["median_tpot_ms"], 25.0)

    def test_reject_wrong_summary_or_failed_requests(self):
        for changes in ({"median_tpot_ms": 24.0}, {"failed": 1}, {"errors": ["failure", ""]}):
            data = copy.deepcopy(self.fixture())
            data.update(changes)
            with self.assertRaises(AssertionError):
                checked_metrics(data)

    def test_existing_official_record(self):
        path = Path(__file__).resolve().parent / "fixtures/pilot_flash.json"
        checked_metrics(json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
