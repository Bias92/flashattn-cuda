import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from bench_vllm_serving import resume_report


class ResumeTest(unittest.TestCase):
    def fixture(self, root):
        original = b"original runner\n"
        digest = hashlib.sha256(original).hexdigest()
        (root / "runner_before_resume.py").write_bytes(original)
        stats = dict(completed=1, failed=0, errors=[""])
        (root / "run0_flash_attn.json").write_text(json.dumps(stats))
        saved = dict(complete=False, settings={"runs": 2, "num_prompts": 1},
                     versions={"vllm": "0.19.0"}, error="another job started",
                     implementation_sha256={"bench/bench_vllm_serving.py": digest,
                                             "kernel.cu": "unchanged"},
                     results=[dict(run=0, backend="FLASH_ATTN")])
        (root / "manifest.json").write_text(json.dumps(saved))
        requested = copy.deepcopy(saved)
        requested["settings"]["resume"] = True
        requested["implementation_sha256"]["bench/bench_vllm_serving.py"] = "new-runner"
        return requested

    def test_preserve_and_attribute_completed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requested = self.fixture(root)
            resumed = resume_report(root, requested)
            self.assertEqual(len(resumed["results"]), 1)
            self.assertEqual(resumed["resume_events"][0]["completed_rows"], 1)
            self.assertEqual(resumed["resume_events"][0]["previous_error"], "another job started")
            self.assertNotEqual(resumed["results"][0]["runner_sha256"], "new-runner")
            self.assertIn("run0_flash_attn.json", resumed["resume_events"][0]["preserved_raw_sha256"])
            self.assertFalse(resumed["complete"])

    def test_reject_protocol_or_kernel_change(self):
        for section, key, value in (("settings", "runs", 3),
                                    ("implementation_sha256", "kernel.cu", "changed"),
                                    ("versions", "vllm", "different")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                requested = self.fixture(root)
                requested[section][key] = value
                with self.assertRaises(ValueError):
                    resume_report(root, requested)

    def test_reject_unproven_runner_or_failed_record(self):
        for invalid in ("archive", "result"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                requested = self.fixture(root)
                if invalid == "archive":
                    (root / "runner_before_resume.py").write_bytes(b"wrong file")
                else:
                    (root / "run0_flash_attn.json").write_text(json.dumps(dict(completed=0, failed=1, errors=["bad"])))
                with self.assertRaises(ValueError):
                    resume_report(root, requested)


if __name__ == "__main__":
    unittest.main()
