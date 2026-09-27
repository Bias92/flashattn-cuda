"""CPU checks for header-aware prefill build identity; no CUDA compilation."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrefillSourceManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "attention_forward.cu"
        self.header = self.root / "flash_fwd_kernel.h"
        self.source.write_text('#include "flash_fwd_kernel.h"\n')
        self.header.write_text("// original header\n")
        self.so = self.root / "mock.so"
        self.so.write_bytes(b"mock library")
        self.load = Mock(return_value=types.SimpleNamespace(__file__=str(self.so)))
        self.cpp = types.ModuleType("torch.utils.cpp_extension")
        self.cpp.load = self.load
        self.mock_modules = patch.dict("sys.modules", {"torch.utils.cpp_extension": self.cpp})
        self.mock_modules.start()
        self.addCleanup(self.mock_modules.stop)
        self.bench = load_file("prefill_build_test", ROOT / "bench/prefill_support.py")
        self.serving = load_file(
            "prefill_loader_test", ROOT / "integrations/vllm_prefill/scratch_vllm_prefill/loader.py")
        self.serving.SOURCE = self.source

    def test_header_only_edit_changes_identity(self):
        before = self.bench.source_manifest(self.source)
        self.header.write_text("// edited header\n")
        after = self.bench.source_manifest(self.source)
        self.assertNotEqual(before["build_sources_sha256"], after["build_sources_sha256"])
        self.assertEqual(before["source_files"][self.source.name], after["source_files"][self.source.name])

    def test_added_cuh_changes_identity_but_docs_do_not(self):
        before = self.bench.source_manifest(self.source)
        (self.root / "README.md").write_text("Documentation\n")
        self.assertEqual(before, self.bench.source_manifest(self.source))
        (self.root / "attention_forward_ops.cuh").write_text("// PTX wrappers\n")
        self.assertNotEqual(before, self.bench.source_manifest(self.source))

    def test_serving_and_benchmark_use_the_same_build_identity(self):
        with patch("builtins.print"):
            _, record = self.bench.extension(self.source)
            bench_name = self.load.call_args.kwargs["name"]
            self.serving.load_prefill_extension()
            serving_name = self.load.call_args.kwargs["name"]
        digest = record["build_sources_sha256"][:12]
        self.assertTrue(bench_name.endswith(digest))
        self.assertTrue(serving_name.endswith(digest))
        self.assertEqual(set(record["source_files"]), {self.source.name, self.header.name})

    def test_source_distribution_includes_forward_headers(self):
        manifest = (ROOT / "MANIFEST.in").read_text()
        self.assertIn("include cuda/README.md", manifest)
        for name in ("kernel_traits.h", "flash_fwd_memory.h", "softmax.h", "flash_fwd_kernel.h",
                     "flash_fwd_launch_template.h", "attention_forward_ops.cuh"):
            self.assertIn(f"include cuda/{name}", manifest)

    def test_fresh_loader_build_identity_changes_on_header_edit(self):
        with patch("builtins.print"):
            self.serving.load_prefill_extension()
            before = self.load.call_args.kwargs["name"]
            # An engine restart normally supplies a fresh per-process cache.
            self.serving.load_prefill_extension.cache_clear()
            self.header.write_text("// changed after engine shutdown\n")
            self.serving.load_prefill_extension()
            after = self.load.call_args.kwargs["name"]
        self.assertNotEqual(before, after)

    def test_serving_campaign_tracks_all_local_headers(self):
        campaign = load_file("prefill_campaign_test", ROOT / "bench/bench_serving_prefill.py")
        headers = {p.relative_to(ROOT).as_posix()
                   for pattern in ("*.h", "*.cuh") for p in (ROOT / "cuda").glob(pattern)}
        self.assertTrue(headers)
        self.assertTrue(headers.issubset(campaign.SOURCES))
        self.assertEqual(len(campaign.SOURCES), len(set(campaign.SOURCES)))
        with patch.object(campaign, "ROOT", self.root), \
                patch.object(campaign, "SOURCES", [self.source.name, self.header.name]):
            before = campaign.source_hashes()
            self.header.write_text("// edited during a campaign\n")
            self.assertNotEqual(before, campaign.source_hashes())


if __name__ == "__main__":
    unittest.main()
