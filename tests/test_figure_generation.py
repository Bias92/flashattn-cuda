"""Check chart units and backend coverage without running GPU benchmarks."""

import importlib.util
import hashlib
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "docs/figures/generate.py"
SPEC = importlib.util.spec_from_file_location("generate_figures", SOURCE)
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)


def sample_rows(study, mode):
    settings = {
        "latency": [(n, 1) for n in (128, 512, 1024, 1920)],
        "long_context": [(n, 1) for n in (2048, 4096, 8192, 16384, 32640)],
        "throughput": [(512, c) for c in (2, 4, 8, 16, 32)],
    }[study]
    rows = []
    for config_index, (config, _, _, _) in enumerate(figures.SERIES):
        for setting_index, (n, concurrency) in enumerate(settings):
            value = 10 * config_index + setting_index + (100 if mode == "graphs" else 0)
            metrics = ("p50_ttft_ms", "p50_tpot_ms", "output_throughput")
            rows.append({"study": study, "mode": mode, "input_len": n,
                         "concurrency": concurrency, "config": config,
                         "values": {metric: [value + 3, value + 1, value + 2] for metric in metrics},
                         "ratios": {metric: [1, 9, 7] for metric in metrics}})
    return rows


class FigureGenerationTests(unittest.TestCase):
    def tearDown(self):
        figures.plt.close("all")

    def render(self, rows, study, panels):
        with patch.object(figures, "save") as save:
            figures.serving_plot(rows, Path("."), study, panels, "unused.png")
        return save.call_args.args[0]

    def test_latency_uses_absolute_run_medians_and_includes_native(self):
        fig = self.render(sample_rows("latency", "graphs"), "latency", [
            ("graphs", "p50_ttft_ms", "TTFT", "ms", 1),
            ("graphs", "p50_tpot_ms", "TPOT", "ms", 1),
        ])
        for ax in fig.axes:
            self.assertEqual(len(ax.patches), 12)
            np.testing.assert_allclose([bar.get_height() for bar in ax.patches],
                                       [102, 103, 104, 105, 112, 113, 114, 115, 122, 123, 124, 125])
            self.assertEqual(ax.get_ylim()[0], 0)
            self.assertEqual(ax.get_ylabel(), "p50 latency (ms)")
            self.assertEqual(ax.get_legend_handles_labels()[1], [s[1] for s in figures.SERIES])

    def test_long_context_converts_ttft_only_to_seconds(self):
        fig = self.render(sample_rows("long_context", "graphs"), "long_context", [
            ("graphs", "p50_ttft_ms", "TTFT", "s", .001),
            ("graphs", "p50_tpot_ms", "TPOT", "ms", 1),
        ])
        self.assertAlmostEqual(fig.axes[0].patches[0].get_height(), .102)
        self.assertEqual(fig.axes[1].patches[0].get_height(), 102)
        self.assertEqual(fig.axes[0].get_ylabel(), "p50 latency (s)")
        self.assertEqual(len(fig.axes[0].patches), 15)

    def test_throughput_keeps_modes_separate_on_one_scale(self):
        rows = sample_rows("throughput", "eager") + sample_rows("throughput", "graphs")
        fig = self.render(rows, "throughput", [
            ("eager", "output_throughput", "Eager", "tokens/s", 1),
            ("graphs", "output_throughput", "Graphs", "tokens/s", 1),
        ])
        self.assertEqual(fig.axes[0].patches[0].get_height(), 2)
        self.assertEqual(fig.axes[1].patches[0].get_height(), 102)
        self.assertEqual(fig.axes[0].get_ylim(), fig.axes[1].get_ylim())
        self.assertEqual(fig.axes[0].get_ylabel(), "Output tokens/s")
        self.assertEqual([label.get_text() for label in fig.axes[0].get_xticklabels()],
                         ["2", "4", "8", "16", "32"])

    def test_tables_use_same_absolute_values(self):
        rows = [row for study in ("latency", "throughput", "long_context")
                for mode in ("eager", "graphs") for row in sample_rows(study, mode)]
        with TemporaryDirectory() as folder:
            figures.serving_tables(rows, Path(folder))
            text = (Path(folder) / "serving-measurements.md").read_text()
        self.assertIn("| 128 | 102.000 | 112.000 | 122.000 |", text)
        self.assertEqual(text.count("**p50 TTFT (ms)**"), 6)
        self.assertEqual(text.count("**Output tokens/s**"), 6)


class PublishedMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = SOURCE.parents[2]
        cls.rows = figures.serving_data(cls.root, {})

    def test_readme_tables_match_raw_run_medians(self):
        readme = (self.root / "README.md").read_text(encoding="utf-8")
        cases = (
            ("latency", "Low Latency", "graphs", "CUDA Graphs", "p50_ttft_ms", "p50 TTFT (ms)", 1, 3),
            ("latency", "Low Latency", "graphs", "CUDA Graphs", "p50_tpot_ms", "p50 TPOT (ms)", 1, 3),
            ("throughput", "High Throughput", "eager", "eager", "output_throughput", "Output tokens/s", 1, 3),
            ("throughput", "High Throughput", "eager", "eager", "p50_tpot_ms", "p50 TPOT (ms)", 1, 3),
            ("throughput", "High Throughput", "graphs", "CUDA Graphs", "output_throughput", "Output tokens/s", 1, 3),
            ("throughput", "High Throughput", "graphs", "CUDA Graphs", "p50_tpot_ms", "p50 TPOT (ms)", 1, 3),
            ("long_context", "Long Context", "graphs", "CUDA Graphs", "p50_ttft_ms", "p50 TTFT (s)", .001, 4),
            ("long_context", "Long Context", "graphs", "CUDA Graphs", "p50_tpot_ms", "p50 TPOT (ms)", 1, 3),
        )
        for study, section, mode, mode_label, metric, title, scale, precision in cases:
            with self.subTest(study=study, mode=mode, metric=metric):
                section_text = readme.split(f"### {section}\n", 1)[1].split("</details>", 1)[0]
                table = section_text.split(f"**{title}, {mode_label}**\n", 1)[1].strip()
                table = table.split("\n\n", 1)[0]
                rows = [r for r in self.rows if r["study"] == study and r["mode"] == mode]
                settings = sorted({(r["input_len"], r["concurrency"]) for r in rows})
                expected = []
                for n, concurrency in settings:
                    values = []
                    for config, _, _, _ in figures.SERIES:
                        row, = [r for r in rows if (r["input_len"], r["concurrency"], r["config"])
                                == (n, concurrency, config)]
                        values.append(f"{median(row['values'][metric]) * scale:.{precision}f}")
                    setting = concurrency if study == "throughput" else n
                    expected.append(f"| {setting:,} | " + " | ".join(values) + " |")
                self.assertEqual(table.splitlines()[2:], expected)

    def test_published_data_and_full_tables_match_sources(self):
        folder = self.root / "docs/figures"
        data = json.loads((folder / "data.json").read_text(encoding="utf-8"))
        self.assertEqual(data["serving"], self.rows)
        self.assertEqual(len(data["input_sha256"]), 256)
        for relative, digest in data["input_sha256"].items():
            with self.subTest(source=relative):
                self.assertEqual(hashlib.sha256((self.root / relative).read_bytes()).hexdigest(), digest)
        with TemporaryDirectory() as temp:
            figures.serving_tables(self.rows, Path(temp))
            expected = (Path(temp) / "serving-measurements.md").read_text(encoding="utf-8")
        self.assertEqual((folder / "serving-measurements.md").read_text(encoding="utf-8"), expected)


if __name__ == "__main__":
    unittest.main()
