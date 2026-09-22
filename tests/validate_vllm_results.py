"""Validate saved smoke and official serving results without using the GPU."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    records = []
    for mode in ("eager", "graphs"):
        a = json.loads((root / f"smoke_flash_{mode}.json").read_text())
        b = json.loads((root / f"smoke_custom_{mode}.json").read_text())
        assert len(a["outputs"]) == len(b["outputs"]) == 3
        logprob_diff = 0
        for x, y in zip(a["outputs"], b["outputs"]):
            assert x["prompt"] == y["prompt"]
            assert x["token_ids"] == y["token_ids"] and len(x["token_ids"]) == 32
            logprob_diff = max(logprob_diff, max(abs(u - v) for u, v in
                              zip(x["selected_logprobs"], y["selected_logprobs"])))
        layers = [layer for worker in b["routing"] for layer in worker]
        assert len(layers) == 22 and all(layer["scratch_calls"] > 0 for layer in layers)
        if mode == "eager":
            assert all(layer["scratch_calls"] == 31 for layer in layers)
        records.append(dict(mode=mode, matching_tokens=96, native_layers=22,
                            selected_logprob_max_diff=logprob_diff))
    pilot = root / "online_pilot_v4"
    manifest = json.loads((pilot / "manifest.json").read_text())
    assert manifest["complete"] and len(manifest["results"]) == 4
    rows = []
    for run in (0, 1):
        a = json.loads((pilot / f"run{run}_flash_attn.json").read_text())
        b = json.loads((pilot / f"run{run}_custom.json").read_text())
        for stats in (a, b):
            assert stats["completed"] == 16 and stats["failed"] == 0
            # v0.19 RandomDataset removes one BOS from the requested length,
            # and records the non-special-token count as prompt_len.
            assert stats["input_lens"] == [511] * 16
            assert stats["output_lens"] == [32] * 16
            assert not any(stats["errors"])
        rows.append(dict(run=run,
                         requested_input_len=512, benchmark_reported_input_len=511,
                         matching_generated_texts=sum(x == y for x, y in
                             zip(a["generated_texts"], b["generated_texts"])),
                         flash_tpot_median_ms=a["median_tpot_ms"],
                         custom_tpot_median_ms=b["median_tpot_ms"],
                         median_reduction_pct=100 * (1 - b["median_tpot_ms"] / a["median_tpot_ms"]),
                         flash_tpot_mean_ms=a["mean_tpot_ms"],
                         custom_tpot_mean_ms=b["mean_tpot_ms"],
                         mean_reduction_pct=100 * (1 - b["mean_tpot_ms"] / a["mean_tpot_ms"]),
                         flash_ttft_median_ms=a["median_ttft_ms"],
                         custom_ttft_median_ms=b["median_ttft_ms"]))
    samples = [s for result in manifest["results"] for s in result["telemetry"]]
    assert not any(sample["other_jobs"] for sample in samples)
    source = Path(__file__).resolve().parents[1] / "cuda" / "attention_decode_paged.cu"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == manifest["source_sha256"]
    sanitizers = {}
    for name, marker in (("memcheck", "ERROR SUMMARY: 0 errors"),
                         ("racecheck", "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)")):
        assert marker in (root / f"paged_{name}.log").read_text()
        checked = json.loads((root / f"paged_{name}.json").read_text())
        assert checked["passed"] == 57 and checked["source_sha256"] == digest
        sanitizers[name] = marker
    result = dict(smoke=records, pilot=rows,
                  sanitizers=sanitizers,
                  max_sampled_gpu_temperature_c=max(float(s["temperature.gpu"]) for s in samples),
                  observed_other_python_jobs=0, source_sha256=digest,
                  caveats=["Two server runs per backend; 16 requests per run; exploratory only.",
                           "No controlled clock lock; no proof of zero Windows graphics load.",
                           "Prefill uses vLLM FlashAttention in both modes.",
                           "Python call counters exclude CUDA Graph replays."])
    (root / "validation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    names = {dist.metadata.get("Name") for dist in importlib.metadata.distributions()
             if dist.metadata.get("Name")}
    environment = {name: importlib.metadata.version(name) for name in names}
    (root / "environment_packages.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
