"""Recompute direct paired comparisons from preserved benchmark samples."""
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys


def estimate(values):
    rng = random.Random(817)
    boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(4000))
    return {"median": statistics.median(values), "ci95": [boot[99], boot[3899]]}


def summarize(path):
    raw = path.read_bytes()
    result = json.loads(raw)
    assert result["meta"]["complete"], path
    cases = []
    for case in result["cases"]:
        samples = case["samples_ms"]
        keys = samples[0].keys()
        comparisons = {key: estimate([s["candidate"] / s[key] for s in samples])
                       for key in keys if key != "candidate"}
        cases.append({"shape": case["shape"], "entry": case["entry"],
                      "median_ms": {key: statistics.median(s[key] for s in samples) for key in keys},
                      "candidate_over": comparisons})
    return {"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "sources": result["meta"]["sources"],
            "gpu_before": result["meta"]["gpu_before"],
            "gpu_after": result["meta"]["gpu_after"],
            "guard_before": result["meta"]["guard_before"],
            "guard_after": result["meta"]["guard_after"], "cases": cases}


if __name__ == "__main__":
    print(json.dumps([summarize(Path(name)) for name in sys.argv[1:]], indent=2))
