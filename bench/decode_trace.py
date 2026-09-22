"""Read Kineto Chrome traces without double-counting nested CPU operators."""
from collections import Counter, defaultdict


def union_duration(intervals):
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def summarize_trace(trace, steps):
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and "dur" in e]
    windows = [e for e in events if e["name"] == "decode.window" and e.get("cat") == "user_annotation"]
    if len(windows) != 1:
        raise ValueError("Expected one synchronized decode.window")
    window = windows[0]
    begin, end = window["ts"], window["ts"] + window["dur"]
    scopes = defaultdict(list)
    host = Counter()
    for event in events:
        if event["name"].startswith("phase::") and event.get("cat") == "user_annotation":
            scopes[event["tid"]].append(event)
            host[event["name"].split("::", 1)[1]] += event["dur"]

    def category(cpu):
        if cpu is None:
            return "unattributed"
        candidates = [s for s in scopes[cpu["tid"]]
                      if s["ts"] <= cpu["ts"] < s["ts"] + s["dur"]]
        return min(candidates, key=lambda s: s["dur"])["name"].split("::", 1)[1] if candidates else "other"

    external = {}
    runtime = {}
    api = Counter()
    api_calls = Counter()
    sync_sites = Counter()
    for event in events:
        args = event.get("args", {})
        if event.get("cat") in ("cpu_op", "user_annotation") and "External id" in args:
            external[args["External id"]] = event
        if event.get("cat") in ("cuda_runtime", "cuda_driver"):
            if "correlation" in args:
                runtime[args["correlation"]] = event
            if not begin <= event["ts"] < end:
                continue
            name = event["name"]
            api[name] += event["dur"]
            api_calls[name] += 1
            if "Synchronize" in name or "Memcpy" in name:
                sync_sites[category(event) + ":" + name] += event["dur"]

    gpu, counts, intervals, names, details = Counter(), Counter(), [], Counter(), []
    for event in events:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        if not begin <= event["ts"] < end:
            continue
        args = event.get("args", {})
        cpu = runtime.get(args.get("correlation"))
        if cpu is None:
            cpu = external.get(args.get("External id"))
        group = category(cpu)
        gpu[group] += event["dur"]
        counts[group] += 1
        names[(group, event["name"])] += event["dur"]
        intervals.append((event["ts"], event["ts"] + event["dur"]))
        if group == "unattributed":
            details.append(dict(name=event["name"], args=args))
    if not intervals:
        raise ValueError("Trace has no GPU activities in the decode window")
    active = union_duration(intervals)
    gpu_span = max(b for _, b in intervals) - min(a for a, _ in intervals)
    scale = 1.0 / (1000 * steps)
    total_device = sum(gpu.values())
    return dict(
        steps=steps, trace_wall_ms_per_token=window["dur"] * scale,
        gpu_active_union_ms_per_token=active * scale,
        gpu_activity_sum_ms_per_token=total_device * scale,
        gpu_span_ms_per_token=gpu_span * scale,
        gaps_between_gpu_activities_ms_per_token=(gpu_span - active) * scale,
        gpu_active_fraction_of_trace_window=active / window["dur"],
        gpu_ms_per_token={k: v * scale for k, v in gpu.most_common()},
        gpu_activity_fraction={k: v / total_device for k, v in gpu.most_common()},
        gpu_calls_per_token={k: v / steps for k, v in counts.most_common()},
        host_phase_ms_per_token={k: v * scale for k, v in host.most_common()},
        cuda_api_ms_per_token={k: v * scale for k, v in api.most_common()},
        cuda_api_calls_per_token={k: v / steps for k, v in api_calls.most_common()},
        sync_sites_ms_per_token={k: v * scale for k, v in sync_sites.most_common()},
        top_gpu_activities=[dict(phase=k[0], kernel=k[1], ms_per_token=v * scale)
                            for k, v in names.most_common(20)],
        unattributed=details[:10],
        caveat="Instrumented trace; host/GPU times overlap and must not be added. Gaps are not proof of CPU causality.",
    )


if __name__ == "__main__":
    import argparse
    import gzip
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as handle:
        print(json.dumps(summarize_trace(json.load(handle), args.steps), indent=2))
