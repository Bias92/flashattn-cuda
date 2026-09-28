"""Render README figures from preserved benchmark JSON; no GPU work."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#222222"
MUTED = "#555555"
BLUE = "#2766b0"
TEAL = "#128075"
ORANGE = "#ca6435"
SERIES = (
    ("FLASH_ATTN", "Native Flash", BLUE, "o"),
    ("SCRATCH_DECODE", "Custom decode-only", ORANGE, "s"),
    ("SCRATCH_FULL", "Custom prefill + decode", TEAL, "^"),
)


def read_json(path, root, sources):
    data = path.read_bytes()
    sources[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def kernel_data(root, sources):
    folder = root / "docs/serving/backend_overview_2026-09-22"
    summary = read_json(folder / "summary.json", root, sources)
    runs = [read_json(folder / f"run{i}.json", root, sources) for i in (1, 2, 3)]
    assert all(r["meta"]["complete"] and len(r["cases"]) == 46 for r in runs)
    key = lambda c: (*c["shape"], c["causal"])
    maps = [{key(c): c for c in run["cases"]} for run in runs]
    rows = []
    for case in summary["cases"]:
        if case["shape"][-1] != 64:
            continue
        ratios = {}
        for backend in ("flash", "cudnn"):
            values = [median(s["ours_o"] / s[backend]
                             for s in run[key(case)]["samples_ms"]) for run in maps]
            assert abs(median(values) - case["ratios"]["ours_o"][backend]) < 1e-12
            ratios[backend] = values
        rows.append({"shape": case["shape"], "causal": case["causal"], "ratios": ratios})
    assert len(rows) == 34
    for group in summary["groups"]:
        if group["D"] != 64:
            continue
        for backend in ("flash", "cudnn"):
            values = [median(r["ratios"][backend]) for r in rows if r["causal"] == group["causal"]]
            actual = [sum(v < .99 for v in values), sum(.99 <= v <= 1.01 for v in values),
                      sum(v > 1.01 for v in values)]
            expected = [group["entries"]["ours_o"][backend][k] for k in ("faster", "similar", "slower")]
            assert actual == expected
    return rows


def serving_data(root, sources):
    folder = root / "docs/serving/prefill_campaign_2026-09-20/serving"
    records = {}
    for mode in ("eager", "graphs"):
        paths = sorted((folder / mode / "cases").glob("*.json"))
        assert len(paths) == 126
        for path in paths:
            c = read_json(path, root, sources)
            assert c["mode"] == mode and c["metrics"]["failed"] == 0
            key = (mode, c["study"], c["input_len"], c["concurrency"], c["config"], c["run"])
            assert key not in records
            records[key] = c
    data = []
    metrics = ("p50_ttft_ms", "p50_tpot_ms", "output_throughput")
    settings = sorted({k[:4] for k in records})
    for mode, study, n, concurrency in settings:
        for config, _, _, _ in SERIES:
            values, ratios = {}, {}
            for metric in metrics:
                values[metric] = [records[(mode, study, n, concurrency, config, r)]["metrics"][metric]
                                  for r in range(3)]
                ratios[metric] = [values[metric][r] /
                                 records[(mode, study, n, concurrency, "FLASH_ATTN", r)]["metrics"][metric]
                                 for r in range(3)]
            data.append({"mode": mode, "study": study, "input_len": n, "concurrency": concurrency,
                         "config": config, "values": values, "ratios": ratios})
    return data


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#777777")
    ax.tick_params(colors=INK, length=3, width=.6, pad=5)
    ax.grid(axis="y", color="#e5e5e5", linewidth=.6)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = list(fig.texts)
    for ax in fig.axes:
        labels += [ax.title, *ax.texts]
        if ax.axison:
            labels += [ax.xaxis.label, ax.yaxis.label,
                       *ax.get_xticklabels(), *ax.get_yticklabels()]
    for label in labels:
        if label.get_visible() and label.get_text():
            bbox = label.get_window_extent(renderer)
            assert bbox.x0 >= -1 and bbox.y0 >= -1, label.get_text()
            assert bbox.x1 <= fig.bbox.width + 1 and bbox.y1 <= fig.bbox.height + 1, label.get_text()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def kernel_plot(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.205, right=.98, bottom=.11, top=.85, wspace=.12)
    handles = []
    shapes = sorted({tuple(r["shape"]) for r in rows})
    assert len(shapes) == 17
    for causal, ax in zip((False, True), axes):
        group = {tuple(r["shape"]): r for r in rows if r["causal"] == causal}
        ax.axvspan(-1, 1, color="#eff2f5", zorder=0)
        ax.axvline(0, color=MUTED, lw=1)
        for backend, label, color, offset, marker in (
                ("flash", "vs SDPA-Flash", BLUE, -.14, "o"),
                ("cudnn", "vs SDPA-cuDNN", ORANGE, .14, "s")):
            values = np.array([group[s]["ratios"][backend] for s in shapes]) * 100 - 100
            middle = np.median(values, axis=1)
            h = ax.errorbar(middle, np.arange(17) + offset,
                            xerr=[middle - values.min(axis=1), values.max(axis=1) - middle],
                            fmt=marker, markersize=5, color=color, capsize=2, lw=1, label=label)
            if not causal:
                handles.append(h)
        ax.set_title("Causal" if causal else "Dense", fontsize=11, pad=10)
        ax.set_yticks(range(17), [f"B{b}  H{h}/{hk}  N{n}" for b, h, hk, n, _ in shapes])
        ax.set_ylim(16.6, -.6)
        ax.set_xlabel("Latency change (%)", labelpad=8)
        style_axes(ax)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.60, .99), ncol=2, frameon=False)
    save(fig, out / "kernel-performance.png")


def serving_plot(rows, out, study, panels, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.subplots_adjust(left=.08, right=.98, bottom=.18, top=.78, wspace=.27)
    for ax, (mode, metric, panel_title, ylabel, divisor) in zip(axes, panels):
        for (config, label, color, marker), line in zip(SERIES, ("-", "--", ":")):
            data = sorted([r for r in rows if r["study"] == study and r["mode"] == mode and r["config"] == config],
                          key=lambda r: (r["concurrency"], r["input_len"]))
            assert len(data) == (4 if study == "latency" else 5)
            labels = [r["concurrency"] if study == "throughput" else r["input_len"] for r in data]
            values = [median(r["values"][metric]) / divisor for r in data]
            ax.plot(labels, values, color=color, marker=marker, label=label,
                    linewidth=1.3, markersize=4, linestyle=line, markerfacecolor="white")
            ax.set_xscale("log", base=2)
            ax.set_xticks(labels, [f"{n:,}" for n in labels], fontsize=10)
            ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Concurrent requests (log scale)" if study == "throughput" else "Input tokens (log scale)", labelpad=10)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, fontsize=11, pad=10)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1.005), ncol=3, frameon=False)
    if panels[0][1] == panels[1][1]:
        ymax = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(0, ymax)
    save(fig, out / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK, "axes.titlecolor": INK})
    sources = {}
    kernel = kernel_data(args.repo, sources)
    serving = serving_data(args.repo, sources)
    kernel_plot(kernel, args.output)
    serving_plot(serving, args.output, "latency", [
        ("graphs", "p50_ttft_ms", "TTFT", "p50 latency (ms)", 1),
        ("graphs", "p50_tpot_ms", "TPOT", "p50 latency (ms)", 1),
    ], "low-latency.png")
    serving_plot(serving, args.output, "throughput", [
        ("eager", "output_throughput", "Eager execution", "Output tokens/s", 1),
        ("graphs", "output_throughput", "CUDA Graphs", "Output tokens/s", 1),
    ], "high-throughput.png")
    serving_plot(serving, args.output, "long_context", [
        ("graphs", "p50_ttft_ms", "TTFT", "p50 latency (s)", 1000),
        ("graphs", "p50_tpot_ms", "TPOT", "p50 latency (ms)", 1),
    ], "long-context.png")
    assert len(sources) == 256
    for relative, digest in sources.items():
        assert hashlib.sha256((args.repo / relative).read_bytes()).hexdigest() == digest
    (args.output / "data.json").write_text(json.dumps({
        "aggregation": "Kernel: median paired ratios. Serving plots: median of three per-run metrics.",
        "input_sha256": sources, "kernel": kernel, "serving": serving,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(kernel)} kernel cases, 252 serving records, {len(sources)} unchanged source files.")


if __name__ == "__main__":
    main()
