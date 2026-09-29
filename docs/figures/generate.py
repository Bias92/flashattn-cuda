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
        times = {}
        for backend in ("ours_o", "flash", "cudnn"):
            times[backend] = median(median(s[backend] for s in run[key(case)]["samples_ms"])
                                    for run in maps)
            assert abs(times[backend] - case["median_ms"][backend]) < 1e-12
        b, h, _, n, d = case["shape"]
        flops = 4 * b * h * n * n * d // (2 if case["causal"] else 1)
        rows.append({"shape": case["shape"], "causal": case["causal"], "ratios": ratios,
                     "median_ms": times, "flops": flops,
                     "effective_tflops": {k: flops / (ms * 1e9) for k, ms in times.items()}})
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
    for legend in fig.legends:
        labels += legend.get_texts()
    boxes = []
    for label in labels:
        if label.get_visible() and label.get_text():
            bbox = label.get_window_extent(renderer)
            assert bbox.x0 >= -1 and bbox.y0 >= -1, label.get_text()
            assert bbox.x1 <= fig.bbox.width + 1 and bbox.y1 <= fig.bbox.height + 1, label.get_text()
            assert label.get_fontsize() * 550 / (72 * fig.get_figwidth()) >= 12, label.get_text()
            boxes.append((label.get_text(), bbox))
    for i, (text, bbox) in enumerate(boxes):
        for other, other_bbox in boxes[i + 1:]:
            assert not bbox.overlaps(other_bbox), (text, other)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def kernel_plot(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 8.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.32, right=.97, bottom=.10, top=.89, wspace=.20)
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
        ax.set_title("Causal" if causal else "Dense", fontsize=13, pad=10)
        ax.set_yticks(range(17), [f"B{b}  H{h}/{hk}  N{n}" for b, h, hk, n, _ in shapes])
        ax.set_ylim(16.6, -.6)
        ax.set_xlabel("Latency change (%)", labelpad=8)
        style_axes(ax)
    left, right = axes[0].get_xlim()
    axes[0].set_xticks([tick for tick in axes[0].get_xticks() if left <= tick <= right])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .995), ncol=2, frameon=False)
    save(fig, out / "kernel-performance.png")


def kernel_throughput_plot(rows, out):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharey=True)
    fig.subplots_adjust(left=.13, right=.98, bottom=.10, top=.88, hspace=.55)
    lengths = [1024, 2048, 4096, 8192]
    x = np.arange(len(lengths))
    for causal, ax in zip((False, True), axes):
        group = sorted([r for r in rows if r["causal"] == causal
                        and r["shape"][:3] == [1, 8, 8]], key=lambda r: r["shape"][3])
        assert [r["shape"][3] for r in group] == lengths
        assert all(r["shape"][4] == 64 for r in group)
        for i, (backend, label, color) in enumerate((
                ("ours_o", "Custom", TEAL),
                ("flash", "SDPA-Flash", BLUE),
                ("cudnn", "SDPA-cuDNN", ORANGE))):
            values = [r["effective_tflops"][backend] for r in group]
            ax.bar(x + (i - 1) * .24, values, width=.22, color=color, label=label)
        ax.set_xticks(x, [f"{n:,}" for n in lengths])
        ax.set_yticks(range(0, 41, 10))
        ax.set_ylim(0, 45)
        ax.set_xlabel("Sequence length", labelpad=8)
        ax.set_ylabel("TFLOP/s")
        ax.set_title("Causal" if causal else "Dense", fontsize=13, pad=10)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.55, .995), ncol=3, frameon=False)
    save(fig, out / "kernel-throughput.png")


def serving_plot(rows, out, study, panels, filename):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.6), sharey=study == "throughput")
    fig.subplots_adjust(left=.15, right=.98, bottom=.10, top=.88, hspace=.58)
    maxima = []
    for ax, (mode, metric, panel_title, unit, scale) in zip(axes, panels):
        maximum = 0
        settings = None
        for i, (config, label, color, _) in enumerate(SERIES):
            data = sorted([r for r in rows if r["study"] == study and r["mode"] == mode and r["config"] == config],
                          key=lambda r: (r["concurrency"], r["input_len"]))
            assert len(data) == (4 if study == "latency" else 5)
            labels = [r["concurrency"] if study == "throughput" else r["input_len"] for r in data]
            if settings is None:
                settings = labels
            assert labels == settings
            values = [median(r["values"][metric]) * scale for r in data]
            assert all(np.isfinite(value) and value > 0 for value in values)
            x = np.arange(len(data)) + (i - 1) * .24
            ax.bar(x, values, width=.22, color=color, label=label)
            maximum = max(maximum, max(values))
        maxima.append(maximum)
        ax.set_xticks(range(len(settings)), [f"{n:,}" for n in settings])
        ax.set_xlim(-.55, len(settings) - .45)
        ax.set_xlabel("Concurrent requests" if study == "throughput" else "Input tokens", labelpad=8)
        ax.set_ylabel("Output tokens/s" if study == "throughput" else f"p50 latency ({unit})")
        ax.set_title(panel_title, fontsize=13, pad=10)
        style_axes(ax)
    for ax, maximum in zip(axes, maxima):
        ax.set_ylim(0, (max(maxima) if study == "throughput" else maximum) * 1.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.55, .995),
               ncol=3, frameon=False, handlelength=1.1, columnspacing=1.0, handletextpad=.4)
    save(fig, out / filename)


def serving_tables(rows, out):
    lines = ["# Serving Measurements", "",
             "Median of three accepted runs per configuration and setting. TTFT and TPOT",
             "are per-run request p50s, not pooled request percentiles. These are the",
             "September 20-21 archived serving results, not measurements of the current build.", "",
             "Times below are in milliseconds; throughput is total output tokens/s.",
             "The long-context TTFT figure converts milliseconds to seconds.", "",
             "| Label | Attention configuration |", "|---|---|",
             "| Native Flash | Native prefill and decode |",
             "| Custom decode-only | Native prefill, custom decode |",
             "| Custom prefill + decode | Custom prefill and decode |", ""]
    for study, title in (("latency", "Low Latency"), ("throughput", "High Throughput"),
                         ("long_context", "Long Context")):
        lines += [f"## {title}", ""]
        for mode, mode_label in (("eager", "Eager"), ("graphs", "CUDA Graphs")):
            lines += [f"### {mode_label}", ""]
            settings = sorted({(r["input_len"], r["concurrency"]) for r in rows
                               if r["study"] == study and r["mode"] == mode})
            for metric, metric_title in (("p50_ttft_ms", "p50 TTFT (ms)"),
                                         ("p50_tpot_ms", "p50 TPOT (ms)"),
                                         ("output_throughput", "Output tokens/s")):
                setting_label = "Concurrent requests" if study == "throughput" else "Input tokens"
                lines += [f"**{metric_title}**", "",
                          f"| {setting_label} | Native Flash | Custom decode-only | Custom prefill + decode |",
                          "|---:|---:|---:|---:|"]
                for n, concurrency in settings:
                    values = []
                    for config, _, _, _ in SERIES:
                        matches = [r for r in rows if (r["study"], r["mode"], r["input_len"],
                                   r["concurrency"], r["config"]) == (study, mode, n, concurrency, config)]
                        assert len(matches) == 1
                        values.append(median(matches[0]["values"][metric]))
                    setting = concurrency if study == "throughput" else n
                    lines.append(f"| {setting:,} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
                lines.append("")
    lines += ["Serving percentage comparisons in the root README use these same medians",
              "before rounding. Historical per-run ratios remain in the campaign report",
              "and data.json.", "",
              "[Raw runs and campaign report](../serving/prefill_campaign_2026-09-20/REPORT.md)",
              "| [Unrounded values and source hashes](data.json)"]
    (out / "serving-measurements.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def throughput_table(rows, out):
    lines = ["# Effective Attention Throughput", "",
             "RTX 4060 Ti 8 GB, FP16 inputs and FP32 accumulation, D=64. These values",
             "are derived from the September 22 warmed O-only API measurements, not a new GPU run.", "",
             "## FLOP Accounting", "",
             "For equal query/KV lengths, the QK and PV products together use the",
             "[FlashAttention benchmark convention](https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py):", "",
             "- Dense: `F = 4 * B * H_q * N^2 * D`.",
             "- Causal: `F = 2 * B * H_q * N^2 * D` (half-dense approximation).",
             "- `effective TFLOP/s = F / (time_ms * 10^9)`; one FMA counts as two FLOPs.", "",
             "This counts useful matrix-product work, excluding softmax and padding/masked",
             "tile work. It is not a count of all hardware instructions, GPU peak utilization,",
             "or whole-model serving FLOP/s. GQA uses the query-head count H_q.", "",
             "Each time is the median of three per-run median API latencies. The paired",
             "ratio plot uses a different aggregation and can differ in sign near parity.", "",
             "## All 34 Cases", "",
             "| B | H_q/H_kv | N | Mask | GFLOP/call | Custom ms | Custom TFLOP/s | Flash TFLOP/s | cuDNN TFLOP/s |",
             "|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda r: (*r["shape"], r["causal"])):
        b, h, hk, n, _ = row["shape"]
        rates = row["effective_tflops"]
        lines.append(f"| {b} | {h}/{hk} | {n} | {'Causal' if row['causal'] else 'Dense'} | "
                     f"{row['flops'] / 1e9:.3f} | {row['median_ms']['ours_o']:.5f} | "
                     f"{rates['ours_o']:.2f} | {rates['flash']:.2f} | {rates['cudnn']:.2f} |")
    lines += ["", "[Source measurements](../serving/backend_overview_2026-09-22/README.md)",
              "| [Exact derived values and input hashes](data.json)"]
    (out / "kernel-throughput.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
                         "axes.labelcolor": INK, "axes.titlecolor": INK})
    sources = {}
    kernel = kernel_data(args.repo, sources)
    serving = serving_data(args.repo, sources)
    kernel_plot(kernel, args.output)
    kernel_throughput_plot(kernel, args.output)
    throughput_table(kernel, args.output)
    serving_plot(serving, args.output, "latency", [
        ("graphs", "p50_ttft_ms", "(a) TTFT", "ms", 1),
        ("graphs", "p50_tpot_ms", "(b) TPOT", "ms", 1),
    ], "low-latency.png")
    serving_plot(serving, args.output, "throughput", [
        ("eager", "output_throughput", "(a) Eager", "tokens/s", 1),
        ("graphs", "output_throughput", "(b) CUDA Graphs", "tokens/s", 1),
    ], "high-throughput.png")
    serving_plot(serving, args.output, "long_context", [
        ("graphs", "p50_ttft_ms", "(a) TTFT", "s", .001),
        ("graphs", "p50_tpot_ms", "(b) TPOT", "ms", 1),
    ], "long-context.png")
    serving_tables(serving, args.output)
    assert len(sources) == 256
    for relative, digest in sources.items():
        assert hashlib.sha256((args.repo / relative).read_bytes()).hexdigest() == digest
    (args.output / "data.json").write_text(json.dumps({
        "aggregation": "Kernel: median paired ratios. TFLOP/s: FLOPs divided by median API time. Serving plots: median of three per-run absolute measurements; long-context TTFT converted from ms to s. Paired serving ratios are retained separately.",
        "input_sha256": sources, "kernel": kernel, "serving": serving,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(kernel)} kernel cases, 252 serving records, {len(sources)} unchanged source files.")


if __name__ == "__main__":
    main()
