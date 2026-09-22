"""Serving comparison of three attention configurations in vLLM 0.19.

    FLASH_ATTN      vLLM's FlashAttention for prefill and decode
    SCRATCH_DECODE  integrations/vllm: scratch paged decode, FlashAttention prefill
    SCRATCH_FULL    integrations/vllm_prefill: scratch prefill and scratch paged decode

The conditions are those of the decode campaign (bench/bench_serving_workloads.py, 2026-09-15):
`vllm serve` and `vllm bench serve`, random dataset at request rate inf, FP16, 128 output
tokens, max-num-seqs 32, 2048 batched tokens per step, prefix caching off, GPU memory fraction
0.55, server seed 42, max(64, 8 C) prompts per case after a warm-up of max(3, 2 C) prompts
with another seed. Eager and CUDA-graph servers are separate campaigns (--mode), never mixed.

    latency        TinyLlama-1.1B, one request at a time, inputs 128 to 1920
    throughput     TinyLlama-1.1B, input 512, 2 to 32 concurrent requests (mixed batches)
    long_context   Qwen2.5-0.5B, one request at a time, inputs 2048 to 32640 (chunked prefill)

One server per (study, configuration, run) serves all cases of the study; the configuration
order rotates from run to run. Server start and warm-up are outside the measured window.

What ran is read from the server, not assumed: the per-route counters of every attention layer
and the server's own token counters are read right before and right after the measured client
run. A SCRATCH_FULL case is rejected if a single token went to the FlashAttention fallback. In
eager mode every token passes through forward(), so the routes have to add up to exactly the
tokens the server processed; with CUDA graphs, steps of nothing but single-token requests are
replayed without passing through forward(), so only the prefill routes are held against the
server's prompt tokens (a one-token piece of a chunked prompt takes the decode route, hence
the 1% allowance) and the decode kernel is shown to have run before the window instead of
inside it. Counting is a few dictionary updates per layer call.

Another program on the GPU ruins a case. Before a session and before every case the runner
waits until the GPU is idle by nvidia-smi and the Windows host reports no busy GPU process
outside the WSL VM (bench/gpu_busy.ps1); the host is asked again right after the case, and a
case with more than 15% there is rejected. Rejected attempts stay on disk under sessions/ and
are listed in failures.jsonl; the case is tried again in a later sweep.

    python3 bench/bench_serving_prefill.py --output-dir docs/serving/<dir> --mode eager
    python3 bench/bench_serving_prefill.py --output-dir docs/serving/<dir> --mode eager --report
"""

import argparse
import hashlib
import json
import os
import platform
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "tinyllama": dict(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6", capacity=2048),
    "qwen": dict(model="Qwen/Qwen2.5-0.5B-Instruct",
                 revision="7ae557604adf67be50417f59c2c2f167def9a775", capacity=32768),
}
STUDIES = {
    # name: (model, [(input tokens, concurrent requests)])
    "latency": ("tinyllama", [(n, 1) for n in (128, 512, 1024, 1920)]),
    "throughput": ("tinyllama", [(512, c) for c in (2, 4, 8, 16, 32)]),
    "long_context": ("qwen", [(n, 1) for n in (2048, 4096, 8192, 16384, 32640)]),
}
CONFIGS = {
    # name: (VLLM_PLUGINS, vLLM attention backend, implementation class the layers must report)
    "FLASH_ATTN": ("", "FLASH_ATTN", "FlashAttentionImpl"),
    "SCRATCH_DECODE": ("scratch_attention", "CUSTOM", "ScratchImpl"),
    "SCRATCH_FULL": ("scratch_prefill", "CUSTOM", "ScratchPrefillImpl"),
}
OUTPUT_LEN = 128
BASE_PROMPTS = 64
OTHER_GPU_USER_LIMIT = 15.0   # percent of the GPU held by a Windows process outside the WSL VM
SOURCES = ["cuda/attention_forward.cu", "cuda/attention_decode_paged.cu",
           "integrations/vllm/scratch_vllm/backend.py", "integrations/vllm/scratch_vllm/loader.py",
           "integrations/vllm_prefill/scratch_vllm_prefill/backend.py",
           "integrations/vllm_prefill/scratch_vllm_prefill/loader.py",
           "integrations/vllm_prefill/scratch_vllm_prefill/audit.py",
           "bench/bench_serving_prefill.py", "bench/gpu_busy.ps1"]


class Rejected(Exception):
    """The case ran but cannot be counted."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def prompts_for(concurrency, warmup):
    return max(3, 2 * concurrency) if warmup else max(BASE_PROMPTS, 8 * concurrency)


def case_key(study, config, run, length, concurrency):
    return f"{study}_{config}_r{run}_n{length}_c{concurrency}"


# ------------------------------------------------------------ the GPU has to be ours

def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,pstate",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    load, temperature, power, pstate = (x.strip() for x in out.stdout.strip().split(","))
    return dict(load=float(load), temperature=float(temperature), power=float(power), pstate=pstate)


def other_gpu_user():
    """Busiest GPU process on the Windows host outside the WSL VM: (percent, name)."""
    script = subprocess.run(["wslpath", "-w", str(ROOT / "bench" / "gpu_busy.ps1")],
                            capture_output=True, text=True, check=True).stdout.strip()
    out = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script],
                         capture_output=True, text=True, check=True, timeout=120)
    percent, name = out.stdout.strip().split(maxsplit=1)
    return float(percent.replace(",", "")), name


def wait_for_idle_gpu(log):
    """Three idle samples in a row, a cool GPU, and nobody else on it. A loaded server that
    serves no request counts as idle, so this also runs between the cases of a session."""
    good = 0
    while True:
        state = gpu_state()
        idle = state["load"] <= 10 or (state["pstate"] == "P8" and state["load"] <= 30)
        good = good + 1 if idle and state["temperature"] <= 65 else 0
        if good >= 3:
            percent, name = other_gpu_user()
            if percent <= OTHER_GPU_USER_LIMIT:
                return dict(state, other_gpu_user=[percent, name])
            state["other_gpu_user"] = [percent, name]
            good = 0
        if not idle or "other_gpu_user" in state:
            log(dict(event="waiting_for_gpu", **state))
        time.sleep(2)


# ------------------------------------------------------------ talking to the server

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(port, token, path, body=None, timeout=40):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def server_counters(port, token):
    """Attention routes summed over layers, and the server's own token counters."""
    ranks = json.loads(http(port, token, "/collective_rpc",
                            dict(method="attention_routing", timeout=30)))["results"]
    layers = [layer for rank in ranks for layer in rank]
    counters = {"scratch_decode_calls": sum(x["scratch_decode_calls"] or 0 for x in layers),
                "flash_calls": sum(x["flash_calls"] or 0 for x in layers)}
    for kind in ("route_calls", "route_tokens"):
        for layer in layers:
            for route, n in (layer[kind] or {}).items():
                counters[f"{kind}.{route}"] = counters.get(f"{kind}.{route}", 0) + n
    for line in http(port, token, "/metrics").splitlines():
        for name in ("prompt_tokens_total", "generation_tokens_total", "request_success_total"):
            if line.startswith(f"vllm:{name}"):
                counters[name] = counters.get(name, 0) + float(line.rsplit(" ", 1)[1])
    implementations = sorted({x["implementation"] for x in layers})
    return counters, implementations, len(layers)


def check_routing(config, mode, before, used, implementations, layers):
    """Raise Rejected unless the measured window ran on the kernels the configuration names.

    Eager: every token passes through forward(), so the counters cover the whole window.

    CUDA graphs: a step of nothing but single-token requests is replayed from a captured graph
    and never reaches forward(), so decode does not appear in `used` for either scratch config.
    What can be shown there is that the decode kernel ran before the window (`before`, which
    covers startup, graph capture and the warm-up run) and that prefill, which is never
    captured, still adds up. A captured graph runs whatever forward() did when it was captured,
    and the implementation class is checked, so this is strong evidence rather than proof.
    """
    if implementations != [CONFIGS[config][2]]:
        raise Rejected(f"layers run {implementations}, wanted {CONFIGS[config][2]}")
    if config == "FLASH_ATTN":
        return
    # Both scratch configs decode on the paged decode kernel of integrations/vllm.
    captured = mode == "graphs"
    decode_calls = before["scratch_decode_calls"] if captured else used["scratch_decode_calls"]
    if decode_calls <= 0:
        raise Rejected("no decode call reached the scratch kernel"
                       + (" before the window, so nothing shows it was captured" if captured
                          else " inside the window"))
    if config == "SCRATCH_DECODE":
        # This config prefills on vLLM's FlashAttention, which is never captured.
        if used["flash_calls"] <= 0:
            raise Rejected("no prefill reached FlashAttention, which this config prefills on")
        return
    tokens = {k.split(".", 1)[1]: v for k, v in used.items() if k.startswith("route_tokens.")}
    if tokens["flash_fallback"]:
        raise Rejected(f"{tokens['flash_fallback']} tokens went to the FlashAttention fallback")
    prefill = (tokens["prefill_dense"] + tokens["prefill_paged"]) / layers
    if mode == "eager":
        # The first output token of a request comes out of its prefill step.
        processed = (used["prompt_tokens_total"] + used["generation_tokens_total"]
                     - used["request_success_total"])
        if sum(tokens.values()) / layers != processed:
            raise Rejected(f"routes hold {sum(tokens.values()) / layers} tokens per layer, "
                           f"the server processed {processed}")
    elif not 0.99 * used["prompt_tokens_total"] <= prefill <= used["prompt_tokens_total"]:
        raise Rejected(f"prefill routes hold {prefill} tokens per layer, "
                       f"the server counted {used['prompt_tokens_total']} prompt tokens")


# ------------------------------------------------------------ one server, all cases of a study

class Campaign:
    def __init__(self, args):
        self.args = args
        self.root = args.output_dir.resolve() / args.mode
        (self.root / "cases").mkdir(parents=True, exist_ok=True)
        (self.root / "sessions").mkdir(exist_ok=True)
        self.started = time.monotonic()

    def log(self, item):
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps(dict(item, at=utc())) + "\n")

    def case_path(self, *key):
        return self.root / "cases" / f"{case_key(*key)}.json"

    def provenance(self):
        import torch
        import transformers
        import vllm
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        record = dict(at=utc(), mode=self.args.mode, python=platform.python_version(),
                      torch=torch.__version__, vllm=vllm.__version__,
                      transformers=transformers.__version__, gpu_and_driver=smi,
                      command=sys.argv, sources={s: sha256(ROOT / s) for s in SOURCES})
        with (self.root / "provenance.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return record["sources"]

    def run_client(self, spec, tokenizer, port, token, run, length, concurrency, directory, warmup):
        name = "warmup" if warmup else "client"
        command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
                   "--backend", "openai", "--model", spec["model"], "--tokenizer", tokenizer,
                   "--host", "127.0.0.1", "--port", str(port), "--dataset-name", "random",
                   "--random-input-len", str(length), "--random-output-len", str(OUTPUT_LEN),
                   "--random-range-ratio", "0.0",
                   "--num-prompts", str(prompts_for(concurrency, warmup)),
                   "--max-concurrency", str(concurrency), "--request-rate", "inf", "--ignore-eos",
                   "--temperature", "0", "--seed", str(1000 + run if warmup else 42 + run),
                   "--save-result", "--save-detailed", "--result-dir", str(directory),
                   "--result-filename", f"{name}.json",
                   "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99"]
        with (directory / f"{name}.log").open("w") as log:
            subprocess.run(command, env=dict(os.environ, OPENAI_API_KEY=token), stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=7200)
        stats = json.loads((directory / f"{name}.json").read_text())
        wanted = prompts_for(concurrency, warmup)
        if stats["completed"] != wanted or set(stats["output_lens"]) != {OUTPUT_LEN}:
            raise Rejected(f"{name}: {stats['completed']} of {wanted} requests completed, "
                           f"output lengths {sorted(set(stats['output_lens']))}")
        return stats, command

    def measure(self, spec, tokenizer, port, token, study, config, run, length, concurrency, directory):
        directory.mkdir()
        gpu_before = wait_for_idle_gpu(self.log)["other_gpu_user"]
        self.run_client(spec, tokenizer, port, token, run, length, concurrency, directory, warmup=True)
        before, _, _ = server_counters(port, token)
        stats, command = self.run_client(spec, tokenizer, port, token, run, length, concurrency,
                                         directory, warmup=False)
        after, implementations, layers = server_counters(port, token)
        gpu_after = other_gpu_user()
        used = {k: after[k] - before.get(k, 0) for k in after}
        record = dict(
            study=study, config=config, mode=self.args.mode, run=run, input_len=length,
            concurrency=concurrency, model=spec["model"], measured_at=utc(),
            metrics={k: v for k, v in stats.items() if isinstance(v, (int, float))},
            used=used, before=before, implementations=implementations, layers=layers,
            other_gpu_user=dict(before=gpu_before, after=gpu_after),
            raw=str((directory / "client.json").relative_to(self.root)),
            raw_sha256=sha256(directory / "client.json"), client_command=command)
        (directory / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        if max(gpu_before[0], gpu_after[0]) > OTHER_GPU_USER_LIMIT:
            raise Rejected(f"another program used the GPU: before {gpu_before}, after {gpu_after}")
        check_routing(config, self.args.mode, before, used, implementations, layers)
        return record

    def session(self, study, config, run, sources):
        model, cases = STUDIES[study]
        spec = MODELS[model]
        cases = [c for c in cases if not self.case_path(study, config, run, *c).exists()]
        if not cases:
            return
        from huggingface_hub import snapshot_download
        tokenizer = snapshot_download(spec["model"], revision=spec["revision"], local_files_only=True)
        attempt = sum(1 for _ in (self.root / "sessions").glob(f"{study}_{config}_r{run}_a*"))
        directory = self.root / "sessions" / f"{study}_{config}_r{run}_a{attempt}"
        directory.mkdir()
        idle = wait_for_idle_gpu(self.log)
        token, port = secrets.token_hex(24), free_port()
        plugins, backend, _ = CONFIGS[config]
        server = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", spec["model"],
                  "--revision", spec["revision"], "--tokenizer-revision", spec["revision"],
                  "--host", "127.0.0.1", "--port", str(port), "--api-key", token,
                  "--served-model-name", spec["model"], "--dtype", "half",
                  "--max-model-len", str(spec["capacity"]), "--max-num-seqs", "32",
                  "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.55",
                  "--no-enable-prefix-caching", "--seed", "42", "--attention-backend", backend,
                  "--worker-extension-cls", "scratch_vllm_prefill.audit.RoutingWorker"]
        server += ["--enforce-eager"] if self.args.mode == "eager" else []
        env = dict(os.environ, VLLM_SERVER_DEV_MODE="1", VLLM_PLUGINS=plugins, HF_HUB_OFFLINE="1",
                   VLLM_NO_USAGE_STATS="1")
        self.log(dict(event="session", directory=directory.name, gpu=idle,
                      server=["<key>" if x == token else x for x in server], VLLM_PLUGINS=plugins))
        with (directory / "server.log").open("w") as log:
            proc = subprocess.Popen(server, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                self.wait_until_healthy(proc, port, token, directory)
                for length, concurrency in cases:
                    key = (study, config, run, length, concurrency)
                    try:
                        record = self.measure(spec, tokenizer, port, token, *key,
                                              directory / f"n{length}_c{concurrency}")
                    except (Rejected, subprocess.SubprocessError, urllib.error.URLError, OSError) as e:
                        self.log(dict(event="rejected", case=case_key(*key), why=repr(e),
                                      directory=directory.name))
                        with (self.root / "failures.jsonl").open("a") as f:
                            f.write(json.dumps(dict(case=case_key(*key), why=repr(e), at=utc(),
                                                    directory=directory.name)) + "\n")
                        if proc.poll() is not None:
                            break
                        continue
                    if sources != {s: sha256(ROOT / s) for s in SOURCES}:
                        raise RuntimeError("a source file changed during the campaign")
                    record["sources_sha256"] = hashlib.sha256(
                        json.dumps(sources, sort_keys=True).encode()).hexdigest()
                    self.case_path(*key).write_text(json.dumps(record, indent=2) + "\n")
                    self.log(dict(event="accepted", case=case_key(*key)))
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGINT)
                    try:
                        proc.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
        loaded = [line.strip() for line in (directory / "server.log").read_text(errors="replace").splitlines()
                  if "source_sha256=" in line]
        (directory / "loaded_kernels.json").write_text(json.dumps(
            [dict(line=line, so_sha256=sha256(line.split("so=", 1)[1]))
             for line in sorted(set(loaded))], indent=2) + "\n")

    def wait_until_healthy(self, proc, port, token, directory):
        start = time.monotonic()
        while time.monotonic() - start < 1500:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}, see {directory / 'server.log'}")
            try:
                http(port, token, "/health", timeout=2)
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(2)
        raise TimeoutError("the server did not come up")

    def run(self):
        sources = self.provenance()
        studies = self.args.studies
        for sweep in range(1 + self.args.retries):   # later sweeps pick up rejected cases
            for run in range(self.args.runs):
                for index, study in enumerate(studies):
                    k = (run + index) % len(self.args.configs)
                    for config in self.args.configs[k:] + self.args.configs[:k]:
                        if time.monotonic() - self.started > self.args.max_hours * 3600:
                            self.log(dict(event="time_limit"))
                            return
                        self.session(study, config, run, sources)


# ------------------------------------------------------------ report

def report(root, configs):
    cases = [json.loads(p.read_text()) for p in sorted((root / "cases").glob("*.json"))]
    base = configs[0]
    shown = [("median_ttft_ms", "TTFT p50 ms", 1), ("p99_ttft_ms", "TTFT p99 ms", 1),
             ("median_tpot_ms", "TPOT p50 ms", 2), ("median_e2el_ms", "E2E p50 ms", 0),
             ("output_throughput", "out tok/s", 1)]
    print(f"Values: median over runs. Ratios: median of the per-run ratios against {base}, the "
          "per-run ratios in brackets. Rows with fewer runs than the others are incomplete.")
    print(f"{len(cases)} accepted cases, measured with "
          f"{len({c['sources_sha256'] for c in cases})} distinct set(s) of source files.\n")
    for study, (_, grid) in STUDIES.items():
        rows = [c for c in cases if c["study"] == study]
        if not rows:
            continue
        print(f"### {study} ({rows[0]['model']}, {rows[0]['mode']})\n")
        print("| input | C | config | runs | " + " | ".join(label for _, label, _ in shown)
              + " | TTFT ratio | TPOT ratio | tok/s ratio |")
        print("|---:|---:|---|---:|" + "---:|" * len(shown) + "---|---|---|")
        for length, concurrency in grid:
            at = {(c["config"], c["run"]): c["metrics"] for c in rows
                  if (c["input_len"], c["concurrency"]) == (length, concurrency)}
            for config in configs:
                runs = sorted(r for cfg, r in at if cfg == config)
                if not runs:
                    continue
                cells = [f"{statistics.median(at[config, r][key] for r in runs):.{digits}f}"
                         for key, _, digits in shown]
                ratios = []
                for key in ("median_ttft_ms", "median_tpot_ms", "output_throughput"):
                    per_run = [at[config, r][key] / at[base, r][key] for r in runs if (base, r) in at]
                    ratios.append(f"{statistics.median(per_run):.3f} [{' / '.join(f'{x:.3f}' for x in per_run)}]"
                                  if per_run else "")
                print(f"| {length} | {concurrency} | {config} | {len(runs)} | " + " | ".join(cells + ratios) + " |")
        print()
    print("### routes inside the measured windows, tokens per layer summed over all accepted cases\n")
    for config in configs:
        rows = [c for c in cases if c["config"] == config]
        totals = {}
        for c in rows:
            for k, v in c["used"].items():
                totals[k] = totals.get(k, 0) + (v / c["layers"] if k.startswith("route_") else v)
        print(f"- {config} ({len(rows)} cases): " + ", ".join(f"{k}={v:.0f}" for k, v in sorted(totals.items())))
    failures = root / "failures.jsonl"
    if failures.exists():
        print(f"\n### rejected attempts ({failures.name})\n")
        for line in failures.read_text().splitlines():
            item = json.loads(line)
            print(f"- {item['at']} {item['case']}: {item['why']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["eager", "graphs"], required=True)
    ap.add_argument("--studies", nargs="+", choices=list(STUDIES), default=list(STUDIES))
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS),
                    help="the first one is the reference of the ratios")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--retries", type=int, default=2, help="extra sweeps over rejected cases")
    ap.add_argument("--max-hours", type=float, default=12)
    ap.add_argument("--report", action="store_true", help="print the tables of what is on disk")
    args = ap.parse_args()
    if args.report:
        return report(args.output_dir.resolve() / args.mode, args.configs)
    Campaign(args).run()
    report(args.output_dir.resolve() / args.mode, args.configs)


if __name__ == "__main__":
    main()
