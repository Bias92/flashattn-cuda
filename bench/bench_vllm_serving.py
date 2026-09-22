"""Run the official serving benchmark against two otherwise identical engines.

Owns and shuts down only the localhost servers it creates. Results are exploratory
serving measurements, not CUDA kernel timings or evidence of an SDPA win.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from bench_attention_decode import exclusive_gpu_check
from serving_gpu_guard import monitored_wait, other_python_jobs, telemetry, wait_idle


MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CLI = [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop_server(proc):
    # The child starts a new session, so this cannot signal another agent's job.
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    else:
        # Startup failures can leave a worker in our dedicated process group.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def ready(proc, port, log, timeout=600):
    start, reported = time.monotonic(), 0
    while time.monotonic() - start < timeout:
        if jobs := other_python_jobs():
            raise RuntimeError(f"Other Python jobs appeared during server startup: {jobs}")
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited {proc.returncode}: {log.read_text()[-6000:]}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        elapsed = int(time.monotonic() - start)
        if elapsed // 30 > reported:
            reported = elapsed // 30
            print(f"Waiting for server ({elapsed}s): {log}", flush=True)
        time.sleep(1)
    raise TimeoutError(f"Server did not become ready: {log}")


def warmup(port, count, input_len, output_len):
    payload = json.dumps(dict(model=MODEL, prompt=[1] + [500] * (input_len - 1),
                              max_tokens=output_len, temperature=0, ignore_eos=True)).encode()
    for _ in range(count):
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                         data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        if result["usage"]["completion_tokens"] != output_len:
            raise RuntimeError("Warmup did not produce the requested number of tokens")


def resume_report(output, requested):
    saved = json.loads((output / "manifest.json").read_text())
    if saved["complete"]:
        raise ValueError("This comparison is already complete")
    for key, value in requested["settings"].items():
        if key != "resume" and saved["settings"].get(key) != value:
            raise ValueError(f"Resume would change a benchmark setting: {key}")
    if saved["versions"] != requested["versions"]:
        raise ValueError("Resume would change package versions")
    runner = "bench/bench_vllm_serving.py"
    for path, digest in requested["implementation_sha256"].items():
        if path != runner and saved["implementation_sha256"].get(path) != digest:
            raise ValueError(f"Resume would change the attention implementation: {path}")
    old_hash = saved["implementation_sha256"][runner]
    new_hash = requested["implementation_sha256"][runner]
    revisions = saved.setdefault("runner_revisions", {})
    if old_hash != new_hash:
        archive = output / "runner_before_resume.py"
        if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != old_hash:
            raise ValueError("The original benchmark runner must be archived before changing it")
        revisions[old_hash] = archive.name
    revisions[new_hash] = "current_workspace_runner"
    seen, raw_hashes = set(), {}
    for record in saved["results"]:
        key = (record["run"], record["backend"])
        if key in seen:
            raise ValueError(f"Duplicate completed result: {key}")
        seen.add(key)
        path = output / f"run{key[0]}_{key[1].lower()}.json"
        stats = json.loads(path.read_text())
        count = requested["settings"]["num_prompts"]
        if stats["completed"] != count or stats["failed"] or any(stats["errors"]):
            raise ValueError(f"Invalid completed record: {path}")
        raw_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        record.setdefault("runner_sha256", old_hash)
    saved["resume_events"] = saved.get("resume_events", []) + [dict(
        utc=datetime.now(timezone.utc).isoformat(), previous_error=saved.pop("error", None),
        completed_rows=len(seen), preserved_raw_sha256=raw_hashes,
        runner_sha256=new_hash,
    )]
    saved["implementation_sha256"][runner] = new_hash
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--idle-timeout", type=int, default=180)
    parser.add_argument("--idle-quiet-seconds", type=int, default=0)
    parser.add_argument("--backends", nargs="+", choices=("FLASH_ATTN", "CUSTOM", "SDPA_AUTO",
                        "SDPA_FLASH", "SDPA_CUDNN", "SDPA_EFFICIENT", "SDPA_MATH"),
                        default=["FLASH_ATTN", "CUSTOM"])
    args = parser.parse_args()
    if not (0 < args.input_len and 1 < args.output_len and
            args.input_len + args.output_len <= 2048 and 1 <= args.concurrency <= 8):
        parser.error("Require positive input, output>1, total<=2048 and concurrency in 1..8")
    if args.runs < 1 or args.num_prompts < 1:
        parser.error("runs and num-prompts must be positive")
    if any(b.startswith("SDPA_") for b in args.backends) and not args.eager:
        parser.error("SDPA comparison requires --eager for every backend in this run")
    if not 0 <= args.idle_quiet_seconds < args.idle_timeout:
        parser.error("Require 0 <= idle-quiet-seconds < idle-timeout")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
               VLLM_PLUGINS="scratch_attention", SCRATCH_DECODE_AUDIT="1")
    source = Path(__file__).resolve().parents[1] / "cuda" / "attention_decode_paged.cu"
    report = dict(complete=False, settings=vars(args) | {"output_dir": str(output)},
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  versions={p: importlib.metadata.version(p) for p in
                            ("vllm", "torch", "transformers", "scratch-vllm-attention")},
                  prefill_backend="FLASH_ATTN in every mode", results=[])
    root = Path(__file__).resolve().parents[1]
    report["implementation_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source, Path(__file__).resolve(),
                     root / "integrations/vllm/scratch_vllm/comparison.py",
                     root / "integrations/vllm/scratch_vllm/backend.py",
                     root / "integrations/vllm/scratch_vllm/__init__.py")
    }
    if args.resume:
        report = resume_report(output, report)
    completed = {(r["run"], r["backend"]) for r in report["results"]}
    manifest = output / "manifest.json"
    def save():
        manifest.write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        for run in range(args.runs):
            order = args.backends if run % 2 == 0 else list(reversed(args.backends))
            for backend in order:
                if (run, backend) in completed:
                    print(f"Keeping completed run{run}_{backend.lower()}", flush=True)
                    continue
                env.pop("SCRATCH_COMPARISON_BACKEND", None)
                dense = backend.startswith("SDPA_")
                if dense:
                    env["SCRATCH_COMPARISON_BACKEND"] = backend
                idle_samples = wait_idle(timeout=args.idle_timeout,
                                         quiet_seconds=args.idle_quiet_seconds)
                port = free_port()
                name = f"run{run}_{backend.lower()}"
                server_log = output / f"{name}_server.log"
                if server_log.exists():
                    suffix = len(report.get("resume_events", []))
                    server_log.rename(output / f"{name}_server.interrupted{suffix}.log")
                server = CLI + ["serve", MODEL, "--host", "127.0.0.1", "--port", str(port),
                    "--served-model-name", MODEL, "--dtype", "half", "--max-model-len", "2048",
                    "--max-num-seqs", "8", "--max-num-batched-tokens", "2048",
                    "--gpu-memory-utilization", "0.55", "--no-enable-prefix-caching",
                    "--attention-backend", "CUSTOM" if dense else backend, "--seed", "42",
                    "--compilation-config", json.dumps({"cudagraph_capture_sizes": [1, 2, 4, 8]})]
                if args.eager:
                    server.append("--enforce-eager")
                print(f"Starting {name} on localhost:{port}", flush=True)
                record = dict(run=run, backend=backend, server_command=server,
                              runner_sha256=report["implementation_sha256"]["bench/bench_vllm_serving.py"],
                              idle_samples=idle_samples, paged_to_dense_included=dense,
                              comparison_backend_env=env.get("SCRATCH_COMPARISON_BACKEND"))
                with server_log.open("w") as log:
                    proc = subprocess.Popen(server, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                    try:
                        ready(proc, port, server_log)
                        warmup(port, 3, args.input_len, args.output_len)
                        record["after_warmup"] = telemetry()
                        client = CLI + ["bench", "serve", "--backend", "openai", "--model", MODEL,
                            "--tokenizer", MODEL, "--host", "127.0.0.1", "--port", str(port),
                            "--dataset-name", "random", "--random-input-len", str(args.input_len),
                            "--random-output-len", str(args.output_len), "--random-range-ratio", "0.0",
                            "--num-prompts", str(args.num_prompts), "--max-concurrency", str(args.concurrency),
                            "--request-rate", "inf", "--ignore-eos", "--temperature", "0",
                            "--seed", str(42 + run),
                            "--save-result", "--save-detailed", "--result-dir", str(output),
                            "--result-filename", f"{name}.json", "--percentile-metrics", "ttft,tpot,itl,e2el"]
                        record["client_command"] = client
                        with (output / f"{name}_client.log").open("w") as client_log:
                            with subprocess.Popen(client, env=env, stdout=client_log,
                                                  stderr=subprocess.STDOUT) as client_proc:
                                record["telemetry"] = monitored_wait(client_proc)
                        stats = json.loads((output / f"{name}.json").read_text())
                        record["metrics"] = {k: v for k, v in stats.items()
                                             if k.startswith(("mean_", "median_", "p99_")) or k == "completed"}
                        print(json.dumps(record["metrics"]), flush=True)
                    finally:
                        stop_server(proc)
                text = server_log.read_text()
                if backend == "CUSTOM" and "SCRATCH_PAGED_DECODE active" not in text:
                    raise RuntimeError("No proof of custom decode dispatch in server log")
                if dense and f"DENSE_SDPA_DECODE active: mode={backend}" not in text:
                    raise RuntimeError("No proof of selected SDPA adapter dispatch in server log")
                record["operator_audit"] = [line for line in text.splitlines()
                                            if "DENSE_SDPA_OPERATOR_AUDIT" in line]
                if dense and not record["operator_audit"]:
                    raise RuntimeError("Missing actual SDPA operator audit")
                time.sleep(2)
                exclusive_gpu_check()
                report["results"].append(record)
                save()
        report["complete"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
    print(f"Completed: {manifest}", flush=True)


if __name__ == "__main__":
    main()
