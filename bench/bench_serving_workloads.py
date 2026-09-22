"""Resumable, monitored local workload matrices. No changes to attention math."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
from statistics import mean, median
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "integrations/vllm")]
from bench_vllm_serving import free_port, stop_server
from serving_gpu_guard import other_python_jobs, telemetry
from validate_sdpa_serving_results import response_differences

MODES = ["FLASH_ATTN", "CUSTOM", "SDPA_AUTO", "SDPA_FLASH", "SDPA_CUDNN",
         "SDPA_EFFICIENT", "SDPA_MATH"]
MODELS = {
    "throughput": dict(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6", capacity=2048,
        layers=22, hq=32, hkv=4, d=64),
    "long_context": dict(model="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775", capacity=32768,
        layers=24, hq=14, hkv=2, d=64),
}
CORE_FILES = ["cuda/attention_decode_paged.cu", "integrations/vllm/scratch_vllm/backend.py",
              "integrations/vllm/scratch_vllm/comparison.py",
              "integrations/vllm/scratch_vllm/__init__.py"]
RUN_FILES = CORE_FILES + ["bench/bench_serving_workloads.py", "bench/bench_vllm_serving.py",
    "bench/run_overnight_workloads.ps1",
    "bench/serving_gpu_guard.py", "tests/validate_workload_shapes.py",
    "tests/validate_sdpa_serving_results.py", "tests/test_decode_paged.py",
    "integrations/vllm/scratch_vllm/workload_audit.py",
    "integrations/vllm/scratch_vllm/loader.py", "integrations/vllm/scratch_vllm/audit.py"]


class StopRequested(Exception):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_stamp(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns]


def capture_runtime(paths):
    files = {}
    for path in paths:
        before = runtime_stamp(path)
        checksum = digest(path)
        after = runtime_stamp(path)
        if before != after:
            raise StopRequested(f"Runtime changed while fingerprinting: {path}")
        files[str(path)] = dict(sha256=checksum, stamp=after)
    return dict(python=sys.version, kernel=os.uname().release, files=files)


def check_runtime(snapshot):
    for path, record in snapshot["files"].items():
        try:
            unchanged = runtime_stamp(path) == record["stamp"]
        except OSError:
            unchanged = False
        if not unchanged:
            raise StopRequested(f"Runtime environment changed: {path}; preserve this campaign and diagnose")


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def percentile(values, percent):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of empty data")
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def request_count(base, concurrency):
    return max(base, 8 * concurrency)


def classify_jobs(jobs, owned_groups, now):
    foreign, owned = [], []
    for job in jobs:
        try:
            group = os.getpgid(job["pid"])
            session = os.getsid(job["pid"])
        except ProcessLookupError:
            continue
        record = dict(job, pgid=group, session=session)
        if group == session and owned_groups.get(group, -1) > now:
            owned.append(record)
        else:
            foreign.append(record)
    return foreign, owned


def validated_metrics(stats, count, input_len, output_len, *, special_tokens=1):
    if (stats["completed"] != count or stats["input_lens"] != [input_len - special_tokens] * count
            or stats["output_lens"] != [output_len] * count or output_len <= 1
            or stats["num_prompts"] != count or stats["failed"] != 0 or any(stats["errors"])
            or len(stats["ttfts"]) != count or len(stats["itls"]) != count):
        raise ValueError("Workload token/count mismatch")
    duration = stats["duration"]
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid client duration")
    if stats["total_output_tokens"] != sum(stats["output_lens"]):
        raise ValueError("Output total mismatch")
    if stats["total_input_tokens"] != sum(stats["input_lens"]):
        raise ValueError("Input total mismatch")
    result = {}
    for name, value in (("output_throughput", stats["total_output_tokens"] / duration),
                        ("request_throughput", count / duration)):
        if not math.isclose(stats[name], value, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Throughput mismatch: {name}")
        result[name] = stats[name]
    tpot = [1000 * sum(gaps) / (n - 1) for gaps, n in
            zip(stats["itls"], stats["output_lens"], strict=True)]
    distributions = dict(ttft=[1000 * x for x in stats["ttfts"]], tpot=tpot,
        itl=[1000 * x for gaps in stats["itls"] for x in gaps],
        e2el=[1000 * (first + sum(gaps)) for first, gaps in
              zip(stats["ttfts"], stats["itls"], strict=True)])
    if any(not values or any(not math.isfinite(v) or v < 0 for v in values)
           for values in distributions.values()):
        raise ValueError("Invalid streaming timing samples")

    # vLLM openai completions timestamps the first token twice. The gap is
    # in TTFT but not ITLs, so mean E2EL reveals the sum of those nonnegative
    # gaps. Every quantile shift is bounded by that sum (no fixed slack).
    epsilon_ms = 1e-6
    official_e2el = stats["mean_e2el_ms"]
    if not math.isfinite(official_e2el) or official_e2el < 0:
        raise ValueError("Invalid official mean E2EL")
    offset_sum = count * (mean(distributions["e2el"]) - official_e2el)
    if offset_sum < -count * epsilon_ms or offset_sum > sum(distributions["ttft"]) + count * epsilon_ms:
        raise ValueError("Impossible first-token timestamp offset")
    offset_bound = max(0.0, offset_sum) + count * epsilon_ms
    bounds = dict(ttft=0.0, itl=0.0, tpot=offset_bound / (output_len - 1), e2el=offset_bound)

    def check_metric(key, reconstructed, bound=0.0):
        official = stats[key]
        if (not math.isfinite(official) or official < 0
                or official > reconstructed + epsilon_ms
                or official < reconstructed - bound - epsilon_ms):
            raise ValueError(f"Timing identity/bound mismatch: {key}")

    residuals = []
    for name in ("ttft", "tpot"):
        for label, aggregate in (("mean", mean), ("median", median)):
            key = f"{label}_{name}_ms"
            value = aggregate(distributions[name])
            check_metric(key, value, bounds[name])
            result[key] = stats[key]
            residuals.append(abs(value - stats[key]))
    expected_mean_tpot = (official_e2el - stats["mean_ttft_ms"]) / (output_len - 1)
    if abs(stats["mean_tpot_ms"] - expected_mean_tpot) > epsilon_ms:
        raise ValueError("Official mean TPOT/E2EL identity mismatch")
    for name, values in distributions.items():
        for p in (50, 95, 99):
            value = percentile(values, p)
            key = f"p{p}_{name}_ms"
            if key in stats:
                check_metric(key, value, bounds[name])
            result[key] = value
    return result | dict(stream_recompute_max_error_ms=max(residuals),
        first_token_timestamp_offset_sum_ms=offset_sum,
        timing_definition="Official means/medians; stream-derived percentiles; first-token offset bounded from mean E2EL",
        duration_s=duration, completed=count, failed=stats["failed"],
        input_tokens=stats["total_input_tokens"], output_tokens=stats["total_output_tokens"])


def model_comparison(actual, baseline):
    checks = []
    for index, (observed, expected) in enumerate(zip(actual["outputs"], baseline["outputs"], strict=True)):
        if observed["prompt"] != expected["prompt"]:
            raise ValueError("Model smoke prompt mismatch")
        a, b = observed["ids"], expected["ids"]
        different = next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)
        item = dict(prompt_index=index, identical=a == b, first_divergent_step=different)
        if different is not None:
            # Only the first mismatch still shares the same preceding token prefix.
            reference_scores = expected["logprobs"][different]
            alternate = reference_scores.get(str(a[different]))
            winner = reference_scores[str(b[different])]
            item.update(reference_token=b[different], actual_token=a[different],
                        reference_logprob_margin=None if alternate is None else winner - alternate)
            if alternate is None or winner - alternate > 0.05:
                raise ValueError(f"Model smoke divergence needs diagnosis: {item}")
        checks.append(item)
    return dict(prompts=checks, all_identical=all(x["identical"] for x in checks),
        note="First-divergence reference top-5 margin <=0.05 is a smoke gate, not a global accuracy guarantee.")


def routes(before, after):
    records = []
    for left, right in zip(before, after, strict=True):
        by_name = {x["name"]: x for x in left["routing"]}
        for layer in right["routing"]:
            old = by_name[layer["name"]]
            record = dict(name=layer["name"], implementation=layer["implementation"],
                scratch_calls=None if layer["scratch_calls"] is None else layer["scratch_calls"] - old["scratch_calls"],
                fallback_calls=None if layer["prefill_calls"] is None else layer["prefill_calls"] - old["prefill_calls"])
            if any(record[k] is not None and record[k] < 0 for k in ("scratch_calls", "fallback_calls")):
                raise ValueError("Routing counters went backwards")
            records.append(record)
    return records


def prometheus_values(body):
    from prometheus_client.parser import text_string_to_metric_families
    prefixes = ("vllm:kv_cache_usage", "vllm:cache_config_info", "vllm:num_requests_running",
                "vllm:num_requests_waiting", "vllm:num_preemptions")
    return [dict(name=s.name, labels=s.labels, value=s.value)
            for family in text_string_to_metric_families(body) for s in family.samples
            if s.name.startswith(prefixes)]


class Experiment:
    def __init__(self, args):
        self.args, self.root = args, args.output_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / "runner.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.started = time.monotonic()
        self.owned_groups = {}
        self.model_paths, self.special_tokens = {}, {}
        self.deadline = self.started + args.max_hours * 3600
        self.env = dict(os.environ, CUDA_HOME="/usr/local/cuda-12.8", HF_HUB_OFFLINE="1",
            VLLM_NO_USAGE_STATS="1", VLLM_PLUGINS="scratch_attention", SCRATCH_DECODE_AUDIT="1",
            WORKLOAD_OWNER_PID=str(os.getpid()), MAX_JOBS="1")
        self.env.pop("VLLM_ALLOW_INSECURE_SERIALIZATION", None)
        self.versions = {p: importlib.metadata.version(p) for p in
                         ("vllm", "torch", "transformers", "scratch-vllm-attention")}
        self.runtime = capture_runtime([sys.executable, "/var/lib/dpkg/status",
            "/lib/x86_64-linux-gnu/libc.so.6", "/lib/x86_64-linux-gnu/libm.so.6",
            "/lib/x86_64-linux-gnu/libssl.so.3", "/lib/x86_64-linux-gnu/libcrypto.so.3"])
        self.hashes = {name: digest(ROOT / name) for name in RUN_FILES}
        self.revision = hashlib.sha256(json.dumps(self.hashes, sort_keys=True).encode()).hexdigest()
        archive = self.root / "source_snapshots" / self.revision
        for name in RUN_FILES:
            target = archive / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != self.hashes[name]:
                raise ValueError(f"Archived source changed: {target}")
            if not target.exists():
                shutil.copy2(ROOT / name, target)
        settings = dict(runs=args.runs, prompts=args.prompts, output_len=128,
                        models=MODELS, eager=True, max_num_seqs=32, token_budget=2048,
                        gpu_memory_fraction=0.55, prefix_caching=False, initial_concurrency=[1, 2, 4, 8],
                        initial_long_inputs=[2048, 4096, 8192, 16384])
        state_path = self.root / "state.json"
        if state_path.exists():
            self.state = json.loads(state_path.read_text())
            if self.state["settings"] != settings or self.state["versions"] != self.versions:
                raise ValueError("Resume would change protocol or package versions")
            if self.state.get("runtime_environment") != self.runtime:
                raise ValueError("Resume would mix runtime environments; retain old evidence and use a new campaign")
            for name in CORE_FILES:
                if self.state["core_hashes"][name] != self.hashes[name]:
                    raise ValueError(f"Resume would change attention math: {name}")
            for record in self.state["results"].values():
                if digest(self.root / record["raw"]) != record["raw_sha256"]:
                    raise ValueError(f"Accepted raw result changed: {record['raw']}")
            if self.revision not in self.state["revisions"] and self.state["failed"]:
                self.state.setdefault("previous_failures", []).append(dict(
                    time=utc(), reason="New archived orchestration revision; retry previously failed cells",
                    previous=self.state["failed"]))
                self.state["failed"] = {}
        else:
            self.state = dict(created=utc(), settings=settings, versions=self.versions,
                runtime_environment=self.runtime,
                core_hashes={name: self.hashes[name] for name in CORE_FILES}, revisions={},
                validation={}, results={}, failed={}, events=[], sessions=[])
        self.state["revisions"][self.revision] = dict(files=self.hashes, created=utc())
        self.update("preparing", "Validating environment and preserving evidence")

    def update(self, status, detail):
        self.state.update(status=status, detail=detail, updated=utc(), pid=os.getpid())
        save(self.root / "state.json", self.state)
        print(f"[{utc()}] {status}: {detail}", flush=True)

    def event(self, kind, **fields):
        item = dict(time=utc(), kind=kind, **fields)
        self.state["events"].append(item)
        with (self.root / "events.jsonl").open("a") as output:
            output.write(json.dumps(item) + "\n")

    def check_stop(self):
        if (self.root / "STOP").exists():
            raise StopRequested("STOP file requested shutdown; no automatic restart until cleared")
        if time.monotonic() > self.deadline:
            raise StopRequested("Invocation time budget reached; state saved for heartbeat continuation")

    def check_sources(self):
        check_runtime(self.runtime)
        for name, checksum in self.hashes.items():
            if digest(ROOT / name) != checksum:
                raise StopRequested(f"Source changed during this invocation: {name}")

    def sample(self):
        self.check_stop()
        check_runtime(self.runtime)
        item = telemetry()
        foreign, owned = classify_jobs(other_python_jobs(), self.owned_groups, time.monotonic())
        item.update(utc=utc(), other_jobs=foreign, owned_teardown_jobs=owned)
        return item

    def spawn(self, command, env, log):
        check_runtime(self.runtime)
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        self.owned_groups[proc.pid] = float("inf")
        self.event("owned_process_started", pid=proc.pid, pgid=proc.pid)
        return proc

    def stop_owned(self, proc):
        try:
            stop_server(proc)
        finally:
            # A resource tracker can briefly outlive its parent at shutdown.
            # Ownership is proved by our session/PGID, never by a process name.
            self.owned_groups[proc.pid] = time.monotonic() + 10
            self.event("owned_process_stopped", pid=proc.pid, returncode=proc.poll())

    def idle(self):
        self.update("waiting_gpu", "Waiting for 30 quiet seconds and a cool, low-load GPU")
        quiet, good, iterations = None, 0, 0
        log = self.root / "idle_samples.jsonl"
        while True:
            item = self.sample()
            now = time.monotonic()
            quiet = None if item["other_jobs"] else (now if quiet is None else quiet)
            low_power = (item["pstate"] == "P8" and float(item["clocks.sm"]) <= 300
                and float(item["clocks.mem"]) <= 1000 and float(item["power.draw"]) <= 20
                and float(item["utilization.gpu"]) <= 30)
            valid = not item["other_jobs"] and float(item["temperature.gpu"]) <= 65 and (
                float(item["utilization.gpu"]) <= 10 or low_power)
            good = good + 1 if valid else 0
            item.update(low_power_desktop_gate=low_power,
                        quiet_s=0 if quiet is None else now - quiet)
            with log.open("a") as output:
                output.write(json.dumps(item) + "\n")
            if good >= 3 and item["quiet_s"] >= 30:
                return item
            iterations += 1
            if iterations % 30 == 0:
                self.update("waiting_gpu", f"temperature={item['temperature.gpu']} C, load={item['utilization.gpu']}%, other_python={len(item['other_jobs'])}")
            time.sleep(1)

    def check_active(self):
        item = self.sample()
        if item["other_jobs"]:
            raise RuntimeError(f"contamination: unrelated Python jobs: {item['other_jobs']}")
        if float(item["temperature.gpu"]) > 80:
            raise RuntimeError("thermal: GPU temperature exceeded 80 C")
        return item

    def request(self, port, token, path, data=None, timeout=15, text=False):
        payload = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode() if text else json.load(response)

    def rpc(self, port, token, method):
        return self.request(port, token, "/collective_rpc", dict(method=method, timeout=30), timeout=40)["results"]

    def monitor(self, proc, path, timeout=1800, port=None, token=None):
        start, count = time.monotonic(), 0
        with path.open("w") as log:
            while proc.poll() is None:
                item = self.check_active()
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"Child timed out after {timeout}s")
                if port is not None:
                    try:
                        body = self.request(port, token, "/metrics", text=True, timeout=2)
                        item["server_metrics"] = prometheus_values(body)
                    except (urllib.error.URLError, TimeoutError) as error:
                        item["metrics_error"] = str(error)
                log.write(json.dumps(item) + "\n")
                log.flush()
                count += 1
                time.sleep(1)
        if proc.returncode:
            raise RuntimeError(f"Child exited {proc.returncode}; see {path.parent}")
        if not count:
            raise RuntimeError("No monitoring samples were collected")

    def validation(self, phase, study=None, backend=None):
        key = "kernels" if phase == "kernels" else f"model_{study}_{backend}"
        if key in self.state["validation"]:
            entry = self.state["validation"][key]
            if digest(self.root / entry["path"]) != entry["sha256"]:
                raise ValueError(f"Validation evidence changed: {key}")
            return
        self.idle()
        self.check_sources()
        directory = self.new_attempt("validation_" + key)
        result = directory / "result.json"
        command = [sys.executable, str(ROOT / "tests/validate_workload_shapes.py"), phase,
                   "--output", str(result)]
        if study:
            spec = MODELS[study]
            command += ["--model", spec["model"], "--revision", spec["revision"],
                        "--backend", backend, "--capacity", str(spec["capacity"])]
        self.update("validating", key)
        with (directory / "process.log").open("w") as log:
            proc = self.spawn(command, self.env, log)
            try:
                self.monitor(proc, directory / "telemetry.jsonl", timeout=1800)
            finally:
                self.stop_owned(proc)
        self.check_sources()
        data = json.loads(result.read_text())
        comparison = None
        if study and backend != "FLASH_ATTN":
            reference = self.state["validation"][f"model_{study}_FLASH_ATTN"]
            comparison = model_comparison(data, json.loads((self.root / reference["path"]).read_text()))
        self.state["validation"][key] = dict(path=str(result.relative_to(self.root)),
            sha256=digest(result), command=command, revision=self.revision, comparison=comparison)
        self.update("validated", key)

    def new_attempt(self, name):
        root = self.root / "attempts" / name
        root.mkdir(parents=True, exist_ok=True)
        for index in range(10000):
            path = root / f"attempt_{index:04d}"
            try:
                path.mkdir()
                return path
            except FileExistsError:
                continue
        raise RuntimeError("Too many attempts")

    def case_key(self, study, backend, run, length, concurrency):
        return f"{study}_{backend}_r{run}_n{length}_c{concurrency}"

    def client_command(self, study, spec, run, length, concurrency, directory, warmup):
        return [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "openai", "--model", spec["model"], "--tokenizer", self.model_paths[study],
            "--host", "127.0.0.1", "--dataset-name", "random", "--random-input-len", str(length),
            "--random-output-len", "128", "--random-range-ratio", "0.0",
            "--num-prompts", str(max(3, 2 * concurrency) if warmup else request_count(self.args.prompts, concurrency)),
            "--max-concurrency", str(concurrency), "--request-rate", "inf",
            "--ignore-eos", "--temperature", "0", "--seed", str(1000 + run if warmup else 42 + run),
            "--save-result", "--save-detailed", "--result-dir", str(directory),
            "--result-filename", "warmup.json" if warmup else "client.json",
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99"]

    def session(self, study, backend, run, cases):
        remaining = [(n, c) for n, c in cases if self.case_key(study, backend, run, n, c)
                     not in self.state["results"] and self.case_key(study, backend, run, n, c)
                     not in self.state["failed"]]
        if not remaining:
            return
        spec = MODELS[study]
        self.idle()
        self.check_sources()
        directory = self.new_attempt(f"{study}_{backend}_run{run}")
        token, port = secrets.token_hex(24), free_port()
        env = dict(self.env, VLLM_SERVER_DEV_MODE="1", OPENAI_API_KEY=token)
        env.pop("SCRATCH_COMPARISON_BACKEND", None)
        if backend.startswith("SDPA_"):
            env["SCRATCH_COMPARISON_BACKEND"] = backend
        server = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", spec["model"],
            "--revision", spec["revision"], "--tokenizer-revision", spec["revision"],
            "--host", "127.0.0.1", "--port", str(port), "--api-key", token,
            "--served-model-name", spec["model"], "--dtype", "half",
            "--max-model-len", str(spec["capacity"]), "--max-num-seqs", "32",
            "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.55",
            "--no-enable-prefix-caching", "--enforce-eager", "--seed", "42",
            "--attention-backend", "CUSTOM" if backend.startswith("SDPA_") else backend,
            "--worker-extension-cls", "scratch_vllm.workload_audit.WorkloadAuditWorker"]
        record = dict(study=study, backend=backend, run=run, start=utc(), revision=self.revision,
            directory=str(directory.relative_to(self.root)), server_command=["<ephemeral-api-key>" if s == token else s for s in server],
            dev_rpc="localhost-only, ephemeral API key, string methods, no insecure pickle")
        save(directory / "session.json", record)
        self.state["sessions"].append(record)
        self.update("starting_server", f"{study} {backend} run {run}")
        with (directory / "server.log").open("w") as log:
            proc = self.spawn(server, env, log)
            try:
                start = time.monotonic()
                while True:
                    self.check_active()
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server exited {proc.returncode}: {(directory / 'server.log').read_text()[-4000:]}")
                    try:
                        request = urllib.request.Request(f"http://127.0.0.1:{port}/health",
                                  headers={"Authorization": f"Bearer {token}"})
                        with urllib.request.urlopen(request, timeout=2) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() - start > 900:
                        raise TimeoutError("Server startup timeout")
                    time.sleep(1)
                record["ready"] = utc()
                self.rpc(port, token, "workload_snapshot")
                for length, concurrency in remaining:
                    self.check_sources()
                    key = self.case_key(study, backend, run, length, concurrency)
                    case_dir = directory / f"n{length}_c{concurrency}"
                    case_dir.mkdir()
                    commands = []
                    before = None
                    for warm in (True, False):
                        self.update("warming" if warm else "measuring", key)
                        if not warm:
                            self.rpc(port, token, "reset_workload_peaks")
                            before = self.rpc(port, token, "workload_snapshot")
                        command = self.client_command(study, spec, run, length, concurrency, case_dir, warm)
                        command += ["--port", str(port)]
                        commands.append(command)
                        label = "warmup" if warm else "client"
                        with (case_dir / f"{label}.log").open("w") as client_log:
                            child = self.spawn(command, env, client_log)
                            try:
                                self.monitor(child, case_dir / f"{label}_telemetry.jsonl", port=port, token=token)
                            finally:
                                self.stop_owned(child)
                        phase_stats = json.loads((case_dir / ("warmup.json" if warm else "client.json")).read_text())
                        validated_metrics(phase_stats, max(3, 2 * concurrency) if warm else request_count(self.args.prompts, concurrency),
                                          length, 128, special_tokens=self.special_tokens[study])
                    self.check_sources()
                    after = self.rpc(port, token, "workload_snapshot")
                    raw = case_dir / "client.json"
                    stats = json.loads(raw.read_text())
                    metrics = validated_metrics(stats, request_count(self.args.prompts, concurrency), length, 128,
                                                special_tokens=self.special_tokens[study])
                    routing = routes(before, after)
                    if backend != "FLASH_ATTN" and not any((r["scratch_calls"] or 0) > 0 for r in routing):
                        raise ValueError("No measured custom/SDPA decode calls")
                    if backend.startswith("SDPA_") and f"DENSE_SDPA_OPERATOR_AUDIT mode={backend}" not in (directory / "server.log").read_text():
                        raise ValueError("Missing forced/automatic SDPA operator audit")
                    accepted = dict(study=study, backend=backend, run=run, input_len=length,
                        special_tokens=self.special_tokens[study],
                        concurrency=concurrency, revision=self.revision, measured_at=utc(),
                        raw=str(raw.relative_to(self.root)), raw_sha256=digest(raw), metrics=metrics,
                        session=record["directory"], commands=commands, routing=routing,
                        before=before, after=after,
                        telemetry=str((case_dir / "client_telemetry.jsonl").relative_to(self.root)))
                    self.state["results"][key] = accepted
                    save(case_dir / "accepted.json", accepted)
                    self.update("accepted", key)
                    self.report()
            finally:
                self.stop_owned(proc)
                record["stopped"] = utc()
                save(directory / "session.json", record)
                save(self.root / "state.json", self.state)

    def guarded(self, operation, label, attempts=3):
        failures = 0
        while failures < attempts:
            self.check_stop()
            try:
                operation()
                return True
            except StopRequested:
                raise
            except Exception as error:
                message = str(error)
                transient = "contamination:" in message or "thermal:" in message
                failures += 0 if transient else 1
                self.event("retry", label=label, error=repr(error), failures=failures,
                           transient=transient)
                self.update("retry_wait", f"{label}: {message[-700:]}")
                time.sleep(30)
        self.state["failed"][label] = dict(error="See retry events and attempt logs", time=utc())
        self.update("failed_item", label)
        return False

    def study(self, name, cases):
        for mode in MODES:
            if not self.guarded(lambda m=mode: self.validation("model", name, m), f"validation_{name}_{mode}"):
                if mode in ("FLASH_ATTN", "CUSTOM"):
                    raise RuntimeError(f"Required model validation failed: {name} {mode}")
        for run in range(self.args.runs):
            order = MODES[:] if run % 2 == 0 else list(reversed(MODES))
            if run >= 2:
                order = order[run:] + order[:run]
            ordered_cases = cases if run % 2 == 0 else list(reversed(cases))
            for mode in order:
                if f"model_{name}_{mode}" not in self.state["validation"]:
                    for n, c in cases:
                        self.state["failed"][self.case_key(name, mode, run, n, c)] = dict(error="Model validation failed")
                    continue
                label = f"session_{name}_{mode}_{run}_{cases}"
                success = self.guarded(lambda m=mode, r=run: self.session(name, m, r, ordered_cases), label)
                if not success:
                    for n, c in cases:
                        key = self.case_key(name, mode, run, n, c)
                        if key not in self.state["results"]:
                            self.state["failed"][key] = dict(error=f"Session failed: {label}")
                self.report()

    def wants_more_concurrency(self, old, current):
        decisions = []
        for mode in ("FLASH_ATTN", "CUSTOM"):
            values = {}
            for concurrency in (old, current):
                rows = [x["metrics"]["output_throughput"] for x in self.state["results"].values()
                        if x["study"] == "throughput" and x["backend"] == mode and x["concurrency"] == concurrency]
                if len(rows) == self.args.runs:
                    values[concurrency] = median(rows)
            if len(values) == 2:
                decisions.append(dict(mode=mode, old=old, current=current,
                                      ratio=values[current] / values[old]))
        self.event("concurrency_extension_check", values=decisions,
                   rule="Extend if either native mode median throughput gains >5%; heuristic, not significance")
        return any(x["ratio"] > 1.05 for x in decisions)

    def report(self):
        rows = list(self.state["results"].values())
        groups = {}
        for row in rows:
            groups.setdefault((row["study"], row["input_len"], row["concurrency"], row["backend"]), []).append(row)
        lines = ["# Overnight Workload Evaluation", "", f"Status: {self.state['status']}",
            f"Updated: {utc()}", "", "All modes eager; only decode attention replaced.",
            "SDPA includes paged gather/repack, Python request loops and any GQA expansion.",
            "These are adapter-inclusive serving results, not pure CUDA kernel rankings.", "",
            "| Study | Input target | Concurrency | Backend | Runs | Output tokens/s median | TTFT p50 median ms | TPOT p50 median ms | TPOT p95 median ms |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for (study, n, c, backend), records in sorted(groups.items()):
            metrics = [r["metrics"] for r in records]
            med = lambda k: median(x[k] for x in metrics)
            lines.append(f"| {study} | {n} | {c} | {backend} | {len(records)} | {med('output_throughput'):.3f} | "
                f"{med('median_ttft_ms'):.3f} | {med('median_tpot_ms'):.3f} | {med('p95_tpot_ms'):.3f} |")
        differences = []
        memory_records = []
        for key, row in self.state["results"].items():
            samples = [json.loads(line) for line in (self.root / row["telemetry"]).read_text().splitlines()]
            server = [value for sample in samples for value in sample.get("server_metrics", [])]
            usage = [x["value"] for x in server if x["name"] == "vllm:kv_cache_usage_perc"]
            cache_configs = [x["labels"] for x in server if x["name"] == "vllm:cache_config_info"]
            spec = MODELS[row["study"]]
            record = dict(key=key, max_device_memory_mib=max(float(x["memory.used"]) for x in samples),
                max_temperature_c=max(float(x["temperature.gpu"]) for x in samples),
                peak_torch_allocated_bytes=max(x["max_memory_allocated"] for x in row["after"]),
                peak_torch_reserved_bytes=max(x["max_memory_reserved"] for x in row["after"]),
                sampled_kv_usage_fraction=max(usage) if usage else None,
                cache_config=cache_configs[-1] if cache_configs else None,
                kv_payload_bytes_per_token=2 * spec["layers"] * spec["hkv"] * spec["d"] * 2,
                metrics_sample_errors=sum("metrics_error" in x for x in samples))
            config = record["cache_config"]
            if config and config.get("num_gpu_blocks", "None") != "None" and config.get("block_size", "None") != "None":
                capacity_bytes = int(config["num_gpu_blocks"]) * int(config["block_size"]) * record["kv_payload_bytes_per_token"]
                record["kv_capacity_payload_bytes"] = capacity_bytes
                record["estimated_sampled_live_kv_bytes"] = None if not usage else max(usage) * capacity_bytes
            memory_records.append(record)
            reference_key = self.case_key(row["study"], "FLASH_ATTN", row["run"], row["input_len"], row["concurrency"])
            baseline = self.state["results"].get(reference_key)
            if baseline and row["backend"] != "FLASH_ATTN":
                actual = json.loads((self.root / row["raw"]).read_text())
                reference = json.loads((self.root / baseline["raw"]).read_text())
                mismatch = response_differences(actual["generated_texts"], reference["generated_texts"])
                differences.append(dict(key=key, matching=len(actual["generated_texts"]) - len(mismatch),
                                        total=len(actual["generated_texts"]), mismatches=mismatch))
        save(self.root / "response_differences.json", differences)
        save(self.root / "memory_summary.json", memory_records)
        lines += ["", "## Interpretation and remaining work", "",
            "- Three planned independent server runs per mode; exploratory, not publication-strength significance.",
            "- At least 64 prompts and eight concurrency-sized groups per condition; raw records contain actual counts.",
            "- Input accounting subtracts each pinned tokenizer's actual special-token count, recorded in state.json.",
            "- Same server is reused for shapes within a mode/run; prefix caching is disabled and each shape is warmed.",
            "- Route deltas and allocator peaks are collected outside client timing via authenticated localhost-only RPC.",
            "- All comparisons preserve native prefill; TTFT is not a custom-prefill result.",
            "- Peak PyTorch allocator memory is not whole-device memory; sampled telemetry and live cache metrics are separate.",
            "- memory_summary.json preserves measured peaks/cache fractions; live KV bytes are derived from sampled usage and capacity, not allocator measurements.",
            "- Clocks are unlocked; polling cannot exclude brief Windows/non-Python interference.",
            "- Per-run raw timing, throughput checks, route counts, memory and telemetry are preserved in state.json and attempts/.",
            "- Free-generation differences are retained in response_differences.json; do not claim universal token equality.",
            "- Any additional near-32K or concurrency extension is recorded explicitly in events.jsonl.",
            "", f"Accepted measurement rows: {len(rows)}. Failure entries: {len(self.state['failed'])}.", ""]
        if self.state["failed"]:
            lines += ["## Failures", "", *[f"- {k}: {v['error']}" for k, v in self.state["failed"].items()], ""]
        (self.root / "RESULTS.md").write_text("\n".join(lines))

    def run(self):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        for name, spec in MODELS.items():
            path = Path(snapshot_download(spec["model"], revision=spec["revision"], local_files_only=True))
            config = json.loads((path / "config.json").read_text())
            if config["hidden_size"] // config["num_attention_heads"] != spec["d"]:
                raise ValueError(f"Model config mismatch: {name}")
            if spec["capacity"] > config["max_position_embeddings"]:
                raise ValueError(f"Unsupported context: {name}")
            if not list(path.glob("*.safetensors")):
                raise ValueError(f"Weights missing in local snapshot: {path}")
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            self.model_paths[name] = str(path)
            self.special_tokens[name] = int(tokenizer.num_special_tokens_to_add())
            self.state.setdefault("model_token_accounting", {})[name] = dict(
                snapshot=str(path), special_tokens=self.special_tokens[name], config_sha256=digest(path / "config.json"))
            self.event("model_cache", study=name, snapshot=str(path), config=config)
        if not self.guarded(lambda: self.validation("kernels"), "kernel_validation"):
            raise RuntimeError("Shape validation failed; no performance claims")
        self.study("throughput", [(512, c) for c in (1, 2, 4, 8)])
        self.study("long_context", [(n, 1) for n in (2048, 4096, 8192, 16384)])
        previous, current = 4, 8
        for following in (16, 32):
            if not self.wants_more_concurrency(previous, current):
                break
            self.study("throughput", [(512, following)])
            previous, current = current, following
        if all(self.case_key("long_context", m, r, 16384, 1) in self.state["results"]
               for m in ("FLASH_ATTN", "CUSTOM") for r in range(self.args.runs)):
            self.event("long_extension", input_target=32640, output_len=128, total_limit=32768)
            self.study("long_context", [(32640, 1)])
        self.check_sources()
        self.update("complete_with_failures" if self.state["failed"] else "complete", "Declared workload matrix accounted for; own servers stopped")
        self.report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--max-hours", type=float, default=12)
    args = parser.parse_args()
    if args.runs < 1 or args.prompts < 32 or args.max_hours <= 0:
        parser.error("Require runs>=1, prompts>=32, max-hours>0")
    os.chdir(ROOT)
    experiment = Experiment(args)
    def stop(signum, frame):
        raise StopRequested(f"Signal {signum}; stopping owned jobs")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        experiment.run()
    except StopRequested as error:
        experiment.event("paused", reason=str(error))
        experiment.update("paused", str(error))
        experiment.report()
    except Exception as error:
        experiment.event("error", reason=repr(error))
        experiment.update("needs_recovery", repr(error))
        experiment.report()
        raise


if __name__ == "__main__":
    main()
