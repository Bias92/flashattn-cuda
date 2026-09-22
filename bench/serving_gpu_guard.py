"""Read-only GPU coordination. Never stop unrelated jobs or change GPU clocks."""

import csv
import io
import os
from pathlib import Path
import subprocess
import time

import psutil


def other_python_jobs():
    owned = {os.getpid()}
    owned.update(p.pid for p in psutil.Process().children(recursive=True))
    jobs = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "exe", "status"]):
        try:
            if proc.pid in owned or proc.info["status"] == psutil.STATUS_ZOMBIE:
                continue
            cmd = proc.info["cmdline"] or []
            if not cmd or "unattended-upgrade" in " ".join(cmd):
                continue
            executable = Path(proc.info["exe"] or cmd[0]).name
            if "python" in executable.lower():
                jobs.append(dict(pid=proc.pid, command=cmd))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return jobs


def telemetry():
    columns = ["temperature.gpu", "utilization.gpu", "clocks.sm", "clocks.mem",
               "power.draw", "pstate", "memory.used"]
    result = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--id=0",
                             "--query-gpu=" + ",".join(columns),
                             "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True, timeout=10)
    row = next(csv.reader(io.StringIO(result.stdout)))
    values = dict(zip(columns, (v.strip() for v in row)))
    values["monotonic_s"] = time.monotonic()
    return values


def wait_idle(timeout=180, consecutive=3, quiet_seconds=0):
    deadline = time.monotonic() + timeout
    good, samples = 0, []
    quiet_since = None
    while time.monotonic() < deadline:
        jobs = other_python_jobs()
        now = time.monotonic()
        if jobs:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = now
        item = telemetry()
        item["other_jobs"] = jobs
        item["quiet_seconds_observed"] = 0 if quiet_since is None else now - quiet_since
        desktop_low_power = (
            item.get("pstate") == "P8" and float(item.get("clocks.sm", "inf")) <= 300
            and float(item.get("clocks.mem", "inf")) <= 1000
            and float(item.get("power.draw", "inf")) <= 20
            and float(item["utilization.gpu"]) <= 30
        )
        item["low_power_desktop_gate"] = desktop_low_power
        samples.append(item)
        good = good + 1 if (not jobs and (float(item["utilization.gpu"]) <= 10 or desktop_low_power)
                            and float(item["temperature.gpu"]) <= 65) else 0
        if good >= consecutive and item["quiet_seconds_observed"] >= quiet_seconds:
            return samples
        if len(samples) % 15 == 0:
            print(f"GPU idle gate: {item}", flush=True)
        time.sleep(1)
    raise RuntimeError(f"GPU did not become idle; unrelated jobs were not stopped: {samples[-1]}")


def monitored_wait(proc, timeout=600):
    samples = []
    deadline = time.monotonic() + timeout
    try:
        while proc.poll() is None:
            item = telemetry()
            item["other_jobs"] = other_python_jobs()
            samples.append(item)
            if item["other_jobs"]:
                raise RuntimeError(f"Benchmark rejected: another Python job started: {item['other_jobs']}")
            if float(item["temperature.gpu"]) > 80:
                raise RuntimeError("Benchmark stopped: GPU temperature exceeded 80 C")
            if time.monotonic() >= deadline:
                raise TimeoutError("Serving benchmark timed out")
            time.sleep(1)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        raise
    return samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--quiet-seconds", type=int, default=0)
    args = parser.parse_args()
    wait_idle(timeout=args.timeout, quiet_seconds=args.quiet_seconds)
    print("GPU start gate passed; unrelated jobs were not stopped", flush=True)
