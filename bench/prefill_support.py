"""Shared build and GPU-state helpers for the current prefill benchmark."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "cuda/attention_forward.cu"
FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]
os.environ.setdefault("MAX_JOBS", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest(source):
    """Hash the entry point and local headers, using the serving loader's build identity."""
    source = Path(source).resolve()
    paths = [source, *sorted(source.parent.glob("*.h")), *sorted(source.parent.glob("*.cuh"))]
    hashes = {p.name: sha(p) for p in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"build_sources_sha256": digest, "source_files": hashes}


def extension(source):
    source = Path(source).resolve()
    manifest = source_manifest(source)
    name = "prefill_layout_" + manifest["build_sources_sha256"][:12]
    module = load(name=name, sources=[str(source)], extra_cuda_cflags=FLAGS, verbose=False)
    record = dict(source=str(source), sha256=sha(source), so=module.__file__,
                  so_sha256=sha(module.__file__), flags=FLAGS, **manifest)
    print(json.dumps(record), flush=True)
    return module, record


def gpu_state():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu,temperature.gpu,"
        "clocks.sm,clocks.mem,power.draw", "--format=csv,noheader"], text=True).strip()


def guard():
    script = subprocess.check_output(["wslpath", "-w", str(ROOT / "bench/gpu_busy.ps1")],
                                     text=True).strip()
    status = subprocess.check_output(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                                      "-File", script], text=True).strip()
    if float(status.split()[0].replace(",", ".")) > 15:
        raise RuntimeError("Other GPU user: " + status)
    return status
