"""Builds cuda/attention_forward.cu with the flags the tests and benchmarks use.

The module name hashes the entry point and CUDA headers, including header-only edits.
The extension is cached once per process; restart the engine after editing sources.
"""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "cuda" / "attention_forward.cu"
FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_89,code=sm_89"]


@lru_cache(maxsize=1)
def load_prefill_extension():
    from torch.utils.cpp_extension import load

    if not SOURCE.is_file():
        raise RuntimeError("Use pip install -e integrations/vllm_prefill from the source checkout")
    # Conservatively include every local CUDA header, including decode's shared helper.
    paths = [SOURCE, *sorted(SOURCE.parent.glob("*.h")), *sorted(SOURCE.parent.glob("*.cuh"))]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    os.environ.setdefault("MAX_JOBS", "1")
    module = load(name=f"attention_forward_{digest[:12]}", sources=[str(SOURCE)],
                  extra_cuda_cflags=FLAGS,
                  verbose=os.environ.get("SCRATCH_BUILD_VERBOSE") == "1")
    print(f"SCRATCH_PREFILL source_sha256={hashes[SOURCE.name]} "
          f"build_sources_sha256={digest} so={module.__file__}", flush=True)
    return module
