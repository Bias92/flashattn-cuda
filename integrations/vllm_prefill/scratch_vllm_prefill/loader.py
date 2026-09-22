"""Builds cuda/attention_forward.cu with the flags the tests and benchmarks use.

The module name carries the source hash, so an edited kernel is never served from an
extension cached under an unchanged name.
"""

import hashlib
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
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    os.environ.setdefault("MAX_JOBS", "1")
    module = load(name=f"attention_forward_{digest[:12]}", sources=[str(SOURCE)],
                  extra_cuda_cflags=FLAGS,
                  verbose=os.environ.get("SCRATCH_BUILD_VERBOSE") == "1")
    print(f"SCRATCH_PREFILL source_sha256={digest} so={module.__file__}", flush=True)
    return module
