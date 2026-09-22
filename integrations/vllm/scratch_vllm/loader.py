"""Content-addressed build; never load an old extension by an unchanged name."""

import hashlib
import os
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "cuda" / "attention_decode_paged.cu"


@lru_cache(maxsize=1)
def load_extension():
    import torch
    from torch.utils.cpp_extension import load

    if not SOURCE.is_file():
        raise RuntimeError("Use pip install -e integrations/vllm from the source checkout")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    os.environ.setdefault("MAX_JOBS", "1")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        capability = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
    module = load(
        name=f"attention_decode_paged_{digest[:12]}",
        sources=[str(SOURCE)],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-Xptxas=-v"],
        extra_cflags=["-O3"],
        verbose=os.environ.get("SCRATCH_BUILD_VERBOSE") == "1",
    )
    print(f"SCRATCH_PAGED source_sha256={digest} so={module.__file__}", flush=True)
    return module


if __name__ == "__main__":
    load_extension()
