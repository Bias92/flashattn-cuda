"""Out-of-tree backend. Use an editable install from this repository."""


def register():
    from importlib.metadata import version
    import os

    if version("vllm") != "0.19.0":
        raise RuntimeError("scratch_vllm is validated only against vLLM 0.19.0")
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    path = ("scratch_vllm.comparison.DenseComparisonBackend"
            if os.environ.get("SCRATCH_COMPARISON_BACKEND")
            else "scratch_vllm.backend.ScratchBackend")
    register_backend(AttentionBackendEnum.CUSTOM, path)
