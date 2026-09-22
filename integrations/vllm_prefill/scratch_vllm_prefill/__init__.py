"""Out-of-tree vLLM backend: scratch prefill and scratch paged decode.

Select it with VLLM_PLUGINS=scratch_prefill and attention backend CUSTOM. It builds on the
decode-only package in integrations/vllm, which has to be installed too.
"""


def register():
    from importlib.metadata import version

    if version("vllm") != "0.19.0":
        raise RuntimeError("scratch_vllm_prefill is validated only against vLLM 0.19.0")
    import vllm.envs as envs

    # The decode-only plugin registers CUSTOM too and both are always installed together, so
    # this one only acts when it is named. With VLLM_PLUGINS unset CUSTOM stays decode-only.
    if "scratch_prefill" not in (envs.VLLM_PLUGINS or ()):
        return
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.CUSTOM,
                     "scratch_vllm_prefill.backend.ScratchPrefillBackend")
