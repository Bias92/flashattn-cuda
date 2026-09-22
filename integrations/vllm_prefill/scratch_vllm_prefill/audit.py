"""Worker-side routing counters, read through collective_rpc."""


class RoutingWorker:
    def attention_routing(self):
        """Per attention layer: the implementation class, and calls and tokens per route.

        ScratchPrefillImpl keeps `route_calls` and `route_tokens` (see its ROUTES). The
        decode-only ScratchImpl of scratch_vllm only has `scratch_calls` (decode kernel) and
        `prefill_calls`, its name for every call handed to FlashAttention, profiling runs
        without metadata included. vLLM's own implementations have none of these; a missing
        counter comes back as None. All of them count executions of forward(), so steps
        replayed from a captured CUDA graph are not in them.
        """
        return [dict(layer=name, implementation=type(module.impl).__name__,
                     route_calls=getattr(module.impl, "route_calls", None),
                     route_tokens=getattr(module.impl, "route_tokens", None),
                     scratch_decode_calls=getattr(module.impl, "scratch_calls", None),
                     flash_calls=getattr(module.impl, "prefill_calls", None))
                for name, module in self.model_runner.model.named_modules()
                if hasattr(module, "impl")]
