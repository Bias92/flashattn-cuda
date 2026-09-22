"""Worker-side dispatch inspection without serializing arbitrary callables."""


class AuditWorker:
    def scratch_routing_snapshot(self):
        return [dict(name=name, implementation=type(module.impl).__name__,
                     scratch_calls=getattr(module.impl, "scratch_calls", None),
                     prefill_calls=getattr(module.impl, "prefill_calls", None),
                     last_route=getattr(module.impl, "last_route", None))
                for name, module in self.model_runner.model.named_modules()
                if hasattr(module, "impl")]
