"""Out-of-timing snapshots for the local eager workload experiment."""

import torch

from .audit import AuditWorker


class WorkloadAuditWorker(AuditWorker):
    def workload_snapshot(self):
        torch.cuda.synchronize()
        return dict(
            routing=self.scratch_routing_snapshot(),
            memory_allocated=torch.cuda.memory_allocated(),
            memory_reserved=torch.cuda.memory_reserved(),
            max_memory_allocated=torch.cuda.max_memory_allocated(),
            max_memory_reserved=torch.cuda.max_memory_reserved(),
        )

    def reset_workload_peaks(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return True
