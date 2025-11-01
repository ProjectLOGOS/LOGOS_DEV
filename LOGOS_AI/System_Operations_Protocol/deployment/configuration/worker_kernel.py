"""
LOGOS V2 Unified Worker Kernel
==============================
Consolidated worker infrastructure from distributed and adaptive_reasoning.
"""

from .system_imports import *
from .unified_classes import UnifiedWorkerConfig


class WorkerKernel:
    """Consolidated worker management kernel"""

    def __init__(self, config: UnifiedWorkerConfig):
        self.config = config
        self.active_workers = {}
        self.worker_queue = []
        self.logger = logging.getLogger("WorkerKernel")

    def start_worker(self, worker_id: str, task: Any) -> bool:
        """Start a worker with given task"""
        if len(self.active_workers) >= self.config.max_workers:
            self.worker_queue.append((worker_id, task))
            return False

        self.active_workers[worker_id] = {
            "task": task,
            "start_time": time.time(),
            "status": "running",
        }
        self.logger.info(f"Started worker {worker_id}")
        return True

    def stop_worker(self, worker_id: str) -> bool:
        """Stop a worker"""
        if worker_id in self.active_workers:
            del self.active_workers[worker_id]
            self.logger.info(f"Stopped worker {worker_id}")

            # Process queue if space available
            if self.worker_queue:
                next_worker_id, next_task = self.worker_queue.pop(0)
                self.start_worker(next_worker_id, next_task)
            return True
        return False

    def get_worker_status(self, worker_id: str) -> Dict[str, Any]:
        """Get status of specific worker"""
        return self.active_workers.get(worker_id, {})

    def get_all_workers_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all workers"""
        return self.active_workers.copy()


# Export consolidated interface
__all__ = ["WorkerKernel"]
