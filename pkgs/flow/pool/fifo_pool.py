
import traceback
from typing import Any, Callable

from flow.pool.base import TaskPool
from flow.task import Task


class FIFOPool(TaskPool):
    def __init__(self, parallelism=1, name=None) -> None:
        super().__init__(parallelism, name)
        self.parallelism = 1

    def run_task_async(self, task: Task, done: Callable[[Task, Any, Exception], Any]):
        super().run_task_async(task, done)

        outputs = None
        ex: Exception = None
        try:
            outputs = task.fn(*task.args, **task.kwargs)
        except Exception as e:
            print(traceback.format_exc())
            ex = e

        done(task, outputs, ex)

    def close(self):
        print(f"[POOL-{self.name}]: closing pool... Nothing to be destroyed")
        return super().close()
