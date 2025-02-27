
from multiprocessing import Pool, Process, cpu_count
from typing import Any, Callable

from flow.task import Task

from .base import TaskPool


class ProcessPool(TaskPool):
    def __init__(self, parallelism=1, name=None) -> None:
        super().__init__(parallelism, name)
        if self.parallelism > cpu_count():
            print("paralellism is more than cpu count")
            self.parallelism = cpu_count()

        self.pool = None

    def run_task_async(self, task: Task, done: Callable[[Task, Any], Any]):
        super().run_task_async(task, done)
        if not self.pool:
            self.pool = Pool(self.parallelism)

        self.pool.apply_async(
            func=task.fn,
            args=task.args,
            kwds=task.kwargs,
            callback=lambda ret: done(task, ret, None),
            error_callback=lambda error: done(task, None, error)
        )

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
        self.pool = None
        return super().close()
