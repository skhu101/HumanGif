
import asyncio
from abc import abstractmethod
from typing import Any, Callable

from flow.task import Task


class TaskPool:
    def __init__(self, parallelism=1, name=None) -> None:
        self.name = name
        self._parallelism = parallelism
        self.running: list[Task] = []
        self.tasks: list[Task] = []
        self.done: list[Task] = []
        self.on_progress_handlers: list[Callable[[TaskPool, Task, Any], Any]] = []

        self.pre_hooks = []
        self.post_hooks = []
        self.context = {}

        self._pre_hook_running = False

    def pre_hook(self, hook: Callable[..., dict[str, Any]]):
        self.pre_hooks.append(hook)
        return self

    def post_hook(self, hook: Callable[..., Any]):
        self.post_hooks.append(hook)
        return self

    def schedule(self, task: Task):
        self.tasks.append(task)
        self._loop()
        return self

    def on_progress(self, fn: Callable[[Any, Task, Any], Any]):
        if not callable(fn):
            raise ValueError("fn is not callable")
        self.on_progress_handlers.append(fn)

    def _invoke_pre_hooks(self):
        for hook in self.pre_hooks:
            self.context = {**self.context, **hook()}

    def _invoke_post_hooks(self):
        for hook in self.post_hooks:
            self.context = {**self.context, **hook(context=self.context)}

    @property
    def parallelism(self):
        return self._parallelism

    @parallelism.setter
    def parallelism(self, value: int):
        self._parallelism = value

    @property
    def is_idle(self):
        return len(self.running) < self.parallelism

    @property
    def is_running(self):
        return len(self.running) > 0

    @property
    def scheduled_count(self):
        return len(self.running) + len(self.done)

    async def wait_for(self):
        while self.scheduled_count < self.total_count or self.is_running:
            await asyncio.sleep(2.0)

        return self.done

    def _loop(self):
        if self._pre_hook_running:
            return
        if self.scheduled_count == 0:
            self._pre_hook_running = True
            self._invoke_pre_hooks()
            self._pre_hook_running = False
        while self.scheduled_count < self.total_count and self.is_idle:
            # run step async with callback
            to_be_run = self.tasks[self.scheduled_count]
            self.running.append(to_be_run)
            self.run_task_async(to_be_run, self._handle_task_done)

    def _handle_task_done(self, task: Task, outputs: Any, error: Exception):
        if error:
            # TODO: handle task error
            print(f"[SCHEDULER-{self.name}]: error {task.name}")
            print(f"ERROR: {task.name} ############################################")
            print(f"ERROR: {task.name} ############################################")
            print(f"args: {task.args}")
            print(f"kwargs: {task.kwargs}")
            print(f"ERROR: {error}")
            print(f"ERROR: {task.name} ############################################")
            print(f"ERROR: {task.name} ############################################")

        else:
            if outputs:
                task.outputs = outputs
            for handler in self.on_progress_handlers:
                handler(self, task, outputs)

        self.running.remove(task)
        self.done.append(task)

        self._loop()

        print(f"[SCHEDULER-{self.name}]: DONE({len(self.done)}/{len(self.tasks)}) {task.name}")
        print(f"[SCHEDULER-{self.name}]: CURRENT {len(self.tasks)}, RUNNING {len(self.running)}, DONE {len(self.done)}")

    @property
    def total_count(self):
        return len(self.tasks)

    @abstractmethod
    def run_task_async(self, task: Task, done: Callable[[Task, Any], Any]):
        print(f"[SCHEDULER-{self.name}]: RUN({len(self.done)}/{len(self.tasks)}) {task.name}")
        if task.kwargs is None:
            task.kwargs = {}
        task.kwargs = {**task.kwargs, **self.context}

    @abstractmethod
    def close(self):
        self._invoke_post_hooks()
        self.tasks = []
        self.running = []
        self.done = []
