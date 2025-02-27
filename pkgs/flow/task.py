
from typing import Callable


class Task:
    def __init__(self, name: str, fn: Callable, *args, **kwargs) -> None:
        self.name = name
        self.fn = fn

        self.args = args
        self.kwargs = kwargs

        self.outputs = None

    def get_argument(self, name: str=None, index=None):
        res = None
        if name is not None:
            res = self.kwargs.get(name)
        elif index is not None:
            res = self.args[0]

        return res

    def __get_state__(self):
        return {
            "name": self.name,
            "fn": self.fn,
            "args": self.args,
            "kwargs": self.args,
            "outputs": self.outputs,
        }
    
    def __setstate__(self, state):
        self.name = state["name"]
        self.fn = state["fn"]
        self.args = state["args"]
        self.kwargs = state["kwargs"]
        self.outputs = state["outputs"]
