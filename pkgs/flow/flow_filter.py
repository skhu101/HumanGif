
from typing import Any, Callable

from .dag.edge import Edge


class FlowFilter(Edge):
    def __init__(self, start: str, end: str, fn: Callable[..., Any]) -> None:
        super().__init__(start, end)
        self.fn = fn
