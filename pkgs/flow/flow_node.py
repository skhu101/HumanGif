from typing import Any, Callable

from .dag.vertex import Vertex


class FlowNode(Vertex):
    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__()
        self.name = name
