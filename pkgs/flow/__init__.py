
from typing import Any, Callable, Iterable

from flow.dag.edge import Edge
from flow.dag.vertex import Vertex
from flow.flow_filter import FlowFilter
from flow.pool.base import TaskPool
from flow.pool.fifo_pool import FIFOPool
from flow.task import Task

from .dag import DAG
from .flow_node import FlowNode


class Flow:
    def __init__(self, default_pool_maker: Callable[[int], TaskPool]) -> None:
        super().__init__()
        self.dag = DAG()
        self.functions: dict[Vertex, Callable] = {}
        self.pools: dict[Vertex, TaskPool] = {}

        self.filters: dict[Edge, Callable[[Any, tuple[list, dict]], tuple[list, dict]]] = {}

        self.default_pool_maker = default_pool_maker
        self.inputs: list[Iterable] = []
        self.context: dict = {}

    def input(self, inputs: Iterable):
        self.inputs.append(inputs)
        self.pipe(
            name="input",
            fn=lambda x: x,
            parallelism=1,
            pool_maker=FIFOPool,
        )
        return self

    def pipe(self,
            name: str,
            fn: Callable[..., Any],
            parallelism=1,
            pool_maker: Callable[[int],TaskPool]=None,
            filter_method: Callable[[Any, tuple[list, dict]], tuple[list, dict]]=None
        ):
        pool_maker = pool_maker if pool_maker is not None else self.default_pool_maker

        node = FlowNode(name)
        self.functions[node] = fn
        pool = pool_maker(parallelism)
        pool.on_progress(lambda _, task, _outputs: self._run_next_hop(self._get_node_by_task(task)))
        self.pools[node] = pool
        self.dag.add_v(node)

        end_vertices = self.dag.get_end_vertices()
        for end_v in end_vertices:
            edge = Edge(end_v.id, node.id)
            self.dag.add_e(edge)
            if filter_method:
                self.filters[edge] = filter_method

        return self

    def start(self):
        self.schedule()
        return self

    def _run_next_hop(self, pre_node: FlowNode):
        for node in self.dag.next_hop(pre_node):
            pass

    def schedule(self, v: FlowNode, *args, **kwargs):
        pool = self.pools[v]
        fn = self.functions[v]

        pool.schedule(Task(
            name=v.name,
            fn=fn,
            args=args,
            kwargs=kwargs
        ))

    def _get_node_by_task(self, task: Task):
        for node in self.dag.vertices.values():
            if node.task == task:
                return node
        return None

    def stop(self):
        return self
