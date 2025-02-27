
from queue import Queue
from typing import Any, Generator

from .edge import Edge
from .vertex import Vertex


class DAG:
    def __init__(self) -> None:
        self.vertices: dict[str, Vertex] = {}
        self.edges = []

    def add_v(self, v: Vertex):
        assert v.is_valid
        assert not self._is_repeat_v(v)

        self.vertices[v.id] = v

    def add_e(self, e: Edge):
        assert e.is_valid
        assert e.start in self.vertex_ids
        assert e.end in self.vertex_ids
        assert not self._is_repeat_e(e)

        self.edges.append(e)
        self.vertices[e.start].out_degree += 1
        self.vertices[e.end].in_degree += 1

    def get_end_vertices(self):
        return list(filter(lambda v: v.out_degree == 0, self.vertices.values()))

    def get_start_vertices(self):
        return list(filter(lambda v: v.in_degree == 0, self.vertices.values()))

    def bfs(self) -> Generator[Vertex, None, Any]:
        queue: Queue[Vertex] = Queue()
        cache: dict[Vertex, int, int] = {}

        start_v = self.get_start_vertices()
        for v in start_v:
            queue.put(v)
            cache[v] = v.in_degree

        while not queue.empty():
            v = queue.get()
            cache[v] = cache[v] - 1

            if cache[v] == 0:
                yield v

            for next_v in self.next_hop(v):
                queue.put(next_v)
                if next_v not in cache:
                    cache[next_v] = next_v.in_degree

    def next_hop(self, v: Vertex):
        vertices: list[Vertex] = []
        if self._is_repeat_v(v):
            next_e = filter(lambda e: e.start == v.id, self.edges)
            for e in next_e:
                vertices.append(self.vertices[e.end])

        return vertices

    def _is_repeat_e(self, e: Edge):
        exist_e = [edge for edge in self.edges if edge.start == e.start and edge.end == e.end]
        return len(exist_e) > 0

    def _is_repeat_v(self, v: Vertex):
        return self.vertices[v.id] is not None


    @property
    def vertex_ids(self):
        return list(map(lambda x: x.id, self.vertices.values))
