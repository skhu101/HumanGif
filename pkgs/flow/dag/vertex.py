
import uuid


class Vertex:
    def __init__(self) -> None:
        self.id = str(uuid.uuid4())
        self.in_degree = 0
        self.out_degree = 0

    @property
    def is_valid(self):
        return bool(self.id)
