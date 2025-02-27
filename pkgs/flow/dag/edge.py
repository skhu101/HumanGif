
class Edge:
    def __init__(self, start: str, end: str) -> None:
        self.start = start
        self.end = end

    @property
    def is_valid(self):
        return self.start and self.end
