
from . import DAG


class Hopper:
    def __init__(self, dag: DAG) -> None:
        self.dag = dag
