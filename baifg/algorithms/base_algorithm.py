from baifg.model.feedback_graph import FeedbackGraph
from typing import NamedTuple, List, Tuple
from abc import abstractmethod, ABC

class Observable(NamedTuple):
    out_vertex: int
    in_vertex: int
    observed_value: float
    
    @property
    def edge(self) -> Tuple[int, int]:
        return (self.in_vertex, self.out_vertex)

class Experience(NamedTuple):
    vertex: int
    observables: List[Observable]


class BaseAlg(ABC):
    NAME: str
    fg: FeedbackGraph

    def __init__(self, name: str, fg: FeedbackGraph):
        self.NAME = name
        self.fg = fg

    @abstractmethod
    def sample(self) -> int:
        raise NotImplementedError("Sample function not imlpemented")
    
    @abstractmethod
    def backward(self, experience: Experience):
        raise NotImplementedError("Backward function not implemented")
