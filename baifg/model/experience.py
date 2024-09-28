from typing import NamedTuple, List, Tuple

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
