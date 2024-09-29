import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Set



class Graph(object):
    K: int
    G: NDArray[np.float64]
    observable_vertices: set
    vertices: set
    in_neighborhood: Dict[int, Set[int]]
    out_neighborhood: Dict[int, Set[int]]
    observable: bool
    
    def __init__(self, G: NDArray[np.float64]):
        assert len(G.shape) == 2 and G.shape[0] == G.shape[1], "G needs to be a square matrix"
        self.G = G.copy()
        self.K = G.shape[0]
        self.vertices = set([i for i in range(self.K)])
        self.observable_vertices = set()
        self.update_observable_vertices()
        self.update_neighborhood()

    @property
    def is_observable(self):
        return self.observable

    def update_graph(self, G: NDArray[np.float64]):
        """ The graph should only be updated through this function to
            ensure that we update the set of observable vertices
        """
        assert len(G.shape) == 2 and G.shape[0] == G.shape[1], "G needs to be a square matrix"
        assert G.shape[0] == self.K, "New matrix is not equivalent to the previousone"
        self.G = G.copy()
        self.update_observable_vertices()
        self.update_neighborhood()

    def update_observable_vertices(self):
        # Compute observable vertices
        for i in range(self.K):
            if np.any(self.G[:,i] > 0):
                self.observable_vertices.add(i)
    
    def update_neighborhood(self):
        self.in_neighborhood = {i: [] for i in range(self.K)}
        self.out_neighborhood = {i: [] for i in range(self.K)}
        for i in range(self.K):
            self.in_neighborhood[i] = set(self.get_in_neighborhood(i))
            self.out_neighborhood[i] = set(self.get_out_neighborhood(i))
        self.observable = np.all(self.G.sum(-1) > 0)

    def get_in_neighborhood(self, u: int) -> List[np.int64]:
        assert u < self.K, f'u={u} is not in the vertex of size {self.K}'
        return np.argwhere(self.G[:,u] > 0).flatten().tolist()

    def get_out_neighborhood(self, u: int) -> List[np.int64]:
        assert u < self.K, f'u={u} is not in the vertex of size {self.K}'
        return np.argwhere(self.G[u,:] > 0).flatten().tolist()
    

