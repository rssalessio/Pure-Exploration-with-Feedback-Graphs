import numpy as np
from numpy.typing import NDArray
from typing import List



class Graph(object):
    K: int
    G: NDArray[np.float64]
    observable_vertices: set
    vertices: set
    
    
    def __init__(self, G: NDArray[np.float64]):
        assert len(G.shape) == 2 and G.shape[0] == G.shape[1], "G needs to be a square matrix"
        self.G = G.copy()
        self.K = G.shape[0]
        self.vertices = set([i for i in range(self.K)])
        self.observable_vertices = set([])

        # Compute observable vertices
        for i in range(self.K):
            if np.any(self.G[:,i] > 0):
                self.observable_vertices.add(i)

    def get_in_neighborhood(self, u: int) -> List[np.int64]:
        assert u < self.K, f'u={u} is not in the vertex of size {self.K}'
        return np.argwhere(self.G[:,u] > 0).flatten().tolist()

    def get_out_neighborhood(self, u: int) -> List[np.int64]:
        assert u < self.K, f'u={u} is not in the vertex of size {self.K}'
        return np.argwhere(self.G[u,:] > 0).flatten().tolist()
    
    