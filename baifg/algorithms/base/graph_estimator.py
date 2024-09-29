from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from baifg.model.experience import Experience, Observable
from baifg.model.graph import Graph

class GraphEstimator(Graph):
    """Generic graph estimator
    """
    informed: bool
    known: bool
    N: NDArray[np.float64]
    Npair: NDArray[np.float64]
    confidence: NDArray[np.float64]

    def __init__(self, G0: Graph, informed: bool, known: bool):
        """Initialize the graph estimator

        Args:
            G0 (Graph): initial graph
            informed (bool): true if the learner knows if the edge was acitvated
            known (bool): true if the learner knows the graph structure and thus it should not
                          be updated.
        """
        assert informed or (not informed and not known), "The graph can only  be informed or (not informed and not known)"
        super().__init__(G0.G)
        self.informed = informed
        self.known = known
        self.N = np.zeros(self.K)
        self.Npair = np.ones((self.K, self.K))

    @staticmethod
    def from_graph(G: Graph, informed: bool, known: bool) -> GraphEstimator:
        """Create a graph estimator from an initial graph
        """
        return GraphEstimator(G, informed, known)

    @staticmethod
    def optimistic_graph(K: int, informed: bool, known: bool) -> GraphEstimator:
        """Create a graph estimator with the probability of all edges set to 1
        """
        g = Graph(np.ones((K, K)))
        return GraphEstimator(g, informed, known)

    def update(self, t: int, experience: Experience):
        """Update the graph according to the observations"""
        self.N[experience.vertex] += 1

        for obs in experience.observables:
            v = obs.in_vertex
            activated = obs.activated if self.informed else not np.isclose(obs.observed_value,0)
            if activated:
                self.Npair[v,obs.out_vertex] += 1
            
        if not self.known:
            self.update_graph(G = self.Npair / np.maximum(1, self.N[:, None]))
            self.confidence = np.sqrt(0.5 * np.log(1 + t) / np.maximum(1, self.N))


