from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from baifg.algorithms.base.base_algorithm import Experience, Observable
from baifg.model.graph import Graph

class RewardEstimator(object):
    """Generic reward estimator
    """
    N: NDArray[np.float64]
    mu: NDArray[np.float64]
    confidence: NDArray[np.float64]
    K: int
    informed: bool

    def __init__(self, K: int, informed: bool):
        """Initialize the reward estimator

        Args:
            K (int): number of vertices
            informed (bool): true if it is the informed case
        """
        assert K > 1, "K needs to be greater than 1"
        self.K = K
        self.M = np.zeros(self.K)
        self.mu = np.ones(self.K)
        self.informed = informed

    def update(self, t: int, experience: Experience):
        """Update rewards according to the observations"""
        for obs in experience.observables:
            for u in obs.out_vertex:
                if self.informed or (not self.informed and not np.isclose(obs.observed_value,0)):
                    self.M[u] += 1
                    self.mu[u] = ((self.M[u]-1) * self.mu[u] + obs.observed_value) / self.M[u]
                    
                    self.confidence = np.sqrt(2 * np.log(1 + t) / np.maximum(1, self.M))


