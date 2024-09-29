from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from baifg.model.experience import Experience, Observable
from baifg.model.graph import Graph
from baifg.model.reward_model import RewardModel, RewardType

class RewardEstimator(RewardModel):
    """Generic reward estimator
    """
    N: NDArray[np.float64]
    confidence: NDArray[np.float64]
    informed: bool
    sigma = 1.0

    def __init__(self, K: int, informed: bool, reward_type: RewardType):
        """Initialize the reward estimator

        Args:
            K (int): number of vertices
            informed (bool): true if it is the informed case
        """
        assert K > 1, "K needs to be greater than 1"
        super().__init__(np.ones(K))
        self.reward_type = reward_type
        self.K = K
        self.M = np.zeros(self.K)
        self.informed = informed

    def update(self, t: int, experience: Experience):
        """Update rewards according to the observations"""
        mu = self.mu.copy()
        for obs in experience.observables:
            u = obs.out_vertex
            activated = obs.activated if self.informed else not np.isclose(obs.observed_value,0)
            if activated:
                self.M[u] += 1
                mu[u] = ((self.M[u]-1) * mu[u] + obs.observed_value) / self.M[u]
        
        self.confidence = np.sqrt(2 * np.log(1 + t) / np.maximum(1, self.M))
        self.update_reward(mu)


