import numpy as np
from typing import NamedTuple

from baifg.algorithms.base.base_algorithm import BaseAlg, Experience, Observable
from baifg.algorithms.base.graph_estimator import GraphEstimator


class EpsilonGreedyParameters(NamedTuple):
    """ Exploration rate """
    exp_rate: float

class EpsilonGreedy(BaseAlg):
    """ Implements an epsilon-greedy algorithm """
    params: EpsilonGreedyParameters

    def __init__(self, graph: GraphEstimator, parameters: EpsilonGreedyParameters):
        super().__init__("Epsilon-greedy", graph)
        self.params = parameters

    def sample(self) -> int:
        """ Sample according to epsilon-greedy strategy. 
            That is, with probability epsilon we sample a random vertex.
            Otherwise we find  the vertex `m` with highest reward
            and then we select the vertex `u` with highest probability G_{u,m}
        """
        if np.random.rand() < self.params.exp_rate:
            return np.random.choice(self.graph.K)
        
        m = self.reward.mu.argmax()
        return self.graph.G[:,m].argmax()
    
    def _backward_impl(self, experience: Experience):
        pass