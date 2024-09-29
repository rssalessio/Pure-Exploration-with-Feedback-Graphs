import numpy as np
from typing import NamedTuple

from baifg.algorithms.base.base_algorithm import BaseAlg, Experience, Observable
from baifg.algorithms.base.graph_estimator import GraphEstimator
from baifg.algorithms.base.reward_estimator import RewardType


class EpsilonGreedyParameters(NamedTuple):
    """ Exploration rate """
    exp_rate: float
    """ If false, runs basic epsilon greedy. If true
        runs the graph aware version
    """
    graph_aware: bool

class EpsilonGreedy(BaseAlg):
    """ Implements an epsilon-greedy algorithm """
    params: EpsilonGreedyParameters

    def __init__(self, graph: GraphEstimator, reward_type: RewardType, delta: float, parameters: EpsilonGreedyParameters):
        super().__init__("Epsilon-greedy", graph, reward_type, delta)
        self.params = parameters

    def sample(self, time: int) -> int:
        """ Sample according to epsilon-greedy strategy. 
            That is, with probability epsilon we sample a random vertex.
            Otherwise we find  the vertex `m` with highest reward
            and then we select the vertex `u` with highest probability G_{u,m}
        """
        if np.random.rand() < self.params.exp_rate:
            return np.random.choice(self.graph.K)
        
        if not self.params.graph_aware:
            m = self.reward.mu.argmax()
            return self.graph.G[:,m].argmax()
        else:
            if np.any(np.isclose(0, self.reward.gaps)):
                return np.random.choice(self.graph.K)
            gaps_inv_sq = 1 / self.reward.gaps ** 2
            p = gaps_inv_sq / gaps_inv_sq.sum(-1)
            return (self.graph.G @ p).argmax()
    
    def _backward_impl(self, experience: Experience):
        pass