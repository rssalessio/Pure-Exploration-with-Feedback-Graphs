import numpy as np
from typing import NamedTuple

from baifg.algorithms.base.base_algorithm import BaseAlg, Experience, Observable
from baifg.algorithms.base.graph_estimator import GraphEstimator
from baifg.algorithms.base.reward_estimator import RewardType
from baifg.utils.utils import approximate_solution


class EpsilonGreedyParameters(NamedTuple):
    """ Exploration rate """
    exp_rate: float
    """ If false, runs basic epsilon greedy. If true
        runs the information greedy version
    """
    information_greedy: bool

class EpsilonGreedy(BaseAlg):
    """ Implements an epsilon-greedy algorithm """
    params: EpsilonGreedyParameters

    def __init__(self, graph: GraphEstimator, reward_type: RewardType, delta: float, parameters: EpsilonGreedyParameters):
        super().__init__("Epsilon-greedy" + ("" if parameters.information_greedy is False else " IG"),
                         graph, reward_type, delta)
        self.params = parameters

    def sample(self, time: int) -> int:
        """ Sample according to epsilon-greedy strategy. 
            That is, with probability epsilon we sample a random vertex.
            Otherwise we find  the vertex `m` with highest reward
            and then we select the vertex `u` with highest probability G_{u,m}
        """
        if np.random.rand() < self.params.exp_rate:
            return np.random.choice(self.graph.K)
        
        if not self.params.information_greedy:
            m = self.reward.mu.argmax()
            return self.graph.G[:,m].argmax()
        else:
            if np.any(np.isclose(0, self.reward.gaps)):
                return np.random.choice(self.graph.K)
            p = approximate_solution(self.reward, self.graph)
            return p.argmax()
    
    def _backward_impl(self, time: int, experience: Experience):
        pass