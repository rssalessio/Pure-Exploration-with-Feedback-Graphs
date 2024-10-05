from typing import NamedTuple
from baifg.algorithms.base.base_algorithm import BaseAlg, Experience, RewardType
from baifg.algorithms.base.graph_estimator import GraphEstimator


class UCBParameters(NamedTuple):
    """ Exploration rate """
    greedy_wrt_connected_edges: bool


class UCB(BaseAlg):
    """ Implements an UCB-like algorithm """

    def __init__(self, graph: GraphEstimator, reward_type: RewardType, delta: float, parameters: UCBParameters):
        super().__init__(graph, reward_type, delta)
        self.params = parameters

    @property
    def NAME(self) -> str:
        return "UCB-FG-E" if self.params.greedy_wrt_connected_edges else "UCB-FG-V"
    
    def sample(self, time: int) -> int:
        """ Sample according to a UCB like strategy """
        
        mu_conf = self.reward.confidence
        g_conf = self.graph.confidence
        mu = self.reward.mu + mu_conf
        G = self.graph.G + g_conf
        if self.params.greedy_wrt_connected_edges:
            return (G @ mu).argmax()

        
        return G[:, mu.argmax()].argmax()
    
    def _backward_impl(self, time: int, experience: Experience):
        pass