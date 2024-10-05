from baifg.algorithms.base.base_algorithm import BaseAlg, Experience, RewardType
from baifg.algorithms.base.graph_estimator import GraphEstimator

class UCB(BaseAlg):
    """ Implements an UCB-like algorithm """

    def __init__(self, graph: GraphEstimator, reward_type: RewardType, delta: float):
        super().__init__("UCB-FG", graph, reward_type, delta)

    
    def sample(self, time: int) -> int:
        """ Sample according to a UCB like strategy """
        
        mu_conf = self.reward.confidence
        g_conf = self.graph.confidence

        mu = self.reward.mu + mu_conf
        G = self.graph.G + g_conf
        m = mu.argmax()
        return G[:,m].argmax()
    
    def _backward_impl(self, time: int, experience: Experience):
        pass