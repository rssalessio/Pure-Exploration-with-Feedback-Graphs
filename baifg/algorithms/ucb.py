from baifg.algorithms.base.base_algorithm import BaseAlg, Experience
from baifg.algorithms.base.graph_estimator import GraphEstimator

class UCB(BaseAlg):
    """ Implements an UCB-like algorithm """

    def __init__(self, graph: GraphEstimator, delta: float):
        super().__init__("UCB", graph, delta)

    
    def sample(self, time: int) -> int:
        """ Sample according to a UCB like strategy """
        
        mu_conf = self.reward.confidence
        g_conf = self.graph.confidence

        m = (self.reward.mu + mu_conf).argmax()
        return (self.graph.G[:,m] + g_conf).argmax()
    
    def _backward_impl(self, experience: Experience):
        pass