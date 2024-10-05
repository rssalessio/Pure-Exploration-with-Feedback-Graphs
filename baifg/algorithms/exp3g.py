import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from baifg.algorithms.base.base_algorithm import BaseAlg, Experience, RewardType, GraphEstimator


class Exp3GParameters(NamedTuple):
    learn_rate: float
    exp_rate: float

class Exp3G(BaseAlg):
    """ Implements the EXP3.G algorithm (@see https://arxiv.org/pdf/1502.07617) """
    params: Exp3GParameters
    q: NDArray[np.float64]
    p: NDArray[np.float64]

    def __init__(self,graph: GraphEstimator, reward_type: RewardType, delta: float,  parameters: Exp3GParameters):
        super().__init__(graph, reward_type, delta)
        self.params = parameters
    
        self.q = np.ones(self.K) / self.K
        self.p = np.ones(self.K) / self.K

    @property
    def NAME(self) -> str:
        return "EXP3.G"
    
    def sample(self, time: int) -> int:
        self.p = (1 - self.params.exp_rate) * self.q + self.params.exp_rate / self.K
        return np.random.choice(self.K, p=self.p)
    
    def _backward_impl(self, time: int, experience: Experience):
        losses = np.zeros(self.K)
        for obs in experience.observables:
            Nin_out = self.graph.in_neighborhood[obs.out_vertex]
            Pout = self.p[list(Nin_out)].sum()
            losses[obs.out_vertex] = -obs.observed_value / Pout
        qtemp = self.q * np.exp(- self.params.learn_rate * losses)
        self.q = qtemp / qtemp.sum()

    