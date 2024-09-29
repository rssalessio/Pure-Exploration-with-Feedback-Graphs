import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from baifg.model.feedback_graph import FeedbackGraph
from baifg.algorithms.base.base_algorithm import BaseAlg, Experience


class Exp3GParameters(NamedTuple):
    learn_rate: float
    exp_rate: float

class Exp3G(BaseAlg):
    """ Implements the EXP3.G algorithm (@see https://arxiv.org/pdf/1502.07617) """
    params: Exp3GParameters
    q: NDArray[np.float64]
    p: NDArray[np.float64]

    def __init__(self, fg: FeedbackGraph, parameters: Exp3GParameters, delta: float):
        super().__init__("EXP3.G", fg, delta)
        self.params = parameters
    
        self.q = np.ones(self.K) / self.K
        self.p = np.ones(self.K) / self.K
    
    def sample(self, time: int) -> int:
        self.p = (1 - self.params.exp_rate) * self.q + self.params.exp_rate / self.K
        return np.random.choice(self.K, p=self.p)
    
    def backward(self, experience: Experience):
        losses = np.zeros(self.K)
        for obs in experience.observables:
            assert experience.vertex == obs.in_vertex, "Experience vertex and in_vertex do not coincide"

            Nin_out = self.graph.in_neighborhood[obs.out_vertex]
            Pout = self.p[Nin_out].sum()
            losses[obs.out_vertex] = -obs.observed_value / Pout
        qtemp = self.q * np.exp(- self.params.learn_rate * losses)
        self.q = qtemp / qtemp.sum()

    