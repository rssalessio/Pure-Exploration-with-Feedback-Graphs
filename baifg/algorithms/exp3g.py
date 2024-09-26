import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

from baifg.model.feedback_graph import FeedbackGraph
from baifg.algorithms.base_algorithm import BaseAlg


class Exp3GParameters(NamedTuple):
    learn_rate: float
    exp_rate: float

class Exp3G(BaseAlg):
    params: Exp3GParameters
    q: NDArray[np.float64]

    def __init__(self, fg: FeedbackGraph, parameters: Exp3GParameters):
        super().__init__("EXP3.G", fg)
        self.params = parameters
    
        self.q = np.ones(self.fg.K) / self.fg.K
    
    def sample(self) -> int:
        p = (1 - self.params.exp_rate) * self.q + self.params.exp_rate / self.fg.K
        return np.random.choice(self.fg.K, p=p)
    
    def backward(self, )

    