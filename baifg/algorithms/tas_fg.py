import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray
from baifg.algorithms.base.base_algorithm import BaseAlg, Experience, Observable
from baifg.algorithms.base.graph_estimator import GraphEstimator
from baifg.algorithms.base.reward_estimator import RewardType
from baifg.utils.utils import approximate_solution
from baifg.utils.characteristic_time import compute_characteristic_time, CharacteriticTimeSolution


class TaSFGParameters(NamedTuple):
    """ Update frequency of the allocation rate """
    update_frequency: int

class TaSFG(BaseAlg):
    """ Implements Track and Stop for Feedback Graphs """
    params: TaSFGParameters
    avg_alloc: NDArray[np.float64]
    alloc: NDArray[np.float64]

    def __init__(self, graph: GraphEstimator, reward_type: RewardType, delta: float, parameters: EpsilonGreedyParameters):
        super().__init__("TaSFG", graph, reward_type, delta)
        self.params = parameters
        self.avg_alloc = np.full(self.K, 1/self.K)
        self.alloc = np.full(self.K, 1/self.K)

    def sample(self, time: int) -> int:
        """ Sample an action """
        # Forced exploration
        St = self.N < np.sqrt(time) - self.K/2
        if np.any(St):
            return np.argmax(St)

        # Tracking
        return (self.N / time - self.avg_alloc).argmin()

    
    def _backward_impl(self, time: int, experience: Experience):
        if self.is_model_regular and time % self.params.update_frequency == 0:
            sol = compute_characteristic_time(
                self.feedback_graph
            )

            self.alloc = sol.wstar
            self.avg_alloc = ((time - 1) * self.avg_alloc + self.alloc) / time 
