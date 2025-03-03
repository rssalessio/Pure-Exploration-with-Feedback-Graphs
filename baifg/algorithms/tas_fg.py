import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray
from baifg.algorithms.base.base_algorithm import BaseAlg, Experience, Observable
from baifg.algorithms.base.graph_estimator import GraphEstimator
from baifg.algorithms.base.reward_estimator import RewardType
from baifg.utils.utils import approximate_solution
from baifg.utils.characteristic_time import compute_characteristic_time, CharacteriticTimeSolution
from baifg.utils.utils import approximate_solution

class TaSFGParameters(NamedTuple):
    """ Update frequency of the allocation rate """
    update_frequency: int
    heuristic: bool

class TaSFG(BaseAlg):
    """ Implements Track and Stop for Feedback Graphs """
    params: TaSFGParameters
    avg_alloc: NDArray[np.float64]
    alloc: NDArray[np.float64]
    num_updates: int

    def __init__(self, graph: GraphEstimator, reward_type: RewardType, delta: float, parameters: TaSFGParameters):
        super().__init__(graph, reward_type, delta)
        self.params = parameters
        self.avg_alloc = np.full(self.K, 1/self.K)
        self.alloc = np.full(self.K, 1/self.K)
        self.num_updates = 0

    @property
    def NAME(self) -> str:
        return 'TaS-FG' if self.params.heuristic is False else 'TaS-FG Heur.'

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

            if self.params.heuristic:
                wstar = approximate_solution(self.feedback_graph, normalize=True)
            else:
                try:
                    sol = compute_characteristic_time(
                        self.feedback_graph
                    )
                    wstar = sol.wstar
                except:
                    return

            if wstar is None:
                return
            self.alloc = wstar
            self.avg_alloc = (self.num_updates * self.avg_alloc + self.alloc) / (self.num_updates + 1)
            self.num_updates += 1 
