import numpy as np
from baifg.model.feedback_graph import FeedbackGraph
from baifg.algorithms.base.graph_estimator import GraphEstimator
from baifg.algorithms.base.reward_estimator import RewardEstimator
from baifg.model.experience import Experience, Observable
from typing import NamedTuple, List, Tuple
from abc import abstractmethod, ABC
from numpy.typing import NDArray


class BaseAlg(ABC):
    """ Base algorithm class """
    NAME: str
    graph: GraphEstimator
    reward: RewardEstimator
    N: NDArray[np.float64]
    K: int
    time: int

    def __init__(self, name: str, graph: GraphEstimator):
        self.NAME = name
        self.graph = graph
        self.reward = RewardEstimator(graph.K, informed=graph.informed)
        self.N = np.zeros(graph.K) 
        self.K = graph.K
        self.time = 1

    @abstractmethod
    def sample(self) -> int:
        raise NotImplementedError("Sample function not imlpemented")

    def backward(self, experience: Experience):
        self.N[experience.vertex] += 1
        self.time += 1
        self.graph.update(self.time, experience)
        self.reward.update(self.time, experience)
        self._backward_impl(experience)

    @abstractmethod
    def _backward_impl(self, experience: Experience):
        raise NotImplementedError("Backward function not implemented")