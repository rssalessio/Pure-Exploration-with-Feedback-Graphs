import numpy as np
from baifg.model.feedback_graph import FeedbackGraph
from baifg.algorithms.base.graph_estimator import GraphEstimator
from baifg.algorithms.base.reward_estimator import RewardEstimator, RewardType
from baifg.model.experience import Experience, Observable
from baifg.utils.characteristic_time import evaluate_characteristic_time
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
    delta: float

    def __init__(self, name: str, graph: GraphEstimator, reward_type: RewardType, delta: float):
        assert delta > 0, 'delta needs to be strictly positive'
        self.NAME = name
        self.graph = graph
        self.reward = RewardEstimator(graph.K, informed=graph.informed, reward_type=reward_type)
        self.N = np.zeros(graph.K) 
        self.K = graph.K
        self.delta = delta

    @abstractmethod
    def sample(self, time: int) -> int:
        raise NotImplementedError("Sample function not imlpemented")
    
    @property
    def estimated_best_vertex(self) -> int:
        return self.reward.mu.argmax()
    
    @property
    def feedback_graph(self) -> FeedbackGraph:
        return FeedbackGraph(reward_model=self.reward, graph=self.graph)

    @property
    def is_model_regular(self) -> bool:
        """ Returns true if the model is observable and best vertex is unique """
        fg = self.feedback_graph
        return fg.is_best_vertex_unique and fg.graph.is_observable
    
    def should_stop(self, time: int) -> bool:
        if time < self.K or not self.is_model_regular: return False
        beta = np.log((1 + np.log(time)) / self.delta)
        Lt = time / max(1, evaluate_characteristic_time(self.N / self.N.sum(), self.feedback_graph))
        return Lt >= beta

    def backward(self, time: int, experience: Experience):
        self.N[experience.vertex] += 1
        self.graph.update(time, experience)
        self.reward.update(time, experience)
        self._backward_impl(time, experience)

    @abstractmethod
    def _backward_impl(self, time: int, experience: Experience):
        raise NotImplementedError("Backward function not implemented")