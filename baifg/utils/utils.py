import numpy as np
from multipledispatch import dispatch
from numpy.typing import NDArray
from baifg.model.reward_model import RewardModel
from baifg.model.graph import Graph
from baifg.model.feedback_graph import FeedbackGraph

@dispatch(RewardModel, Graph)
def approximate_solution(reward_model: RewardModel, graph: Graph, normalize: bool = True) -> NDArray[np.float64]:
    """Compute an approximate solution of w* as w*=Gy, with y_u propto 1/Delta_u^2
    

    Args:
        reward_model (RewardModel): Reward model of the graph
        graph (Graph): Graph
        normalize (bool, optional): normalize w to belong to the simplex. Defaults to True.

    Returns:
        NDArray[np.float64]: Approximate allocation w*
    """
    if np.any(np.isclose(0, reward_model.gaps)):
        return np.full(graph.K, 1/graph.K)
    gaps_inv_sq = 1 / (reward_model.gaps ** 2)
    p = gaps_inv_sq
    w = graph.G @ p

    if normalize:
        w = np.abs(w)
        w /= np.sum(w)
    return w

@dispatch(FeedbackGraph)
def approximate_solution(fg: FeedbackGraph, normalize: bool = True) -> NDArray[np.float64]:
    """Compute an approximate solution of w* as w*=Gy, with y_u propto 1/Delta_u^2

    Args:
        fg (FeedbackGraph): feedback graph model
        normalize (bool, optional): normalize w to belong to the simplex. Defaults to True.

    Returns:
        NDArray[np.float64]: Approximate allocation w*
    """
    return approximate_solution(fg.reward_model, fg.graph, normalize=normalize)