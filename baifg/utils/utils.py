import numpy as np
from baifg.model.reward_model import RewardModel
from baifg.model.graph import Graph

def approximate_solution(reward_model: RewardModel, graph: Graph):
    if np.any(np.isclose(0, reward_model.gaps)):
        return np.full(graph.K, 1/graph.K)
    gaps_inv_sq = 1 / reward_model.gaps ** 2
    p = gaps_inv_sq / gaps_inv_sq.sum(-1)
    return graph.G @ p