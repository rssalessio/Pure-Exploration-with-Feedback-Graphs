import cvxpy as cp
import numpy as np

from typing import NamedTuple
from numpy.typing import NDArray
from baifg.model.reward_model import RewardType
from baifg.model.feedback_graph import FeedbackGraph

class CharacteriticTimeSolution(NamedTuple):
    value: float
    wstar: NDArray[np.float64]
    mstar: NDArray[np.float64]

def compute_characteristic_time(fg: FeedbackGraph) -> CharacteriticTimeSolution:
    if fg.reward_model.reward_type == RewardType.GAUSSIAN:
        return compute_characteristic_time_gaussian(fg)
    raise Exception("Only gaussian rewards are implemented")

def evaluate_characteristic_time(w: NDArray[np.float64], fg: FeedbackGraph) -> float:
    if fg.reward_model.reward_type == RewardType.GAUSSIAN:
        return evaluate_characteristic_time_gaussian(w, fg)
    raise Exception("Only gaussian rewards are implemented")
    

def compute_characteristic_time_gaussian(fg: FeedbackGraph) -> CharacteriticTimeSolution:
    astar = fg.reward_model.astar
    var = fg.reward_model.sigma ** 2

    w = cp.Variable(fg.K, nonneg=True)
    m = cp.Variable(fg.K, nonneg=True)
    p = cp.Variable(1, nonneg=True)

    constraints = [cp.sum(w) == 1, m == w @ fg.graph.G]

    for u in range(fg.K):
        if u != astar:
            gap_u = fg.reward_model.gaps[u] ** 2
            constraints.append(p >= (cp.inv_pos(m[u]) + cp.inv_pos(m[astar])) * (2 * var / gap_u))
    
    obj = cp.Minimize(p)
    problem = cp.Problem(obj, constraints)
    sol = problem.solve(solver=cp.GUROBI, reoptimize=True)
    return CharacteriticTimeSolution(sol, w.value, m.value)


def evaluate_characteristic_time_gaussian(w: NDArray[np.float64], fg: FeedbackGraph) -> float:
    astar = fg.reward_model.astar
    var = fg.reward_model.sigma ** 2
    gaps = fg.reward_model.gaps ** 2
    w = np.array(w)
    m = np.zeros_like(w)

    assert np.isclose(w.sum(), 1), "w does not sum up to 1"
    assert len(w) == fg.K, f"w needs to have {fg.K} elements"

    
    m = np.array(
        [np.sum([w[v] * fg.graph.G[v,u]  for v in fg.graph.get_in_neighborhood(u)]) for u in range(fg.K)]
    )

    mstar = m[astar]
    if not np.all(m > 0):
        print(fg.graph.G)
    assert np.all(m > 0), 'some elements in m are 0'
    idxs = np.arange(fg.K)
    gaps = gaps[idxs != astar]
    
    values = (1/m[idxs != astar]  + 1/mstar) * 2 * var / gaps
    return np.max(values)
