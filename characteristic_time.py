import cvxpy as cp
import numpy as np

from reward_model import RewardType
from feedback_graph import FeedbackGraph

def compute_characteristic_time(fg: FeedbackGraph):
    if fg.reward_model.reward_type == RewardType.GAUSSIAN:
        return compute_characteristic_time_gaussian(fg)
    raise Exception("Only gaussian rewards are implemented")
    

def compute_characteristic_time_gaussian(fg: FeedbackGraph):
    astar = fg.reward_model.astar
    var = fg.reward_model.sigma ** 2

    w = cp.Variable(fg.K, nonneg=True)
    m = cp.Variable(fg.K, nonneg=True)
    p = cp.Variable(1, nonneg=True)

    constraints = [cp.sum(w) == 1]

    for u in range(fg.K):
        constraints.append(
            m[u] == cp.sum([w[v] * fg.graph.G[v,u]  for v in fg.graph.get_in_neighborhood(u)])
        )

        if u != astar:
            gap_u = fg.reward_model.gaps[u] ** 2
            constraints.append(p >= (cp.inv_pos(m[u]) + cp.inv_pos(m[astar])) * (2 * var / gap_u))
    
    obj = cp.Minimize(p)
    problem = cp.Problem(obj, constraints)
    sol = problem.solve()
    return (sol, w.value, m.value, p.value)


        