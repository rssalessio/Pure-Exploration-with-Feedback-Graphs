import numpy as np
from numpy.typing import NDArray
from baifg.model.feedback_graph import FeedbackGraph, Graph
from baifg.model.reward_model import GaussianRewardModel
from typing import NamedTuple


def make_symmetric_graph(p: float = 0.5, p_prime: float = 0.5, q: float = 1) -> FeedbackGraph:
    """Creates a symmetric graph with 3 vertices.

    Args:
        p (float, optional): Self loop probabilities. Defaults to 0.5.
        p_prime (float, optional): Probability of observing the bet vertex. Defaults to 0.5.
        q (float, optional): Probability of observing a non-optimal vertex. Defaults to 1.

    Returns:
        FeedbackGraph: returns a feedback graph object
    """
    G = np.array([
        [p, q, p_prime],
        [0., 0, 0],
        [p, q, p_prime]
    ])
    R = GaussianRewardModel([0, 1, 0])
    G = Graph(G)
    fg = FeedbackGraph(R, G)
    return fg

def make_loopystar_graph(p: float, q: float, r: float, K: int, a1_optimal: bool = False) -> FeedbackGraph:
    """Create a loopystar graph with rewards 0.5 for non-optimal arm
       and reward 1 for the optimal arm.

    Args:
        p (float): probability of G[0,astar] and max(0,1-2p) for G[i,i] with i!=0 and i!=astar;
                   1-p is the probability of G[astar,astar]
        q (float): probability of G[0,0]
        r (float): probability of G[0,i] with i != astar
        K (int): number of vertices
        a1_optimal (bool): if true, sets the first vertex to be the optimal one, otherwise
                           it's the last vertex. Defaults to False.

    Returns:
        FeedbackGraph: Feedback graph object
    """
    G = np.zeros((K,K))
    G[0] =  [q]+ [r] * (K-2) + [p]
    G[1:,1:] = np.eye(K-1) * max(0, 1-2*p)
    G[-1,-1] = 1-p

    rewards = [0.5] * (K-1) + [1] if a1_optimal is False else [1]+[0.5] * (K-1)
    R = GaussianRewardModel(rewards)

    G = Graph(G)
    fg = FeedbackGraph(R, G)
    return fg

def make_loopless_clique(p: float, mu: NDArray[np.float64]) -> FeedbackGraph:
    """Creates a loopless clique where G_{u,u} = 0 and
       G_{u,v} = p / u for every v!=u and v uneven, where u is the vertex index, or
       G_{u,v} = 1-(p / u) for every v!=u and v even.

    Args:
        p (float): edge probability parameter
        mu (NDArray[float]): average rewards for each vertex

    Returns:
        FeedbackGraph: Return a feedback graph object
    """
    mu = np.array(mu).flatten()
    K = len(mu)
    G = np.zeros((K,K))
    idxs = np.arange(K)
    for i in range(K):
        G[i,:] = p / (i+1)
        G[i, idxs % 2==0] = 1 - (p / (i + 1))
    
    G[idxs, idxs] = 0

    R = GaussianRewardModel(mu)
    G = Graph(G)
    fg = FeedbackGraph(R, G)
    return fg



def make_ring_graph(p: float, mu: NDArray[np.float64]) -> FeedbackGraph:
    """Creates a ring graph where G_{u,v} = p if v= u+1 (0 if u+1 >=K)
        and G_{u,v} = 1-p if v=u-1 (K-1 if u-1<0).

        All other edges have 0 probability

    Args:
        p (float): edge probability parameter
        mu (NDArray[float]): average rewards for each vertex

    Returns:
        FeedbackGraph: Return a feedback graph object
    """
    mu = np.array(mu).flatten()
    K = len(mu)
    G = np.zeros((K,K))
    for i in range(K):
        left = i-1 if i > 0 else K-1
        right = i+1 if i < K-1 else 0
        G[i, left] = 1-p
        G[i, right] = p

    R = GaussianRewardModel(mu)
    G = Graph(G)
    fg = FeedbackGraph(R, G)
    return fg


