import numpy as np
from numpy.typing import NDArray
from baifg.model.feedback_graph import FeedbackGraph, Graph
from baifg.model.reward_model import GaussianRewardModel

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

def make_loopystar_graph(p: float, q: float, r: float, K: int) -> FeedbackGraph:
    """Create a loopystar graph with rewards 0.5 for non-optimal arm
       and reward 1 for the optimal arm.

    Args:
        p (float): probability of G[0,astar] and max(0,1-2p) for G[i,i] with i!=0 and i!=astar;
                   1-p is the probability of G[astar,astar]
        q (float): probability of G[0,0]
        r (float): probability of G[0,i] with i != astar
        K (int): number of vertices

    Returns:
        FeedbackGraph: Feedback graph object
    """
    G = np.zeros((K,K))
    G[0] =  [q]+ [r] * (K-2) + [p]
    G[1:,1:] = np.eye(K-1) * max(0, 1-2*p)
    G[-1,-1] = 1-p

    R = GaussianRewardModel([0.5] * (K-1) + [1])

    G = Graph(G)
    fg = FeedbackGraph(R, G)
    return fg

def make_loopless_clique(p: float, mu: NDArray[np.float64]) -> FeedbackGraph:
    """Creates a loopless clique where even vertices have outgoing edges
    with probability p. non-even vertices have outgoing edges with probability 1-p.

    Args:
        p (float): edge probability. Even vertices have outgoing edges
                   with probability p. non-even vertices have outgoing edges
                   with probability 1-p.
        mu (NDArray[float]): average rewards for each 

    Returns:
        FeedbackGraph: Return a feedback graph object
    """
    mu = np.array(mu).flatten()
    K = len(mu)
    G = np.zeros((K,K))
    idxs = np.arange(K)
    G[:] =  np.array([1-p] * K)
    G[idxs % 2 == 0] = np.array([p] * K)

    R = GaussianRewardModel(mu)
    G = Graph(G)
    fg = FeedbackGraph(R, G)
    return fg

