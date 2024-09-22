import numpy as np
from numpy.typing import NDArray
from enum import Enum

class RewardType(Enum):
    NONE = "None",
    GAUSSIAN = "Gaussian"

class RewardModel(object):
    reward_type: RewardType = RewardType.NONE
    K: int
    mu: NDArray[np.float64]
    gaps: NDArray[np.float64]
    astar: int
    mustar: float
    
    def __init__(self, mu: NDArray[np.float64]):
        assert len(mu.flatten().shape) == 1, "mu needs to be a 1-D vector"
        self.mu = mu.flatten().copy()
        self.K = len(self.mu)

        self.astar = np.argmax(self.mu)
        self.mustar = self.mu[self.astar]
        self.gaps = self.mu - self.mustar


class GaussianRewardModel(RewardModel):
    reward_type: RewardType = RewardType.GAUSSIAN
    sigma: float
    
    def __init__(self, mu: NDArray[np.float64], sigma: float = 1.0):
        super().__init__(mu)
        assert sigma > 0, "Sigma needs to be strictly positive"
        self.sigma = sigma


    
    
    