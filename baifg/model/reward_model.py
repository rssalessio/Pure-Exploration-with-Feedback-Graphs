import numpy as np
from numpy.typing import NDArray
from enum import Enum
from typing import List

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
        self.update_reward(mu)
        self.K = len(self.mu)

    def update_reward(self, mu: NDArray[np.float64]):
        mu = np.array(mu)
        assert len(mu.flatten().shape) == 1, "mu needs to be a 1-D vector"
        self.mu = mu.flatten().copy()
        self.astar = np.argmax(self.mu)
        self.mustar = self.mu[self.astar]
        self.gaps = self.mustar - self.mu 
        idxs = self.gaps > 0
        if np.any(idxs):
            self.gaps[self.astar] = self.gaps[idxs].min()

    def sample(self, idxs: List[int]) -> NDArray[np.float64]:
        raise NotImplementedError('Sample function not implemented')

class GaussianRewardModel(RewardModel):
    reward_type: RewardType = RewardType.GAUSSIAN
    sigma: float
    
    def __init__(self, mu: NDArray[np.float64], sigma: float = 1.0):
        super().__init__(mu)
        assert sigma > 0, "Sigma needs to be strictly positive"
        self.sigma = sigma

    def sample(self) -> NDArray[np.float64]:
        return np.random.normal(self.mu, scale=self.sigma)
    