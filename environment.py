import numpy as np
import math 
import random
from baifg.model.feedback_graph import FeedbackGraph
from baifg.algorithms.base.base_algorithm import BaseAlg
from baifg.model.experience import Experience, Observable
from typing import NamedTuple, Dict, List

class RunStatistics(NamedTuple):
    estimated_best_vertex: int
    stopping_time: int


class RunParameters(NamedTuple):
    name: str
    description: str
    delta: float
    informed: bool
    known: bool
    fg: FeedbackGraph
    results: Dict[str, List[RunStatistics]]




def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def run_experiment(fg: FeedbackGraph, algo: BaseAlg, seed: int):
    set_seed(seed)

    t = 1
    while not algo.should_stop(time=t):
        Vt = algo.sample(t)
        Yt = np.random.binomial(n=1, p=fg.graph.G[Vt]).astype(np.float64)
        Rt = fg.reward_model.sample()
        Zt = Yt * Rt

        observables = [
            Observable(out_vertex=u, in_vertex=Vt, observed_value=Zt[u], activated=Yt[u] > 0)
                for u in range(fg.K)
        ]

        experience = Experience(vertex=Vt, observables=observables)
        algo.backward(time=t, experience=experience)
        t += 1
    return RunStatistics(algo.estimated_best_vertex, stopping_time=t)
