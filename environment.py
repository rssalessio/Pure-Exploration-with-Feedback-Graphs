import numpy as np
import math 
import random
from baifg.model.feedback_graph import FeedbackGraph
from baifg.algorithms.base.base_algorithm import BaseAlg
from baifg.model.experience import Experience, Observable

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def run_experiment(fg: FeedbackGraph, algo: BaseAlg, seed: int):
    set_seed(seed)

    t = 1
    while not algo.should_stop(time=t):
        Vt = algo.sample(t)
        Yt = np.random.binomial(n=1, p=fg.graph.G[Vt])
        Rt = fg.reward_model.sample()
        Zt = Yt * Rt

        observables = [
            Observable(out_vertex=u, in_vertex=Vt, observed_value=Zt[idx], activated=True)
                for idx, u in enumerate(fg.graph.out_neighborhood[Vt])
        ]
        for u in range(fg.K):
            if u not in fg.graph.out_neighborhood[Vt]:
                observables.append(
                    Observable(out_vertex=u, in_vertex=Vt, observed_value=0, activated=False)
                )
            
        experience = Experience(vertex=Vt, observables=observables)
        algo.backward(time=t, experience=experience)
        print(f'---------- Time {t} -------------')
        print(algo.N)
        print(algo.reward.mu)
        print(algo.graph.G)
        print('-------------------')
        t += 1
    return algo.estimated_best_vertex
