import numpy as np
import pandas as pd
import multiprocessing as mp
import lzma
import pickle
import os
from copy import deepcopy
from environment import run_experiment, RunParameters, RunStatistics
from baifg.model.feedback_graph import FeedbackGraph
from baifg.model.reward_model import GaussianRewardModel, RewardType
from baifg.algorithms.eps_greedy import EpsilonGreedy, EpsilonGreedyParameters
from baifg.algorithms.ucb import UCB
from baifg.algorithms.exp3g import Exp3G, Exp3GParameters
from baifg.algorithms.tas_fg import TaSFG, TaSFGParameters
from baifg.algorithms.base.graph_estimator import GraphEstimator
from baifg.algorithms.base.base_algorithm import BaseAlg
from baifg.utils.graphs import make_loopless_clique, make_loopystar_graph, make_ring_graph
from baifg.utils.characteristic_time import compute_characteristic_time
from itertools import product
from typing import List, NamedTuple, Tuple, Dict, Callable, Type
from tqdm import tqdm
from datetime import datetime


def make_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
        return True
    return False



def make_model(algo_name: Type[BaseAlg], algo_params: Dict[str, float | int | bool],
               K: int, fg: FeedbackGraph, delta: float, informed: bool) -> BaseAlg:
    """ """
    if algo_name == EpsilonGreedy:
        return EpsilonGreedy(
            GraphEstimator.optimistic_graph(K, informed=informed, known=False),
            fg.reward_model.reward_type,
            delta=delta,
            parameters=EpsilonGreedyParameters(exp_rate=0.3, **algo_params))
    elif algo_name == UCB:
        return UCB(
            GraphEstimator.optimistic_graph(K, informed=informed, known=False),
            reward_type=fg.reward_model.reward_type,
            delta=delta)
    elif algo_name == Exp3G:
        return Exp3G(
            GraphEstimator.optimistic_graph(K, informed=informed, known=False),
            reward_type=fg.reward_model.reward_type,
            delta=delta,
            parameters= Exp3GParameters(exp_rate=0.3, learn_rate=1/(2*K), **algo_params)
        )
    elif algo_name == TaSFG:
        return TaSFG(
            GraphEstimator.optimistic_graph(K, informed=informed, known=False),
            reward_type=fg.reward_model.reward_type,
            delta=delta, parameters=TaSFGParameters(update_frequency=K*5, **algo_params)
        )
    raise Exception('Algorithm not found')

def run_exp(seed: int, env: RunParameters,  algorithms: Tuple[BaseAlg, NamedTuple]) -> Dict[str, List[RunStatistics]]:
    results: Dict[str, List[RunStatistics]] = {}
  
    for algo, algo_params in algorithms:
        algo = make_model(algo_name=algo, algo_params=algo_params,
                        K=env.fg.K, fg=env.fg, delta=env.delta, informed=env.informed)
        res = run_experiment(fg=env.fg, algo=algo, seed=seed)

        if algo.NAME not in results:
            results[algo.NAME] = []
        results[algo.NAME].append(res)
    return results




if __name__ == '__main__':
    NUM_PROCESSES = 25
    Nsims = 100
    envs: List[RunParameters] = []
    Kvalues = [5, 10, 15]
    delta = np.exp(-np.linspace(1, 7, 6))
    PATH =  f"./data/{datetime.today().strftime('%Y-%m-%d-%H-%M')}/"

    make_dir(PATH)
    algorithms = [
        (EpsilonGreedy, {'information_greedy': False}),
        (EpsilonGreedy, {'information_greedy': True}),
        (Exp3G, {}),
        (TaSFG, {'heuristic': False}),
        (TaSFG, {'heuristic': True}),
        (UCB, {})
    ]

    for K, delta, informed in product(Kvalues, delta, [False]):
        envs.append(
            RunParameters('Loopless clique', f'p=0.5, K={K}, delta={np.log(1/delta).round(2)}, informed={informed}', delta, informed=True,
                        known=False, fg=make_loopless_clique(p=0.5, mu=np.linspace(0, 1, K)),
                        results = {})
        )
        envs.append(
            RunParameters('Loopystar', f'p=0.2, q=0.25, r=0.25 K={K}, delta={np.log(1/delta).round(2)}, informed={informed}', delta, informed=True,
                        known=False, fg=make_loopystar_graph(p=0.2, q=0.25, r=0.25, K=K),
                        results = {})
        )
        envs.append(
            RunParameters('Ring', f'p=0.3 K={K}, delta={np.log(1/delta).round(2)}, informed={informed}', delta, informed=True,
                        known=False, fg=make_ring_graph(p=0.3, mu=np.linspace(0, 1, K)),
                        results = {})
        )
        q=0.25
        r = 0.25*(1-2*q)/(K-1)
        envs.append(
            RunParameters('Loopystar (hard)', f'p=0, q=0.25, r={np.round(r,2)} K={K}, delta={np.log(1/delta).round(2)}, informed={informed}', delta, informed=True,
                        known=False, fg=make_loopystar_graph(p=0., q=q, r=r, K=K, a1_optimal=True),
                        results = {})
        )

    
    df = pd.DataFrame({},  columns =  ["env", "K", "seed", "algorithm", "delta", "stopping_time", "identified_optimal_arm", "characteristic_time"])

    with mp.Pool(NUM_PROCESSES) as pool:
        for env in envs:
            print(f'Running {env.name} - {env.description}')
            df_env = pd.DataFrame({},  columns =  ["env", "K", "seed", "algorithm", "delta", "stopping_time", "identified_optimal_arm","characteristic_time"])

            results = pool.starmap(run_exp, [(n, deepcopy(env), algorithms) for n in range(Nsims)])

            sol = compute_characteristic_time(env.fg)
            for res in results:
                for algo_name in res.keys():
                    if algo_name not in env.results:
                        env.results[algo_name] = []
                    env.results[algo_name].extend(res[algo_name])

                    for n, run_in_res in enumerate(res[algo_name]):
                        df.loc[len(df.index)] = [
                            env.name, env.fg.K, n, algo_name, env.delta, run_in_res.stopping_time, 
                            run_in_res.estimated_best_vertex == env.fg.reward_model.astar, sol.value]
                        df_env.loc[len(df.index)] = [
                            env.name, env.fg.K, n, algo_name, env.delta, run_in_res.stopping_time, 
                            run_in_res.estimated_best_vertex == env.fg.reward_model.astar, sol.value]


            filename = f'{PATH}/{env.name}_{env.description}.lzma'
            with open(filename, 'wb') as f:
                pickle.dump({'df': df_env, 'env': env}, f)

    filename = f'{PATH}/full_data.lzma'
    with open(filename, 'wb') as f:
        pickle.dump({'df': df, 'env': envs}, f)
    
        