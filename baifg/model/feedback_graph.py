import numpy as np
from baifg.model.graph import Graph
from baifg.model.reward_model import RewardModel


class FeedbackGraph(object):
    vertices: set
    K: int

    def __init__(self, reward_model: RewardModel, graph: Graph):
        assert reward_model.K == graph.K, "The reward model and the graph" \
             + " need to have the same number of vertices"
        assert graph.K > 1, "There needs to be more than 1 vertex"
        assert len(graph.observable_vertices) == graph.K, "The graph is not observable"
        assert sum(np.isclose(reward_model.mu,reward_model.mustar)) == 1, "The best vertex is not unique"

        self.reward_model = reward_model
        self.graph = graph
        self.K = graph.K

    
       

