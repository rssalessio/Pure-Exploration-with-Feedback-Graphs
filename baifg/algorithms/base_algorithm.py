from baifg.model.feedback_graph import FeedbackGraph
from typing import NamedTuple, Set

class Experience(NamedTuple):
    vertex: int
    Z: Set[float]


class BaseAlg(object):
    NAME: str
    fg: FeedbackGraph

    def __init__(self, name: str, fg: FeedbackGraph):
        self.NAME = name
        self.fg = fg
