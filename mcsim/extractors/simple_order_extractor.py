"""

"""

import numpy as np
import pyzx

from .base_extractor import BaseExtractor
from .mansikka_graph import MansikkaGraph

from mcsim.mcsim_tensor import mcsim_tensorfy
import random

class SimpleOrderExtractor(BaseExtractor):
    def extract(self, graph: pyzx.graph):


        working_graph = MansikkaGraph([k for k in graph.vertices()], graph.edge_set().copy())
        working_graph = working_graph.construct_dual()


        initial_order = [k for k in working_graph.vertices]
        print("initial_order 0:",initial_order)
        if self.params["order"] == 0:
            contraction_order = initial_order
        elif self.params["order"] == 1:
            random.shuffle(initial_order)
            print("initial_order 1:", initial_order)
            contraction_order = initial_order

        return graph.to_matrix(
            my_tensorfy = mcsim_tensorfy,
            contraction_order = contraction_order
        )

