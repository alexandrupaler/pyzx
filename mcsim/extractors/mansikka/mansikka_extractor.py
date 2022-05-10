"""

"""

import numpy as np
import pyzx
from typing import Dict, List

from .base_extractor import BaseExtractor
from .graph import Graph
from .treewidth_from_order import find_treewidth_from_order
from .greedy_treewidth_deletion import greedy_treewidth_deletion


class MansikkaExtractor(BaseExtractor):
    def extract(self, graph: pyzx.graph, show_changes=0):
        """
        Model of extractor that tries to take advance of specific distribution of degre/rerows
        :graph: the pyzx graph whose matrix we want to extract
        :show_changes: just for learning can be deleted when merge
        :return: graph matrix
        """

        working_graph = Graph([k for k in graph.vertices()], graph.edge_set())

        dual_graph = working_graph.construct_dual()
        # print("dual vert:", dual_graph.vertices)
        # print("dual edges: ", dual_graph.edges)

        contraction_order = get_contraction_order(
            dual_graph, self.params["m"], self.params["nr_iter"]
        )
        # print("contraction order:", contraction_order)

        return graph.to_matrix()


def get_contraction_order(graph, m, nr_iter=10):

    working_graph = Graph(graph.vertices.copy(), graph.edges.copy())
    initial_order = [k for k in graph.vertices]
    contraction_order = []

    initial_tw = find_treewidth_from_order(working_graph, contraction_order)

    if initial_tw:
        return initial_order

    while nr_iter > 0:
        nr_iter = nr_iter - 1
        reduced_g, reduced_order, tw, removing_order = greedy_treewidth_deletion(
            working_graph, initial_order, m
        )
        for node in removing_order:
            contraction_order.append(node)

        initial_order = reduced_order.copy()
        working_graph = reduced_g
        if tw == 1:
            break

    for node in reduced_order:
        contraction_order.append(node)

    return contraction_order
