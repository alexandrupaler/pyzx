"""

"""

import numpy as np
import pyzx
from typing import Dict, List

from mcsim import BaseExtractor


class MansikkaExtractor(BaseExtractor):
    def distribution_f(self, params) -> List[float]:
        return simple_distribution(params)

    def extract(self, graph: pyzx.graph, show_changes=0):
        """
        Model of extractor that tries to take advance of specific distribution of degre/rerows
        :graph: the pyzx.graph whose matrix we want to extract
        :show_changes: just for learning can be deleted when mrge
        :return: graph matrix
        """

        # deserialization of parameters#
        tolerance = self.params["tolerance"]
        ##

        if show_changes:
            graph_info(graph, show_changes)

        dist = self.distribution_f(self.params)
        graph = fill_degree(graph, dist, tolerance)

        if show_changes:
            graph_info(graph, show_changes)

        return graph.to_matrix()


def fill_degree(graph: pyzx.graph, contraction_order: List[float]) -> pyzx.graph:
    """
     This function rearange the vertices of graph on rows to fit dist
    :graph: pyzx the graph whose nodes we want to rearrange to match distribution.
    :contraction_order: degree distribution
    :return: the pyzx graph with the nodes rearranged on the rows.
    """

    vertices = graph.vertices()

    return 1
