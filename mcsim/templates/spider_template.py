"""
    A simple template with calls for performing optimisations of spiders
"""

from typing import List

import pyzx
from pyzx.rules import MatchPivotType
from pyzx.graph.base import BaseGraph, VT, ET

from mcsim.optimizers.base_optimizer import BaseOptimizer
from mcsim.mcsim_pyzx_simplify import (
    match_spider,
    spider,
    simp,
    unspider,
    flip_spider_type,
    phase_free_simp,
)


class SpiderOptimizer(BaseOptimizer):
    """
    The spider optimizer is searching to minimize the average
    degree/arity of the spiders. The hope is to reduce CNOTs.
    """

    @staticmethod
    def match_min_degree(g: BaseGraph[VT, ET], min_degree) -> List[MatchPivotType[VT]]:
        """
        Matches vertices of degree max param
        """
        ret = []

        all_vertices = []

        for vertex in g.vertices():
            degree = g.vertex_degree(vertex)

            if degree >= min_degree:

                # Param1: Uncomment if random size cut
                # import random
                # cut = random.randint(0, degree - 2)
                cut = 2
                ngh_slice = list(g.neighbors(vertex))[0:cut]

                processed_ngh = [x for x in ngh_slice if x not in all_vertices]

                # Param2: Modify new_phase to reflect the phase of
                # the spider that is resulting after unspider rule.
                new_phase = g.phase(vertex)
                ret.append([vertex, processed_ngh, new_phase])

                all_vertices.append(vertex)
                all_vertices += processed_ngh

        return ret

    def optimize(self, graph: pyzx.Graph) -> pyzx.Graph:
        """
        Main heuristic for optimizing a PyZX grapgh
        :param graph: PyZX graph to be optimized
        :return: optimized PyZX graph
        """
        graph_copy = graph.copy()

        simp(graph_copy, match_spider, spider, quiet=True)
        print(graph_copy.stats())

        # Parameter: the number of steps to apply the procedure
        nr_steps = 10
        # Parameter: the degree of the spiders to unspider
        min_degree = 3
        for step_i in range(nr_steps):

            if step_i == 0:
                print(f"{step_i}, {graph_copy}, ", end=" ")

            new_spiders = []

            # Find the nodes to perform unspider
            matches = self.match_min_degree(graph_copy, min_degree)

            # print(f" -> len {len(matches)}")
            for match in matches:
                n_spider = unspider(graph_copy, match)
                new_spiders.append(n_spider)

            # Flip the type of the unspidered spiders
            for n_spider in new_spiders:
                flip_spider_type(graph_copy, n_spider)

            # Perform phase free simplification:
            # a. spiders of the same type are merged
            # b. bialgebra rule is applied
            phase_free_simp(graph_copy, quiet=True)

        print(f"{nr_steps}, {graph_copy}")
        print(graph_copy.stats())
        print("----------")

        return graph_copy
