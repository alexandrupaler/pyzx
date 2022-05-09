"""

"""

import numpy as np
import pyzx
from typing import Dict, List

from mcsim.extractors import BaseExtractor


class Extractor_Dist(BaseExtractor):

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
        tolerance = self.params['tolerance']
        ##

        if show_changes:
            graph_info(graph, show_changes)

        dist = self.distribution_f(self.params)
        graph = fill_degree(graph, dist, tolerance)

        if show_changes:
            graph_info(graph, show_changes)

        return graph.to_matrix()


def graph_info(graph, level):  # where I shuld mouve this or delet
    """
    Print informations about a graph.
    :graph:
    :leve:
    """
    if level == 1:
        print(graph.stats())
        print('rows:', graph.rows())
    if level == 2:
        print('phases:', graph.phases())
        print()
        print('types:', graph.types())
        print()
        print('depth:', graph.depth())
        print()
        print('vertices:', graph.vertices)
    pyzx.draw(graph, labels=True)


def fill_degree(graph: pyzx.graph, dist: List[float], tolerance: float) -> pyzx.graph:
    """
    This function rearange the vertices of graph on rowsto fit dist
    :graph: pyzx the graph whose nodes we want to rearrange to match distribution.
    :dist: degree distribution
    :tolerange: initial tolerance that is accepted between dezire degree on the row and real one.
    :return: the pyzx graph with the nodes rearranged on the rows.
    """

    vertices = graph.vertices()
    importances = get_importance(graph)
    distribution = distributin_to_importance(graph, dist)

    vert_i=0# vert indices in vertices
    for vert in vertices:
        vert_i=vert_i+1
        if graph.type(vert) == 0 and graph.row(vert) != 0:
            graph.set_row(vert, len(distribution))

        if graph.type(vert) != 0:
            t_tolerance = tolerance
            insert = 1
            while insert:
                for k in range(len(distribution)):
                    if distribution[k] + t_tolerance >= importances[vert_i-1]:
                        distribution[k] = distribution[k] - importances[vert_i-1]
                        graph.set_row(vert, k)
                        insert = 0
                        break
                t_tolerance = t_tolerance + 1

    return graph

'''
def distributin_to_degree(graph: pyzx.graph, dist: List[float]) -> List[float]:
    """
    Create a degree distribution close to the desired one.
    :graph: pyzx the graph whose nodes we want to rearrange to match distribution.
    :dist: target de degree distribution
    :return: degree distribution.
    """
    vertices = graph.vertices()
    total_degree = 0
    for vert in vertices:
        total_degree = total_degree + graph.vertex_degree(vert)

    s = 0
    for i in dist:
        s = s + i

    for i in range(len(dist)):
        dist[i] = (dist[i] / s) * total_degree

    return dist
'''
def distributin_to_importance(graph: pyzx.graph, dist: List[float]) -> List[float]:
    """
    Create a degree distribution close to the desired one.
    :graph: pyzx the graph whose nodes we want to rearrange to match distribution.
    :dist: target de degree distribution
    :return: degree distribution.
    """
    vertices = graph.vertices()
    importances = get_importance(graph)
    total_imp = 0

    k=0 # vert indices in vertices
    for vert in vertices:
        total_imp = total_imp + importances[k]
        k=k+1
    s = 0
    for i in dist:
        s = s + i

    for i in range(len(dist)):
        dist[i] = (dist[i] / s) * total_imp

    return dist

def simple_distribution(params: Dict) -> List[float]:
    """
    Desire distribution
    :nr_sections: number of rows that will be used to reduce the tensor
    :sigma:sigma
    :mu: mu
    :return: list of elements proportional with degrees/row for eatch row
    """

    # deserialization of parameters#
    nr_sections = int(params['nr_section'])
    sigma = params['sigma']
    mu = params['mu'] * nr_sections  # !! Here I  need some sugestion
    ##

    dristribution = []
    for i in range(nr_sections):
        dristribution.append(gaussin_n(sigma, mu, i))

    return dristribution


def gaussin_n(sigma: float, mu: float, x: float) -> float:
    """
    Normalized gaussian function
    :sigma:sigma
    :mu: mu
    :x: coordonate
    :return: g(x)
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -1 * ((x - mu) * (x - mu) / (2 * sigma * sigma))
    )


def get_importance(graph):
    """

    """
    importances = []

    k=0
    for spider in graph.vertices():
        k=k+1
        deg = graph.vertex_degree(spider)
        neighbours = graph.neighbors(spider)
        neighbour_contribution = 0
        for ne in neighbours:
            neighbour_contribution = neighbour_contribution + graph.vertex_degree(ne)

        value = (neighbour_contribution + deg) * np.log(neighbour_contribution + deg)
        importances.append(value)



    return importances



