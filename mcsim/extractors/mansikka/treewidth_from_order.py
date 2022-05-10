"""
Return the treewidth for the TD graph for a Graph G and an elimination order
"""
from .graph import Graph


def find_treewidth_from_order(target_graph, elimination_order):
    """ """

    working_graph = Graph(target_graph.vertices.copy(), target_graph.edges.copy())
    treewidth = 1

    for u in elimination_order:

        nb = working_graph.neighbourhood(u)  # neighbourhood of vertices V

        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                if (nb[j], nb[i]) not in working_graph.edges:
                    working_graph.edges.add((nb[i], nb[j]))

        if len(working_graph.neighbourhood(u)) > treewidth:
            treewidth = len(working_graph.neighbourhood(u))

        working_graph.vertices.remove(u)
        edges = working_graph.edges.copy()

        for edge in edges:
            if u in edge:
                working_graph.edges.remove(edge)

    return treewidth
