"""

"""

from .graph import Graph
from .treewidth_from_order import find_treewidth_from_order


def greedy_treewidth_deletion(
    target_graph: Graph, elimination_order, m, option=0, direct_minimization=False
):
    """ """
    removed_vertices = []
    new_graph = Graph(target_graph.vertices, target_graph.edges)
    new_order = elimination_order.copy()

    for j in range(m):

        if direct_minimization:
            u = removal_recommendation(
                new_graph, new_order
            )  # if new order is given direct treewidth metric is used
        else:
            u = removal_recommendation(new_graph)

        # update new graph

        new_graph.vertices.remove(u)

        edges = new_graph.edges.copy()

        for edge in edges:
            if u in edge:
                new_graph.edges.remove(edge)

        # update removed_vertices
        if u not in removed_vertices:
            removed_vertices.append(u)

        # update new_order
        new_order.remove(u)
        if option == 1:
            new_order = tree_decomposition(new_graph)

    tw = find_treewidth_from_order(new_graph, new_order)

    return new_graph, new_order, tw, removed_vertices


def removal_recommendation(target_graph, order=None):
    # this will be updated to betweenness centrality

    if order != None:
        return direct_treewidth_minimization(target_graph, order)

    nr_neighbors = 0
    recommendation = target_graph.vertices[0]
    for v in target_graph.vertices:
        nr_nb = len(target_graph.neighbourhood(v))
        if nr_nb > nr_neighbors:
            recommendation = v
            nr_neighbors = nr_nb

    return recommendation


def direct_treewidth_minimization(target_graph, elimination_order):

    working_graph = Graph(target_graph.vertices.copy(), target_graph.edges.copy())

    tw = find_treewidth_from_order(working_graph, elimination_order.copy())
    delta = 0
    recommendation = target_graph.vertices[0]

    for u in target_graph.vertices:

        new_order = elimination_order.copy()
        new_order.remove(u)

        new_graph = Graph(target_graph.vertices.copy(), target_graph.edges.copy())
        new_graph.vertices.remove(u)
        edges = new_graph.edges.copy()

        for edge in edges:
            if u in edge:
                new_graph.edges.remove(edge)

        n_tw = find_treewidth_from_order(new_graph, new_order)

        new_delta = tw - n_tw

        if new_delta <= delta:
            delta = new_delta
            recommendation = u

    return recommendation


def tree_decomposition(target_graph):

    return 0
