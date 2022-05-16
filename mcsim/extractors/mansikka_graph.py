"""

"""
import copy


class MansikkaGraph:
    """
    Graph class that we use during the heuristic to stay independent of pyzx graph structure.
    """

    def __init__(self, vertices, edges):

        self.vertices = vertices
        self.edges = edges

    def neighbourhood(self, node):
        """
        vertice:
        return: list of all the neighbors of the node V
        """
        nb = []

        for edge in self.edges:
            if node in edge:
                for v in edge:
                    if v != node and v not in nb:
                        nb.append(v)

        return nb

    def construct_dual(self):
        dual_vert = [v for v in self.edges]
        dual_edges = set()

        for d_node in dual_vert:
            n1 = d_node[0]
            n2 = d_node[1]
            for d_node2 in dual_vert:
                if d_node2 != d_node:
                    if n1 in d_node2 or n2 in d_node2:
                        if (d_node2, d_node) not in dual_edges:
                            dual_edges.add((d_node, d_node2))

        return MansikkaGraph(dual_vert, dual_edges)


    def find_treewidth_from_order(self, elimination_order):

        # Work on a copy
        working_graph = copy.deepcopy(self)
        # working_graph = Graph(self.vertices.copy(), self.edges.copy())
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


def greedy_treewidth_deletion(
    self, elimination_order, nr_tensors_to_rem, option=0, direct_minimization=False
):
    """

    Args:
        self:
        elimination_order: ??
        nr_tensors_to_rem: Parameter m from the paper
        option:
        direct_minimization:

    Returns:

    """
    removed_vertices = []
    new_graph = copy.deepcopy(self)
    # new_graph = Graph(self.vertices, self.edges)
    new_order = elimination_order.copy()

    for j in range(nr_tensors_to_rem):

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

    tw = new_graph.find_treewidth_from_order(new_order)

    return new_graph, new_order, tw, removed_vertices


def removal_recommendation(self, order=None):
    # this will be updated to betweenness centrality

    if order != None:
        return self.direct_treewidth_minimization(order)

    nr_neighbors = 0
    recommendation = self.vertices[0]
    for v in self.vertices:
        nr_nb = len(self.neighbourhood(v))
        if nr_nb > nr_neighbors:
            recommendation = v
            nr_neighbors = nr_nb

    return recommendation


def direct_treewidth_minimization(self, elimination_order):

    # copy
    working_graph = copy.deepcopy(self)
    # working_graph = Graph(tensor_graph.vertices.copy(), tensor_graph.edges.copy())

    tw = working_graph.find_treewidth_from_order(elimination_order.copy())
    delta = 0
    recommendation = self.vertices[0]

    for u in self.vertices:

        new_order = elimination_order.copy()
        new_order.remove(u)

        # TODO Alexandru: copy new_graph or working_graph?
        new_graph = copy.deepcopy(self)
        # new_graph = Graph(self.vertices.copy(), self.edges.copy())

        new_graph.vertices.remove(u)
        edges = new_graph.edges.copy()

        for edge in edges:
            if u in edge:
                new_graph.edges.remove(edge)

        n_tw = new_graph.find_treewidth_from_order(new_order)

        new_delta = tw - n_tw

        if new_delta <= delta:
            delta = new_delta
            recommendation = u

    return recommendation


def tree_decomposition(self):

    return None
