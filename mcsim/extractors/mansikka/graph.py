"""

"""


class Graph:
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

        return Graph(dual_vert, dual_edges)
