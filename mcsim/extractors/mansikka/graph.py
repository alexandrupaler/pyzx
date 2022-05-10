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
