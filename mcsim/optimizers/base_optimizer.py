"""
    Base Optimizer to be injected into a simulation pipeline
"""
import pyzx


# pylint: disable=R0201
class BaseOptimizer:
    """
    Base class for optimization heuristics of a pyzx graph.
    """

    def __init__(self, params=None):
        self._params = params if params else {}

    def optimize(self, graph: pyzx.Graph) -> pyzx.Graph:
        """
        Main heuristic for optimizing a pyzx grapgh
        :param graph: pyzx graph to be optimized
        :return: optimized pyzx graph
        """
        graph_copy = graph.copy()
        pyzx.full_reduce(graph_copy)

        return graph_copy

    @property
    def params(self):
        """Parameters of the heuristic used for the circuit optimization"""
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = new_params
