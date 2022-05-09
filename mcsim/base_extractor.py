"""
    Matrix extractor module
"""
import numpy as np
import pyzx


# pylint: disable=R0201
class BaseExtractor:
    """
    Base class for matrix extraction from a pyzx graph.
    """

    def __init__(self, params=None):
        self._params = params if params else {}

    def extract(self, graph: pyzx.Graph) -> np.array:
        """
        Base extraction method of the matrix of a given pyzx graph
        :param graph: pyzx graph pf a given optimized circuit
        :return: matrix of the circuit
        """
        return graph.to_matrix()

    @property
    def params(self):
        """ Parameters of the heuristic used for the matrix extraction"""
        return self._params

    @params.setter
    def params(self, new_extractor_params):
        self._params = new_extractor_params
