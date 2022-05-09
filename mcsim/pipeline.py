"""
    Simulation pipeline
"""
from typing import Union

import numpy as np

import pyzx

from mcsim.optimizers import BaseOptimizer
from mcsim.extractors import BaseExtractor
from mcsim.exceptions import McSimSimulatorTypeError, McSimLoaderError
from mcsim.constants import CircFormat, SimulatorType
from mcsim.performance_indicators.measurements import timeit, MeasurementDict


# pylint: disable=R0201
class McSimPipeline:
    """
    This class is the main pipeline class for running a simulation.
    The pipeline is:
        1. load(circuit)
        2. optimize(graph)
        3. extract(optimized_graph)
        4. evaluate(initial_state, matrix)  or get_circuit(graph, format)
    The pipeline described above is implemented in the `simulate` method,
    but a user can easily access or replace any node of the pipeline individually
    by creating a customized pipeline.
    """

    def __init__(
        self,
        optimizer: BaseOptimizer = BaseOptimizer(),
        extractor: BaseExtractor = BaseExtractor(),
        simulator: str = SimulatorType.MCSIM,
        name: str = "",
    ):
        """
        :param optimizer_class: Heuristic logic for the optimization of a pyzx graph.
                                Inherits from BaseOptimizer
        :param extractor_class: Heuristic for extracting the matrix/list of matices from
                                a pyzx graph. Inherits from BaseExtractor
        :param simulator: String representing the simulator type: "mcsimulator", "qsim"
        :param name: optional pipeline id for logging purposes
        """
        self.optimizer = optimizer
        self.extractor = extractor
        self.simulator = simulator
        self.name = name
        self.timing = MeasurementDict()

    @timeit
    def load(self, circuit: Union[pyzx.Circuit]) -> Union[pyzx.Circuit, None]:
        """
        Loads and converts circuit to a pyzx compatible circuit and graph instances.
        :param circuit: circuit in the supported format: cirq.Circuit or pyzx.Circuit
        :return: pyzx circuit and pyzx graph
        """

        loaded_circuit = None
        print(type(circuit))
        if isinstance(circuit, pyzx.Circuit):
            loaded_circuit = circuit

        if loaded_circuit is None:
            raise McSimLoaderError("Unknown circuit type")

        circuit_graph = loaded_circuit.to_graph()
        print(circuit_graph)
        return loaded_circuit, circuit_graph

    @timeit
    def extract(self, graph: pyzx.Graph) -> np.array:
        """
        Extracts the matrix from a graph using the supplied extractor class
        :param graph: pyzx optimized graph
        :return: the matix associated wiht the graph
        """
        matrix = self.extractor.extract(graph)

        return matrix

    @timeit
    def optimize(self, graph: pyzx.Graph) -> pyzx.Graph:
        """
         Optimizes a given graph using the supplied optimizer class
        :param graph: pyzx graph to be optimized
        :return: optimized pyzx graph
        """
        optimized_graph = self.optimizer.optimize(graph)

        return optimized_graph

    @timeit
    def simulate(
        self, initial_state: np.array, circuit: Union[pyzx.Circuit]
    ) -> np.array:
        """
        Runs the default pipeline steps.
        :param initial_state: prepared initial state
        :param circuit: circuit to be simulaed, in compatible format (see load method)
        :return: circuit simulation result starting from the ininital_state
        """
        _, graph = self.load(circuit)
        optimized_graph = self.optimize(graph)

        if self.simulator == SimulatorType.MCSIM:
            result = self.evaluate_with_mcsim(initial_state, optimized_graph)

        else:
            raise McSimSimulatorTypeError("Unknown simulator type")

        return result

    @timeit
    def evaluate_with_mcsim(
        self, initial_state: np.array, optimized_graph: pyzx.Graph
    ) -> np.array:
        """
            Simulation of a circuit from an optimized graph, using mcsim basic simulator
        :param initial_state: prepared initial state
        :param optimized_graph: graph optimized using the supplied optimizer class
        :return: circuit simulation result starting from the ininital_state
        """
        matrix = self.extract(optimized_graph)
        result = self.evaluate(initial_state, matrix)

        return result

    @timeit
    def evaluate(self, initial_state: np.array, matrix: np.array) -> np.array:
        """
        Evalates the result of the circuit matirx applied to the initial_state
        :param initial_state: prepared initial state
        :param matrix: extracted matirx of the circuit
        :return: circuit simulation result starting from the ininital_state
        """

        return np.dot(matrix, initial_state)

    @timeit
    def get_circuit(
        self, graph: pyzx.Graph, circuit_format: CircFormat = CircFormat.PYZX
    ) -> Union[pyzx.Circuit, None]:
        """
        Converts a pyzx graph into a supported type using connectors
        :param graph: pyzx grapg to be converted
        :param circuit_format: supported type
        :return: circuit in the chosen format
        """

        pyzx_circuit = pyzx.extract_circuit(graph)

        if circuit_format == CircFormat.PYZX:
            return pyzx_circuit

        return None
