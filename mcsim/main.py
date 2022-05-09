"""
    The main module for pipeline demonstration
"""


import numpy as np

import pyzx

from mcsim.constants import CircFormat
from mcsim import McSimPipeline


if __name__ == "__main__":

    qubits = 3
    depth = 5
    circuit = pyzx.generate.CNOT_HAD_PHASE_circuit(qubits, depth, clifford=True)

    initial_state = np.zeros((2**qubits,))
    initial_state[0] = 1

    pipeline = McSimPipeline(name="sim1")
    result = pipeline.simulate(initial_state, circuit)

    pipelineGamma = McSimPipeline(name="simGamma")
    loaded_circ, loaded_graph = pipelineGamma.load(circuit)
    optimized_graph = pipelineGamma.optimize(loaded_graph)
    matrix = pipelineGamma.extract(optimized_graph)
    result = pipelineGamma.evaluate(initial_state, matrix)
    retrieved_circuit = pipelineGamma.get_circuit(
        optimized_graph, circuit_format=CircFormat.PYZX
    )

    print(result)
