"""
    The main module for pipeline demonstration
"""
from typing import Dict

import numpy as np
import cirq

from cirq import ops

# from mcsim.constants import CircFormat
# from mcsim.optimizers import BaseOptimizer
# from mcsim.extractors import BaseExtractor
from mcsim import McSimPipeline

GATE_DOMAIN: Dict[ops.Gate, int] = {
    ops.CNOT: 2,
    ops.H: 1,
    ops.S: 1,
    ops.T: 1,
    ops.X: 1,
    ops.Z: 1,
}

if __name__ == '__main__':

    # Random check
    circuit = circuit = cirq.testing.random_circuit(
        qubits=5, n_moments=5, op_density=1, gate_domain=GATE_DOMAIN, random_state=42
    )
    initial_state = np.zeros((2 ** 5,))
    initial_state[0] = 1
    initial_state[11] = 1

    pipeline = McSimPipeline(name="sim1")
    result = pipeline.simulate(initial_state, circuit)

    # pipelineGamma = McSimPipeline(name="simGamma")
    # loaded_circ, loaded_graph = pipelineGamma.load(circuit)
    # optimized_graph = pipelineGamma.optimize(loaded_graph)
    # matrix = pipelineGamma.extract(optimized_graph)
    # result = pipelineGamma.evaluate(initial_state, matrix)
    # retrieved_circuit = pipelineGamma.get_circuit(optimized_graph, circuit_format=CircFormat.CIRQ)

    print(result)
