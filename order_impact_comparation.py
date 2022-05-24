import random
try:
    import cupy as np
    print("CUPY0!")
except:
    import numpy as np
    np.set_printoptions(suppress=True)
import pyzx

from mcsim import McSimPipeline
from mcsim.extractors import MansikkaExtractor, MansikkaGraph, SimpleOrderExtractor


# Pipeline example  #
print("\n\n ## Mansikka example ##")

baseline_pipeline = McSimPipeline(name="baseline")

mansikka_extractor = MansikkaExtractor(params={"m": 2, "nr_iter": 6})
mansikka_pipeline = McSimPipeline(name="mansikka", extractor=mansikka_extractor)

simple_extractor = SimpleOrderExtractor(params={"order": 1,})
simple_pipeline = McSimPipeline(name="simple", extractor=simple_extractor)

qubits = 5
depth = 50

# Force seed
random.seed(1)
circuit = pyzx.generate.CNOT_HAD_PHASE_circuit_mixt(qubits, depth, p_had=0.5, p_cnot=0.5)


## Visualising the circuit
# pyzx.draw_matplotlib(circuit.to_graph(), labels=True, figsize=(8, 4), h_edge_draw='blue', show_scalar=False,
#                      rows=None).savefig("circuit_0.png")


baseline_circ, baseline_graph = baseline_pipeline.load(circuit)
mansikka_circ, mansikka_graph = mansikka_pipeline.load(circuit)
simple_circ, simple_graph = simple_pipeline.load(circuit)

# matrix_0 = baseline_pipeline.extract(baseline_graph)
# matrix_1 = mansikka_pipeline.extract(mansikka_graph)
# matrix_2 = mansikka_pipeline.extract(simple_graph)
#
# print("Equals matrix_0=matrix_1: ", np.allclose(matrix_0, matrix_1))
# print("Equals matrix_0=matrix_2: ", np.allclose(matrix_0, matrix_2))




print("## Time performace##:\n")

initial_state = np.zeros((2 ** qubits,), dtype=np.complex64)
initial_state[0] = 1

result_0 = baseline_pipeline.evaluate_with_mcsim(initial_state, baseline_graph)
result_1 = mansikka_pipeline.evaluate_with_mcsim(initial_state, mansikka_graph)
result_2 = simple_pipeline.evaluate_with_mcsim(initial_state, simple_graph)

print("\n####baseline:####\n", baseline_pipeline.timing)
print("\n####mansika:####\n", mansikka_pipeline.timing)
print("\n####simple:####\n", simple_pipeline.timing)

print("-Done-")


