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
#from memory_profiler import profile
import gc

random.seed(1)


# Load circuit
# qubits = 7
# depth = 50
# circuit = pyzx.generate.CNOT_HAD_PHASE_circuit_mixt(qubits, depth, p_had=0.5, p_cnot=0.5)

# Load circuit_from file
circuit_path ="C:/Users/tomut/Documents/GitHub/pyzx/circuits/Arithmetic_and_Toffoli"+"/vbe_adder_3_before"#"/gf2^4_mult_before"
circuit = pyzx.Circuit.load(circuit_path)
qubits = circuit.qubits
print("nr_qubits:", qubits)

## Visualising the circuit
pyzx.draw_matplotlib(circuit.to_graph(), labels=True, figsize=(8, 4), h_edge_draw='blue', show_scalar=False,
                      rows=None).savefig("circuit_0.png")


baseline_pipeline = McSimPipeline(name="baseline")

mansikka_extractor = MansikkaExtractor(params={"m": 2, "nr_iter": 100})
mansikka_pipeline = McSimPipeline(name="mansikka", extractor=mansikka_extractor)

simple_extractor = SimpleOrderExtractor(params={"order": 1,})
simple_pipeline = McSimPipeline(name="simple", extractor=simple_extractor)



baseline_circ, baseline_graph = baseline_pipeline.load(circuit)
mansikka_circ, mansikka_graph = mansikka_pipeline.load(circuit)
simple_circ, simple_graph = simple_pipeline.load(circuit)

matrix_0 = baseline_pipeline.extract(baseline_graph)
matrix_1 = mansikka_pipeline.extract(mansikka_graph)
matrix_2 = simple_pipeline.extract(simple_graph)



s = 0
for i in range(len(matrix_0)):
    for j in range(len(matrix_0)):
        s = s + (abs(matrix_0[i][j] - matrix_2[i][j])) ** 2
print("Dif:", s)

print("Equals matrix_0=matrix_1: ", np.allclose(matrix_0, matrix_1))
print("Equals matrix_0=matrix_2: ", np.allclose(matrix_0, matrix_2))


#@profile()
# def get_memeo():
#     print("## Time performace##:\n")
#
#     initial_state = np.zeros((2 ** qubits,), dtype=np.complex64)
#     initial_state[0] = 1
#
#     result_0 = baseline_pipeline.evaluate_with_mcsim(initial_state, baseline_graph)
#     del(result_0)
#     gc.collect()
#     result_1 = mansikka_pipeline.evaluate_with_mcsim(initial_state, mansikka_graph)
#     result_2 = simple_pipeline.evaluate_with_mcsim(initial_state, simple_graph)
#
#     print("\n####baseline:####\n", baseline_pipeline.timing)
#     print("\n####mansika:####\n", mansikka_pipeline.timing)
#     print("\n####simple:####\n", simple_pipeline.timing)
#
#     print("-Done-")
# get_memeo()

