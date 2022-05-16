import pyzx
from mcsim import phase_free_simp, McSimPipeline
from mcsim.constants import CircFormat


print("run numpy/cupy/pyzx")
qubits = 3
depth = 5
testc = pyzx.generate.CNOT_HAD_PHASE_circuit(qubits, depth, clifford=True)
matrix = testc.to_matrix()
print(matrix)


print("run mcsim")
# Test simplification from mcsim

phase_free_simp(testc.to_graph())

# Test base extractor from mcsim
print("run mcsim pipeline")

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