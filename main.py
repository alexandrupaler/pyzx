import pyzx

print("run numpy/cupy/pyzx")
qubits = 3
depth = 5
testc = pyzx.generate.CNOT_HAD_PHASE_circuit(qubits, depth, clifford=True)
matrix = testc.to_matrix()
print(matrix)


print("run mcsim")
# Test simplification from mcsim
from mcsim import phase_free_simp

phase_free_simp(testc.to_graph())

# Test base extractor from mcsim
