import pyzx

qubits = 3
depth = 5
testc = pyzx.generate.CNOT_HAD_PHASE_circuit(qubits, depth, clifford=True)
matrix = testc.to_matrix()

print(matrix)