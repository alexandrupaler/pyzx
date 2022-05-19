import random
import numpy as np
import pyzx

from mcsim import McSimPipeline
from mcsim.extractors import MansikkaExtractor, MansikkaGraph


## Pipeline example  ##
print("\n\n ## Mansikka example ##")

baseline_pipeline = McSimPipeline(name="baseline")

mansikka_extractor = MansikkaExtractor(params={"m": 2, "nr_iter": 6})
mansikka_pipeline = McSimPipeline(name="mansikka", extractor=mansikka_extractor)

qubits = 7
depth = 29

# Force seed
random.seed(1)
circuit = pyzx.generate.CNOT_HAD_PHASE_circuit(qubits, depth, p_had=0.0, clifford=True)
# Visualising the circuit
pyzx.draw_matplotlib(circuit.to_graph(), labels=True, figsize=(8, 4), h_edge_draw='blue', show_scalar=False, rows=None).savefig("circuit_0.png")

baseline_circ, baseline_graph = baseline_pipeline.load(circuit)
matrix_0 = baseline_pipeline.extract(baseline_graph)

mansikka_circ, mansikka_graph = mansikka_pipeline.load(circuit)
matrix_1 = mansikka_pipeline.extract(mansikka_graph)

print("Baseline matrix \n", matrix_0)
print("        Mansikka\n", matrix_1)
print("Equals:         \n", np.equal(matrix_0, matrix_1))

s=0
for i in range(len(matrix_0)):
    for j in range(len(matrix_0)):
        s=s+(matrix_0[i][j]-matrix_1[i][j])**2

print("Dif:",s)
# for state in range(2**qubits):
#     print("##########\n state:{} #######".format(state))
#     initial_state = np.zeros((2 ** qubits,))
#     initial_state[state] = 1
#
#     # The baseline result
#     result_0 = baseline_pipeline.evaluate(initial_state, matrix_0)
#
#     # Our contraction order + tensorfy
#     result_1 = mansikka_pipeline.evaluate(initial_state, matrix_1)
#
#     print("##########s state:{} #######".format(state))
#     print("baseline:", result_0)
#     print("    ours:", result_1)

print("\n\n Mansikka Done !")

#####################################################

def test_trewidth(circuit):
    zx_graph = circuit.to_graph()
    pyzx.draw_matplotlib(zx_graph, labels=True, figsize=(8, 2), h_edge_draw='blue', show_scalar=False, rows=None).savefig("circuit.png")

    example_graph = MansikkaGraph([k for k in zx_graph.vertices()], zx_graph.edge_set().copy())
    elimination_order = [k for k in zx_graph.vertices()]

    tw = example_graph.find_treewidth_from_order(elimination_order)
    print("treewidth :", tw)

"""
###################################################
## Example from paper ##

vertices = ["i", "j", "k", "l", "m", "n"]
edges = {
    ("i", "i"),
    ("i", "j"),
    ("i", "k"),
    ("j", "k"),
    ("j", "l"),
    ("k", "l"),
    ("k", "m"),
    ("l", "n"),
    ("m", "n"),
}
example_graph = Graph(vertices, edges)
elimination_order = ["i", "j", "k", "l", "m", "n"]

tw = find_treewidth_from_order(example_graph, elimination_order)
print("treewidth :", tw)
print("######\n")
m = 1
reduced_g, reduced_order, tw, removing_order = greedy_treewidth_deletion(
    example_graph, elimination_order, m
)
print("new_vertices:", reduced_g.vertices)
print("new_edges:", reduced_g.edges)
print("reduced_order:", reduced_order)
print("treewidth:", tw)
print("removing_order:", removing_order)


#####################################################
"""

"""
######## ###########################################
## Extra example ##

vertices = ["i", "j", "k", "l", "m", "n", "o"]
edges = {
    ("i", "i"),
    ("i", "j"),
    ("i", "k"),
    ("j", "k"),
    ("j", "l"),
    ("k", "l"),
    ("k", "m"),
    ("l", "n"),
    ("m", "n"),
    ("n", "o"),
    ("m", "l"),
    ("o", "k"),
}
example_graph = Graph(vertices, edges)
elimination_order = ["i", "j", "k", "l", "m", "n", "o"]

tw = find_treewidth_from_order(example_graph, elimination_order)
print("\n\nn ## Extra example ##")
print("treewidth :", tw)
print("######\n")
m = 2
reduced_g, reduced_order, tw, removing_order = greedy_treewidth_deletion(
    example_graph, elimination_order, m
)

print("new_vertices:", reduced_g.vertices)
print("new_edges:", reduced_g.edges)
print("reduced_order:", reduced_order)
print("treewidth:", tw)
print("removing_order:", removing_order)

#####################################################
"""

"""
######## ###########################################
## Pyzx example  ##

qubits = 3
depth = 5
circuit = pyzx.generate.CNOT_HAD_PHASE_circuit(qubits, depth, clifford=True)
zx_graph = circuit.to_graph()

example_graph = Graph([k for k in zx_graph.vertices()], zx_graph.edge_set())
elimination_order = [k for k in zx_graph.vertices()]

tw = find_treewidth_from_order(example_graph, elimination_order)
print("\n\n ## Pyzx example ##")
print("treewidth :", tw)
print("######\n")
m = 3
reduced_g, reduced_order, tw, removing_order = greedy_treewidth_deletion(
    example_graph, elimination_order, m
)

print("new_vertices:", reduced_g.vertices)
print("new_edges:", reduced_g.edges)
print("reduced_order:", reduced_order)
print("treewidth:", tw)
print("removing_order:", removing_order)

#####################################################
"""