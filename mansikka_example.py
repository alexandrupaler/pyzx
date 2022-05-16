"""

"""


import numpy as np

import pyzx

from mcsim import McSimPipeline
from mcsim.constants import CircFormat
from mcsim.extractors import MansikkaExtractor, MansikkaGraph


## Pipeline example  ##
print("\n\n ## Mansikka example ##")


qubits = 2
depth = 1

circuit = pyzx.generate.CNOT_HAD_PHASE_circuit(qubits, depth, clifford=True)
zx_graph = circuit.to_graph()
print("@@@@@@@@")
pyzx.draw_matplotlib(zx_graph, labels=True, figsize=(8, 2), h_edge_draw='blue', show_scalar=False, rows=None).savefig("circuit.png")

example_graph = MansikkaGraph([k for k in zx_graph.vertices()], zx_graph.edge_set().copy())
elimination_order = [k for k in zx_graph.vertices()]

tw = example_graph.find_treewidth_from_order(elimination_order)
print("treewidth :", tw)


initial_state = np.zeros((2**qubits,))
initial_state[1] = 1

# pyzx.draw_matplotlib(example_graph, labels=False, figsize=(8, 2), h_edge_draw='blue', show_scalar=False, rows=None).savefig("graph_0.png")

pipeline = McSimPipeline(name="sim1")
result0 = pipeline.simulate(initial_state, circuit)
print("r0:", result0)

params = {"m": 2, "nr_iter": 6}
mansikka_extractor = MansikkaExtractor(params=params)
pipelineMansikka = McSimPipeline(name="basicMansikka", extractor=mansikka_extractor)
loaded_circ, loaded_graph = pipelineMansikka.load(circuit)

matrix = pipelineMansikka.extract(loaded_graph)
result = pipelineMansikka.evaluate(initial_state, matrix)
# retrieved_circuit = pipelineMansikka.get_circuit(zx_graph, circuit_format=CircFormat.PYZX)

print("r0:", result0)
print("r9:", result)



for state in range(2**qubits):
    print("##########\n state:{} #######".format(state))
    initial_state = np.zeros((2 ** qubits,))
    initial_state[state] = 1

    pipeline = McSimPipeline(name="sim1")
    result0 = pipeline.simulate(initial_state, circuit)

    params = {"m": 2, "nr_iter": 6}
    mansikka_extractor = MansikkaExtractor(params=params)
    pipelineMansikka = McSimPipeline(name="basicMansikka", extractor=mansikka_extractor)
    loaded_circ, loaded_graph = pipelineMansikka.load(circuit)

    matrix = pipelineMansikka.extract(loaded_graph)
    result = pipelineMansikka.evaluate(initial_state, matrix)

    print("##########s state:{} #######".format(state))
    print("r0:", result0)
    print("r9:", result)


#####################################################


print("\n\n Done !")


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