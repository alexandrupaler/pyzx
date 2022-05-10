"""

"""


import numpy as np

import pyzx


from mcsim.extractors.mansikka import Graph, find_treewidth_from_order
from mcsim.extractors.mansikka import greedy_treewidth_deletion


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
