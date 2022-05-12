from math import pi, sqrt

import pyzx

try:
    import cupy as np
except:
    import numpy as np

    np.set_printoptions(suppress=True)


# typing imports
from typing import TYPE_CHECKING, List, Dict, Union
from .utils import FractionLike, FloatInt, VertexType, EdgeType

if TYPE_CHECKING:
    from .graph.base import BaseGraph, VT, ET
    from .circuit import Circuit
TensorConvertible = Union[np.ndarray, "Circuit", "BaseGraph"]


def Z_to_tensor(arity: int, phase: float) -> np.ndarray:
    m = np.zeros([2] * arity, dtype=complex)
    if arity == 0:
        m[()] = 1 + np.exp(1j * phase)
        return m
    m[(0,) * arity] = 1
    m[(1,) * arity] = np.exp(1j * phase)
    return m


def X_to_tensor(arity: int, phase: float) -> np.ndarray:
    m = np.ones(2**arity, dtype=complex)
    if arity == 0:
        m[()] = 1 + np.exp(1j * phase)
        return m
    for i in range(2**arity):
        if bin(i).count("1") % 2 == 0:
            m[i] += np.exp(1j * phase)
        else:
            m[i] -= np.exp(1j * phase)
    return np.power(np.sqrt(0.5), arity) * m.reshape([2] * arity)


def H_to_tensor(arity: int, phase: float) -> np.ndarray:
    m = np.ones(2**arity, dtype=complex)
    if phase != 0:
        m[-1] = np.exp(1j * phase)
    return m.reshape([2] * arity)


# Nu cred ca e bine de  ce e ls fel??
def input_to_tensor() -> np.ndarray:
    return np.identity(2)  # np.identity(2) #np.array([1, 0])


def output_to_tensor() -> np.ndarray:
    return np.identity(2)  # np.array([1, 0])


def mcs_tensorfy(
    contraction_order,
    g: "BaseGraph[VT,ET]",
    preserve_scalar: bool = True,
) -> np.ndarray:
    print(
        "\n############################################## msc tensorfy ##############################################"
    )

    nodes, edge_list = get_nodes_edges(g)

    had = 1 / sqrt(2) * np.array([[1, 1], [1, -1]])

    named_contraction_order = [
        get_edges_from_g(g).index(ce) for ce in contraction_order
    ]
    print("graph edge_list :", edge_list)
    print("contraction_order 0:", named_contraction_order)

    while len(named_contraction_order) > 0:

        contraction_edge = named_contraction_order[0]
        print("\n## contraction_edge in named_contraction_order ##")

        edge = edge_list[contraction_edge]
        input_node = nodes[edge["inp"]]
        output_node = nodes[edge["out"]]

        print("edge under contraction:", contraction_edge)
        print("input:{}\noutput:{}".format(edge["inp"], edge["out"]))

        # get the joint e
        # edges between the nodes and contraction axes.
        ni_axes = []
        no_axes = []
        je = []

        print("\ninput node edges:", input_node.edges)
        op = [(edge_list[k]["inp"], edge_list[k]["out"]) for k in input_node.edges]
        print("input node edges:", op)

        for i, inp_edge in enumerate(input_node.edges):
            print("##")
            print("edge_axes in input:{}\nedge:{}".format(i, inp_edge))
            if inp_edge in output_node.edges:
                ni_axes.append(i)
                no_axes.append(output_node.edges.index(inp_edge))
                je.append(inp_edge)
            else:
                # update the new ends of new edges
                if edge_list[inp_edge]["inp"] == edge["inp"]:
                    edge_list[inp_edge]["inp"] = edge["out"]
                if edge_list[inp_edge]["out"] == edge["inp"]:
                    edge_list[inp_edge]["out"] = edge["out"]

        # calcualte the new tensor and update

        """
        print("\n##calculate new tensor##")
        if output_node.index in output_nodes:
            print("!!output reached!!")
        elif input_node in input_nodes:
            print("!!input node!!")
        """
        print("\n##calculate new tensor##")
        new_tensor = np.tensordot(
            input_node.tensor, output_node.tensor, axes=(ni_axes, no_axes)
        )
        output_node.set_tensor(new_tensor)

        # update node edges
        new_edges = []
        for e in input_node.edges:
            if e not in je:
                new_edges.append(e)
        for e in output_node.edges:
            if e not in je:
                new_edges.append(e)
        output_node.set_edges(new_edges)

        # remove the node
        nodes.pop(edge["inp"])
        print("# remaining nodes#")
        print(" \nnodes:{} \n".format(nodes.keys()))

        # update the edge_list
        # remove contracted edges
        print("deprecate edges:", je)
        for deprecate_edge in je:
            if deprecate_edge in named_contraction_order:
                named_contraction_order.remove(deprecate_edge)
                edge_list.pop(deprecate_edge)
        print("\n##updated edge list##")
        print("update contraction order:", named_contraction_order)
        print("update edge list:", edge_list)
        print("######################")

    print("####\n Remaining edges:{} \n \n####".format(named_contraction_order))
    print("\n ###### final nodes:{}\n#### ".format(nodes))

    for node in nodes:
        tensor = nodes[node].tensor
        break

    print("tensor:", tensor)
    if preserve_scalar:
        tensor *= g.scalar.to_number()
    print("tensor shape:", tensor.shape)
    print("final tensor:", tensor)

    print("t_dot")
    return tensor


def get_nodes_edges(g: "BaseGraph[VT,ET]"):

    nodes = {}
    edges = {}
    edge_index = 0
    for edg in get_edges_from_g(g):
        edges[edge_index] = {"inp": min(edg), "out": max(edg), "type": g.edge_type(edg)}
        edge_index = edge_index + 1

    for v in g.vertices():
        node = Node(v, g)
        nodes[v] = node

    return nodes, edges


class Node:
    def __init__(self, v, g):
        self.index = v
        self.tensor = get_tensor_from_g(g, v)
        self.edges = [
            i for i in range(len(get_edges_from_g(g))) if v in get_edges_from_g(g)[i]
        ]

    def set_tensor(self, tensor):
        self.tensor = tensor

    def set_edges(self, new_edges):
        self.edges = new_edges


def get_tensor_from_g(g, v):

    phase = pi * g.phases()[v]
    v_type = g.types()[v]
    arity = len(g.neighbors(v))

    # input
    # output
    if v in g.inputs():
        return input_to_tensor()
    if v in g.outputs():
        return output_to_tensor()  # np.identity(2)

    if v_type == 1:
        t = Z_to_tensor(arity, phase)
    elif v_type == 2:
        t = X_to_tensor(arity, phase)
    elif v_type == 3:
        t = H_to_tensor(arity, phase)
    else:
        raise ValueError(
            "Vertex %s has non-ZXH type but is not an input or output" % str(v)
        )

    return t


def get_edges_from_g(g):
    edg = []
    for edge in g.edge_set():
        edg.append(edge)
    return edg


def print_edgelist(edgelist):
    el = [(edgelist[k]["inp"], edgelist[k]["out"]) for k in edgelist.keys()]
    print("edgelist:", el)
