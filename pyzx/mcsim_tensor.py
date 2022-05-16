from math import pi, sqrt
from .tensor import tensor_to_matrix
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
    print("@Z_tensor@")
    print("shape:", m.shape)
    print("tensor:", m)
    return m


def X_to_tensor(arity: int, phase: float) -> np.ndarray:
    m = np.ones(2 ** arity, dtype=complex)
    if arity == 0:
        m[()] = 1 + np.exp(1j * phase)
        return m
    for i in range(2 ** arity):
        if bin(i).count("1") % 2 == 0:
            m[i] += np.exp(1j * phase)
        else:
            m[i] -= np.exp(1j * phase)
    tens = np.power(np.sqrt(0.5), arity) * m.reshape([2] * arity)

    print("@X_tensor@")
    print("shape:", tens.shape)
    print("tensor:", tens)
    return tens


def H_to_tensor(arity: int, phase: float) -> np.ndarray:
    m = np.ones(2 ** arity, dtype=complex)
    if phase != 0:
        m[-1] = np.exp(1j * phase)
    return m.reshape([2] * arity)



def input_to_tensor() -> np.ndarray:
    return np.identity(2)  # np.identity(2) #np.array([1, 0])

def output_to_tensor() -> np.ndarray:
    return np.identity(2)  # np.array([1, 0])


def mcs_tensorfy(g, contraction_order, preserve_scalar: bool = True ) -> np.ndarray:
    """

    """

    print(
        "\n############################################## msc tensorfy ##############################################"
    )

    # Hadamard gate tensor will be used for the Hadamard edges.
    had = 1 / sqrt(2) * np.array([[1, 1], [1, -1]])

    # Dictionaries with the nodes and edges in the graph.
    nodes, edge_list = get_nodes_edges(g)

    # move the ede with  inputs at the end
    inp_order = [ contraction_order[0] for i in range(len(g.inputs())) ]

    co_copy = contraction_order.copy()
    for edge in co_copy:
        if edge[0] in g.inputs():
            position=len(g.inputs())-edge[0]-1
            print("index", position)
            inp_order[position] = edge

    for edge in inp_order:
        contraction_order.remove(edge)
        contraction_order.append(edge)
    print("input_order:",inp_order )


    # move the ede with  output at the end
    output_order = [ contraction_order[0] for i in range(len(g.outputs())) ]
    nr_vert=len(g.vertices())

    co_copy = contraction_order.copy()
    for edge in co_copy:
        if edge[1] in g.outputs():
            position= len(g.outputs())-(nr_vert-edge[1]-1) -1
            print("index", nr_vert-edge[1]-1)
            output_order[position] = edge

    for edge in output_order:
        contraction_order.remove(edge)
        contraction_order.append(edge)
    print("output_order:",output_order )


    # Contracting order is provided like a list of tuples, and now we change it into a list of indexes.
    named_contraction_order = [
        get_edges_from_g(g).index(ce) for ce in contraction_order
    ]


    print("graph edge_list :", edge_list)
    print("contraction_order 0:", named_contraction_order)

    # Now, we contract the edges until the contraction list is empty
    while len(named_contraction_order) > 0:
        print("\n## contraction_edge in named_contraction_order ##\n")

        contraction_edge_index = named_contraction_order[0]
        edge = edge_list[contraction_edge_index]
        input_node = nodes[edge["inp"]]
        output_node = nodes[edge["out"]]

        print("## edge under contraction:", contraction_edge_index)
        print("## input:{} | output:{}".format(edge["inp"], edge["out"]))


        ni_axes = [] # contraction axes for the input node.
        no_axes = [] # contraction axes for the output node.
        je = []      # joint edges between the nodes.
                     # We will need ned to contract over all of the joint edges.

        print("\ninput node edges:", input_node.edges)
        op = [(edge_list[k]["inp"], edge_list[k]["out"]) for k in input_node.edges]
        print("input node edges:", op)

        # Populate the contraction axes and joint edges
        for i, inp_edge in enumerate(input_node.edges):
            print("#### Populate the contraction axes and joint edges ####")
            print("#### edge_axes in input:{}\n edge:{}".format(i, inp_edge))
            if inp_edge in output_node.edges:
                if edge_list[inp_edge]["inp"] in g.inputs():
                    ni_axes.append(1)#1
                else:
                    ni_axes.append(i)
                no_axes.append(output_node.edges.index(inp_edge))
                je.append(inp_edge)
            else:
                # update the new ends of edge
                if edge_list[inp_edge]["inp"] == edge["inp"]:
                    edge_list[inp_edge]["inp"] = edge["out"]
                if edge_list[inp_edge]["out"] == edge["inp"]:
                    edge_list[inp_edge]["out"] = edge["out"]




#######################################################################################################
        # treat the Hadamard edges
        # need to be verrified
        for c_e in je:
            print("#### Treat Hadamard edges ####")
            if edge_list[c_e]["type"] == EdgeType.HADAMARD:
                # get the contraction axes in input tensor
                inp_axis_connected_with_had = [input_node.edges.index(c_e)]
                # contract with the hadamard and update the input tensor
                new_tensor = np.tensordot(
                    input_node.tensor,had, axes=(inp_axis_connected_with_had,[1] )
                )
                input_node.set_tensor(new_tensor)

                # remove contracted edge and add it again at the right place
                print("#### input node edges before H:", input_node.edges)
                input_node.edges.remove(c_e)
                input_node.edges.insert(0,c_e)#append(c_e)
                print("#### input node edges after H:", input_node.edges)

                # recalculate contraction axes for the modified tensor
                print("#### recalculate input axes ")
                ni_axes = []
                for i, inp_edge in enumerate(input_node.edges):
                    print("## edge_axes in input:{}\nedge:{}".format(i, inp_edge))
                    if inp_edge in output_node.edges:
                        ni_axes.append(i)
######################################################################################################


        print("\n##!!calculate new tensor!!##")

        print("inp _tensor:\n",input_node.tensor)
        print("output _tensor:\n",output_node.tensor)
        print("ni:{}|no{}".format(ni_axes,no_axes))
        new_tensor = np.tensordot(
             input_node.tensor,output_node.tensor, axes=(ni_axes, no_axes)
        )
        output_node.set_tensor(new_tensor)
        if (len(named_contraction_order)>=1):
            print("@@@@@@@@@@@tensor@@@@@@@@@@@@@@@")
            print(new_tensor)
            print("@@matrix@@")
            print(tensor_to_matrix(new_tensor,2,2))
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
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

    if (len(nodes.keys())>1):
        raise Exception("You have more than one final node. This means that you can treat your circuit as two separate circuits!")
        return 1

    for node in nodes:
        tensor = nodes[node].tensor
        break

    perm = []
    for i in range(2 * len(g.inputs())):
        #perm.append(len(g.inputs())-i-1)
        perm=[3,2,1,0]
    tensor = np.transpose(tensor, perm)


    print("tensor:", tensor)
    if preserve_scalar:
        tensor *= g.scalar.to_number()
    print("tensor shape:", tensor.shape)
    print("final tensor:", tensor)
    print("final mat:", tensor_to_matrix(tensor,2,2))

    return tensor


def get_nodes_edges(g: "BaseGraph[VT,ET]"):

    nodes = {}
    edges = {}

    # The index represents the key to the edge in dic. edges.  Each edge has an input, an output amnd a type.
    # In the beginning, the nodes are labeled in such a way that the one with the smaller index is the input.
    # The type may indicate the presence of a Hadamard gate between the end vertices.
    edge_index = 0
    for edg in get_edges_from_g(g):
        edges[edge_index] = {
            "inp": min(edg),
            "out": max(edg),
            "type": g.edge_type(g.edge(edg[0], edg[1])),
        }
        edge_index = edge_index + 1

    # The key of a node will be its initial index
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
