import math
try:
    import cupy as np
except:
    import numpy as np

    np.set_printoptions(suppress=True)

from pyzx.utils import EdgeType
import pyzx.tensor as pyzxtensor

from .mcsim_node import MansikkaNode

def mcsim_tensorfy(pyzx_graph, contraction_edge_list, preserve_scalar: bool = True) -> np.ndarray:
    """

    """

    print(
        "\n############################################## msc tensorfy ##############################################"
    )

    # Hadamard gate tensor will be used for the Hadamard edges.
    had = 1 / math.sqrt(2) * np.array([[1, 1], [1, -1]])

    # Dictionaries with the nodes and edges in the graph.
    mansikka_node_map, edge_list = get_nodes_edges(pyzx_graph)
    nr_vert = pyzx_graph.num_vertices()

    reorder_contraction_edge_list(contraction_edge_list, nr_vert, pyzx_graph)

    # Contracting order is provided like a list of tuples, and now we change it into a list of ids.
    contraction_ids = [
        pyzx_graph.edges().index(edge) for edge in contraction_edge_list
    ]

    print("graph edge_list :", edge_list)
    print("contraction_order 0:", contraction_ids)

    # Now, we contract the edges until the contraction list is empty
    while len(contraction_ids) > 0:
        print("\n## contraction_edge in named_contraction_order ##\n")

        contraction_edge_index = contraction_ids[0]

        edge = edge_list[contraction_edge_index]
        input_node = mansikka_node_map[edge["inp"]]
        output_node = mansikka_node_map[edge["out"]]

        print("## edge under contraction:", contraction_edge_index)
        print("## input:{} | output:{}".format(edge["inp"], edge["out"]))


        ni_axes = [] # contraction axes for the input node.
        no_axes = [] # contraction axes for the output node.
        # joint_edges = []      # joint edges between the nodes.
                     # We will need ned to contract over all of the joint edges.

        print("\ninput node edges:", input_node.edges)
        op = [(edge_list[k]["inp"], edge_list[k]["out"]) for k in input_node.edges]
        print("input node edges:", op)

        # Populate the contraction axes and joint edges
        for i, inp_edge in enumerate(input_node.edges):
            print("#### Populate the contraction axes and joint edges ####")
            print("#### edge_axes in input:{}\n edge:{}".format(i, inp_edge))
            if inp_edge in output_node.edges:
                if edge_list[inp_edge]["inp"] in pyzx_graph.inputs():
                    ni_axes.append(1)#1
                else:
                    ni_axes.append(i)
                no_axes.append(output_node.edges.index(inp_edge))
                joint_edges.append(inp_edge)
            else:
                # update the new ends of edge
                if edge_list[inp_edge]["inp"] == edge["inp"]:
                    edge_list[inp_edge]["inp"] = edge["out"]
                if edge_list[inp_edge]["out"] == edge["inp"]:
                    edge_list[inp_edge]["out"] = edge["out"]

#######################################################################################################
#         # treat the Hadamard edges
#         # need to be verrified
#         for c_e in joint_edges:
#             print("#### Treat Hadamard edges ####")
#             if edge_list[c_e]["type"] == EdgeType.HADAMARD:
#                 # get the contraction axes in input tensor
#                 inp_axis_connected_with_had = [input_node.edges.index(c_e)]
#                 # contract with the hadamard and update the input tensor
#                 new_tensor = np.tensordot(
#                     input_node.tensor,had, axes=(inp_axis_connected_with_had,[1] )
#                 )
#                 input_node.set_tensor(new_tensor)
#
#                 # remove contracted edge and add it again at the right place
#                 print("#### input node edges before H:", input_node.edges)
#                 input_node.edges.remove(c_e)
#                 input_node.edges.insert(0,c_e)#append(c_e)
#                 print("#### input node edges after H:", input_node.edges)
#
#                 # recalculate contraction axes for the modified tensor
#                 print("#### recalculate input axes ")
#                 ni_axes = []
#                 for i, inp_edge in enumerate(input_node.edges):
#                     print("## edge_axes in input:{}\nedge:{}".format(i, inp_edge))
#                     if inp_edge in output_node.edges:
#                         ni_axes.append(i)
# ######################################################################################################


        print("\n##!!calculate new tensor!!##")

        print("inp _tensor:\n",input_node.tensor)
        print("output _tensor:\n",output_node.tensor)
        print("ni:{}|no{}".format(ni_axes,no_axes))
        new_tensor = np.tensordot(
             input_node.tensor,output_node.tensor, axes=(ni_axes, no_axes)
        )
        output_node.set_tensor(new_tensor)
        if (len(contraction_ids)>=1):
            print("@@@@@@@@@@@tensor@@@@@@@@@@@@@@@")
            print(new_tensor)
            print("@@matrix@@")
            print(pyzxtensor.tensor_to_matrix(new_tensor, 2, 2))
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@")

        # update node edges
        output_node.update_edges(input_node)

        # remove the node
        mansikka_node_map.pop(edge["inp"])
        print("# remaining nodes#")
        print(" \nnodes:{} \n".format(mansikka_node_map.keys()))

        # update the edge_list
        # remove contracted edges
        print("deprecate edges:", joint_edges)
        for deprecate_edge in joint_edges:
            if deprecate_edge in contraction_ids:
                contraction_ids.remove(deprecate_edge)
                edge_list.pop(deprecate_edge)
        print("\n##updated edge list##")
        print("update contraction order:", contraction_ids)
        print("update edge list:", edge_list)
        print("######################")

    print("####\n Remaining edges:{} \n \n####".format(contraction_ids))
    print("\n ###### final nodes:{}\n#### ".format(mansikka_node_map))

    if (len(mansikka_node_map.keys())>1):
        raise Exception("You have more than one final node. This means that you can treat your circuit as two separate circuits!")
        return 1

    for node in mansikka_node_map:
        tensor = mansikka_node_map[node].tensor
        break

    perm = []
    for i in range(2 * len(pyzx_graph.inputs())):
        perm.append(len(pyzx_graph.inputs())-i-1)
        # perm=[3,2,1,0]
    tensor = np.transpose(tensor, perm)


    print("tensor:", tensor)
    if preserve_scalar:
        tensor *= pyzx_graph.scalar.to_number()

    print("tensor shape:", tensor.shape)
    print("final tensor:", tensor)
    print("final mat:", pyzxtensor.tensor_to_matrix(tensor, 2, 2))

    return tensor


def reorder_contraction_edge_list(contraction_edge_list, nr_vert, pyzx_graph):
    # move the ede with  inputs at the end
    input_edge_list = [-1] * pyzx_graph.num_inputs()
    # move the ede with  output at the end
    output_edge_list = [-1] * pyzx_graph.num_outputs()
    for edge in contraction_edge_list:
        if edge[0] in pyzx_graph.inputs():
            position = pyzx_graph.num_inputs() - edge[0] - 1
            input_edge_list[position] = edge
        elif edge[1] in pyzx_graph.outputs():
            position = pyzx_graph.num_outputs() - (nr_vert - edge[1] - 1) - 1
            output_edge_list[position] = edge
        # print("index", position)
    for edge in input_edge_list:
        contraction_edge_list.remove(edge)
    for edge in output_edge_list:
        contraction_edge_list.remove(edge)
    contraction_edge_list.extend(input_edge_list)
    contraction_edge_list.extend(output_edge_list)
    print("input_order:", input_edge_list)
    print("output_order:", output_edge_list)


def get_nodes_edges(pyzx_graph: "BaseGraph[VT,ET]"):
    node_map = {}
    edge_map = {}

    # The the key to the edge in edge_map is an integer
    # Each edge has an input, an output amnd a type.
    # In the beginning, the nodes are labeled in such a way that
    # the one with the smaller index is the input.
    # The type may indicate the presence of a Hadamard between the end vertices.
    edge_key = 0
    for edg in pyzx_graph.edges():
        edge_map[edge_key] = {
            "inp": min(edg),
            "out": max(edg),
            "type": pyzx_graph.edge_type(edg),
        }
        edge_key = edge_key + 1

    # The key of a node will be its initial index
    for v in pyzx_graph.vertices():
        node = MansikkaNode(v, pyzx_graph)
        node_map[v] = node

    return node_map, edge_map


def input_to_tensor() -> np.ndarray:
    return np.identity(2)  # np.identity(2) #np.array([1, 0])

def output_to_tensor() -> np.ndarray:
    return np.identity(2)  # np.array([1, 0])

def get_tensor_from_g(pyzx_graph, v):
    phase = math.pi * pyzx_graph.phases()[v]
    v_type = pyzx_graph.types()[v]
    arity = len(pyzx_graph.neighbors(v))

    # input
    # output
    if v in pyzx_graph.inputs():
        return input_to_tensor()
    if v in pyzx_graph.outputs():
        return output_to_tensor()  # np.identity(2)

    if v_type == 1:
        t = pyzxtensor.Z_to_tensor(arity, phase)
    elif v_type == 2:
        t = pyzxtensor.X_to_tensor(arity, phase)
    elif v_type == 3:
        t = pyzxtensor.H_to_tensor(arity, phase)
    else:
        raise ValueError(
            "Vertex %s has non-ZXH type but is not an input or output" % str(v)
        )

    return t

def print_edgelist(edgelist):
    el = [(edgelist[k]["inp"], edgelist[k]["out"]) for k in edgelist.keys()]
    print("edgelist:", el)