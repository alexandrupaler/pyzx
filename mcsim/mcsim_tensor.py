import math

import numpy

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
        "\n################## msc tensorfy ####################"
    )

    # Hadamard gate tensor will be used for the Hadamard edges.
    had = 1 / math.sqrt(2) * np.array([[1, 1], [1, -1]])

    # Dictionaries with the nodes and edges in the graph.
    mansikka_node_map, mansikka_edge_map = get_nodes_edges(pyzx_graph)

    # Eliminate H edges:
    for edge_key in mansikka_edge_map:
        edge = mansikka_edge_map[edge_key]
        if edge["type"] == 2: # hadamard
            had_tensor = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])#--0--H--1
            if edge["inp"] not in pyzx_graph.inputs():
                new_tensor = numpy.tensordot(mansikka_node_map[edge["inp"]].tensor,had_tensor, axes=([mansikka_node_map[edge["inp"]].edge_ids.index(edge_key)],[0]))
                new_edge_order = mansikka_node_map[edge["inp"]].edge_ids
                new_edge_order.remove(edge_key)
                new_edge_order.append(edge_key)
                transposition_order =[mansikka_node_map[edge["inp"]].edge_ids.index(k) for k in new_edge_order ]
                new_tensor = new_tensor.transpose(transposition_order)
                mansikka_node_map[edge["inp"]].set_tensor(new_tensor)
            else :
                new_tensor = numpy.tensordot(had_tensor, mansikka_node_map[edge["out"]].tensor, axes=([1],[mansikka_node_map[edge["out"]].edge_ids.index(edge_key)]))
                new_edge_order = mansikka_node_map[edge["out"]].edge_ids
                new_edge_order.remove(edge_key)
                new_edge_order.insert(0,edge_key)
                transposition_order = [mansikka_node_map[edge["out"]].edge_ids.index(k) for k in new_edge_order]
                new_tensor = new_tensor.transpose(transposition_order)
                mansikka_node_map[edge["out"]].set_tensor(new_tensor)


    nr_vert = pyzx_graph.num_vertices()

    reorder_contraction_edge_list(contraction_edge_list, nr_vert, pyzx_graph)

    # Contracting order is provided like a list of tuples, and now we change it into a list of ids.
    edge_list = list(pyzx_graph.edges())
    contraction_ids = [
        edge_list.index(edge) for edge in contraction_edge_list
    ]

    print("graph edge_list :", mansikka_edge_map)
    print("contraction_order 0:", contraction_ids)




    # Do not contract edges  connecting to input and output nodes
    nr_do_not_contract = pyzx_graph.num_outputs()+pyzx_graph.num_inputs()
    while len(contraction_ids) > nr_do_not_contract:

        print("\n## contraction_edge in named_contraction_order ##\n")

        contraction_edge_index = contraction_ids[0]

        edge = mansikka_edge_map[contraction_edge_index]
        mansikka_input_node = mansikka_node_map[edge["inp"]]
        mansikka_output_node = mansikka_node_map[edge["out"]]

        print("## edge under contraction:", contraction_edge_index)
        print("## input:{} | output:{}".format(edge["inp"], edge["out"]))

        input_axes = [] # contraction axes for the input node.
        output_axes = [] # contraction axes for the output node.

        edge_id_and, edge_id_xor = mansikka_output_node.edge_set_and_xor(mansikka_input_node)

        # These are the axes the will be removed
        for edgex in edge_id_and:
            if mansikka_edge_map[edgex]["inp"] in pyzx_graph.inputs():
                input_axes.append(1)  # 1
            else:
                input_axes.append(mansikka_input_node.edge_ids.index(edgex))

            output_axes.append(mansikka_output_node.edge_ids.index(edgex))

        # For remaining edges, update the end points
        for edgex in edge_id_xor:
            if mansikka_edge_map[edgex]["inp"] == edge["inp"]:
                mansikka_edge_map[edgex]["inp"] = edge["out"]
            if mansikka_edge_map[edgex]["out"] == edge["inp"]:
                mansikka_edge_map[edgex]["out"] = edge["out"]

        # remove the input node from the map
        mansikka_node_map.pop(edge["inp"])
        print("# remaining nodes#")
        print(" \nnodes:{} \n".format(mansikka_node_map.keys()))

        # update the edge list
        # remove contracted edges
        for deprecate_edge_id in edge_id_and:
            if deprecate_edge_id in contraction_ids:
                contraction_ids.remove(deprecate_edge_id)
                mansikka_edge_map.pop(deprecate_edge_id)

        # calculate new tensor and update the output node tensor
        print("ni:{}|no{}".format(input_axes,output_axes))
        new_tensor = np.tensordot(
             mansikka_input_node.tensor,mansikka_output_node.tensor, axes=(input_axes, output_axes)
        )
        mansikka_output_node.set_tensor(new_tensor)

        # update output node
        mansikka_output_node.update_edges_in_tensor(mansikka_input_node, mansikka_edge_map)


    tensor = mansikka_output_node.tensor
    if preserve_scalar:
        tensor *= pyzx_graph.scalar.to_number()

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




#######################################################################################################
        # # treat the Hadamard edges
        # # need to be verrified
        # for c_e in edge_id_and:
        #     print("#### Treat Hadamard edges ####")
        #     if mansikka_edge_map[c_e]["type"] == EdgeType.HADAMARD:
        #         # get the contraction axes in input tensor
        #         print("mansfikka_input_node.edge_ids:",mansikka_input_node.edge_ids)
        #         inp_axis_connected_with_had = [mansikka_input_node.edge_ids.index(c_e)]
        #         # contract with the hadamard and update the input tensor
        #         new_tensor = np.tensordot(
        #             mansikka_input_node.tensor,had, axes=(inp_axis_connected_with_had,[1] )
        #         )
        #         mansikka_input_node.set_tensor(new_tensor)
        #
        #         # remove contracted edge and add it again at the right place
        #         print("#### input node edges before H:", mansikka_input_node.edge_ids)
        #         mansikka_input_node.edge_ids.remove(c_e)
        #         mansikka_input_node.edge_ids.insert(0,c_e)#append(c_e)
        #         print("#### input node edges after H:", mansikka_input_node.edge_ids)
        #
        #         # recalculate contraction axes for the modified tensor
        #         print("#### recalculate input axes ")
        #         input_axes = []
        #         for i, inp_edge in enumerate(mansikka_input_node.edge_ids):
        #             print("## edge_axes in input:{}\nedge:{}".format(i, inp_edge))
        #             if inp_edge in mansikka_output_node.edge_ids:
        #                 input_axes.append(i)
# ######################################################################################################




