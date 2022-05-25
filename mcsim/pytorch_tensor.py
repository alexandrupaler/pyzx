import math
import torch

from pyzx.utils import EdgeType
import pyzx.tensor as pyzxtensor

from .mcsim_node import MansikkaNode

import time
#from  memory_profiler import profile
import gc

#from memory_profiler import profile

#@profile()
def mcsim_tensorfy(pyzx_graph, contraction_edge_list, preserve_scalar: bool = True):
    """

    """
    tic = time.perf_counter()
    print("\n################## pytorch tensorfy ####################")

    # Dictionaries with the nodes and edges in the graph.
    mansikka_node_map, mansikka_edge_map = get_nodes_edges(pyzx_graph)

    # Eliminate H edges:
    convert_hadamard_edges(mansikka_edge_map, mansikka_node_map, pyzx_graph)

    nr_vert = pyzx_graph.num_vertices()
    reorder_contraction_edge_list(contraction_edge_list, nr_vert, pyzx_graph)

    # Contracting order is provided like a list of tuples, and now we change it into a list of ids.
    edge_list = list(pyzx_graph.edges())
    contraction_ids = [
        edge_list.index(edge) for edge in contraction_edge_list
    ]
    # print("graph edge_list :", mansikka_edge_map)
    # print("contraction_order 0:", contraction_ids)

    # TODO  Alexandru: Check if it's a compact circuit

    nr_edges_to_contract = len(contraction_ids) - (pyzx_graph.num_outputs() + pyzx_graph.num_inputs())

    # Do not contract edges  connecting to input and output nodes
    nr_do_not_contract = pyzx_graph.num_outputs() + pyzx_graph.num_inputs()
    while len(contraction_ids) > nr_do_not_contract:

        #print("Contraction status: {}/{}".format(len(contraction_ids) - nr_do_not_contract, nr_edges_to_contract))
        # print("\n## contraction_edge in named_contraction_order ##\n")

        contraction_edge_index = contraction_ids[0]
        edge = mansikka_edge_map[contraction_edge_index]
        mansikka_input_node = mansikka_node_map[edge["inp"]]
        mansikka_output_node = mansikka_node_map[edge["out"]]

        # print("## edge under contraction:", contraction_edge_index)
        # print("## input:{} | output:{}".format(edge["inp"], edge["out"]))

        input_axes = []  # contraction axes for the input node.
        output_axes = []  # contraction axes for the output node.

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
        del mansikka_node_map[edge["inp"]]
        #gc.collect()
        # print("# remaining nodes#")
        # print(" \nnodes:{} \n".format(mansikka_node_map.keys()))

        # update the edge list
        # remove contracted edges
        for deprecate_edge_id in edge_id_and:
            if deprecate_edge_id in contraction_ids:
                contraction_ids.remove(deprecate_edge_id)
                mansikka_edge_map.pop(deprecate_edge_id)
        del deprecate_edge_id
        #gc.collect()
        # calculate new tensor and update the output node tensor
        # print("ni:{}|no{}".format(input_axes, output_axes))
        # new_tensor = np.tensordot(
        #     mansikka_input_node.tensor, mansikka_output_node.tensor, axes=(input_axes, output_axes)
        # )
        mansikka_output_node.set_tensor(torch.tensordot(
            a=mansikka_input_node.tensor, b=mansikka_output_node.tensor, dims=(input_axes, output_axes)
        ))

        # update output node
        mansikka_output_node.update_edges_in_tensor(mansikka_input_node, mansikka_edge_map)
        del(mansikka_input_node)
        #gc.collect()

    tensor = mansikka_output_node.tensor
    if preserve_scalar:
        tensor *= pyzx_graph.scalar.to_number()

    # print("\n################## msc tensorfy -done- ####################")
    toc = time.perf_counter()
    print(f"Time in mcsim_tensorfy {toc - tic:0.4f} seconds")

    return tensor.numpy()

#@profile()
def convert_hadamard_edges(mansikka_edge_map, mansikka_node_map, pyzx_graph):
    # Eloiminate Hadamrd edges
    had_tensor = 1 / math.sqrt(2) * torch.tensor([[1, 1], [1, -1]],dtype=torch.cfloat)  # --0--H--1--
    for edge_key in mansikka_edge_map:
        edge = mansikka_edge_map[edge_key]
        if edge["type"] == EdgeType.HADAMARD:  # hadamard
            mansikka_edge_map[edge_key]["type"] = EdgeType.SIMPLE
            if edge["inp"] not in pyzx_graph.inputs():
                new_tensor = torch.tensordot(mansikka_node_map[edge["inp"]].tensor, had_tensor,
                                          dims=([mansikka_node_map[edge["inp"]].edge_ids.index(edge_key)], [0]))
                new_edge_order = mansikka_node_map[edge["inp"]].edge_ids
                new_edge_order.remove(edge_key)
                new_edge_order.append(edge_key)
                # transposition_order = [mansikka_node_map[edge["inp"]].edge_ids.index(k) for k in new_edge_order]
                mansikka_node_map[edge["inp"]].set_tensor(new_tensor)
            else:
                new_tensor = torch.tensordot(a=had_tensor, b=mansikka_node_map[edge["out"]].tensor,
                                            dims=([1], [mansikka_node_map[edge["out"]].edge_ids.index(edge_key)]))
                new_edge_order = mansikka_node_map[edge["out"]].edge_ids
                new_edge_order.remove(edge_key)
                new_edge_order.insert(0, edge_key)
                # transposition_order = [mansikka_node_map[edge["out"]].edge_ids.index(k) for k in new_edge_order]
                mansikka_node_map[edge["out"]].set_tensor(new_tensor)

#@profile()
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
    # print("input_order:", input_edge_list)
    # print("output_order:", output_edge_list)

#@profile()
def get_nodes_edges(pyzx_graph):
    node_map = {}
    edge_map = {}

    # The key to the edge in edge_map is an integer
    # Each edge has an input, an output and a type.
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



def get_tensor_from_g(pyzx_graph, v):
    phase = math.pi * pyzx_graph.phases()[v]
    v_type = pyzx_graph.types()[v]
    arity = len(pyzx_graph.neighbors(v))

    # input
    # output
    if v in pyzx_graph.inputs():
        return 0
    if v in pyzx_graph.outputs():
        return 0  # np.identity(2)

    if v_type == 1:
        t = torch.from_numpy(pyzxtensor.Z_to_tensor(arity, phase)).type(torch.cfloat)
    elif v_type == 2:
        t = torch.from_numpy(pyzxtensor.X_to_tensor(arity, phase)).type(torch.cfloat)
    elif v_type == 3:
        t = torch.from_numpy(pyzxtensor.H_to_tensor(arity, phase)).type(torch.cfloat)
    else:
        raise ValueError(
            "Vertex %s has non-ZXH type but is not an input or output" % str(v)
        )

    return t
