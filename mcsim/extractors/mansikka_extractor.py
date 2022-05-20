"""

"""

import numpy as np
import pyzx

from .base_extractor import BaseExtractor
from .mansikka_graph import MansikkaGraph

from mcsim.mcsim_tensor import mcsim_tensorfy


class MansikkaExtractor(BaseExtractor):
    def extract(self, graph: pyzx.graph, show_changes=0):
        """
        Model of extractor that tries to take advance of specific distribution of degre/rerows
        :graph: the pyzx graph whose matrix we want to extract
        :show_changes: just for learning can be deleted when merge
        :return: graph matrix
        """

        # print("graph_ edges:", graph.edge_set())
        working_graph = MansikkaGraph([k for k in graph.vertices()], graph.edge_set().copy())
        working_graph = working_graph.construct_dual()
        # print("dual vert:", working_graph.vertices)
        # print("dual edges: ", working_graph.edges)

        contraction_order = get_contraction_order(
            working_graph, self.params["m"], self.params["nr_iter"]
        )
        # print("contraction order: ", contraction_order)
        #
        # print("Ok here !!")

        # return graph.to_matrix()
        return graph.to_matrix(
            my_tensorfy = mcsim_tensorfy,
            contraction_order = contraction_order
        )  # mcs_tensorfy(contraction_order, graph, preserve_scalar=True)


def get_contraction_order(graph, nr_tensors_to_rem, nr_iter=10):

    working_graph = MansikkaGraph(graph.vertices.copy(), graph.edges.copy())
    initial_order = [k for k in graph.vertices]
    contraction_order = []

    initial_tw = working_graph.find_treewidth_from_order(initial_order)

    if initial_tw == 1:
        return initial_order

    while nr_iter > 0:
        nr_iter = nr_iter - 1
        reduced_g, reduced_order, tw, removing_order = \
            working_graph.greedy_treewidth_deletion(
            initial_order, nr_tensors_to_rem
        )
        for node in removing_order:
            contraction_order.append(node)

        initial_order = reduced_order.copy()
        working_graph = reduced_g
        if tw == 1:
            break

    for node in reduced_order:
        contraction_order.append(node)

    return contraction_order


def reorder_indices(
    pyzx_graph,
    contraction_order,
):

    start_index = max(contraction_order) + 1
    node_map = {}

    for v in range(len(contraction_order)):
        node_map[contraction_order[v]] = start_index + v

    new_graph = pyzx.Graph()

    for v in pyzx_graph.vertices():
        new_graph.add_vertex_indexed(index=node_map[v])
        new_graph.set_type(node_map[v], pyzx_graph.type(v))
        new_graph.set_qubit(node_map[v], pyzx_graph.qubit(v))
        new_graph.set_row(node_map[v], pyzx_graph.row(v))
        new_graph.set_phase(node_map[v], pyzx_graph.phase(v))

    for edge in pyzx_graph.edge_set():
        new_graph.add_edge(
            new_graph.edge(node_map[edge[0]], node_map[edge[1]]),
            edgetype=pyzx_graph.edge_type(edge),
        )

    inp = []
    for i in pyzx_graph.inputs():
        inp.append(node_map[i])
    new_graph.set_inputs(inp)

    out = []
    for o in pyzx_graph.outputs():
        out.append(node_map[o])
    new_graph.set_outputs(out)

    new_graph.scalar = pyzx_graph.scalar

    return new_graph
