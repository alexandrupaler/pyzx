"""

"""

import numpy as np
import pyzx

from .base_extractor import BaseExtractor
from .graph import Graph
from .treewidth_from_order import find_treewidth_from_order
from .greedy_treewidth_deletion import greedy_treewidth_deletion


class MansikkaExtractor(BaseExtractor):
    def extract(self, graph: pyzx.graph, show_changes=0):
        """
        Model of extractor that tries to take advance of specific distribution of degre/rerows
        :graph: the pyzx graph whose matrix we want to extract
        :show_changes: just for learning can be deleted when merge
        :return: graph matrix
        """

        working_graph = Graph([k for k in graph.vertices()], graph.edge_set().copy())
        # working_graph = working_graph.construct_dual()
        # print("dual vert:", dual_graph.vertices)
        # print("dual edges: ", dual_graph.edges)
        """
        contraction_order = get_contraction_order(
            working_graph, self.params["m"], self.params["nr_iter"]
        )
        print("contraction order: ", contraction_order)

        new_graph = reorder_indices(graph, contraction_order)

        pyzx.draw_matplotlib(
            graph,
            labels=False,
            figsize=(8, 2),
            h_edge_draw="blue",
            show_scalar=False,
            rows=None,
        ).savefig("graph_1.png")
        pyzx.draw_matplotlib(
            new_graph,
            labels=False,
            figsize=(8, 2),
            h_edge_draw="blue",
            show_scalar=False,
            rows=None,
        ).savefig("graph_2.png")
        
        return new_graph.to_matrix()
        """
        print("Ok here !!")
        return graph.to_matrix()


def get_contraction_order(graph, m, nr_iter=10):

    working_graph = Graph(graph.vertices.copy(), graph.edges.copy())
    initial_order = [k for k in graph.vertices]
    contraction_order = []

    initial_tw = find_treewidth_from_order(working_graph, initial_order)

    if initial_tw == 1:
        return initial_order

    while nr_iter > 0:
        nr_iter = nr_iter - 1
        reduced_g, reduced_order, tw, removing_order = greedy_treewidth_deletion(
            working_graph, initial_order, m
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
