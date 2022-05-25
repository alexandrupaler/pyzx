import pyzx
import torch
import mcsim


class MansikkaNode:
    def __init__(self, v, pyzx_graph):
        self.index = v
        self.tensor = mcsim.pytorch_tensor.get_tensor_from_g(pyzx_graph, v)

        assert (isinstance(pyzx_graph, pyzx.graph.graph_s.GraphS))

        self.edge_ids = sort_edges(pyzx_graph, v)
        """
        self.edge_ids = []
        for i, edge in enumerate(pyzx_graph.edges()):
            if v in edge:
                self.edge_ids.append(i)
        """

    def set_tensor(self, tensor):
        self.tensor = tensor

    def update_edges_in_tensor(self, other_node, mansikka_edge_map):
        joint_edges, xor_edges = self.edge_set_and_xor(other_node)

        # The order of edges after numpy tensor dot product
        n_edge_ids = []
        for e in other_node.edge_ids:
            if e not in joint_edges:
                n_edge_ids.append(e)

        for e in self.edge_ids:
            if e not in joint_edges:
                n_edge_ids.append(e)

        # Sort the edges according to the transversal order of the contraction
        input_edges = []
        output_edges = []
        for edge_id in n_edge_ids:
            if mansikka_edge_map[edge_id]["inp"] == self.index:
                input_edges.append(edge_id)
            else:
                output_edges.append(edge_id)

        input_edges.sort(key=lambda edg: (mansikka_edge_map[edg]["inp"], mansikka_edge_map[edg][
            "out"], edg))
        output_edges.sort(key=lambda edg: (mansikka_edge_map[edg]["inp"], mansikka_edge_map[edg][
            "out"], edg))
        input_edges.extend(output_edges)

        transposition_order = [n_edge_ids.index(e) for e in input_edges]
        self.tensor = torch.permute(self.tensor,transposition_order)
        self.edge_ids = input_edges

    def edge_set_and_xor(self, other_node):
        intersection = []
        xor_edges = []

        for edge_id in self.edge_ids:
            if edge_id in other_node.edge_ids:
                intersection.append(edge_id)
            else:
                xor_edges.append(edge_id)

        for edge_id in other_node.edge_ids:
            if edge_id not in self.edge_ids:
                xor_edges.append(edge_id)

        return intersection, xor_edges


def sort_edges(pyzx_graph, v):
    neighbors = sorted(pyzx_graph.neighbors(v))
    edge_ids = [-1 for _ in range(len(neighbors))]
    for i, edge in enumerate(pyzx_graph.edges()):
        if v == edge[1]:
            edge_ids[neighbors.index(edge[0])] = i
        if v == edge[0]:
            edge_ids[neighbors.index(edge[1])] = i
    return edge_ids
