import pyzx

class MansikkaNode:
    def __init__(self, v, pyzx_graph):
        self.index = v
        self.tensor = get_tensor_from_g(pyzx_graph, v)

        assert (pyzx_graph.edges() is pyzx.graph.graph_s.GraphS)

        self.edge_ids = []
        for i, edge in enumerate(pyzx_graph.edges()):
            if v in edge:
                self.edge_ids.append(i)

    def set_tensor(self, tensor):
        self.tensor = tensor

    def update_edges(self, other_node):
        joint_edges, xor_edges = self.edge_set_and_xor(other_node)

        n_edge_ids = []
        for e in other_node.edges:
            if e not in joint_edges:
                n_edge_ids.append(e)

        for e in self.edges:
            if e not in joint_edges:
                n_edge_ids.append(e)

        self.edges_ids = n_edge_ids

    def edge_set_and_xor(self, other_node):
        intersection = []
        xor_edges = []

        for edge_id in self.edges_ids:
            if edge_id in other_node.edge_ids:
                intersection.append(edge_id)
            else:
                xor_edges.append(edge_id)

        for edge_id in other_node.edges_ids:
            if edge_id not in self.edge_ids:
                xor_edges.append(edge_id)

        return intersection, xor_edges