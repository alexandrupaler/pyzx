"""

"""

from typing import Optional

import numpy as np
import pyzx as zx

from mcsim import BaseExtractor


class ReorderExtractor(BaseExtractor):
    def extract(self, graph, help_view=1):
        """"""

        # this part is just for verification and checking
        if help_view:
            print(graph.stats())
            print("rows:", graph.rows())
            print(graph.rows()[0])
            print("phases:", graph.phases())
            print()
            print("types:", graph.types())
            print()
            print("depth:", graph.depth())
            print()
            print("vertices:", graph.vertices)
            zx.draw(graph, labels=True)

        ##################################

        # heur 1
        initial_depth = graph.depth()
        l = 1
        new = initial_depth + 1
        k = len(graph.inputs)
        # print("k===",k)
        for g in graph.vertices():
            if (graph.row(g) != 0) and (graph.row(g) != graph.depth()):
                if graph.vertex_degree(g) > 3:
                    graph.set_row(g, new)
                    l = l + 1
                    if l % k == 0:
                        new = new + 1
                        l = 1
            if graph.row(g) == initial_depth:
                graph.set_row(g, new + 1)

        # this part is just for verification and checking
        if help_view:
            print("rows:", graph.rows())
            print(graph.rows()[0])
            print("phases:", graph.phases())
            print()
            print("types:", graph.types())
            print()
            print("depth:", graph.depth())
            print()
            print("vertices:", graph.vertices)
            zx.draw(graph, labels=True)
        ##################################

        return graph.to_matrix()
