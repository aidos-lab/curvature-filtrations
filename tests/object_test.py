
from cfggme.curvature import Curvature
import networkx as nx
import numpy as np

def test_should_work():
    assert 1 + 1 == 2

def test_create_object():
    obj = Curvature()
    assert type(Curvature) is Curvature

# class TestFormanCurvature:
#     def test_line_graph(self):
#         G = nx.path_graph(5)

#         nx.set_edge_attributes(G, 1.0, "weight")
#         nx.set_node_attributes(G, 1.0, "weight")

#         curvature = forman_curvature(G)
#         weighted_curvature = forman_curvature(G, "weight")

#         assert all(curvature == weighted_curvature)

#         G[1][2]["weight"] = 2.0
#         G[2][3]["weight"] = 2.0

#         curvature = forman_curvature(G)
#         weighted_curvature = forman_curvature(G, "weight")

#         assert not all(curvature == weighted_curvature)

#         nx.set_edge_attributes(G, 1.0, "weight")
#         #propagate_edge_attribute_to_nodes(G, "weight")