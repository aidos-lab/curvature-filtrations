
from curvature import Curvature
import networkx as nx
import numpy as np
import methods

def test_pytest():
    assert 1 + 1 == 2

# Create of Curvature object works with no inputs
def test_create_object():
    obj = Curvature()
    assert type(obj) == Curvature

# Checks that default values are assigned properly
def test_defaults():
    obj = Curvature()
    assert obj.method == "forman_curvature"
    assert obj.weight == None
    assert obj.alpha ==  0.0
    assert obj.alpha == 0.0
    assert obj.prob_fn == None

# Curvature computation tests on ER graphs
def test_er_forman():
    G = nx.erdos_renyi_graph(100, 0.1)
    obj = Curvature()
    curvature_1 = obj.fit(G)
    curvature_2 = methods.forman_curvature(G)
    assert all(curvature_1 == curvature_2)

def test_er_orc():
    G = nx.erdos_renyi_graph(100, 0.1)
    obj = Curvature(method="ollivier_ricci_curvature")
    curvature_1 = obj.fit(G)
    curvature_2 = methods.ollivier_ricci_curvature(G)
    assert all(curvature_1 == curvature_2)

def test_er_resistance():
    G = nx.erdos_renyi_graph(100, 0.1)
    obj = Curvature(method="resistance_curvature")
    curvature_1 = obj.fit(G)
    curvature_2 = methods.resistance_curvature(G)
    assert all(curvature_1 == curvature_2)

    # alpha_obj = Curvature(method="resistance_curvature", alpha=0.1)
    # alpha_1 = alpha_obj.fit(G)
    # alpha_2 = methods.resistance_curvature(G, alpha=0.1)
    # assert all(alpha_1 == alpha_2)


# Test adapted from Nammu
def test_weighted_forman():
    unweighted = Curvature()
    weighted = Curvature(weight = "weight")
    G = nx.path_graph(5)

    nx.set_edge_attributes(G, 1.0, "weight")
    nx.set_node_attributes(G, 1.0, "weight")

    curvature = unweighted.fit(G)
    #notice here instead of adding 'weight' to parameters, we change the curvature object
    weighted_curvature = weighted.fit(G)

    assert all(curvature == weighted_curvature)

    # changing weights
    G[1][2]["weight"] = 2.0
    G[2][3]["weight"] = 2.0

    curvature = weighted.fit(G)
    weighted_curvature = unweighted.fit(G)

    assert not all(curvature == weighted_curvature)

# Topology basic tests
def test_make_landscape():
    G = nx.erdos_renyi_graph(100, 0.1)
    obj = Curvature()
    obj.make_landscape(G)