from cfggme.curvature import Curvature
from cfggme.compare import GraphComparator
import networkx as nx
import numpy as np


def test_pytest():
    assert 1 + 1 == 2


def test_self():
    curv = Curvature()
    graph1 = nx.erdos_renyi_graph(100, 0.1)
    graph2 = graph1
    comp = GraphComparator(graph1, graph2, curv)
    assert comp.calc_total_curv_diff() == 0


def test_placeholder_comparison():
    curv = Curvature()
    graph1 = nx.erdos_renyi_graph(100, 0.2)
    graph2 = nx.erdos_renyi_graph(100, 0.1)
    comp = GraphComparator(graph1, graph2, curv)
    assert comp.calc_total_curv_diff() != 0


def test_same_landscape():
    curv = Curvature()
    graph1 = nx.erdos_renyi_graph(100, 0.1)
    graph2 = graph1
    comp = GraphComparator(graph1, graph2, curv)
    assert comp.curvature_filtration_distance() == 0


def test_diff_landscape():
    curv = Curvature()
    graph1 = nx.erdos_renyi_graph(100, 0.1)
    graph2 = nx.erdos_renyi_graph(100, 0.15)
    comp = GraphComparator(graph1, graph2, curv)
    assert comp.curvature_filtration_distance() != 0

    graph3 = nx.erdos_renyi_graph(100, 0.3)
    comp2 = GraphComparator(graph1, graph3, curv)

    # graphs 1 and 2 should have more similar curvature than graphs 1 and 3
    assert (
        comp.curvature_filtration_distance()
        < comp2.curvature_filtration_distance()
    )
