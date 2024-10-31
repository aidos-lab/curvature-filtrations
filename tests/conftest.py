import pytest
import networkx as nx


@pytest.fixture
def graph():
    return nx.erdos_renyi_graph(100, 0.1)


@pytest.fixture
def empty_graph():
    return nx.Graph()


@pytest.fixture
def regular_homology_dims():
    return [0, 1]
