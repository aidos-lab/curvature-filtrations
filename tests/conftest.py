import pytest
import networkx as nx
import numpy as np


@pytest.fixture
def graph():
    return nx.erdos_renyi_graph(100, 0.1)


@pytest.fixture
def graph2():
    return nx.erdos_renyi_graph(100, 0.1)


@pytest.fixture
def small_graph():
    return nx.erdos_renyi_graph(20, 0.5)


@pytest.fixture
def empty_graph():
    return nx.Graph()


@pytest.fixture
def graph_distribution1():
    return [nx.erdos_renyi_graph(100, np.random.rand()) for _ in range(8)]


@pytest.fixture
def graph_distribution2():
    return [nx.erdos_renyi_graph(100, np.random.rand()) for _ in range(10)]


@pytest.fixture
def regular_homology_dims():
    return [0, 1]


@pytest.fixture
def toy_diagram1():
    """Toy persistence diagram with 0D and 1D homology features."""
    return [{0: np.array([[0, 1], [1, 2]]), 1: np.array([[0.5, 2.0], [1.5, 2.5]])}]


@pytest.fixture
def toy_diagram2():
    """Toy persistence diagram with 0D and 1D homology features."""
    return [{0: np.array([[0, 1], [1, 3]]), 1: np.array([[0.5, 2.1], [1.5, 3.0]])}]


@pytest.fixture
def dummy_diagram():
    """Toy input for making a Diagram object (output of the persistent homology )"""
    return {0: np.array([[-24.0, -20.0], [-23.0, -19.0], [-19.0, -18.0]]), 1: np.array([-21, -20])}
