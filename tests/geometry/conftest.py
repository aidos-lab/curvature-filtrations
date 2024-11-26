import pytest
import networkx as nx


@pytest.fixture
def simple_graph():
    """Fixture to create a simple graph for testing."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return G


@pytest.fixture
def main_figure_graph():
    """Returns a simple graph with Forman curvature."""
    G = nx.Graph()
    # Add nodes
    G.add_nodes_from(range(8))

    # Add edges
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 5),
        (4, 5),
        (5, 6),
        (3, 7),
    ]
    G.add_edges_from(edges)

    return G
