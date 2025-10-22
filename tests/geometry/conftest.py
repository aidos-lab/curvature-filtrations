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


@pytest.fixture
def string_node_graph():
    """Fixture to create a graph with string node labels."""
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("A", "C")])
    return G


@pytest.fixture
def mixed_node_graph():
    """Fixture to create a graph with mixed node types."""
    G = nx.Graph()
    G.add_edges_from([("A", 1), (1, "B"), ("B", 2), (2, "C"), ("A", 2)])
    return G


@pytest.fixture
def tuple_node_graph():
    """Fixture to create a graph with tuple node labels."""
    G = nx.Graph()
    G.add_edges_from(
        [
            ((0, 0), (0, 1)),
            ((0, 1), (1, 1)),
            ((1, 1), (1, 0)),
            ((1, 0), (0, 0)),
            ((0, 0), (1, 1)),  # diagonal edge
        ]
    )
    return G


@pytest.fixture
def weighted_string_graph():
    """Fixture to create a weighted graph with string nodes."""
    G = nx.Graph()
    G.add_edge("A", "B", weight=2.0)
    G.add_edge("B", "C", weight=1.5)
    G.add_edge("C", "D", weight=3.0)
    G.add_edge("A", "D", weight=1.0)
    return G
