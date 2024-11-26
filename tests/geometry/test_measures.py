import pytest

import pytest
import networkx as nx
import numpy as np
from scott.geometry.measures import (
    forman_curvature,
    ollivier_ricci_curvature,
    balanced_forman_curvature,
    resistance_curvature,
)


class TestCurvatureMeasures:
    def test_forman_curvature(self, simple_graph, main_figure_graph):
        """Test Forman curvature function."""
        curvatures = forman_curvature(simple_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (
            len(simple_graph.edges()),
        )  # Number of edges in the graph

        curvatures = forman_curvature(main_figure_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (
            len(main_figure_graph.edges()),
        )  # Number of edges in the graph

    def test_ollivier_ricci_curvature(self, simple_graph, main_figure_graph):
        """Test Ollivier-Ricci curvature function."""
        curvatures = ollivier_ricci_curvature(simple_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (
            len(simple_graph.edges()),
        )  # Number of edges in the graph

        curvatures = ollivier_ricci_curvature(main_figure_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (
            len(main_figure_graph.edges()),
        )  # Number of edges in the graph

        # Here we can pass a custom probability function (for simplicity, we use None or a basic one)
        def custom_prob_fn(G, node, node_to_index):
            prob = np.zeros(len(G.nodes))
            prob[node_to_index[node]] = 0.5
            return prob

        # Check that we default to custom_orc if prob_fn is passed
        curvature2 = ollivier_ricci_curvature(
            simple_graph, prob_fn=custom_prob_fn
        )
        assert curvature2.shape == (
            len(simple_graph.edges()),
        )  # Number of edges in the graph

    def test_balanced_forman_curvature(self, simple_graph, main_figure_graph):
        """Test balanced Forman curvature function."""
        curvatures = balanced_forman_curvature(simple_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (
            len(simple_graph.edges()),
        )  # Number of edges in the graph

        curvatures = balanced_forman_curvature(main_figure_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (
            len(main_figure_graph.edges()),
        )  # Number of edges in the graph

    def test_resistance_curvature(self, simple_graph, main_figure_graph):
        """Test resistance curvature function."""
        curvatures = resistance_curvature(simple_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (
            len(simple_graph.edges()),
        )  # Number of edges in the graph

        curvatures = resistance_curvature(main_figure_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (
            len(main_figure_graph.edges()),
        )  # Number of edges in the graph
