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
        assert curvatures.shape == (len(simple_graph.edges()),)  # Number of edges in the graph

        curvatures = forman_curvature(main_figure_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(main_figure_graph.edges()),)  # Number of edges in the graph

    def test_ollivier_ricci_curvature(self, simple_graph, main_figure_graph):
        """Test Ollivier-Ricci curvature function."""
        curvatures = ollivier_ricci_curvature(simple_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(simple_graph.edges()),)  # Number of edges in the graph

        curvatures = ollivier_ricci_curvature(main_figure_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(main_figure_graph.edges()),)  # Number of edges in the graph

        # Here we can pass a custom probability function (for simplicity, we use None or a basic one)
        def custom_prob_fn(G, node, node_to_index):
            prob = np.zeros(len(G.nodes))
            prob[node_to_index[node]] = 0.5
            return prob

        # Check that we default to custom_orc if prob_fn is passed
        curvature2 = ollivier_ricci_curvature(simple_graph, prob_fn=custom_prob_fn)
        assert curvature2.shape == (len(simple_graph.edges()),)  # Number of edges in the graph

    def test_balanced_forman_curvature(self, simple_graph, main_figure_graph):
        """Test balanced Forman curvature function."""
        curvatures = balanced_forman_curvature(simple_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(simple_graph.edges()),)  # Number of edges in the graph

        curvatures = balanced_forman_curvature(main_figure_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(main_figure_graph.edges()),)  # Number of edges in the graph

    def test_resistance_curvature(self, simple_graph, main_figure_graph):
        """Test resistance curvature function."""
        curvatures = resistance_curvature(simple_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(simple_graph.edges()),)  # Number of edges in the graph

        curvatures = resistance_curvature(main_figure_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(main_figure_graph.edges()),)  # Number of edges in the graph

    # Tests for different node types
    def test_forman_curvature_different_node_types(
        self,
        string_node_graph,
        mixed_node_graph,
        tuple_node_graph,
        weighted_string_graph,
    ):
        """Test Forman curvature with different node types."""
        # String nodes
        curvatures = forman_curvature(string_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(string_node_graph.edges()),)

        # Mixed node types
        curvatures = forman_curvature(mixed_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(mixed_node_graph.edges()),)

        # Tuple nodes
        curvatures = forman_curvature(tuple_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(tuple_node_graph.edges()),)

        # Weighted string nodes
        curvatures = forman_curvature(weighted_string_graph, weight="weight")
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(weighted_string_graph.edges()),)

    def test_ollivier_ricci_curvature_different_node_types(
        self, string_node_graph, mixed_node_graph, tuple_node_graph
    ):
        """Test Ollivier-Ricci curvature with different node types."""
        # String nodes
        curvatures = ollivier_ricci_curvature(string_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(string_node_graph.edges()),)

        # Mixed node types
        curvatures = ollivier_ricci_curvature(mixed_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(mixed_node_graph.edges()),)

        # Tuple nodes
        curvatures = ollivier_ricci_curvature(tuple_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(tuple_node_graph.edges()),)

    def test_balanced_forman_curvature_different_node_types(
        self,
        string_node_graph,
        mixed_node_graph,
        tuple_node_graph,
        weighted_string_graph,
    ):
        """Test balanced Forman curvature with different node types."""
        # String nodes
        curvatures = balanced_forman_curvature(string_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(string_node_graph.edges()),)

        # Mixed node types
        curvatures = balanced_forman_curvature(mixed_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(mixed_node_graph.edges()),)

        # Tuple nodes
        curvatures = balanced_forman_curvature(tuple_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(tuple_node_graph.edges()),)

        # Weighted string nodes
        curvatures = balanced_forman_curvature(weighted_string_graph, weight="weight")
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(weighted_string_graph.edges()),)

    def test_resistance_curvature_different_node_types(
        self, string_node_graph, mixed_node_graph, tuple_node_graph
    ):
        """Test resistance curvature with different node types."""
        # String nodes
        curvatures = resistance_curvature(string_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(string_node_graph.edges()),)

        # Mixed node types
        curvatures = resistance_curvature(mixed_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(mixed_node_graph.edges()),)

        # Tuple nodes
        curvatures = resistance_curvature(tuple_node_graph)
        assert isinstance(curvatures, np.ndarray)
        assert curvatures.shape == (len(tuple_node_graph.edges()),)

    def test_parallelization_consistency(self, weighted_string_graph):
        """Test that parallel and serial computations yield the same results for all curvature measures."""

        # Test Forman curvature
        serial_curvatures = forman_curvature(weighted_string_graph, n_jobs=1)
        parallel_curvatures = forman_curvature(weighted_string_graph, n_jobs=2)
        assert np.allclose(
            serial_curvatures, parallel_curvatures
        ), "Forman curvature parallel/serial mismatch"

        # Test Ollivier-Ricci curvature
        serial_curvatures = ollivier_ricci_curvature(weighted_string_graph, n_jobs=1)
        parallel_curvatures = ollivier_ricci_curvature(weighted_string_graph, n_jobs=2)
        assert np.allclose(
            serial_curvatures, parallel_curvatures
        ), "Ollivier-Ricci curvature parallel/serial mismatch"

        # Test balanced Forman curvature
        serial_curvatures = balanced_forman_curvature(weighted_string_graph, n_jobs=1)
        parallel_curvatures = balanced_forman_curvature(weighted_string_graph, n_jobs=2)
        assert np.allclose(
            serial_curvatures, parallel_curvatures
        ), "Balanced Forman curvature parallel/serial mismatch"

        # Test resistance curvature
        serial_curvatures = resistance_curvature(weighted_string_graph, n_jobs=1)
        parallel_curvatures = resistance_curvature(weighted_string_graph, n_jobs=2)
        assert np.allclose(
            serial_curvatures, parallel_curvatures
        ), "Resistance curvature parallel/serial mismatch"
