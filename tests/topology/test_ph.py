import pytest
import networkx as nx
import numpy as np
from gudhi import SimplexTree
from scott.topology.representations import PersistenceDiagram
from scott.topology.ph import GraphHomology


class TestGraphHomology:

    def test_initialize(self, GH):
        """Test initialization of the GraphHomology object."""
        assert GH.homology_dims == [0, 1]
        assert GH.filter_attribute == "curvature"
        assert GH.max_dimension == 2

    def test_build_simplex_tree(
        self,
        main_figure_graph_with_forman_curvature,
        extended_forman_GH,
    ):
        """Test the _build_simplex_tree method returns a SimplexTree."""
        st = extended_forman_GH._build_simplex_tree(
            main_figure_graph_with_forman_curvature
        )
        assert isinstance(st, SimplexTree)
        # Verify simplex tree contains expected simplices
        simplices = list(st.get_simplices())
        nodes = []
        edges = []
        curvatures = []
        for simplex in simplices:
            if len(simplex[0]) == 1:
                nodes.append(simplex)
            elif len(simplex[0]) == 2:
                edges.append(simplex)
                curvatures.append(simplex[1])

        assert len(nodes) == 8
        assert len(edges) == 10
        assert np.array_equal(
            curvatures, [1.0, 3.0, -1.0, -2.0, 2.0, 0.0, -3.0, -1.0, -1.0, 0.0]
        )

    def test_build_simplex_tree_missing_attribute(
        self, graph_with_random_curvature, GH
    ):
        """Test the _build_simplex_tree method raises an error for missing attributes."""
        G = graph_with_random_curvature.copy()
        # Remove curvature from one edge
        G.edges[0, 1].pop("curvature")
        with pytest.raises(
            ValueError,
            match="Edge \\(0, 1\\) is missing required attribute 'curvature'.",
        ):
            GH._build_simplex_tree(G)

    def test_calculate_persistent_homology(
        self,
        main_figure_graph_with_forman_curvature,
        forman_GH,
    ):
        """Test the calculate_persistent_homology method."""
        diagram = forman_GH.calculate_persistent_homology(
            main_figure_graph_with_forman_curvature
        )
        assert isinstance(diagram, PersistenceDiagram)
        assert isinstance(diagram.persistence_pts, dict)
        for dim in forman_GH.homology_dims:
            assert dim in diagram.persistence_pts
            assert isinstance(diagram.persistence_pts[dim], np.ndarray)

    def test_calculate_persistent_homology_extended(
        self,
        main_figure_graph_with_forman_curvature,
        extended_forman_GH,
    ):
        """Test the calculate_persistent_homology method with extended persistence."""
        diagram = extended_forman_GH.calculate_persistent_homology(
            main_figure_graph_with_forman_curvature,
        )
        assert isinstance(diagram, PersistenceDiagram)
        assert isinstance(diagram.persistence_pts, dict)
        for dim in extended_forman_GH.homology_dims:
            assert dim in diagram.persistence_pts
            assert isinstance(diagram.persistence_pts[dim], np.ndarray)

        assert len(diagram.persistence_pts[0]) == len(
            main_figure_graph_with_forman_curvature.nodes
        )

        non_trivial_ccs = [
            1 if x[1] > x[0] else 0 for x in diagram.persistence_pts[0]
        ]
        assert sum(non_trivial_ccs) == 2

        # Match expected persistence pairs in main figure
        assert len(diagram.persistence_pts[1]) == 3
        non_trivial_loops = [
            1 if x[1] > x[0] else 0 for x in diagram.persistence_pts[1]
        ]
        assert sum(non_trivial_loops) == 2

    def test_format_persistence_diagrams(self, graph_with_random_curvature, GH):
        """Test the _format_persistence_diagrams method."""
        st = GH._build_simplex_tree(graph_with_random_curvature)
        st.make_filtration_non_decreasing()
        st.expansion(GH.max_dimension)
        st.persistence()
        diagram = GH._format_persistence_diagrams(st)
        assert isinstance(diagram, PersistenceDiagram)
        assert isinstance(diagram.persistence_pts, dict)

    def test_mask_infinities(self):
        """Test the _mask_infinities static method."""
        input_array = np.array([[0, 1], [2, np.inf], [3, 4]])
        output_array = GraphHomology._mask_infinities(input_array)
        assert isinstance(output_array, np.ndarray)
        assert output_array.shape == input_array.shape
        assert np.all(output_array[:, 1] < np.inf)
        assert output_array[1, 1] == 5
