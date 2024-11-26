import pytest
from curvature_filtrations.kilt import KILT
import networkx as nx
import numpy as np
import curvature_filtrations.geometry.measures as measures
from curvature_filtrations.topology.ph import PersistenceDiagram


class TestKILT:
    """Class designed to test the functionality of the KILT class."""

    def test_create_object(self):
        """Test initialization of KILT instance."""
        klt = KILT()
        assert type(klt) == KILT

    def test_defaults(self):
        """Test default attributes of KILT object."""
        klt = KILT()
        assert klt.measure == "forman_curvature"
        assert klt.weight == None
        assert klt.alpha == 0.0
        assert klt.alpha == 0.0
        assert klt.prob_fn == None

    def test_wrong_measure(self):
        """Checks that initialization fails when unsupported measure is inputted."""
        with pytest.raises(AssertionError):
            KILT(measure="wrong_measure")

    def test_G_get_set(self, graph):
        """Tests setter and getter methods for attribute G."""
        klt = KILT()
        assert klt.G is None
        klt.G = graph
        assert klt.G is not None
        assert isinstance(klt.G, nx.Graph)
        assert klt.G != graph  # We're copying!

    def test_curvature_get_set(self, graph):
        """Tests setter and getter methods for attribute curvature."""
        klt = KILT()
        # Get Curvature before computing raises error
        assert klt.curvature is None

        # Set Graph
        klt.G = graph
        assert klt.G is not None
        assert isinstance(klt.curvature, np.ndarray)
        assert len(klt.curvature) == 0

        # Set Curvature
        klt.fit(graph)
        # Get Curvature from KILT.G
        assert len(klt.curvature) == len(graph.edges)
        assert klt.curvature.shape == (len(graph.edges),)

    def test_kilt_forman(self, graph):
        """Tests forman curvature calculation."""
        klt = KILT(measure="forman_curvature")
        klt.fit(graph)
        curvature = measures.forman_curvature(graph)
        assert len(klt.curvature) == len(curvature)
        assert np.array_equal(klt.curvature, curvature)

    def test_kilt_orc(self, graph):
        """Tests olliver-ricci curvature calculation."""
        klt = KILT(measure="ollivier_ricci_curvature")
        klt.fit(graph)
        curvature = measures.ollivier_ricci_curvature(graph)
        assert len(klt.curvature) == len(curvature)
        assert np.array_equal(klt.curvature, curvature)

    def test_kilt_resistance(self, small_graph):
        """Tests resistance curvature calculation."""
        klt = KILT(measure="resistance_curvature")
        klt.fit(small_graph)
        curvature = measures.resistance_curvature(small_graph)
        assert len(klt.curvature) == len(curvature)
        assert np.array_equal(klt.curvature, curvature)

    def test_kilt_balanced_forman(self, small_graph):
        """Tests resistance curvature calculation."""
        klt = KILT(measure="balanced_forman_curvature")
        klt.fit(small_graph)
        curvature = measures.balanced_forman_curvature(small_graph)
        assert len(klt.curvature) == len(curvature)
        assert np.array_equal(klt.curvature, curvature)

    def test_fit(self, graph):
        """Tests fit() method."""
        klt = KILT(measure="forman_curvature")
        assert klt.G is None
        assert klt.fit(graph) is None
        assert klt.G is not None
        assert isinstance(klt.G, nx.Graph)
        assert isinstance(klt.curvature, np.ndarray)

    def test_transform(self, graph, regular_homology_dims):
        """Tests transform() method."""
        klt = KILT(measure="forman_curvature")

        with pytest.raises(AssertionError):
            klt.transform(graph)

        klt.G = graph
        with pytest.raises(AssertionError):
            klt.transform(regular_homology_dims)

        # Manually set curvature
        klt.curvature = measures.forman_curvature(graph)
        klt.transform(regular_homology_dims)

        # Use fit to set curvature
        klt.fit(graph)
        diagram = klt.transform(homology_dims=regular_homology_dims)
        assert isinstance(diagram, PersistenceDiagram)
        for dim in regular_homology_dims:
            assert dim in diagram.homology_dims
            isinstance(diagram.get_pts_for_dim(dim), np.ndarray)

    def test_fit_transform(self, graph, graph2, small_graph):
        """Tests fit_transform() method."""
        klt = KILT(measure="forman_curvature")
        diagram1 = klt.fit_transform(graph)
        curv1 = klt.curvature
        assert isinstance(diagram1, PersistenceDiagram)

        small_diagram = klt.fit_transform(small_graph)
        small_curv = klt.curvature
        assert isinstance(small_diagram, PersistenceDiagram)

        assert not len(curv1) == len(small_curv)

        diagram2 = klt.fit_transform(graph2)
        curv2 = klt.curvature
        assert isinstance(diagram2, PersistenceDiagram)

        if len(curv1) == len(curv2):
            assert not np.array_equal(curv1, curv2)
