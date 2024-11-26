import pytest
from scott.compare import Comparator
from scott.kilt import KILT
from scott.topology.ph import GraphHomology, PersistenceDiagram
from scott.topology.distances import (
    LandscapeDistance,
    ImageDistance,
)
import numpy as np


class TestComparator:
    """Class designed to test the functionality of the Comparator class."""

    def test_create_object(self):
        """Test initialization of Comparator instance."""
        comp = Comparator()
        assert type(comp) == Comparator

    def test_defaults(self):
        """Test default attributes."""
        comp = Comparator()
        assert isinstance(comp.kilt, KILT)
        assert isinstance(comp.homology_dims, list)
        assert isinstance(comp.extended_persistence, bool)

        assert comp.kilt.measure == "forman_curvature"
        assert comp.kilt.weight == None
        assert comp.kilt.alpha == 0.0
        assert comp.kilt.prob_fn == None

        assert comp.homology_dims == [0, 1]

        assert comp.descriptor1 == None
        assert comp.descriptor2 == None

    def test_wrong_measure(self):
        """Test initialization failure when unsupported measure is inputted."""
        with pytest.raises(AssertionError):
            Comparator(measure="wrong_measure")

    def test_set_up_landscape_distance(self, toy_diagram1, toy_diagram2):
        """Test setup of LandscapeDistance object."""
        comp = Comparator()
        cls = comp._setup_distance("landscape")
        distance = cls(toy_diagram1, toy_diagram2, norm=2, resolution=1000)
        assert isinstance(distance, LandscapeDistance)
        with pytest.raises(AssertionError):
            comp._setup_distance("wrong_metric")

    def test_set_up_image_distance(self, toy_diagram1, toy_diagram2):
        """Test setup of ImageDistance object."""
        comp = Comparator()
        cls = comp._setup_distance("image")
        distance = cls(toy_diagram1, toy_diagram2, norm=2, resolution=1000)
        assert isinstance(distance, ImageDistance)
        with pytest.raises(AssertionError):
            comp._setup_distance("wrong_metric")

    def test_format_inputs(self, graph):
        """Test format_inputs helper method."""
        comp = Comparator()
        assert isinstance(comp._format_inputs(graph), list)
        assert len(comp._format_inputs(graph)) == 1
        assert isinstance(comp._format_inputs([graph, graph]), list)
        assert len(comp._format_inputs([graph, graph])) == 2

        with pytest.raises(ValueError):
            comp._format_inputs(1)

    def test_curvature_filtration(self, graph_distribution1):
        """Test that curvature is being computed for each graph in distribution."""
        comp = Comparator()
        holder = np.zeros(1)

        for G in comp._format_inputs(graph_distribution1):
            comp.kilt.fit(G)
            # KILT is fitting new curvature for each graph!
            assert not np.array_equal(holder, comp.kilt.curvature)
            holder = comp.kilt.curvature

    def test_fit_landscape(
        self,
        graph,
        empty_graph,
        graph_distribution1,
        graph_distribution2,
    ):
        """Test fit() method (converting to topological descriptors)."""
        comp = Comparator()
        distribution = [graph, empty_graph]
        with pytest.raises(AssertionError):
            comp.fit(distribution, distribution)

        distribution.pop()
        comp.fit(distribution, distribution)  # metric=landscape by default
        assert comp.descriptor1 is not None
        assert comp.descriptor2 is not None

        comp.fit(graph_distribution1, graph_distribution2)

    def test_fit_image(
        self,
        graph,
        empty_graph,
        graph_distribution1,
        graph_distribution2,
    ):
        """Test fit() method (converting to topological descriptors)."""
        comp = Comparator()
        distribution = [graph, empty_graph]
        with pytest.raises(AssertionError):
            comp.fit(distribution, distribution)

        distribution.pop()
        comp.fit(distribution, distribution, metric="image")
        assert comp.descriptor1 is not None
        assert comp.descriptor2 is not None

        comp.fit(graph_distribution1, graph_distribution2)

    def test_fit_transform(self, graph, graph2):
        """Test fit_transform method."""
        comp = Comparator(measure="ollivier_ricci_curvature")
        distance = comp.fit_transform(graph, graph2, metric="image")
        assert isinstance(distance, float)

    def test_kilterator(self, graph):
        """Test kilterator helper method."""
        comp = Comparator()
        pd = comp._kilterator(graph)
        assert isinstance(pd, PersistenceDiagram)

    def test_curvature_filtration(self, graph_distribution1):
        """Test curvature_filtration helper method."""
        comp = Comparator()
        pds = comp._curvature_filtration(graph_distribution1)
        assert isinstance(pds, list)
        assert isinstance(pds[0], PersistenceDiagram)
