import pytest
from curvature_filtrations.compare import Comparator
from curvature_filtrations.kilt import KILT
from curvature_filtrations.topology.ph import GraphHomology
from curvature_filtrations.topology.distances import (
    LandscapeDistance,
)
import numpy as np


class TestComparator:
    def test_create_object(self):
        comp = Comparator()
        assert type(comp) == Comparator

    def test_defaults(self):
        comp = Comparator()
        assert isinstance(comp.kilt, KILT)
        assert isinstance(comp.ph, GraphHomology)

        assert comp.kilt.measure == "forman_curvature"
        assert comp.kilt.weight == None
        assert comp.kilt.alpha == 0.0
        assert comp.kilt.prob_fn == None

        assert comp.ph.homology_dims == [0, 1]

        assert comp.descriptor1 == None
        assert comp.descriptor2 == None

    def test_wrong_measure(self):
        with pytest.raises(AssertionError):
            Comparator(measure="wrong_measure")

    def test_set_up_distance(self, toy_diagram1, toy_diagram2):
        comp = Comparator()
        cls = comp._setup_distance("landscape")
        distance = cls(toy_diagram1, toy_diagram2, norm=2, resolution=1000)
        assert isinstance(distance, LandscapeDistance)
        with pytest.raises(AssertionError):
            comp._setup_distance("wrong_metric")

    def test_format_inputs(self, graph):
        comp = Comparator()
        assert isinstance(comp._format_inputs(graph), list)
        assert len(comp._format_inputs(graph)) == 1
        assert isinstance(comp._format_inputs([graph, graph]), list)
        assert len(comp._format_inputs([graph, graph])) == 2

        with pytest.raises(ValueError):
            comp._format_inputs(1)

    def test_curvature_filtration(
        self, graph, graph2, small_graph, empty_graph
    ):
        comp = Comparator()
        holder = np.zeros(1)
        distribution = [graph, graph2, small_graph, empty_graph]

        for G in comp._format_inputs(distribution):
            comp.kilt.fit(G)
            # KILT is fitting new curvature for each graph!
            assert not np.array_equal(holder, comp.kilt.curvature)
            holder = comp.kilt.curvature

    def test_fit(self, graph, graph2, small_graph, empty_graph):
        comp = Comparator()
        distribution = [graph, graph2, small_graph, empty_graph]
        with pytest.raises(AssertionError):
            comp.fit(distribution, distribution)

        distribution.pop()
        comp.fit(distribution, distribution)
        assert comp.descriptor1 is not None
        assert comp.descriptor2 is not None
