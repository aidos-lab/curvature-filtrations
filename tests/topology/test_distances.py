import pytest
import numpy as np
from curvature_filtrations.topology.distances import (
    LandscapeDistance,
)
from curvature_filtrations.topology.representations import PersistenceDiagram, PersistenceLandscape


class TestLandscapeDistance:

    def test_init(self, toy_diagram1, toy_diagram2):
        LD = LandscapeDistance(toy_diagram1, toy_diagram2, norm=2, resolution=1000)
        assert LD.diagram1 == toy_diagram1
        assert LD.diagram2 == toy_diagram2

    def test_distribution_support(self, setup_landscape_distance):
        """Test the support for distributions."""
        assert setup_landscape_distance.supports_distribution() is True

    def test_pd_to_landscape(self, setup_landscape_distance):
        """Test conversion from persistence diagram to landscape."""
        landscapes = setup_landscape_distance._convert_to_landscape(
            setup_landscape_distance.diagram1
        )
        assert isinstance(landscapes, list)
        assert isinstance(landscapes[0], PersistenceLandscape)
        assert 0 in landscapes[0]

    def test_average_landscape(self, setup_landscape_distance):
        """Test calculation of average persistence landscape."""
        landscapes = setup_landscape_distance._convert_to_landscape(
            setup_landscape_distance.diagram1
        )
        avg_landscape = setup_landscape_distance._average_landscape(landscapes)
        assert isinstance(avg_landscape, dict)
        assert 0 in avg_landscape

    def test_transform(self, setup_landscape_distance):
        """Test the distance calculation between landscapes."""
        avg1, avg2 = setup_landscape_distance.fit()
        distance = setup_landscape_distance.transform(avg1, avg2)
        assert isinstance(distance, float)

    def test_subtract_landscapes(self, setup_landscape_distance):
        """Test subtraction of landscapes."""
        landscapes1 = setup_landscape_distance._convert_to_landscape(
            setup_landscape_distance.diagram1
        )
        landscapes2 = setup_landscape_distance._convert_to_landscape(
            setup_landscape_distance.diagram2
        )
        diff = setup_landscape_distance._subtract_landscapes(landscapes1[0], landscapes2[0])
        assert isinstance(diff, dict)
        assert 0 in diff
        assert 1 in diff
        assert isinstance(diff[0], np.ndarray)
        assert isinstance(diff[1], np.ndarray)
