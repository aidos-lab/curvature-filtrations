import pytest
import numpy as np
import gudhi as gd
from scott.topology.distances import LandscapeDistance, ImageDistance
from scott.topology.representations import (
    PersistenceLandscape,
    PersistenceImage,
)


class TestLandscapeDistance:
    """Class designed to test the functionality of the LandscapeDistance class (subclass of TopologicalDistance)."""

    def test_init(self, toy_diagram1, toy_diagram2):
        """Test initialization of LandscapeDistance object."""
        LD = LandscapeDistance(
            toy_diagram1, toy_diagram2, norm=2, resolution=1000
        )
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
        assert 0 in landscapes[0].functions

    def test_average_landscape(self, setup_landscape_distance):
        """Test calculation of average persistence landscape."""
        landscapes = setup_landscape_distance._convert_to_landscape(
            setup_landscape_distance.diagram1
        )
        avg_landscape = setup_landscape_distance._average_landscape(landscapes)
        assert isinstance(avg_landscape, PersistenceLandscape)
        assert 0 in avg_landscape.functions

    def test_transform(self, setup_landscape_distance):
        """Test the distance calculation between landscapes."""
        avg1, avg2 = setup_landscape_distance.fit()
        distance = setup_landscape_distance.transform(avg1, avg2)
        assert isinstance(distance, float)

    def test_fit_transform(self, toy_pd, toy_pd2):
        """Test that fit_transform() method executes as desired."""
        LD = LandscapeDistance(toy_pd, toy_pd2, norm=2)
        distance1 = LD.fit_transform()
        avg1, avg2 = LD.fit()
        distance2 = LD.transform(avg1, avg2)
        assert type(distance2) == np.float64
        assert distance1 == distance2

    def test_subtract_landscapes(self, setup_landscape_distance):
        """Test subtraction of landscapes."""
        landscapes1 = setup_landscape_distance._convert_to_landscape(
            setup_landscape_distance.diagram1
        )
        landscapes2 = setup_landscape_distance._convert_to_landscape(
            setup_landscape_distance.diagram2
        )
        diff = setup_landscape_distance._subtract_landscapes(
            landscapes1[0], landscapes2[0]
        )
        assert isinstance(diff, PersistenceLandscape)
        assert 0 in diff.functions
        assert 1 in diff.functions
        assert isinstance(diff.functions[0], np.ndarray)
        assert isinstance(diff.functions[1], np.ndarray)

    def test_same_pds(self, toy_pd):
        """Ensure that distance between same persistence diagrams is 0."""
        LD = LandscapeDistance(toy_pd, toy_pd)
        assert LD.fit_transform() == 0


class TestImageDistance:
    """Class designed to test the functionality of the ImageDistance class (subclass of TopologicalDistance)."""

    def test_init(self, toy_diagram1, toy_diagram2):
        """Test initialization of ImageDistance instance."""
        ID = ImageDistance(toy_diagram1, toy_diagram2, norm=2)
        assert ID.diagram1 == toy_diagram1
        assert ID.diagram2 == toy_diagram2

    def test_dist_support(self, toy_diagram1, toy_diagram2):
        """Tests support for distributions"""
        ID = ImageDistance(toy_diagram1, toy_diagram2)
        assert ID.supports_distribution() == True

    def test_defaults(self, toy_diagram1, toy_diagram2):
        """Test defaults for ImageDistance object"""
        ID = ImageDistance(toy_diagram1, toy_diagram2)
        assert ID.bandwidth == 1.0
        assert ID.resolution == [20, 20]

    def test_img_transformer_init(self, toy_diagram1, toy_diagram2):
        """Test creation of of gudhi transformer."""
        ID = ImageDistance(toy_diagram1, toy_diagram2, norm=2)
        assert type(ID.image_transformer) == type(
            gd.representations.vector_methods.PersistenceImage()
        )

    def test_pd_to_img(self, toy_pd, toy_pd2):
        """Test conversion of PersistenceDiagram to PersistenceImage"""
        ID = ImageDistance(toy_pd, toy_pd2, norm=2)
        img = ID._convert_to_image(toy_pd)
        assert type(img) == list
        assert type(img[0]) == PersistenceImage
        assert img[0].pixels != None

    def test_fit(self, toy_pd, toy_pd2):
        """Test fit method."""
        ID = ImageDistance(toy_pd, toy_pd2, norm=2)
        avg1, avg2 = ID.fit()
        assert type(avg1) == PersistenceImage
        assert type(avg2) == PersistenceImage
        assert avg1.pixels != None
        assert avg2.pixels != None

    def test_transform(self, toy_pd, toy_pd2):
        """Test transform method."""
        ID = ImageDistance(toy_pd, toy_pd2, norm=2)
        avg1, avg2 = ID.fit()
        assert type(ID.transform(avg1, avg2)) == np.float64

    def test_fit_transform(self, toy_pd, toy_pd2):
        """Test to ensure fit_transform method behaves as expected."""
        ID = ImageDistance(toy_pd, toy_pd2, norm=2)
        distance1 = ID.fit_transform()
        avg1, avg2 = ID.fit()
        distance2 = ID.transform(avg1, avg2)
        assert type(distance2) == np.float64
        assert distance1 == distance2

    def test_same_pds(self, toy_pd):
        """Ensure that distance between same persistence diagrams is 0."""
        ID = ImageDistance(toy_pd, toy_pd)
        assert ID.fit_transform() == 0
