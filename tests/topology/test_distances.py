import pytest
import numpy as np
import gudhi as gd
from curvature_filtrations.topology.distances import LandscapeDistance, ImageDistance
from curvature_filtrations.topology.representations import (
    PersistenceDiagram,
    PersistenceLandscape,
    PersistenceImage,
)


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

    def test_subtract_landscapes(self, setup_landscape_distance):
        """Test subtraction of landscapes."""
        landscapes1 = setup_landscape_distance._convert_to_landscape(
            setup_landscape_distance.diagram1
        )
        landscapes2 = setup_landscape_distance._convert_to_landscape(
            setup_landscape_distance.diagram2
        )
        diff = setup_landscape_distance._subtract_landscapes(landscapes1[0], landscapes2[0])
        assert isinstance(diff, PersistenceLandscape)
        assert 0 in diff.functions
        assert 1 in diff.functions
        assert isinstance(diff.functions[0], np.ndarray)
        assert isinstance(diff.functions[1], np.ndarray)


class TestImageDistance:
    """Class designed to test the functionality of the ImageDistance class (subclass of TopologicalDistance)"""

    def test_init(self, toy_diagram1, toy_diagram2):
        LD = ImageDistance(toy_diagram1, toy_diagram2, norm=2)
        assert LD.diagram1 == toy_diagram1
        assert LD.diagram2 == toy_diagram2

    def test_img_transformer_init(self, toy_diagram1, toy_diagram2):
        ID = ImageDistance(toy_diagram1, toy_diagram2, norm=2)
        assert type(ID.image_transformer) == type(
            gd.representations.vector_methods.PersistenceImage()
        )

    def test_pd_to_img(self, toy_pd, toy_pd2):
        ID = ImageDistance(toy_pd, toy_pd2, norm=2)
        img = ID._convert_to_image(toy_pd)
        assert type(img) == list
        assert type(img[0]) == PersistenceImage
        print(f"Type of pixels attribute: {type(img[0].pixels)}")
        print(f"Shape of pixels np.array: {img[0].pixels[0].shape}")
        print(f"Pixels example: {img[0].pixels}")
