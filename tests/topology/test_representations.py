import pytest
import networkx as nx
import numpy as np
from scott.topology.representations import (
    PersistenceDiagram,
    PersistenceLandscape,
    PersistenceImage,
)
from scott.kilt import KILT
from scott.topology.distances import LandscapeDistance, ImageDistance


class TestDiagram:
    """Class designed to test the functionality of the PersistenceDiagram class."""

    def test_create_object(self):
        """Test the creation of a simple PersistanceDiagram object."""
        diagram = PersistenceDiagram()
        assert type(diagram) == PersistenceDiagram

    def test_defaults(self):
        """Test that the defaults are set correctly."""
        diagram = PersistenceDiagram()
        assert diagram.homology_dims == [0, 1]
        assert diagram.persistence_pts == None

    def test_set_persistence_pts(self, dummy_diagram):
        """Test the setter and getter methods for attaching persistence points to the PersistenceDiagram object."""
        diagram = PersistenceDiagram()
        assert diagram._persistence_pts == None
        # setting persistence points
        diagram.persistence_pts = dummy_diagram
        assert diagram.persistence_pts == dummy_diagram

    def test_get_pts_for_dim(self, dummy_diagram):
        """Test retrieval of persistence points for a specific dimension."""
        diagram = PersistenceDiagram()
        # set and get
        diagram.persistence_pts = dummy_diagram
        assert np.all(diagram.get_pts_for_dim(0) == dummy_diagram[0])

    def test_ph_calc(self, graph):
        """Test that fit_transform on a KILT object returns a PersistenceDiagram object."""
        klt = KILT()
        ph = klt.fit_transform(graph)
        assert type(ph) == PersistenceDiagram


class TestLandscape:
    """Class designed to test the functionality of the PersistenceLandscape class."""

    def test_create_object(self):
        """Test instantiation of a PersistenceLandscape object."""
        pl = PersistenceLandscape()
        assert type(pl) == PersistenceLandscape

    def test_defaults(self):
        """Test that the defaults are set correctly."""
        pl = PersistenceLandscape()
        assert pl.homology_dims == [0, 1]
        assert pl._functions == None

    def test_diagram_to_landscape(self, toy_pd):
        """Test the method that converts a PersistenceDiagram in to a PersistenceLandscape."""
        assert type(toy_pd[0]) == PersistenceDiagram
        dist = LandscapeDistance(None, None)
        pl = dist._convert_to_landscape(toy_pd)[0]
        assert type(pl) == PersistenceLandscape
        assert pl.functions != None

    def test_avg_landscape(self, toy_pd, toy_pd2):
        """Test the functionality for averaging multiple PersistenceLandscapes."""
        dist = LandscapeDistance(None, None)
        pl1 = dist._convert_to_landscape(toy_pd)[0]
        pl2 = dist._convert_to_landscape(toy_pd2)[0]
        avg = dist._average_landscape([pl1, pl2])
        assert type(avg) == PersistenceLandscape
        assert avg.functions != None

    def test_fixture_toy_landscape1(self, toy_landscape1):
        """Checks the toy Landscape object used for tests."""
        assert type(toy_landscape1) == PersistenceLandscape
        assert (
            len(toy_landscape1.functions[toy_landscape1.homology_dims[0]])
            == toy_landscape1.num_functions * toy_landscape1.resolution
        )


class TestImage:
    """Class designed to test the functionality of the PersistenceImage class."""

    def test_create_object(self):
        """Test instantiation of a PersistenceImage object."""
        img = PersistenceImage()
        assert type(img) == PersistenceImage

    def test_defaults(self):
        """Test that the defaults are set correctly."""
        img = PersistenceImage()
        assert img.homology_dims == [0, 1]
        assert img.bandwidth == 1.0
        # TODO: Check weight attribute
        assert img.resolution == [20, 20]

    def test_diagram_to_img(self, toy_pd, toy_pd2):
        """Test conversion of PersistenceDiagram to PersistenceImage"""
        ID = ImageDistance(toy_pd, toy_pd2)
        pi1 = ID._convert_to_image(toy_pd)[0]
        pi2 = ID._convert_to_image(toy_pd2)[0]
        avg = ID._average_image([pi1, pi2])
        assert type(avg) == PersistenceImage
        assert avg.pixels != None
        assert np.all(avg.get_pixels_for_dim(0) == avg.pixels[0])
        assert (
            len(avg.get_pixels_for_dim(0))
            == avg.resolution[0] * avg.resolution[1]
        )
