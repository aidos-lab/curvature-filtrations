import pytest
import numpy as np
from curvature_filtrations.topology.distances import LandscapeDistance
from curvature_filtrations.topology.ph import PersistenceDiagram


@pytest.fixture
def toy_diagram1():
    """Toy persistence diagram with 0D and 1D homology features."""
    return [{0: np.array([[0, 1], [1, 2]]), 1: np.array([[0.5, 2.0], [1.5, 2.5]])}]


@pytest.fixture
def toy_diagram2():
    """Toy persistence diagram with 0D and 1D homology features."""
    return [{0: np.array([[0, 1], [1, 3]]), 1: np.array([[0.5, 2.1], [1.5, 3.0]])}]


@pytest.fixture
def toy_diagram3():
    """Toy persistence diagram with ONLY 0D homology features."""
    return [{0: np.array([[0, 1], [1, 8]])}]


@pytest.fixture
def toy_diagram4():
    """Toy persistence diagram with ONLY 1D homology features."""
    return [{1: np.array([[0, 1], [1, 8]])}]


@pytest.fixture
def toy_pd(toy_diagram1):
    """Toy PersistenceDiagram object"""
    input_dict = {}
    dims = []
    for dim, array in toy_diagram1[0].items():
        input_dict[dim] = array
        dims.append(dim)
    pd = PersistenceDiagram(dims)
    pd.persistence_pts = input_dict
    return [pd]


@pytest.fixture
def toy_pd2(toy_diagram2):
    """Toy PersistenceDiagram object"""
    input_dict = {}
    dims = []
    for dim, array in toy_diagram2[0].items():
        input_dict[dim] = array
        dims.append(dim)
    pd = PersistenceDiagram(dims)
    pd.persistence_pts = input_dict
    return [pd]


@pytest.fixture
def toy_landscape1(toy_pd):
    """Toy PersistenceLandscape Object (created from toy_pd)"""
    dist = LandscapeDistance(None, None)
    return dist._convert_to_landscape(toy_pd)[0]


@pytest.fixture
def toy_landscape2(toy_pd2):
    """Toy PersistenceLandscape Object (created from toy_pd2)"""
    dist = LandscapeDistance(None, None)
    return dist._convert_to_landscape(toy_pd2)[0]


@pytest.fixture
def setup_landscape_distance(toy_pd, toy_pd2):
    """Fixture to set up the LandscapeDistance instance and sample diagrams."""
    landscape_distance = LandscapeDistance(toy_pd, toy_pd2, norm=2, resolution=1000)
    return landscape_distance
