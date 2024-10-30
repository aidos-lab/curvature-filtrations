import pytest
import numpy as np
from curvature_filtrations.topology.distances import LandscapeDistance


@pytest.fixture
def toy_diagram1():
    """Toy persistence diagram with 0D and 1D homology features."""
    return [
        {0: np.array([[0, 1], [1, 2]]), 1: np.array([[0.5, 2.0], [1.5, 2.5]])}
    ]


@pytest.fixture
def toy_diagram2():
    """Toy persistence diagram with 0D and 1D homology features."""
    return [
        {0: np.array([[0, 1], [1, 3]]), 1: np.array([[0.5, 2.1], [1.5, 3.0]])}
    ]


@pytest.fixture
def toy_diagram3():
    """Toy persistence diagram with ONLY 0D homology features."""
    return [{0: np.array([[0, 1], [1, 8]])}]


@pytest.fixture
def toy_diagram4():
    """Toy persistence diagram with ONLY 1D homology features."""
    return [{1: np.array([[0, 1], [1, 8]])}]


@pytest.fixture
def setup_landscape_distance(toy_diagram1, toy_diagram2):
    """Fixture to set up the LandscapeDistance instance and sample diagrams."""
    landscape_distance = LandscapeDistance(
        toy_diagram1, toy_diagram2, norm=2, resolution=1000
    )
    return landscape_distance
