from scott.topology.distances.base import TopologicalDistance
from scott.topology.distances.image import ImageDistance
from scott.topology.distances.landscape import LandscapeDistance

supported_distances = {"landscape": LandscapeDistance, "image": ImageDistance}

__all__ = ["TopologicalDistance", "ImageDistance", "LandscapeDistance"]
