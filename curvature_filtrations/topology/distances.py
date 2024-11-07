from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List
import gudhi as gd
from curvature_filtrations.topology.ph import PersistenceDiagram


class TopologicalDistance(ABC):
    """Abstract base class for computing topological distances.

    Takes in two persistence diagrams and handles necessary transformations
    to compute a specific distance function.
    """

    def __init__(self, diagram1, diagram2, norm=2) -> None:
        super().__init__()
        self.diagram1 = diagram1
        self.diagram2 = diagram2
        self.norm_order = norm

        # Ensure support for distributions if provided
        if self._is_distribution(diagram1) or self._is_distribution(diagram2):
            assert (
                self.supports_distribution()
            ), "Distribution of persistence diagrams is not supported by this distance type."

    def norm(self, x):
        """Compute norm of a vector x according to the specified order."""
        return np.linalg.norm(x, ord=self.norm_order)

    @staticmethod
    def _is_distribution(diagram):
        """Check if the given diagram is a distribution (i.e., a list of diagrams)."""
        return isinstance(diagram, list)

    @abstractmethod
    def supports_distribution(self) -> bool:
        """Indicates if the distance type supports distributions of persistence diagrams."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, **kwargs):
        """Convert persistence diagrams to topological descriptors."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, descriptor1, descriptor2) -> float:
        """Defines how to compute distances between topological descriptors."""
        raise NotImplementedError


class LandscapeDistance(TopologicalDistance):
    """Computes distances between persistence landscapes."""

    def __init__(self, diagram1, diagram2, norm=2, resolution=1000) -> None:
        super().__init__(diagram1, diagram2, norm)
        self.resolution = resolution
        self.landscape_transformer = gd.representations.Landscape(resolution=resolution)

    def supports_distribution(self) -> bool:
        """Indicates support for distributions of persistence diagrams."""
        return True

    def fit(self, **kwargs):
        """Compute and return average persistence landscapes for both diagrams."""
        landscapes1 = self._convert_to_landscape(self.diagram1)
        landscapes2 = self._convert_to_landscape(self.diagram2)

        avg1 = self._average_landscape(landscapes1)
        avg2 = self._average_landscape(landscapes2)
        return avg1, avg2

    def transform(self, landscape1: Dict[int, np.array], landscape2: Dict[int, np.array]) -> float:
        """Compute the norm-based distance between two persistence landscapes."""
        common_dims = set(landscape1.keys()).intersection(landscape2.keys())
        difference = self._subtract_landscapes(landscape1, landscape2)

        distance = sum(self.norm(difference[dim]) for dim in common_dims)
        return distance

    def _convert_to_landscape(
        self, diagrams: List[PersistenceDiagram]
    ) -> List[Dict[int, np.array]]:  # TODO: add PD
        """Convert each persistence diagram to a persistence landscape for each dimension."""
        landscapes = []

        for diagram in diagrams:
            # from before change to PersistenceDiagram object:
            # landscape = {
            #     dim: self.landscape_transformer.fit_transform([points])[0]
            #     for dim, points in diagram.items()
            # }
            landscape = {}
            for dim, points in diagram.persistence_pts.items():
                transformed_points = self.landscape_transformer.fit_transform([points])[0]
                landscape[dim] = transformed_points
            landscapes.append(landscape)

        return landscapes

    @staticmethod
    def _average_landscape(landscapes: List[Dict[int, np.array]]) -> Dict[int, np.array]:
        """Compute the average persistence landscape across multiple landscapes."""
        avg_landscape = {}
        for landscape in landscapes:
            for dim, values in landscape.items():
                if dim not in avg_landscape:
                    avg_landscape[dim] = np.zeros_like(values)
                avg_landscape[dim] += values

        for dim in avg_landscape:
            avg_landscape[dim] /= len(landscapes)
        return avg_landscape

    @staticmethod
    def _subtract_landscapes(
        landscape1: Dict[int, np.array], landscape2: Dict[int, np.array]
    ) -> Dict[int, np.array]:
        """Subtract two landscapes for each common dimension."""
        return {dim: landscape1[dim] - landscape2[dim] for dim in landscape1.keys()}

    def __str__(self):
        return f"LandscapeDistance object between (1) [{self.diagram1}] and (2) [{self.diagram2}]"


# Example dictionary to link supported distances
supported_distances = {
    "landscape": LandscapeDistance,
}


# TODO: Implement other distances
# Some can support distributions!
class ImageDistance(TopologicalDistance):
    pass


class SilhouetteDistance(TopologicalDistance):
    pass


supported_distances = {
    "landscape": LandscapeDistance,
}
