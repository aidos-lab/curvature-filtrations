from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List
import gudhi as gd
from curvature_filtrations.topology.representations import (
    PersistenceDiagram,
    PersistenceLandscape,
    PersistenceImage,
)


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

    def __init__(self, diagram1, diagram2, norm=2, resolution=1000, num_functions=5) -> None:
        super().__init__(diagram1, diagram2, norm)
        self.resolution = resolution
        self.num_functions = num_functions
        self.landscape_transformer = gd.representations.Landscape(
            resolution=resolution, num_landscapes=num_functions
        )

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

    def transform(
        self, landscape1: PersistenceLandscape, landscape2: PersistenceLandscape
    ) -> float:
        """Compute the norm-based distance between two persistence landscapes."""
        common_dims = set(landscape1.homology_dims).intersection(landscape2.homology_dims)
        difference = self._subtract_landscapes(landscape1, landscape2)

        distance = sum(self.norm(difference.functions[dim]) for dim in common_dims)
        return distance

    def _convert_to_landscape(
        self, diagrams: List[PersistenceDiagram]
    ) -> List[PersistenceLandscape]:
        """Convert each persistence diagram to a persistence landscape for each dimension."""
        landscapes = []

        for diagram in diagrams:
            landscape = PersistenceLandscape(
                homology_dims=diagram.homology_dims,
                num_functions=self.num_functions,
                resolution=self.resolution,
            )
            landscape_functions = {}
            for dim, points in diagram.persistence_pts.items():
                # TODO: check this... H1 is not responding to changes in PD input
                transformed_points = self.landscape_transformer.fit_transform([points])[0]
                landscape_functions[dim] = transformed_points
            # set to PersistenceLandscape functions attribute
            landscape.functions = landscape_functions
            # add to list of Landscapes
            landscapes.append(landscape)
        return landscapes

    @staticmethod
    def _average_landscape(landscapes: List[PersistenceLandscape]) -> PersistenceLandscape:
        """Compute the average persistence landscape across multiple landscapes."""
        avg_landscape = {}
        # sum landscape functions
        for landscape in landscapes:
            for dim, functions in landscape.functions.items():
                if dim not in avg_landscape:
                    # instantiate new np.array of proper length with 0s
                    avg_landscape[dim] = np.zeros_like(functions)
                # add landscape functions with existing np.array
                avg_landscape[dim] += functions
        # divide by # of landscapes
        for dim in avg_landscape:
            avg_landscape[dim] /= len(landscapes)
        # creating PersistenceLandscape object to return
        avg_pl = PersistenceLandscape(
            homology_dims=list(avg_landscape.keys()),
            num_functions=landscapes[0].num_functions,
            resolution=landscapes[0].resolution,
        )
        # adding functions to PersistenceLandscape object
        avg_pl.functions = avg_landscape
        return avg_pl

    @staticmethod
    def _subtract_landscapes(
        landscape1: PersistenceLandscape, landscape2: PersistenceLandscape
    ) -> PersistenceLandscape:
        """Subtract two landscapes for each common dimension."""
        # Initialize PersistenceLandscape to return
        diff_pl = PersistenceLandscape(landscape1.homology_dims)

        diff_functions = {}
        for dim in landscape1.homology_dims:
            diff_functions[dim] = landscape1.get_fns_for_dim(dim) - landscape2.get_fns_for_dim(dim)
        diff_pl.functions = diff_functions
        return diff_pl

    def __str__(self):
        return f"LandscapeDistance object between (1) [{self.diagram1}] and (2) [{self.diagram2}]"


class ImageDistance(TopologicalDistance):

    def __init__(
        self, diagram1, diagram2, norm=2, bandwidth=1.0, weight=lambda x: 1, resolution=[20, 20]
    ) -> None:
        super().__init__(diagram1, diagram2, norm)
        self.bandwidth = bandwidth
        self.weight = weight
        self.resolution = resolution
        self.image_transformer = gd.representations.PersistenceImage(
            bandwidth=bandwidth, weight=weight, resolution=resolution
        )

    def supports_distribution(self) -> bool:
        """Indicates support for distributions of persistence images."""
        return True

    def fit(self, **kwargs):
        """Compute and return average persistence images for both diagrams."""
        raise NotImplementedError

    def transform(self, descriptor1, descriptor2) -> float:
        """Compute the norm-based distance between two persistence images."""
        raise NotImplementedError

    def _convert_to_image(self, diagrams: List[PersistenceDiagram]) -> List[PersistenceImage]:
        """Convert each persistence diagram in the given list into to a persistence image in the returned list."""
        images = []

        for diagram in diagrams:
            img = PersistenceImage(
                bandwidth=self.bandwidth, weight=self.weight, resolution=self.resolution
            )
            pixels = {}
            for dim, points in diagram.persistence_pts.items():
                transformed_points = self.image_transformer.fit_transform([points])[0]
                print(type(transformed_points))
                pixels[dim] = transformed_points
            img.pixels = pixels
            images.append(img)
        return images

    def __str__(self):
        pass


class SilhouetteDistance(TopologicalDistance):
    pass


supported_distances = {"landscape": LandscapeDistance, "image": ImageDistance}
