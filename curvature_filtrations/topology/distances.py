from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List
import gudhi as gd
from curvature_filtrations.topology.representations import PersistenceDiagram, PersistenceLandscape


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
            functions = {}
            for dim, points in diagram.persistence_pts.items():
                # TODO: check this... H1 is not responding to changes in PD input
                transformed_points = self.landscape_transformer.fit_transform([points])[0]
                landscape_fns = []
                for fn_num in range(0, self.num_functions):
                    start_idx = int(fn_num * self.resolution)
                    indices = range(start_idx, start_idx + 1000)
                    fn = transformed_points[indices]
                    landscape_fns.append(fn)
                functions[dim] = landscape_fns
            # set to Persistence Landscape
            landscape.functions = functions
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
                    total_functions = np.concatenate(functions)
                    # instantiate np.array of proper length with 0s
                    avg_landscape[dim] = np.zeros_like(total_functions)
                # add data
                avg_landscape[dim] += total_functions
        # divide by # of landscapes
        for dim in avg_landscape:
            avg_landscape[dim] /= len(landscapes)

        # taking parameters from 1st persistence landscape in input
        avg_pl = PersistenceLandscape(
            list(avg_landscape.keys()),
            num_functions=landscapes[0].num_functions,
            resolution=landscapes[0].resolution,
        )
        # TODO: use concatenate method
        avg_fns = {}
        for dim in avg_pl.homology_dims:
            landscape_fns = []
            for fn_num in range(0, avg_pl.num_functions):
                start_idx = int(fn_num * avg_pl.resolution)
                indices = range(start_idx, start_idx + avg_pl.resolution)
                fn = avg_landscape[dim][indices]
                landscape_fns.append(fn)
            avg_fns[dim] = landscape_fns
        avg_pl.functions = avg_fns
        return avg_pl

    @staticmethod
    def _subtract_landscapes(
        landscape1: PersistenceLandscape, landscape2: PersistenceLandscape
    ) -> PersistenceLandscape:
        """Subtract two landscapes for each common dimension."""
        diff_pl = PersistenceLandscape(landscape1.homology_dims)
        diff_functions = {}
        for dim in landscape1.homology_dims:
            landscape1_data = np.concatenate(landscape1.get_fns_for_dim(dim))
            landscape2_data = np.concatenate(landscape2.get_fns_for_dim(dim))
            diff_functions[dim] = landscape1_data - landscape2_data
        diff_pl.functions = diff_functions
        return diff_pl

    @staticmethod
    def _convert_to_function_lists(
        concatenated_functions: Dict[int, np.array], num_functions, resolution
    ) -> Dict[int, np.array]:
        function_list_by_dim = {}
        for dim in concatenated_functions.keys():
            function_list = []
            for fn_num in range(0, num_functions):
                start_idx = fn_num * resolution
                indices = range(start_idx, start_idx + resolution)
                fn = concatenated_functions[dim][indices]
                function_list.append(fn)
            function_list_by_dim[dim] = function_list
        return function_list_by_dim

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
