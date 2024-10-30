from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import gudhi as gd


class TopologicalDistance(ABC):
    """Abstract class for computing topological distances.

    Takes in two persistence diagrams and handles necessary transformations to be able to use a specific distance function.
    """

    def __init__(self, diagram1, diagram2, norm) -> None:
        super().__init__()
        self.diagram1 = diagram1
        self.diagram2 = diagram2

        if self._is_distribution(diagram1) or self._is_distribution(diagram2):
            assert (
                self.distrubution_support()
            ), "Distribution of persistence diagrams is not supported."

        self.norm = norm

    def norm(self, x):
        return np.linalg.norm(x, ord=self.norm)

    @abstractmethod
    def distrubution_support() -> bool:
        """Check if the Class supports distributions of persistence diagrams."""
        raise NotImplementedError

    @abstractmethod
    def fit(self) -> None:
        """Translate to persistence landscape or average of persistence landscapes.

        This method assigns:
        self.top_descriptor1
        self.top_descriptor2
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, descriptor1, descriptor2) -> float:
        """This method defines how to compute distances between topological descriptors."""
        raise NotImplementedError

    @staticmethod
    def _is_distribution(D):
        return isinstance(D, list)


class LandscapeDistance(TopologicalDistance):
    """Class for computing distances between persistence landscapes."""

    def __init__(self, diagram1, diagram2, resolution=1000) -> None:
        super().__init__(diagram1, diagram2)
        self.resolution = resolution
        self.landscape = gd.representations.Landscape(resolution=resolution)

    def distrubution_support() -> bool:
        return True

    def fit(self):
        """Fit persistence landscapes for both diagrams and aggregate if they are distributions."""

        landscapes1 = self._pd_to_landscape(self.diagram1)
        landscapes2 = self._pd_to_landscape(self.diagram2)
        return self._average_landscape(landscapes1), self._average_landscape(
            landscapes2
        )

    def transform(
        self, landscape1: Dict[int, np.array], landscape2: Dict[int, np.array]
    ) -> float:
        """Compute Distance Between Landscapes."""
        distance = 0.0
        common_dims = set(landscape1.keys()).intersection(landscape2.keys())

        diff = self._subtract_landscapes(landscape1, landscape2)
        for dim in common_dims:
            distance += self.norm(diff[dim])
        return distance

    def _pd_to_landscape(
        self, diagrams: List[Dict[int, np.array]]
    ) -> List[Dict[int, np.array]]:
        """Convert each persistence diagram into a persistence landscape for each dimension."""
        landscapes = []
        for diagram in diagrams:
            landscape = {}
            for dim, points in diagram.items():
                landscape[dim] = self.landscape.fit_transform([points])[0]
            landscapes.append(landscape)
        return landscapes

    @staticmethod
    def _average_landscape(L: List[Dict[int, np.array]]) -> Dict[int, np.array]:
        """Compute the average persistence landscape of a list of landscapes."""
        avg = {}
        for landscape in L:
            for dim, values in landscape.items():
                if dim not in avg:
                    avg[dim] = np.zeros_like(values)
                avg[dim] += values
        for dim in avg:
            avg[dim] /= len(L)
        return avg

    @staticmethod
    def _subtract_landscapes(
        landscapeX: Dict[int, np.array], landscapeY: Dict[int, np.array]
    ) -> Dict[int, np.array]:
        res = dict()
        for i in landscapeX.keys():
            res[i] = landscapeX[i] - landscapeY[i]
        return res


# TODO: Implement other distances
class ImageDistance(TopologicalDistance):
    pass


class SilhouetteDistance(TopologicalDistance):
    pass


supported_distances = {
    "landscape": LandscapeDistance,
}
