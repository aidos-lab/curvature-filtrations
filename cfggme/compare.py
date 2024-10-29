import numpy as np
from cfggme.curvature import Curvature
import cfggme.topology as topology
from abc import ABC, abstractmethod
from typing import Dict


class Comparator(ABC):

    def __init__(
        self, method="forman_curvature", weight=None, alpha=0.0, prob_fn=None
    ) -> None:

        # intialize curvature object with input parameters
        self.curv = Curvature(method, weight, alpha, prob_fn)
        print(
            "Curvature will be calculated according to the following specifications: "
            + self.curv
        )
        self.top_descriptor1 = None
        self.top_descriptor2 = None

    # def execute_filtration():
    #     pass

    # def compute_persistent_homology():
    #     pass

    def __graph_to_persistence_landscape(self, graph) -> Dict[int, np.array]:
        return self.curv.make_landscape(graph)

    @abstractmethod
    def fit(
        self,
    ) -> (
        None
    ):  # TODO: do we want to return the two persistence landscapes?? would be [Dict[int, np.array], Dict[int, np.array]]
        """Translate to persistence landscape or average of persistence landscapes

        Here our fit method needs to assign:
        self.top_descriptor1
        self.top_descriptor2
        """
        pass

    def transform(self) -> float:
        """Returns a distance! Yay! Done!"""
        assert (
            self.top_descriptor1
        ), self.top_descriptor2  # check that these are not None

        return np.linalg.norm(np.array(landscape1) - np.array(landscape2))

    def fit_transform(self) -> float:

        self.fit()
        return self.transform()

    class GraphComparator:
        """Class for comparing the curvature between two graphs."""

        def __init__(
            self,
            graph1,
            graph2,
            method="forman_curvature",
            weight=None,
            alpha=0.0,
            prob_fn=None,
        ) -> None:
            """Initializes GraphComparator object."""
            super().__init__(method, weight, alpha, prob_fn)
            # TODO: check whether input is a list of graphs or single graph -> suggest DistributionComparator if necessary
            self.graph1 = graph1
            self.graph2 = graph2

        # distance between persistance landscapes
        def curvature_filtration_distance(
            self,
        ):

            # TODO: do I need to put defaults here??
            """Calculates the curvature filtration distance between two graphs. (Note: Not two DISTRIBUTIONS of graphs)."""
            landscape1 = self.curv.make_landscape(self.graph1)
            landscape2 = self.curv.make_landscape(self.graph2)

            return np.linalg.norm(np.array(landscape1) - np.array(landscape2))

    class DistributionComparator:
        """Class for comparing the curvature distribution between two lists of graphs"""

        def __init__(
            self,
            graphs1,
            graphs2,
            method="forman_curvature",
            weight=None,
            alpha=0.0,
            prob_fn=None,
        ) -> None:
            """Initializes DistributionComparator object."""
            super().__init__(method, weight, alpha, prob_fn)
            # TODO: check whether input is a list of graphs or single graph -> suggest GraphComparator if necessary
            self.graphs1 = graphs1
            self.graphs2 = graphs2

        def fit():
            """Return a landscape from a distribution of graphs. First make curvature filtration, then landscape for each graph, then return average landscape."""
            pass

        def make_landscapes(self):
            landscapes1 = []
            landscapes2 = []
            for graph in self.graphs1:
                landscapes1.append(self.__distance_between_landscapes(graph))
            for graph in self.graphs2:
                landscapes2.append(self.__distance_between_landscapes(graph))
            return landscapes1, landscapes2  # TODO: think about this

        def average_landscape(landscapes, exact=True):
            assert exact, "Only exact landsape computations supported at the moment."

            sum_ = landscapes[0]
            N = len(landscapes)
            for landscape in landscapes[1:]:
                sum_ += landscape
            return sum_.__truediv__(N)
