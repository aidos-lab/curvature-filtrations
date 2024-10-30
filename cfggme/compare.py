import numpy as np
import networkx as nx
from cfggme.geometry.curvature import Curvature
from cfggme.topology.distances import TopologicalDistance
from cfggme.topology.ph import GraphHomology
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict


class Comparator(ABC):
    """Compare Graphs or Graph Distributions using Curvature Filtrations as in https://openreview.net/forum?id=Dt71xKyabn.


    TODO: Add more details about the class here and brief description of how we combine geometry and topology, along with normal documentation. Include high level description of Subclasses.
    """

    def __init__(
        self,
        measure="forman_curvature",
        weight=None,
        alpha=0.0,
        prob_fn=None,
        homology_dims: Optional[List[int]] = None,
    ) -> None:

        # Initialize Curvature and GraphHomology objects
        self.curv = Curvature(measure, weight, alpha, prob_fn)
        self.ph = GraphHomology(homology_dims, measure)

        self.descriptor1 = None
        self.descriptor2 = None

    @abstractmethod
    def curvature(self, G1, G2):
        """Geometry: How our comparators should compute curvature."""
        raise NotImplementedError(
            "This method must be implemented in a subclass."
        )

    @abstractmethod
    def filtration(self, G1, G2):
        """Topology: Compute Filtrations."""
        raise NotImplementedError(
            "This method must be implemented in a subclass."
        )

    @abstractmethod
    def configure_distance(
        self, metric="landscape_distance"
    ) -> TopologicalDistance:
        """Distance: Compute distance between topological descriptors."""
        raise NotImplementedError(
            "This method must be implemented in a subclass."
        )

    def fit(self, G1, G2, metric="landscape_distance", **kwargs) -> None:
        """Initializes a Topological Distance object and computes the topological descriptors that will be compared."""

        self.curvature(G1, G2)
        self.filtration(G1, G2)
        self.distance = self.configure_distance(metric)

        # Figure out whether I am a distribution or a single graph and then configure the Distance object accordingly

        pass

    def transform(self) -> float:
        """Returns a distance! Yay! Done!"""
        assert (
            self.descriptor1 and self.descriptor2
        ), "You have not fitted topological descriptors. Try running `fit` or `fit_transform`"  # check that these are not None

        return self.distance.transform(self.descriptor1, self.descriptor2)

    def fit_transform(self, metric, **kwargs) -> float:

        self.fit()
        return self.transform()


class DistributionComparator(Comparator):
    """Class for comparing the curvature distribution between two lists of graphs"""

    def __init__(
        self,
        graphs1: List[nx.Graph],
        graphs2: List[nx.Graph],
        method: str = "forman_curvature",
        weight=None,
        alpha=0.0,
        prob_fn=None,
    ) -> None:
        """Initializes DistributionComparator object."""
        super().__init__(method, weight, alpha, prob_fn)
        # TODO: check whether input is a list of graphs or single graph -> suggest GraphComparator if necessary
        self.graphs1 = graphs1
        self.graphs2 = graphs2

        self.curvatures1 = []
        self.curvatures2 = []

        self.diagrams1 = []
        self.diagrams2 = []

    def curvature(self, G1, G2):
        # TODO: parallelize this
        self.graphs1 = [self._curvature_iterator(G) for G in G1]
        self.graphs2 = [self._curvature_iterator(G) for G in G2]

    def filtration(self, G1, G2):
        """Topology: Compute Filtrations."""
        for G in G1:
            self.diagrams1.append(self.ph.calculate_persistent_homology(G))
        for G in G2:
            self.diagrams2.append(self.ph.calculate_persistent_homology(G))

    def configure_distance(
        self, metric="landscape_distance"
    ) -> TopologicalDistance:
        """Distance: Compute distance between topological descriptors."""
        raise NotImplementedError(
            "This method must be implemented in a subclass."
        )

    def _curvature_iterator(self, graph):
        curvature = self.curv.curvature(graph)
        nx.set_edge_attributes(graph, curvature, self.curv.measure)
        return graph


class GraphComparator(Comparator):
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

    def curvature(self, G1, G2):
        """Geometry: How our comparators should compute curvature."""
        raise NotImplementedError(
            "This method must be implemented in a subclass."
        )

    def filtration(self, G1, G2):
        """Topology: Compute Filtrations."""
        raise NotImplementedError(
            "This method must be implemented in a subclass."
        )

    def configure_distance(
        self, metric="landscape_distance"
    ) -> TopologicalDistance:
        """Distance: Compute distance between topological descriptors."""
        raise NotImplementedError(
            "This method must be implemented in a subclass."
        )
