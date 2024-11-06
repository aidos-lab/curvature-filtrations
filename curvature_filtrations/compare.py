import numpy as np
import networkx as nx
from curvature_filtrations.kilt import KILT
from curvature_filtrations.topology.distances import (
    TopologicalDistance,
    supported_distances,
)
from curvature_filtrations.topology.ph import GraphHomology
from typing import List, Tuple, Optional, Union, Dict


class Comparator:
    """Compare Graphs or Graph Distributions using Curvature Filtrations as in https://openreview.net/forum?id=Dt71xKyabn using `KILT`.


    TODO: Add more details about the class here and brief description of how we combine geometry and topology, along with normal documentation. Include high level description of Subclasses.
    """

    def __init__(
        self,
        measure="forman_curvature",
        weight=None,
        alpha=0.0,
        prob_fn=None,
        homology_dims: Optional[List[int]] = [0, 1],
    ) -> None:

        # Initialize Curvature and GraphHomology objects
        self.kilt = KILT(measure, weight, alpha, prob_fn)
        self.ph = GraphHomology(homology_dims, measure)

        self.descriptor1 = None
        self.descriptor2 = None

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Key Members: Fit and Transform                           │
    #  ╰──────────────────────────────────────────────────────────╯

    def fit(self, G1, G2, metric="landscape", **kwargs) -> None:
        """Initializes a Topological Distance object and computes the topological descriptors that will be compared."""
        cls = self._setup_distance(metric)

        pd_1 = self._curvature_filtration(G1)
        pd_2 = self._curvature_filtration(G2)

        self.distance = cls(
            diagram1=pd_1,
            diagram2=pd_2,
            **kwargs,
        )  # This should error if theres no distribution support

        self.descriptor1, self.descriptor2 = self.distance.fit(**kwargs)

    def transform(self) -> float:
        """Returns a distance! Yay! Done!"""
        assert (
            self.descriptor1 and self.descriptor2
        ), "You have not fitted topological descriptors. Try running `fit` or `fit_transform`"  # check that these are not None

        return self.distance.transform(self.descriptor1, self.descriptor2)

    def fit_transform(self, G1, G2, metric="landscape") -> float:

        self.fit(G1, G2, metric)
        return self.transform()

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Helper Functions                                         │
    #  ╰──────────────────────────────────────────────────────────╯

    def _setup_distance(self, metric) -> TopologicalDistance:

        # Check that the metric is supported
        assert metric in supported_distances, f"Metric {metric} is not supported."

        return supported_distances[metric]

    def _curvature_filtration(self, G):
        graph_iterable = self._format_inputs(G)
        return [self._kilterator(g) for g in graph_iterable]

    def _kilterator(self, graph):
        return self.kilt.fit_transform(graph, self.ph.homology_dims)

    @staticmethod
    def _format_inputs(G):
        if Comparator._is_distribution(G):
            return G
        elif Comparator._is_graph(G):
            return [G]
        else:
            raise ValueError(
                "Input must be a networkx.Graph or a list of networkx.Graphs"
            )

    @staticmethod
    def _is_distribution(G):
        return isinstance(G, list)

    @staticmethod
    def _is_graph(G):
        return isinstance(G, nx.Graph)

    def __str__(self) -> str:
        """Return a user-friendly string representation of the Comparator object."""
        name = f"Comparator object with: \n\tCurvature method defined by: [{self.kilt}]\n\tDescriptor 1: {self.descriptor1}\n\tDescriptor 2: {self.descriptor2}"
        return name

    def __repr__(self) -> str:
        """Return a string representation of the Comparator object helpful to the developer."""
        return f"Comparator({self.kilt}, {self.descriptor1}, {self.descriptor2}, {self.ph})"
