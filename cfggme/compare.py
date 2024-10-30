import numpy as np
import networkx as nx
from cfggme.kilt import KILT
from cfggme.topology.distances import TopologicalDistance, supported_distances
from cfggme.topology.ph import GraphHomology
from typing import List, Tuple, Optional, Union, Dict


class Comparator:
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
        self.kilt = KILT(measure, weight, alpha, prob_fn)
        self.ph = GraphHomology(homology_dims, measure)

        self.descriptor1 = None
        self.descriptor2 = None

    def curvature_filtration(self, G):
        graph_iterable = self._format_inputs(G)
        return [self._kilterator(g) for g in graph_iterable]

    def fit(self, G1, G2, metric="landscape", **kwargs) -> None:
        """Initializes a Topological Distance object and computes the topological descriptors that will be compared."""
        self.distance = self._setup_distance(metric)

        pd_1 = self.curvature_filtration(G1)
        pd_2 = self.curvature_filtration(G2)

        self.distance(
            pd_1,
            pd_2,
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

    def _setup_distance(self, metric) -> TopologicalDistance:

        # Check that the metric is supported
        assert (
            metric in supported_distances
        ), f"Metric {metric} is not supported."

        return supported_distances[metric]

    def _kilterator(self, graph):
        return self.kilt.fit_transform(graph, self.homology_dims)

    def _format_inputs(self, G):
        if self._is_distribution(G):
            return G
        elif self._is_graph(G):
            return [G]
        else:
            raise ValueError(
                "Input must be a networkx.Graph or a list of networkx.Graphs"
            )

    def _is_distribution(self, G):
        return isinstance(G, list)

    def _is_graph(self, G):
        return isinstance(G, nx.Graph)
