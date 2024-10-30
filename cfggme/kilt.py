import networkx as nx
import numpy as np
import gudhi as gd
from typing import List, Tuple, Optional, Union


import cfggme.geometry.measures as measures
from cfggme.topology.ph import GraphHomology


class KILT:
    """Krvature-Informed Links via Topology (KILT) is a class that computes the topological distance between two graphs using curvature as a filtration function."""

    def __init__(
        self, measure="forman_curvature", weight=None, alpha=0.0, prob_fn=None
    ) -> None:
        """Defines the specifications for the desired measure of computing curvature in a graph."""

        # Check that curvature method is supported
        assert method in [
            "forman_curvature",
            "ollivier_ricci_curvature",
            "resistance_curvature",
        ]
        self.measure = measure

        self.weight = weight
        self.alpha = alpha
        self.prob_fn = prob_fn

    def __str__(self) -> str:
        """Return a string representation of the Curvature and any custom attributes."""
        name = f"Measure: {self.measure}"
        if self.weight != None:
            name += f"\nCustom Weight Attribute: {self.weight}"
        if self.alpha != 0.0:
            name += f"\nCustom Alpha: {self.alpha}"
        if self.prob_fn != None:
            name += f"\nCustom Probability Function: {self.prob_fn}"
        return name

    ## Geometric Section (i.e. computing curvature)
    def fit(self, graph) -> np.array:
        """Computes curvature values for the given graph according to the specifications of the Curvature object."""
        # Search through our measures to find the one specified by user
        curvature_fn = getattr(measures, self.measure)
        # Check that function is callable
        # assert callable(curvature_fn(graph))

        if self.measure == "ollivier_ricci_curvature":
            # Ollivier Ricci measure supports extra inputs
            return curvature_fn(graph, self.alpha, self.weight, self.prob_fn)
        else:
            # Forman and Resistance measures only require graph and optional weight
            return curvature_fn(graph, self.weight)

    def transform(self, graph, curvature_values) -> nx.Graph:
        """Assigns the values of the given curvature_values np.array to the respective edges of the given graph."""
        nx.set_edge_attributes(graph, curvature_values, name="curvature")
        return graph

    def fit_transform(self, graph) -> nx.Graph:
        """Computes the curvature values for the given graph according to the specifications of the Curvature object,
        and assigns them to their respective edges."""
        curvature_values = self.fit(graph)
        fitted_graph = self.transform(graph, curvature_values)
        return fitted_graph
        # streamlined: return self.transform(graph, self.fit(graph))

    ## Topological Section (outcome is persistence diagram)
    def curvature_filtration(
        self,
        graph,
        homology_dims=[0, 1],
    ) -> List[List[Tuple[float, float]]]:
        """Use the curvature values of the given graph to create a topological descriptor i.e. persistence diagram."""

        # NOTE: Do we want to refit here? What if our users are working with very large graphs? E.g. some physician referral networks. We should think about which objects we want to store and which we want to recompute and then stay consistent.
        curvature = self.fit(graph)
        curvature = {e: c for e, c in zip(graph.edges(), curvature)}
        nx.set_edge_attributes(graph, curvature, self.measure)

        # We return the simplest possible descriptor (diagrams) and allow for different comparisons
        ph = GraphHomology(homology_dims, self.measure)

        return ph.calculate_persistent_homology(graph)
