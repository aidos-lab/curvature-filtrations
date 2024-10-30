import networkx as nx
import numpy as np
import gudhi as gd
from typing import List, Tuple, Optional


import cfggme.geometry.measures as measures
from cfggme.topology.ph import GraphHomology


class KILT:
    """Krvature-Informed Links and Topology (KILT) is a class that faciltates computing discrete curvature values for graphs. Curvature values can be used as a filtration for persistent homology (as done in https://openreview.net/forum?id=Dt71xKyabn), providing even more expressive descriptors for the structure of a graph."""

    def __init__(
        self,
        measure="forman_curvature",
        weight=None,
        alpha=0.0,
        prob_fn=None,
    ) -> None:
        """Defines the specifications for the desired measure of computing curvature in a graph."""

        # Check that curvature method is supported
        assert measure in [
            "forman_curvature",
            "ollivier_ricci_curvature",
            "resistance_curvature",
        ]
        self.measure = measure
        self.weight = weight
        self.alpha = alpha
        self.prob_fn = prob_fn

        self.G = None

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

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Geometry                                                 │
    #  ╰──────────────────────────────────────────────────────────╯

    def _compute_curvature(self, graph) -> np.array:
        curvature_fn = getattr(measures, self.measure)
        if self.measure == "ollivier_ricci_curvature":
            # Ollivier Ricci measure supports extra inputs
            return curvature_fn(graph, self.alpha, self.weight, self.prob_fn)
        else:
            # Forman and Resistance measures only require graph and optional weight
            return curvature_fn(graph, self.weight)

    def fit(self, graph) -> None:
        """Computes curvature values for the given graph according to the specifications of the Curvature object."""
        # Search through our measures to find the one specified by user
        edge_curvatures = self._compute_curvature(graph)
        self.G = nx.set_edge_attributes(
            graph.copy(), edge_curvatures, name=self.measure
        )

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Topology                                                 │
    #  ╰──────────────────────────────────────────────────────────╯
    def transform(
        self,
        graph,
        homology_dims: Optional[List[int]] = [0, 1],
    ) -> List[List[Tuple[float, float]]]:  # TODO: clean up types
        ph = GraphHomology(homology_dims, self.measure)
        return ph.calculate_persistent_homology(graph)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Both: Curvature Filtration!                              │
    #  ╰──────────────────────────────────────────────────────────╯
    def fit_transform(
        self, graph, homology_dims
    ) -> List[List[Tuple[float, float]]]:
        """Computes the curvature values for the given graph according to the specifications of the Curvature object,
        and assigns them to their respective edges."""
        if self.G is None:
            self.fit(graph)

        return self.transform(self.G, homology_dims)
