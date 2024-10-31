import networkx as nx
import numpy as np
import gudhi as gd
from typing import List, Tuple, Optional


import curvature_filtrations.geometry.measures as measures
from curvature_filtrations.topology.ph import GraphHomology


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

        self._G = None

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

    def __repr__(self) -> str:
        """Return a string repreåsentation of the Curvature object."""
        return (
            f"KILT({self.measure}, {self.weight}, {self.alpha}, {self.prob_fn})"
        )

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Getters and Setters                                      │
    #  ╰──────────────────────────────────────────────────────────╯

    @property
    def G(self) -> nx.Graph:
        """Return the graph associated with the KILT object."""
        return self._G

    @G.setter
    def G(self, graph) -> None:
        """Copy a user-provided graph to the KILT object."""
        self._G = graph.copy()

    @property
    def curvature(self) -> np.array:
        """Return the curvature values of the graph."""
        if self.G is None:
            return None
        return self._unpack_curvature_values(
            nx.get_edge_attributes(self.G, name=self.measure)
        )

    @curvature.setter
    def curvature(self, values):
        """Set the curvature values of the graph."""
        edge_map = self._pack_curvature_values(self.G.edges, values)
        nx.set_edge_attributes(self.G, edge_map, name=self.measure)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Geometry                                                 │
    #  ╰──────────────────────────────────────────────────────────╯

    def fit(self, graph) -> None:
        """Computes curvature values for the given graph according to the specifications of the Curvature object."""
        self._graph_is_not_empty(graph)
        self.G = graph
        edge_curvatures = self._compute_curvature(graph)
        self.curvature = edge_curvatures

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Topology                                                 │
    #  ╰──────────────────────────────────────────────────────────╯

    def transform(
        self,
        homology_dims: Optional[List[int]] = [0, 1],
    ) -> List[List[Tuple[float, float]]]:  # TODO: clean up types

        ph = GraphHomology(homology_dims, self.measure)
        assert (
            self._curvature_values_exist()
        ), "Curvature values have not been computed. Please run `fit` first."
        return ph.calculate_persistent_homology(self.G)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Both: A Curvature Filtration!                            │
    #  ╰──────────────────────────────────────────────────────────╯

    def fit_transform(
        self,
        graph,
        homology_dims: Optional[List[int]] = [0, 1],
    ) -> List[List[Tuple[float, float]]]:
        """Computes the curvature values for the given graph according to the specifications of the Curvature object,
        and assigns them to their respective edges."""
        self.fit(graph)
        return self.transform(homology_dims)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Helper Functions                                         │
    #  ╰──────────────────────────────────────────────────────────╯

    def _compute_curvature(self, graph) -> np.array:
        curvature_fn = getattr(measures, self.measure)
        if self.measure == "ollivier_ricci_curvature":
            # Ollivier Ricci measure supports extra inputs
            return curvature_fn(graph, self.alpha, self.weight, self.prob_fn)
        else:
            # Forman and Resistance measures only require graph and optional weight
            return curvature_fn(graph, self.weight)

    def _curvature_values_exist(self) -> bool:
        return self.curvature is not None and len(self.curvature) > 0

    def _graph_is_not_empty(self, graph) -> None:
        assert (
            len(graph.nodes) > 0
        ), "Graph must have nodes to compute curvature"

    @staticmethod
    def _unpack_curvature_values(dict) -> np.array:
        return np.array(list(dict.values()))

    @staticmethod
    def _pack_curvature_values(edges, values) -> dict:
        return dict(zip(edges, values))
