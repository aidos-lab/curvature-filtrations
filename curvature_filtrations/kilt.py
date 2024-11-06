import networkx as nx
import numpy as np
import gudhi as gd
from typing import List, Tuple, Optional, Dict


import curvature_filtrations.geometry.measures as measures
from curvature_filtrations.topology.ph import (
    GraphHomology,
)

CURVATURE_MEASURES = [
    "forman_curvature",
    "ollivier_ricci_curvature",
    "resistance_curvature",
]  # methods for calculating curvature that KILT currently supports


class KILT:
    """Krvature-Informed Links and Topology (KILT) is a class that faciltates computing discrete curvature values for graphs.
     Curvature values can be used as a filtration for persistent homology (as done in https://openreview.net/forum?id=Dt71xKyabn), providing even more expressive descriptors for the structure of a graph.

    Attributes
    ----------
    measure : string, default= "foreman_curvature"
        The type of curvature measure to be calculated. See CURVATURE_MEASURES for the methods that KILT functionality currently supports.

    alpha : float, default=0.0
        Only used if Olivier--Ricci curvature is calculated, with default 0.0. Provides laziness parameter for default probability measure. The
        measure is not compatible with a user-defined `prob_fn`. If such
        a function is set, `alpha` will be ignored.

    weight : str or None, default=None
        Can be specified for (weighted) Forman--Ricci, Olivier--Ricci and Resistance curvature measures. Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated. Notice that for Ollivier--Ricci curvature, if `prob_fn` is provided, this parameter will have no effect for the calculation of probability measures, but it will be used for the calculation of shortest-path distances.

    prob_fn : callable or None, default=None
        Used on if Ollivier--Ricci curvature is calculated, with default None. If set, should refer to a function that calculate a probability measure for a given graph and a given node. This callable needs to satisfy the following signature:

        ``prob_fn(G, node, node_to_index)``

        Here, `G` refers to the graph, `node` to the node whose measure is to be calculated, and `node_to_index` to the lookup map that maps a node identifier to a zero-based index.

        If `prob_fn` is set, providing `alpha` will not have an effect.

    Methods
    -------
    G:
        Getter (self -> nx.Graph) and setter (self, nx.Graph -> None) for attribute self._G, the graph associated with the KILT object.

    curvature:
        Getter (self -> np.array) and setter (self, np.array -> None) for np.array of the graph's curvature values.

    fit(self, graph: nx.Graph) -> None
        Calculates the curvature values for the edges of the given graph according to the specifications of the KILT object, which can be retrieved from the curvature property.

    transform(self, homology_dims: Optional[List[int]] = [0, 1],) -> Dict[int, np.array]:
        Executes a curvature filtration for the given homology dimensions. Can only be called after fit() is performed. Returns a persistence diagram stored in dictionary format.

    fit_transform(self, graph, homology_dims: Optional[List[int]] = [0, 1]) -> Dict[int, np.array]:
        Combines fit and transform methods; i.e. calculates curvature and executes a filtration for the given homology dimensions. Returns a persistence diagram stored in dictionary format.
    """

    def __init__(
        self,
        measure="forman_curvature",
        weight=None,
        alpha=0.0,
        prob_fn=None,
    ) -> None:
        """Creates an instance of the KILT class with the desired specifications for the method of computing curvature in a graph."""

        # Check that curvature method is supported
        assert (
            measure in CURVATURE_MEASURES
        ), "The given curvature measure is not yet supported by KILT."

        # Assign curvature attributes
        self.measure = measure
        self.weight = weight
        self.alpha = alpha
        self.prob_fn = prob_fn

        # Initialize graph
        self._G = None

    def __str__(self) -> str:
        """Return a string representation of the KILT object and any custom attributes."""
        name = f"KILT Object with Measure: {self.measure}, Graph: {self.G}"
        if self.weight != None:
            name += f", Custom Weight Attribute: {self.weight}"
        if self.alpha != 0.0:
            name += f", Custom Alpha: {self.alpha}"
        if self.prob_fn != None:
            name += f", Custom Probability Function: {self.prob_fn}"
        return name

    def __repr__(self) -> str:
        """Return a string representation of the KILT object."""
        return f"KILT({self.measure}, {self.weight}, {self.alpha}, {self.prob_fn})"

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Getters and Setters                                      │
    #  ╰──────────────────────────────────────────────────────────╯

    @property
    def G(self) -> nx.Graph:
        """Get the graph associated with the KILT object.
        Returns
        -------
        nx.Graph
            The graph assigned to the KILT object.
        """
        return self._G

    @G.setter
    def G(self, graph) -> None:
        """Set the given graph as the KILT object's graph attribute.
        Parameters
        ----------
        graph : nx.Graph
            The graph to be assigned to the KILT object.
        """
        self._G = graph.copy()

    @property
    def curvature(self) -> np.array:
        """Return the curvature values of the graph.
        Returns
        -------
        np.array
            The curvature values for the edges of the graph associated with the KILT object.
        """
        if self.G is None:
            return None
        return self._unpack_curvature_values(
            nx.get_edge_attributes(self.G, name=self.measure)
        )

    @curvature.setter
    def curvature(self, values) -> None:
        """Set the curvature values of the graph.
        Parameters
        ----------
        values : list
            The graph to be assigned to the KILT object.
        """
        edge_map = self._pack_curvature_values(self.G.edges, values)
        nx.set_edge_attributes(self.G, edge_map, name=self.measure)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Geometry                                                 │
    #  ╰──────────────────────────────────────────────────────────╯

    def fit(self, graph) -> None:
        """
        Calculates the curvature values for the edges of the given graph according to the specifications of the KILT object, which can then be retrieved via the curvature attribute.

        Parameters
        ----------
        graph : networkx.Graph
            Input graph for which curvature values will be calculated.

        Returns
        -------
        None
        """
        self._graph_is_not_empty(graph)
        self.G = graph
        edge_curvatures = self._compute_curvature(graph)
        self.curvature = edge_curvatures

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Topology                                                 │
    #  ╰──────────────────────────────────────────────────────────╯

    def transform(
        self,
        homology_dims: Optional[List[int]] = [
            0,
            1,
        ],
    ) -> Dict[int, np.array]:
        """
        Executes a curvature filtration for the given homology dimensions.

        Parameters
        ----------
        homology_dims : List[int]
            A list of the homology dimensions for which persistence points should be calculated, with default being [0,1].

        Returns
        -------
        Dict[int, np.array]
            A persistence diagram.
                Keys: Integers that correspond to homology dimensions.
                Values: np.arrays that contain all persistence pairs in the form of [birth, death] tuples.
        """
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
        homology_dims: Optional[List[int]] = [
            0,
            1,
        ],
    ) -> Dict[int, np.array]:
        """Computes the curvature values for the given graph according to the specifications of the KILT object and executes a filtration for the given homology dimensions.

        Parameters
        ----------
        graph : networkx.Graph
            Input graph for which curvature values will be calculated.
        homology_dims : List[int]
            A list of the homology dimensions for which persistence points should be calculated, with default being [0,1].

        Returns
        -------
        Dict[int, np.array]
            A persistence diagram.
                Keys: Integers that correspond to homology dimensions.
                Values: np.arrays that contain all persistence pairs in the form of [birth, death] tuples.
        """
        self.fit(graph)
        return self.transform(homology_dims)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Helper Functions                                         │
    #  ╰──────────────────────────────────────────────────────────╯

    def _compute_curvature(self, graph) -> np.array:
        """Helper function for calculating the edge curvature values for the given graph according to the specifications of the KILT object (i.e. measure, weight, etc.)."""
        curvature_fn = getattr(measures, self.measure)
        if self.measure == "ollivier_ricci_curvature":
            # Ollivier Ricci measure supports extra inputs
            return curvature_fn(
                graph,
                self.alpha,
                self.weight,
                self.prob_fn,
            )
        else:
            # Forman and Resistance measures only require graph and optional weight
            return curvature_fn(graph, self.weight)

    def _curvature_values_exist(self) -> bool:
        """Checks whether curvature values have already been computed and stored in the KILT object's curvature attribute."""
        return self.curvature is not None and len(self.curvature) > 0

    def _graph_is_not_empty(self, graph) -> None:
        """Asserts that the inputted graph is not empty (i.e. has nodes and edges)."""
        assert len(graph.nodes) > 0, "Graph must have nodes to compute curvature"

    @staticmethod
    def _unpack_curvature_values(
        dict,
    ) -> np.array:
        """Converts the curvature values from a dictionary to a numpy array."""
        return np.array(list(dict.values()))

    @staticmethod
    def _pack_curvature_values(edges, values) -> dict:
        """Converts corresponding lists of edges and curvature values into a dictionary mapping edges to their curvature value."""
        return dict(zip(edges, values))
