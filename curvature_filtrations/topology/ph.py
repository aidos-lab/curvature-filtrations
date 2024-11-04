import collections
import multiprocessing
from typing import List, Tuple, Optional, Dict

import gudhi as gd
import gudhi.representations
import gudhi.wasserstein
import networkx as nx
import numpy as np
from joblib import Parallel, delayed


class GraphHomology:
    """
    Compute persistent homology on graphs by filtering over edge attributes.

    This class uses Gudhi's `SimplexTree` to build a clique complex on a given
    graph and compute persistence diagrams for homology groups, using an edge
    attribute (e.g. curvature) as a filtration function.
    """

    def __init__(
        self,
        homology_dims: Optional[List[int]] = None,
        filter_attribute: str = "curvature",  # TODO: maybe we pass specific attribute names so there can be multiple curvatures for a single graph?
    ):
        """
        Initializes the GraphHomology class with parameters for homology computation.

        Parameters
        ----------
        homology_dims : list of int, optional
            Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1).
            Default is [0, 1].
        filter_attribute : str, optional
            The edge attribute to use as the filtration value. Default is "curvature".
        """
        self.homology_dims = (
            homology_dims if homology_dims is not None else [0, 1]
        )
        self.filter_attribute = filter_attribute
        self.max_dimension = max(self.homology_dims)

    def calculate_persistent_homology(
        self, G: nx.Graph, extended_persistence: bool = False
    ) -> Dict[int, List[Tuple[float, float]]]:
        """
        Calculates persistent homology of the graph's clique complex.

        Parameters
        ----------
        G : networkx.Graph
            Input graph with edges that contain attributes for filtration.

        Returns
        -------
        List of lists of tuples
            Persistence diagrams for each dimension up to `max_dimension`.
            Each list contains tuples of (birth, death) values for each persistence pair.

        Raises
        ------
        ValueError
            If the specified `filter_attribute` is not found in an edge's attributes.
        """
        simplex_tree = self._build_simplex_tree(G)
        simplex_tree.make_filtration_non_decreasing()
        simplex_tree.expansion(self.max_dimension)
        if extended_persistence:
            simplex_tree.extended_persistence()
        else:
            simplex_tree.persistence(persistence_dim_max=True)

        diagrams = self._format_persistence_diagrams(simplex_tree)
        return diagrams

    def _build_simplex_tree(self, G: nx.Graph) -> gd.SimplexTree:
        """
        Builds a simplex tree from the graph's edges and edge attributes.

        Parameters
        ----------
        G : networkx.Graph
            Input graph where edges contain the `filter_attribute` to use as filtration values.

        Returns
        -------
        gudhi.SimplexTree
            Simplex tree built from the graph, using specified edge attribute as filtration.

        Raises
        ------
        ValueError
            If the specified `filter_attribute` is not found in an edge's attributes.
        """
        st = gd.SimplexTree()

        for u, v, w in G.edges(data=True):
            if self.filter_attribute in w:
                weight = w[self.filter_attribute]
                st.insert([u, v], filtration=weight)
            else:
                raise ValueError(
                    f"Edge ({u}, {v}) is missing required attribute '{self.filter_attribute}'."
                )

        return st

    def _format_persistence_diagrams(self, simplex_tree: gd.SimplexTree):
        """
        Formats persistence pairs into diagrams for each specified homology dimension.

        Parameters
        ----------
        persistence_pairs : list of tuples
            Persistence pairs returned by Gudhi's persistence computation. Each tuple
            is of the form (dimension, (birth, death)).

        Returns
        -------
        List of lists of tuples
            A list of persistence diagrams, each corresponding to a homology dimension. Each
            inner list contains (birth, death) tuples for persistence pairs in that dimension.
        """
        diagrams = {}
        for dim in self.homology_dims:
            persistence_pairs = self._mask_infinities(
                simplex_tree.persistence_intervals_in_dimension(dim)
            )
            diagrams[dim] = persistence_pairs
        return diagrams

    @staticmethod
    def _mask_infinities(array):
        return array[array[:, 1] < np.inf]
