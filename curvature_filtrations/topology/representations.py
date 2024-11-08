from typing import List, Tuple, Optional, Dict
import gudhi as gd
import gudhi.representations
import gudhi.wasserstein
import networkx as nx
import numpy as np


class PersistenceDiagram:
    """A wrapper object for the data housed in a persistence diagram for the specified homology dimensions.

    Attributes
    ----------
    homology_dims : List[int]
        Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1).
        Default is [0, 1].

    _persistence_points : None or Dict[int, np.array]
        A dictionary that maps the homology dimension to a np.array of its persistence pairs.
        Each np.array contains tuples of (birth, death) values for each persistence pair.
        Note that the attribute homology_dims must be a subset of the list of keys (hom. dims.) in this dictionary.
        Initialized to None, set using setter method.

    Methods
    -------
    persistence_points:
        Getter (self -> Dict[int, np.array]) and setter (self, Dict[int, np.array] -> None) for attribute self._persistence_pts, the dictionary that .

    get_pts_for_dim(self, dimension):
        Getter method for the np.array of persistence points for the given homology dimension.
    """

    def __init__(self, homology_dims=[0, 1]):
        """Initializes an instance of the PersistenceDiagram class."""
        self.homology_dims = homology_dims

        # Initialize empty persistence diagram
        self._persistence_pts = None

    @property
    def persistence_pts(self) -> Dict[int, np.array]:
        """Get the PersistenceDiagram's dictionary of persistence points.
        Returns
        -------
        Dict[int,np.array]
            A dictionary that maps np.arrays of persistence point tuples (values) to each homology dimenion (key).
            Will return None if persistence_pts have not yet been set.
        """
        return self._persistence_pts

    @persistence_pts.setter
    def persistence_pts(self, points: Dict[int, np.array]) -> None:
        """Set the PersistenceDiagram's dictionary of persistence points.
        Parameters
        ----------
        points: Dict[int, np.array]
            A dictionary that maps np.arrays of persistence point tuples (values) to each homology dimenion (key)
        """
        assert type(points) == dict
        self._persistence_pts = points

    def get_pts_for_dim(self, dimension: int) -> np.array:
        # Returns a list of birth, death pairs for the dimension
        assert (
            self.persistence_pts != None
        ), "Persistence points have not been added to the PersistenceDiagram object"
        return self.persistence_pts[dimension]
