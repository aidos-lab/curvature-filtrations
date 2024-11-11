from typing import List, Tuple, Optional, Dict
import gudhi as gd
import gudhi.representations
import gudhi.wasserstein
import networkx as nx
import numpy as np


class PersistenceDiagram:
    """A wrapper object for the data housed in a persistence diagram for the specified homology dimensions.
    Main attribute (persistence_points) is a Dict[int, np.array] that maps a homology dimension key to a np.array that contains (birth,death) tuples for all the persistence pairs.

    Attributes
    ----------
    homology_dims : List[int], default=[0, 1]
            Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1).
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
        # Returns a np.array of birth, death pairs for the dimension
        assert (
            self.persistence_pts != None
        ), "Persistence points have not been added to the PersistenceDiagram object"
        return self.persistence_pts[dimension]

    def __str__(self) -> str:
        name = "This is a PersistenceDiagram object with the following (birth, death) pairs: \n\t"
        for dim in self.homology_dims:
            name += "H" + str(dim) + ":"
            for pair in self.get_pts_for_dim(dim):
                name += f"({pair[0]}, {pair[1]})"
            name += "\n\t"
        return name


class PersistenceLandscape:
    """
    A wrapper object for the data housed in a persistence landscape for the specified homology dimensions.

    Attributes
    ----------
    homology_dims : List[int], default=[0, 1]
        Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1).
    num_functions : int, default=5
        The number of landscape functions to compute for each homology dimension.
        These are usually denoted lambda_1, lambda_2, ..., lambda_{num_functions}.
        Recall that for each successive landscape function, the values of the highest peak function at each x-value is removed from consideration.
    resolution: int, default = 1000
        Determines how finely the landscape functions are defined over the range (i.e. how many points are calculated).
    _functions : Dict[int, np.array]
        A dictionary that maps the homology dimension to a np.array of its landscape functions.
            Each np.array contains {num_functions} lists, one for each landscape function.
            Each of these lists has length equal to {resolution}, and contains the points of the landscape function along the range.
        Initialized to None, set using setter method.

    Methods
    -------
    functions:
        Getter (self -> Dict[int, np.array]) and setter (self, Dict[int, np.array] -> None) for attribute self._functions, which stores the landscape functions.
    get_fns_for_dim(self, dim : int) -> np.array:
        Getter method for the np.array containing the landscape functions for the given homology dimension.
    get_fn_for_dim(self, dim : int, fn_num : int) -> List[float]:
        Getter method for the {fn_num}th landscape function for the given homology dimension.
    """

    def __init__(self, homology_dims=[0, 1], num_functions=5, resolution=1000):
        """Initializes an object of the PersistenceDiagram class."""
        self.homology_dims = homology_dims
        self.num_functions = num_functions
        self.resolution = resolution

        # Initialize landscape functions to None
        self._functions = None

    @property
    def functions(self) -> Dict[int, np.array]:
        """Getter for the complete dictionary of landscape functions."""
        return self._functions

    @functions.setter
    def functions(self, functions: Dict[int, np.array]) -> None:
        """Setter for the complete dictionary of landscape functions."""
        assert type(functions) == dict
        self._functions = functions

    def get_fns_for_dim(self, dimension: int) -> np.array:
        # Returns a np.array of all landscape functions for the dimension
        assert (
            self._functions != None
        ), "Landscape functions have not yet been added to the PersistenceLandscape object."
        return self.functions[dimension]

    def get_fn_for_dim(self, dimension: int, fn_num: int) -> np.array:
        # Returns a list of points for landscape function associated with the given dimension and function number.
        assert (
            self._functions != None
        ), "Landscape functions have not yet been added to the PersistenceLandscape object."
        return self.functions[dimension][fn_num - 1]

    def __str__(self):
        # TODO: Implement meaningful string representation
        name = f"This is a PersistenceLandscape object with the following attributes: Homology Dimensions ({self.homology_dims}), Number of Landscape Functions per Dimension ({self.num_functions}), resolution i.e. pts/landscape function ({self.resolution})"
        return name
