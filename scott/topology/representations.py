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
        Determines the number of samples for the piece-wise landscape functions (i.e. how many points are calculated).
    _functions : Dict[int, np.array]
        A dictionary that maps the homology dimension to a np.array of its landscape functions.
            Each np.array contains {num_functions * resolution} floats, which are the {resolution} number of samples for all {num_functions} landscape functions concatenated together.
        Initialized to None, set using setter method.

    Methods
    -------
    functions:
        Getter (self -> Dict[int, np.array]) and setter (self, Dict[int, np.array] -> None) for attribute self._functions, which stores the landscape functions.
    get_fns_for_dim(self, dim : int) -> np.array:
        Getter method for the np.array containing the landscape functions for the given homology dimension.
    get_fn_for_dim(self, dim : int, fn_num : int) -> List[float]:
        Getter method for the {fn_num}th landscape function for the given homology dimension.
    _separate_landscape_functions(self) -> Dict[int, np.array]:
        Returns an amended function dictionary, where np.array values hold {num_functions} lists, each with {resolution} floats, as opposed to simply {num_functions * resolution} floats.
        Thus the landscape functions are stored into separate lists within the np.array, rather than being concatenated.
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
        for dim in self.homology_dims:
            assert (
                len(functions[dim]) == self.num_functions * self.resolution
            ), f"The length of the landscape function samples ({len(functions[dim])}) is not the product of the specified resolution ({self.resolution}) and number of landscape functions ({self.num_functions})."
        self._functions = functions

    def get_fns_for_dim(self, dimension: int) -> np.array:
        """Returns a np.array of concatenated landscape functions for the specified dimension."""
        assert (
            self._functions != None
        ), "Landscape functions have not yet been added to the PersistenceLandscape object."
        return self.functions[dimension]

    def get_fn_for_dim(self, dimension: int, fn_num: int) -> np.array:
        """Returns a list of points for landscape function associated with the given dimension and function number."""
        assert (
            self._functions != None
        ), "Landscape functions have not yet been added to the PersistenceLandscape object."
        start_idx = fn_num * self.resolution
        indices = range(start_idx, start_idx + self.resolution)
        return self.functions[dimension][indices]

    def _separate_landscape_functions(self) -> Dict[int, np.array]:
        """Returns an amended function dictionary.
        Keys are still homology dimensions, but np.array values hold {num_functions} lists, each with {resolution} floats, as opposed to simply {num_functions * resolution} floats.
        Thus the landscape functions are stored into separate lists within the np.array, rather than being concatenated.
        """
        function_list_by_dim = {}
        for dim in self.homology_dims:
            function_list = []
            for fn_num in range(0, self.num_functions):
                start_idx = fn_num * self.resolution
                indices = range(start_idx, start_idx + self.resolution)
                fn = self.functions[dim][indices]
                function_list.append(fn)
            function_list_by_dim[dim] = function_list
        return function_list_by_dim

    def __str__(self):
        """Returns a string representation of the PersistenceLandscape object."""
        name = f"This is a PersistenceLandscape object with the following attributes: Homology Dimensions ({self.homology_dims}), Number of Landscape Functions per Dimension ({self.num_functions}), resolution i.e. pts/landscape function ({self.resolution})"
        return name


class PersistenceImage:
    """
    A wrapper object for the data housed in the image vectorization of a persistence diagram.

    Attributes
    ----------
    homology_dims : List[int], default=[0, 1]
        Dimensions of the homology groups to compute (e.g., [0, 1] for H_0 and H_1).
    bandwidth : double, default = 1.0
        Controls the Gaussian kernel for the probability distribution calculated for each point on the birth/persistence diagram.
        See gudhi documentation for more information.
    weight : function, default = lambda x: 1
        Defines the weight function used to compute the weighted sum of the probability distributions for each point on the bith/persistence diagram, i.e. the persistence surface.
        Default is a constant function. Other common choices are a linear function or a bump function, which put greater emphasis on points with longer persistence.
        See gudhi documentation for more information.
    resolution : List[int, int], default = [20,20]
        The dimensions of the persistence image in pixels.
    _pixels : Dict[int, np.array]
        A dictionary that maps each homology dimension to its persistence image in the form of a np.array.
            Each np.array has shape (resolution[0] * resolution[1], 1), and contains the values of the persistence surface at each pixel.
        Initialized to None, set using setter method.


    Methods
    -------
    pixels:
        Getter (self -> Dict[int, np.array]) and setter (self, Dict[int, np.array] -> None) for attribute self._pixels, which stores the persistence image data.
    get_img_for_dim(self, dimension : int) -> np.array:
        Returns the np.array value for the given homology dimension in the pixels dictionary.
        This np.array has shape (resolution[0] * resolution[1], 1) and stores the values of the persistence surface at each pixel (for the given homology dimension).
    """

    def __init__(
        self,
        homology_dims=[0, 1],
        bandwidth=1.0,
        weight=lambda x: 1,
        resolution=[20, 20],
    ):
        """Initializes an object of the PersistenceImage class."""
        self.homology_dims = homology_dims
        self.bandwidth = bandwidth
        self.weight = weight
        self.resolution = resolution

        # Initialize image to None
        self._pixels = None

    @property
    def pixels(self) -> Dict[int, np.array]:
        """Getter for the dictionary of persistence images."""
        return self._pixels

    @pixels.setter
    def pixels(self, pixels: Dict[int, np.array]) -> None:
        """Setter for the dictionary of persistence images."""
        assert type(pixels) == dict
        # Assert length matches resolution
        for dim in self.homology_dims:
            assert len(pixels[dim]) == self.resolution[0] * self.resolution[1]
        self._pixels = pixels

    def get_pixels_for_dim(self, dimension: int) -> np.array:
        """Returns a np.array representing the persistence image for the specified dimension."""
        assert (
            self._pixels != None
        ), "Persistence image pixel values have not yet been added to the PersistenceImage object."
        return self.pixels[dimension]

    def __str__(self):
        """Returns a string representation of the Image object."""
        return f"PersistenceImage object with the following attributes: Homology Dims ({self.homology_dims}), Bandwidth ({self.bandwidth}), Weight ({self.weight}), Resolution ({self.resolution})"
