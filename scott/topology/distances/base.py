from abc import ABC, abstractmethod
import numpy as np


class TopologicalDistance(ABC):
    """Abstract base class for computing topological distances.

    Takes in two persistence diagrams (or lists of persistence diagrams) and handles necessary transformations
    to compute a specific distance function.

    Attributes
    ----------
    diagram 1 : PersistenceDiagram or List[PersistenceDiagram]
        The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram2.
    diagram 2 : PersistenceDiagram or List[PersistenceDiagram]
        The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram1.
    norm : int, default=2.
        Defines what norm will be used for calculations. Default is 2.

    Core Methods
    -------
    norm(x) -> float:
        Returns the norm of vector x according to the order specified in the norm attribute.
    _is_distribution(diagram) -> bool:
        Returns True if input is a list (i.e. distribution), False otherwise.
    supports_distribution() -> bool:
        Returns True if the distance type supports comparing distributions of persistence diagrams.
    fit -> avg1, avg2 :
        Converts diagram1 and diagram2 into average topological descriptors.
    transform(descriptor1, descriptor2) -> float:
        Computes the distance between inputted topological descriptors and returns it as a float.
    """

    def __init__(self, diagram1, diagram2, norm=2) -> None:
        super().__init__()
        self.diagram1 = diagram1
        self.diagram2 = diagram2
        self.norm_order = norm

        # Ensure support for distributions if provided
        if self._is_distribution(diagram1) or self._is_distribution(diagram2):
            assert (
                self.supports_distribution()
            ), "Distribution of persistence diagrams is not supported by this distance type."

    def norm(self, x):
        """Compute norm of a vector x according to the specified order."""
        return np.linalg.norm(x, ord=self.norm_order)

    @staticmethod
    def _is_distribution(diagram):
        """Check if the given diagram is a distribution (i.e., a list of diagrams)."""
        return isinstance(diagram, list)

    @abstractmethod
    def supports_distribution(self) -> bool:
        """Indicates if the distance type supports distributions of persistence diagrams."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, **kwargs):
        """Convert persistence diagrams to topological descriptors."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, descriptor1, descriptor2) -> float:
        """Defines how to compute distances between topological descriptors."""
        raise NotImplementedError
