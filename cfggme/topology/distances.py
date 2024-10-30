from abc import ABC, abstractmethod


class TopologicalDistance(ABC):
    """Abstract class for computing topological distances.


    Takes in two persistence diagrams and handles necessary transformations to be able to use a specific distance function.
    """

    def __init__(self, diagram1, diagram2) -> None:
        super().__init__()
        self.diagram1 = diagram1
        self.diagram2 = diagram2

    @abstractmethod
    def summarize_distribution():
        """Summarize the distribution of persistence diagrams, e.g. average, median, etc."""
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
    ) -> (
        None
    ):  # TODO: do we want to return the two persistence landscapes?? would be [Dict[int, np.array], Dict[int, np.array]]
        """Translate to persistence landscape or average of persistence landscapes

        Here our fit method needs to assign:
        self.top_descriptor1
        self.top_descriptor2
        """
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, descriptor1, descriptor2) -> float:
        """This method defines the how you compute distances between topological descriptors.

        We can easily support different topological representations e.g.
        * persistence landscapes
        * persistence images
        * persistence silhouettes



        #TODO: we can support a few different options and provide a tutorial for how to implement a `CustomComparator` class with a custom distance function.

        #NOTE: We can also support multiple distances for DistributionComparator, as long as there is a well defined notion of average. Let's do this at least for 1 other descriptor as well as landscapes
        """
        raise NotImplementedError


class LandscapeDistance(TopologicalDistance):
    pass


class ImageDistance(TopologicalDistance):
    pass


class SilhouetteDistance(TopologicalDistance):
    pass
