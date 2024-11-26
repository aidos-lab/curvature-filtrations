import gudhi as gd
import numpy as np
from typing import List
from scott.topology.distances.base import TopologicalDistance
from scott.topology.representations import PersistenceDiagram, PersistenceImage


class ImageDistance(TopologicalDistance):
    """
    Takes in persistence diagrams, converts them to persistence images, and computes the distance between them.
    This TopologicalDistance subclass supports comparison of distributions (i.e. input of lists of persistence diagrams).

    Attributes
    ----------
    diagram 1 : PersistenceDiagram or List[PersistenceDiagram]
        The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram2.
    diagram 2 : PersistenceDiagram or List[PersistenceDiagram]
        The PersistenceDiagram or List[PersistenceDiagram] to be compared with diagram1.
    norm : int, default=2.
        Defines what norm will be used for calculations. Default is 2.
    bandwidth : double, default = 1.0
        Controls the Gaussian kernel for the probability distribution calculated for each point on the birth/persistence diagram.
        See gudhi documentation for more information.
    weight : function, default = lambda x: 1
        Defines the weight function used to compute the weighted sum of the probability distributions for each point on the bith/persistence diagram, i.e. the persistence surface.
        Default is a constant function. Other common choices are a linear function or a bump function, which put greater emphasis on points with longer persistence.
        See gudhi documentation for more information.
    resolution : List[int, int], default = [20,20]
        The dimensions of the persistence image in pixels.
    image_transformer : gudhi package object
        The object that powers the transformation of persistence diagrams into persistence images.

    Methods
    supports_distribution() -> True :
        Indicates that ImageDistance supports comparison between distributions.
    fit() -> PersistenceImage, PersistenceImage:
        Converts persistence diagrams into persistence images.
        If diagram1 and/or diagram2 are distributions, the average persistence image for each distribution is returned.
    transform(image1 : PersistenceImage, image2 : PersistenceImage) -> float :
        Takes in two persistence images and computes the distance between them.
    fit_transform() -> float :
        Runs fit() to create two persistence images and then transform() to calculate the distance between them.

    """

    def __init__(
        self,
        diagram1,
        diagram2,
        norm=2,
        bandwidth=1.0,
        weight=lambda x: 1,
        resolution=[20, 20],
    ) -> None:
        super().__init__(diagram1, diagram2, norm)
        self.bandwidth = bandwidth
        self.weight = weight
        self.resolution = resolution
        self.image_transformer = gd.representations.PersistenceImage(
            bandwidth=bandwidth, weight=weight, resolution=resolution
        )

    def supports_distribution(self) -> bool:
        """Indicates support for distributions of persistence images."""
        return True

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Core Methods: Fit & Transform                            │
    #  ╰──────────────────────────────────────────────────────────╯

    def fit(self, **kwargs):
        """Computes and returns persistence images from the persistence diagrams, diagram1 and diagram2 (or the average persistence images, if diagram1 and/or diagram2 are distributions.

        Returns
        -------
        avg1 : PersistenceImage
            A wrapper object for the (average) persistence landscape for diagram1.
            Attribute called 'pixels' stores a Dict[int, np.array] that a maps homology dimension key to a np.array that stores image data.

        avg2 : PersistenceImage
            A wrapper object for the (average) persistence landscape for diagram1.
            Attribute called 'pixels' stores a Dict[int, np.array] that a maps homology dimension key to a np.array that stores image data.
        """

        """Compute and return average persistence images for both diagrams."""
        img1 = self._convert_to_image(self.diagram1)
        img2 = self._convert_to_image(self.diagram2)

        avg1 = self._average_image(img1)
        avg2 = self._average_image(img2)
        return avg1, avg2

    def transform(self, image1, image2) -> float:
        """Computes the norm-based distance between two persistence images.
        Cannot be executed before fit() method has been run to generate persistence images.

        Parameters
        ----------
        image1 : PersistenceImage
            A wrapper object for the first persistence image to be compared.
            Attribute called 'pixels' stores a Dict[int, np.array] that a maps homology dimension key to a np.array with shape (resolution[0] * resolution[1], 1), which contains the values of the persistence surface at each pixel.
        image2 : PersistenceImage
            A wrapper object for the second persistence image to be compared.
            Attribute called 'pixels' stores a Dict[int, np.array] that a maps homology dimension key to a np.array with shape (resolution[0] * resolution[1], 1), which contains the values of the persistence surface at each pixel.

        Returns
        -------
        float :
            The norm-based distance between the two given persistence landscapes.
        """
        common_dims = set(image1.homology_dims).intersection(
            image2.homology_dims
        )
        difference = self._subtract_images(image1, image2)
        distance = sum(self.norm(difference.pixels[dim]) for dim in common_dims)
        return distance

    def fit_transform(self) -> float:
        """Generates persistence images for diagram1, diagram2 and computes the distance between them.
        Returns
        -------
        float :
            The distance between diagram1 and diagram2, computed via image vectorization.
        """
        avg1, avg2 = self.fit()
        return self.transform(avg1, avg2)

    #  ╭──────────────────────────────────────────────────────────╮
    #  │ Helper Functions                                         │
    #  ╰──────────────────────────────────────────────────────────╯

    def _convert_to_image(
        self, diagrams: List[PersistenceDiagram]
    ) -> List[PersistenceImage]:
        """Convert each persistence diagram in the given list into to a persistence image in the returned list."""
        images = []

        for diagram in diagrams:
            img = PersistenceImage(
                homology_dims=diagram.homology_dims,
                bandwidth=self.bandwidth,
                weight=self.weight,
                resolution=self.resolution,
            )
            pixels = {}
            for dim, points in diagram.persistence_pts.items():
                transformed_points = self.image_transformer.fit_transform(
                    [points]
                )[0]
                pixels[dim] = transformed_points
            img.pixels = pixels
            images.append(img)
        return images

    def _average_image(
        self, images: List[PersistenceImage]
    ) -> PersistenceImage:
        """Computes the average persistence image given multiple persistence images."""
        avg_pixels = {}

        # sum up pixel values across all images
        for img in images:
            for dim, pixels in img.pixels.items():
                # if dim not yet instantiated in avg_image, create np.array of 0s with correct length
                if dim not in avg_pixels:
                    avg_pixels[dim] = np.zeros_like(pixels)
                # add pixel values to existing pixel values
                avg_pixels[dim] += pixels

        # divide pixel values by # of images
        for dim in avg_pixels:
            avg_pixels[dim] /= len(images)

        # create a PersistenceImage to return
        avg_img = PersistenceImage(
            homology_dims=list(avg_pixels.keys()),
            bandwidth=self.bandwidth,
            weight=self.weight,
            resolution=self.resolution,
        )

        # add avg. image values to avg. PersistenceImage object
        avg_img.pixels = avg_pixels

        return avg_img

    # TODO: check that weight and other parameters are the same between images
    def _subtract_images(
        self, image1: PersistenceImage, image2: PersistenceImage
    ) -> PersistenceImage:
        """Subtracts two images for each common dimension, returning a PersistenceImage that represents the difference."""
        common_dims = set(image1.homology_dims).intersection(
            image2.homology_dims
        )
        assert (
            image1.resolution == image2.resolution
        ), "Cannot subtract images with different resolutions."
        # Initialize PersistenceImage to return
        diff_pi = PersistenceImage(
            homology_dims=common_dims,
            bandwidth=self.bandwidth,
            weight=self.weight,
            resolution=self.resolution,
        )
        # subtraction
        diff_pixels = {}
        for dim in common_dims:
            diff_pixels[dim] = image1.get_pixels_for_dim(
                dim
            ) - image2.get_pixels_for_dim(dim)
        diff_pi.pixels = diff_pixels
        return diff_pi

    def __str__(self):
        """Return user-friendly string representation of the ImageDistance instance."""
        return f"ImageDistance object between (1) [{self.diagram1}] and (2) [{self.diagram2}]"
